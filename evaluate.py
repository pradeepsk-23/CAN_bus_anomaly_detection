#!/usr/bin/env python3
"""
evaluate.py — Evaluation and reporting pipeline

Usage
-----
    python evaluate.py
    python evaluate.py --config configs/config.yaml
    python evaluate.py --calibrate   # re-calibrate threshold on val set
    python evaluate.py --explain     # run signal correlation / root-cause analysis
    python evaluate.py --no-gpu

Data contract
-------------
Uses the same load_data() helper as train.py → returns (df_ambient, df_attacks).
Test set is always df_attacks (ground-truth labels from Label column).
Val set is the held-out 20% of df_ambient used for threshold calibration.

Outputs (artifacts/reports/)
-----------------------------
    metrics.json
    pr_curve.png
    roc_curve.png
    score_distribution.png
    confusion_matrix.png
    per_type_f1.png
    root_cause_summary.csv     (requires --explain)
    dashboard.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from src.correlation.signal_correlator import SignalCorrelator
from src.data.can_simulator import CANBusSimulator
from src.data.dataset import (
    SignalNormaliser,
    VehicleDatasetBuilder,
    extract_windows,
    load_road_dataset,
)
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_per_type_f1,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
    plot_training_history,
)
from src.models.ensemble import EnsembleAnomalyDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vehicle Anomaly Detection — Evaluation")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--calibrate",  action="store_true")
    p.add_argument("--explain",    action="store_true")
    p.add_argument("--no-gpu",     action="store_true")
    return p.parse_args()


def load_data(cfg):
    """Same as train.py — returns (df_ambient, df_attacks)."""
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        syn = cfg["data"]["synthetic"]
        sim = CANBusSimulator(seed=cfg["project"]["seed"])
        df_ambient = sim.generate_ambient(
            n_vehicles=syn["n_ambient_vehicles"],
            duration_s=syn["duration_ambient_s"],
            sampling_hz=syn["sampling_hz"],
            noise_std=syn["noise_std"],
        )
        df_attacks = sim.generate_attacks(
            n_vehicles=syn["n_attack_vehicles"],
            duration_s=syn["duration_attack_s"],
            sampling_hz=syn["sampling_hz"],
            anomaly_ratio=syn["anomaly_ratio"],
            noise_std=syn["noise_std"],
        )
        return df_ambient, df_attacks
    elif mode == "road":
        return load_road_dataset(cfg["data"]["road_dataset_path"])
    raise ValueError(f"Unknown data mode: {mode}")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    setup_logger(cfg["paths"]["logs"])
    device = torch.device("cpu") if args.no_gpu or not torch.cuda.is_available() else torch.device("cuda")
    report_dir = Path(cfg["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint or cfg["paths"]["checkpoints"]

    channel_cols = cfg["data"]["channels"]
    window_size  = cfg["data"]["window"]["size"]
    stride       = cfg["data"]["window"]["stride"]
    val_frac     = cfg["data"]["val_split"]

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Loading data (mode={})...", cfg["data"]["mode"])
    df_ambient, df_attacks = load_data(cfg)

    # ── Re-create splits using saved normaliser ────────────────────────
    normaliser = SignalNormaliser.load(str(Path(ckpt_dir) / "normaliser.pkl"))

    N_amb     = len(df_ambient)
    val_start = int(N_amb * (1.0 - val_frac))
    df_val_raw  = df_ambient.iloc[val_start:].copy()
    df_test_raw = df_attacks.copy()

    for split_df in [df_val_raw, df_test_raw]:
        split_df[channel_cols] = normaliser.transform(
            split_df[channel_cols].values.astype("float32")
        )

    X_val,  y_val,  _          = extract_windows(df_val_raw,  channel_cols, window_size, stride)
    X_test, y_test, types_test = extract_windows(df_test_raw, channel_cols, window_size, stride)

    logger.info("Evaluation windows — val={:,}  test={:,}", len(X_val), len(X_test))

    # ── Load ensemble ──────────────────────────────────────────────────
    lstm_ae  = LSTMAutoencoder.from_config(cfg, seq_len=window_size)
    ensemble = EnsembleAnomalyDetector.load(ckpt_dir, lstm_ae, cfg, device)

    if args.calibrate:
        logger.info("Re-calibrating threshold on validation set...")
        ensemble.calibrate_threshold(X_val, y_val,
                                     strategy=cfg["evaluation"]["threshold_strategy"])
        ensemble.save(ckpt_dir)

    # -- Score test set
    if len(X_test) == 0:
        logger.error("Test set is empty — nothing to evaluate.")
        return

    logger.info("Scoring {:,} test windows...", len(X_test))
    scores = ensemble.score(X_test)
    threshold = ensemble.threshold

    # -- Compute metrics
    metrics = compute_metrics(y_test, scores, threshold, anom_types=types_test)

    serialisable = {k: v for k, v in metrics.items()}
    with open(report_dir / "metrics.json", "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info("Metrics saved → {}", report_dir / "metrics.json")

    # -- Generate plots
    y_pred = (scores >= threshold).astype(int)
    plot_pr_curve(y_test, scores, threshold, save_path=str(report_dir / "pr_curve.png"))
    plot_roc_curve(y_test, scores,           save_path=str(report_dir / "roc_curve.png"))
    plot_score_distribution(scores, y_test, threshold,
                            save_path=str(report_dir / "score_distribution.png"))
    plot_confusion_matrix(y_test, y_pred,   save_path=str(report_dir / "confusion_matrix.png"))
    if metrics.get("per_type_f1"):
        plot_per_type_f1(metrics["per_type_f1"],
                         save_path=str(report_dir / "per_type_f1.png"))

    # -- Signal correlation (root-cause analysis)
    if args.explain:
        anom_mask = y_test == 1
        X_anom    = X_test[anom_mask][:200]
        s_anom    = scores[anom_mask][:200]
        if len(X_anom) == 0:
            logger.warning("No anomalous windows — skipping root-cause analysis.")
        else:
            residuals = ensemble.explain(X_anom)["channel_residuals"]
            correlator = SignalCorrelator(cfg)
            root_causes = correlator.analyse(
                windows=X_anom, residuals=residuals, scores=s_anom,
                channel_names=channel_cols,
                sampling_hz=cfg["data"]["synthetic"].get("sampling_hz", 10),
            )
            summary_df = correlator.summarise(root_causes)
            summary_df.to_csv(report_dir / "root_cause_summary.csv", index=False)
            logger.info("Root-cause summary → {}", report_dir / "root_cause_summary.csv")
            logger.info("Sub-system attribution:\n{}",
                        summary_df["primary_subsystem"].value_counts().to_string())

    # -- Final summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("  Test windows  : {:,}", len(X_test))
    logger.info("  Anomaly rate  : {:.2%}", y_test.mean())
    logger.info("  Precision     : {:.3f}", metrics["precision"])
    logger.info("  Recall        : {:.3f}", metrics["recall"])
    logger.info("  F1            : {:.3f}", metrics["f1"])
    logger.info("  ROC-AUC       : {:.3f}", metrics["roc_auc"])
    logger.info("  Avg Precision : {:.3f}", metrics["average_precision"])
    logger.info("  Threshold     : {:.4f}", metrics["threshold"])
    logger.info("  Reports in    : {}", report_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
