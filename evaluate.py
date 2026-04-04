#!/usr/bin/env python3
"""
evaluate.py — Evaluation and reporting pipeline

Usage:
    python evaluate.py                           # uses default config + latest checkpoint
    python evaluate.py --checkpoint artifacts/checkpoints/
    python evaluate.py --calibrate               # re-calibrate threshold on val set
    python evaluate.py --explain                 # run signal correlation analysis

Generates:
    artifacts/reports/
        metrics.json              — all metrics in machine-readable form
        pr_curve.png
        roc_curve.png
        score_distribution.png
        confusion_matrix.png
        per_type_f1.png
        root_cause_summary.csv    — if --explain flag set
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
from src.data.dataset import SignalNormaliser, VehicleDatasetBuilder, extract_windows
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_per_type_f1,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
)
from src.models.ensemble import EnsembleAnomalyDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vehicle Anomaly Detection — Evaluation")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint directory")
    p.add_argument("--calibrate", action="store_true", help="Re-calibrate threshold on val set")
    p.add_argument("--explain", action="store_true", help="Run signal correlator on anomalous test windows")
    p.add_argument("--no-gpu", action="store_true")
    return p.parse_args()


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
    window_size = cfg["data"]["window"]["size"]

    # -- Regenerate test data (same seed → same split)
    logger.info("Regenerating test data...")
    syn_cfg = cfg["data"]["synthetic"]

    sim = CANBusSimulator(seed=cfg["project"]["seed"])
    df, _ = sim.generate_fleet(
        n_vehicles=syn_cfg["n_vehicles"],
        duration_seconds=syn_cfg["duration_seconds"],
        sampling_hz=syn_cfg["sampling_hz"],
        anomaly_ratio=syn_cfg["anomaly_ratio"],
    )

    normaliser = SignalNormaliser.load(str(Path(ckpt_dir) / "normaliser.pkl"))
    train_frac = cfg["data"]["window"]["train_split"]
    val_frac = cfg["data"]["window"]["val_split"]
    N = len(df)
    val_end = int(N * (train_frac + val_frac))
    df_test = df.iloc[val_end:].copy()
    df_val = df.iloc[int(N * train_frac):val_end].copy()

    for split_df in [df_val, df_test]:
        split_df[channel_cols] = normaliser.transform(split_df[channel_cols].values.astype("float32"))

    X_val, y_val, types_val = extract_windows(df_val, channel_cols, window_size, cfg["data"]["window"]["stride"])
    X_test, y_test, types_test = extract_windows(df_test, channel_cols, window_size, cfg["data"]["window"]["stride"])

    # -- Load ensemble
    lstm_ae = LSTMAutoencoder.from_config(cfg, seq_len=window_size)
    ensemble = EnsembleAnomalyDetector.load(ckpt_dir, lstm_ae, cfg, device)

    if args.calibrate:
        logger.info("Re-calibrating threshold on validation set...")
        ensemble.calibrate_threshold(X_val, y_val, strategy=cfg["evaluation"]["threshold_strategy"])
        ensemble.save(ckpt_dir)

    # -- Score test set
    logger.info("Scoring test set ({:,} windows)...", len(X_test))
    scores = ensemble.score(X_test)
    threshold = ensemble.threshold

    # -- Compute metrics
    metrics = compute_metrics(y_test, scores, threshold, anom_types=types_test)

    # Save JSON
    metrics_path = report_dir / "metrics.json"
    # Ensure per_type_f1 is serialisable
    serialisable_metrics = {k: v for k, v in metrics.items() if k != "per_type_f1"}
    if "per_type_f1" in metrics:
        serialisable_metrics["per_type_f1"] = metrics["per_type_f1"]
    with open(metrics_path, "w") as f:
        json.dump(serialisable_metrics, f, indent=2)
    logger.info("Metrics saved → {}", metrics_path)

    # -- Generate plots
    y_pred = (scores >= threshold).astype(int)

    plot_pr_curve(y_test, scores, threshold, save_path=str(report_dir / "pr_curve.png"))
    plot_roc_curve(y_test, scores, save_path=str(report_dir / "roc_curve.png"))
    plot_score_distribution(scores, y_test, threshold, save_path=str(report_dir / "score_distribution.png"))
    plot_confusion_matrix(y_test, y_pred, save_path=str(report_dir / "confusion_matrix.png"))

    if "per_type_f1" in metrics and metrics["per_type_f1"]:
        plot_per_type_f1(metrics["per_type_f1"], save_path=str(report_dir / "per_type_f1.png"))

    # -- Signal correlation (root-cause analysis)
    if args.explain:
        logger.info("Running signal correlation analysis on anomalous test windows...")
        anomaly_mask = y_test == 1
        X_anom = X_test[anomaly_mask]
        scores_anom = scores[anomaly_mask]

        if len(X_anom) == 0:
            logger.warning("No anomalous windows in test set — skipping correlation analysis.")
        else:
            # Limit to first 200 anomalous windows for speed
            limit = min(200, len(X_anom))
            X_anom = X_anom[:limit]
            scores_anom = scores_anom[:limit]

            residuals = ensemble.explain(X_anom)["channel_residuals"]

            correlator = SignalCorrelator(cfg)
            root_causes = correlator.analyse(
                windows=X_anom,
                residuals=residuals,
                scores=scores_anom,
                channel_names=channel_cols,
                sampling_hz=syn_cfg["sampling_hz"],
            )
            summary_df = correlator.summarise(root_causes)
            summary_path = report_dir / "root_cause_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info("Root-cause summary saved → {}", summary_path)

            # Subsystem breakdown
            subsys_counts = summary_df["primary_subsystem"].value_counts()
            logger.info("Sub-system anomaly attribution:\n{}", subsys_counts.to_string())

    # -- Final summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("  Test windows   : {:,}", len(X_test))
    logger.info("  Anomaly rate   : {:.2%}", y_test.mean())
    logger.info("  Precision      : {:.3f}", metrics["precision"])
    logger.info("  Recall         : {:.3f}", metrics["recall"])
    logger.info("  F1             : {:.3f}", metrics["f1"])
    logger.info("  ROC-AUC        : {:.3f}", metrics["roc_auc"])
    logger.info("  Avg Precision  : {:.3f}", metrics["average_precision"])
    logger.info("  Threshold      : {:.4f}", metrics["threshold"])
    logger.info("=" * 60)
    logger.info("Reports written to: {}", report_dir)


if __name__ == "__main__":
    main()
