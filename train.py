#!/usr/bin/env python3
"""
train.py — Main training pipeline

Usage:
    python train.py                          # synthetic data, default config
    python train.py --config configs/config.yaml
    python train.py --data-mode road         # real ROAD dataset (download first)
    python train.py --epochs 50 --lr 5e-4

This script:
  1. Loads / generates data
  2. Builds the sliding-window dataset with temporal splits
  3. Trains the LSTM Autoencoder (unsupervised, normal windows only)
  4. Trains the Isolation Forest on the same normal windows
  5. Calibrates the ensemble threshold on the validation set
  6. Saves artefacts: model weights, normaliser, config snapshot, training plots
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from src.data.can_simulator import CANBusSimulator
from src.data.dataset import VehicleDatasetBuilder, load_road_dataset
from src.evaluation.metrics import plot_training_history
from src.models.ensemble import EnsembleAnomalyDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.training.trainer import LSTMAETrainer
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vehicle Anomaly Detection — Training Pipeline")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data-mode", choices=["synthetic", "road", "custom"], default=None,
                   help="Override data.mode from config")
    p.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    p.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-gpu", action="store_true", help="Force CPU even if CUDA available")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.data_mode:
        cfg["data"]["mode"] = args.data_mode
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.seed:
        cfg["project"]["seed"] = args.seed
    return cfg


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(no_gpu: bool = False) -> torch.device:
    if not no_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: {}", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def load_data(cfg: dict) -> "pd.DataFrame":
    mode = cfg["data"]["mode"]
    syn_cfg = cfg["data"]["synthetic"]

    if mode == "synthetic":
        logger.info("Generating synthetic CAN-bus fleet data...")
        sim = CANBusSimulator(seed=cfg["project"]["seed"])
        df, events = sim.generate_fleet(
            n_vehicles=syn_cfg["n_vehicles"],
            duration_seconds=syn_cfg["duration_seconds"],
            sampling_hz=syn_cfg["sampling_hz"],
            anomaly_ratio=syn_cfg["anomaly_ratio"],
        )
        return df

    elif mode == "road":
        logger.info("Loading ROAD dataset from {}", cfg["data"]["road_dataset_path"])
        return load_road_dataset(cfg["data"]["road_dataset_path"], cfg["data"]["channels"])

    else:
        raise ValueError(f"Unsupported data mode: {mode}. Implement a custom loader.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # -- Setup
    setup_logger(cfg["paths"]["logs"])
    seed_everything(cfg["project"]["seed"])
    device = get_device(args.no_gpu)
    t_start = time.time()

    # -- Ensure output dirs exist
    for key in ["checkpoints", "reports", "data_cache"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    # -- Data
    logger.info("=== STEP 1 / 5: DATA PREPARATION ===")
    df = load_data(cfg)
    channel_cols = cfg["data"]["channels"]
    logger.info("Dataset shape: {}", df.shape)

    builder = VehicleDatasetBuilder(cfg)
    splits = builder.build(df, channel_cols)
    X_train, y_train, types_train = splits["train"]
    X_val,   y_val,   types_val   = splits["val"]
    X_test,  y_test,  types_test  = splits["test"]

    logger.info("Split sizes — train={:,} val={:,} test={:,}", len(X_train), len(X_val), len(X_test))
    logger.info("Anomaly rates — train={:.2%} val={:.2%} test={:.2%}",
                y_train.mean(), y_val.mean(), y_test.mean())

    # Save normaliser
    normaliser_path = Path(cfg["paths"]["checkpoints"]) / "normaliser.pkl"
    builder.save_normaliser(str(normaliser_path))

    # -- LSTM Autoencoder
    logger.info("=== STEP 2 / 5: TRAIN LSTM AUTOENCODER ===")
    window_size = cfg["data"]["window"]["size"]
    lstm_ae = LSTMAutoencoder.from_config(cfg, seq_len=window_size)

    trainer = LSTMAETrainer(
        model=lstm_ae,
        cfg=cfg,
        device=device,
        checkpoint_dir=cfg["paths"]["checkpoints"],
        tensorboard_dir=cfg["paths"]["tensorboard"],
    )
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Load best weights
    best_ckpt = Path(cfg["paths"]["checkpoints"]) / "best_model.pt"
    ckpt = torch.load(best_ckpt, map_location=device)
    lstm_ae.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded best LSTM-AE checkpoint (epoch={}, val_loss={:.6f})",
                ckpt["epoch"], ckpt["val_loss"])

    # -- Isolation Forest
    logger.info("=== STEP 3 / 5: TRAIN ISOLATION FOREST ===")
    normal_mask = y_train == 0
    if_detector = IsolationForestDetector(cfg)
    if_detector.fit(X_train[normal_mask])
    if_detector.save(str(Path(cfg["paths"]["checkpoints"]) / "isolation_forest.joblib"))

    # -- Ensemble + calibration
    logger.info("=== STEP 4 / 5: ENSEMBLE CALIBRATION ===")
    ensemble = EnsembleAnomalyDetector(lstm_ae, if_detector, cfg, device)
    threshold = ensemble.calibrate_threshold(
        X_val, y_val, strategy=cfg["evaluation"]["threshold_strategy"]
    )
    ensemble.save(cfg["paths"]["checkpoints"])

    # -- Save training plots
    logger.info("=== STEP 5 / 5: SAVE ARTEFACTS ===")
    plot_training_history(
        history,
        save_path=str(Path(cfg["paths"]["reports"]) / "training_history.png"),
    )

    # Save config snapshot for reproducibility
    snapshot_path = Path(cfg["paths"]["checkpoints"]) / "config_snapshot.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    elapsed = time.time() - t_start
    logger.info("Training pipeline complete in {:.1f}s ({:.1f}min)", elapsed, elapsed / 60)
    logger.info("Run `python evaluate.py` to generate full evaluation report.")


if __name__ == "__main__":
    main()
