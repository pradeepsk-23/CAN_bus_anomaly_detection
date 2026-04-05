#!/usr/bin/env python3
"""
train.py — Main training pipeline

Usage
-----
    python train.py                          # synthetic data, default config
    python train.py --config configs/config.yaml
    python train.py --data-mode road         # real ROAD dataset
    python train.py --epochs 50 --lr 5e-4

Data contract
-------------
Both modes call:
    df_ambient, df_attacks = load_data(cfg)

Then pass both into:
    builder.build(df_ambient, df_attacks)

This means every downstream step is identical regardless of data source.

Pipeline
--------
    1. Load / generate (df_ambient, df_attacks)
    2. VehicleDatasetBuilder -> train / val / test splits
    3. Train LSTM Autoencoder   (normal windows only)
    4. Train Isolation Forest   (normal windows only)
    5. Calibrate ensemble threshold on val set
    6. Save artefacts
"""

import argparse
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
    p.add_argument("--data-mode", choices=["synthetic", "road"], default=None,
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


def load_data(cfg):
    """
    Returns (df_ambient, df_attacks) regardless of data mode.
    Schema: Signal_1_of_ID … Signal_22_of_ID | Label | anomaly_type | vehicle_id
    """
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        syn = cfg["data"]["synthetic"]
        logger.info("Generating synthetic data (mode=synthetic)...")
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
        logger.info("Loading ROAD dataset from {}", cfg["data"]["road_dataset_path"])
        return load_road_dataset(cfg["data"]["road_dataset_path"])

    else:
        raise ValueError(f"Unknown data mode: {mode}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # -- Setup
    setup_logger(cfg["paths"]["logs"])
    seed_everything(cfg["project"]["seed"])
    device = get_device(args.no_gpu)
    t0     = time.time()

    # -- Ensure output dirs exist
    for key in cfg["paths"].values():
        Path(key).mkdir(parents=True, exist_ok=True)

    # -- Data
    logger.info("=== STEP 1 / 5: DATA PREPARATION ===")
    df_ambient, df_attacks = load_data(cfg)

    channel_cols = cfg["data"]["channels"]
    builder = VehicleDatasetBuilder(cfg)
    splits  = builder.build(df_ambient, df_attacks, channel_cols)

    X_train, y_train, _           = splits["train"]
    X_val,   y_val,   _           = splits["val"]
    X_test,  y_test,  types_test  = splits["test"]

    logger.info("Windows — train={:,}  val={:,}  test={:,}", len(X_train), len(X_val), len(X_test))

    ckpt_dir = cfg["paths"]["checkpoints"]
    builder.save_normaliser(str(Path(ckpt_dir) / "normaliser.pkl"))

    # -- LSTM Autoencoder
    logger.info("=== STEP 2 / 5: TRAIN LSTM AUTOENCODER ===")
    window_size = cfg["data"]["window"]["size"]
    lstm_ae = LSTMAutoencoder.from_config(cfg, seq_len=window_size)

    trainer = LSTMAETrainer(
        model=lstm_ae, cfg=cfg, device=device,
        checkpoint_dir=ckpt_dir,
        tensorboard_dir=cfg["paths"]["tensorboard"],
    )
    history = trainer.train(X_train, y_train, X_val, y_val)

    best_ckpt = torch.load(Path(ckpt_dir) / "best_model.pt", map_location=device)
    lstm_ae.load_state_dict(best_ckpt["model_state_dict"])
    logger.info("Best checkpoint: epoch={}, val_loss={:.6f}",
                best_ckpt["epoch"], best_ckpt["val_loss"])

    # -- Isolation Forest
    logger.info("=== STEP 3 / 5: TRAIN ISOLATION FOREST ===")
    normal_mask = y_train == 0
    if_detector = IsolationForestDetector(cfg)
    if_detector.fit(X_train[normal_mask])
    if_detector.save(str(Path(ckpt_dir) / "isolation_forest.joblib"))

    # -- Ensemble + calibration
    logger.info("=== STEP 4 / 5: ENSEMBLE CALIBRATION ===")
    ensemble = EnsembleAnomalyDetector(lstm_ae, if_detector, cfg, device)
    ensemble.calibrate_threshold(
        X_val, y_val, strategy=cfg["evaluation"]["threshold_strategy"]
    )
    ensemble.save(ckpt_dir)

    # -- Save training plots
    logger.info("=== STEP 5 / 5: SAVE ARTEFACTS ===")
    plot_training_history(
        history,
        save_path=str(Path(cfg["paths"]["reports"]) / "training_history.png"),
    )
    import yaml as _yaml
    with open(Path(ckpt_dir) / "config_snapshot.yaml", "w") as f:
        _yaml.dump(cfg, f, default_flow_style=False)

    elapsed = time.time() - t0
    logger.info("Training complete in {:.1f}s ({:.1f}min)", elapsed, elapsed / 60)
    logger.info("Run `python evaluate.py` to generate the full evaluation report.")


if __name__ == "__main__":
    main()
