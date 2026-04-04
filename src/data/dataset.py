"""
src/data/dataset.py

Sliding-window dataset builder with:
  - Z-score normalisation (fit on normal train data only — avoids leakage)
  - Sliding windows with configurable size and stride
  - Feature engineering: rolling statistics appended to raw channels
  - PyTorch Dataset wrappers for DataLoader compatibility
  - Support for ROAD dataset loading alongside synthetic data
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


# ---------------------------------------------------------------------------
# Normaliser — fit on normal (train) data only; NO label leakage
# ---------------------------------------------------------------------------
class SignalNormaliser:
    """
    Channel-wise Z-score normalisation.
    Fit only on windows labelled as 'normal' from the training split.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SignalNormaliser":
        """X: (N, T, C) or (N, C) — fit across first two dims."""
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8   # epsilon guards against dead channels
        logger.debug("Normaliser fitted — mean={}, std={}", self.mean_.round(2), self.std_.round(2))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() before transform()"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)
        logger.info("Normaliser saved → {}", path)

    @classmethod
    def load(cls, path: str) -> "SignalNormaliser":
        with open(path, "rb") as f:
            state = pickle.load(f)
        n = cls()
        n.mean_ = state["mean"]
        n.std_ = state["std"]
        logger.info("Normaliser loaded ← {}", path)
        return n


# ---------------------------------------------------------------------------
# Sliding window extractor
# ---------------------------------------------------------------------------
def extract_windows(
    df: pd.DataFrame,
    channel_cols: List[str],
    window_size: int = 50,
    stride: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract overlapping sliding windows from a multi-channel DataFrame.

    Returns:
        X         (N, T, C) — signal windows
        y         (N,)      — window-level label (1 if any sample anomalous)
        anom_type (N,)      — most frequent anomaly type in the window
    """
    signals = df[channel_cols].values.astype(np.float32)   # (T_total, C)
    labels = df["is_anomaly"].values if "is_anomaly" in df.columns else np.zeros(len(df))
    anom_types = df["anomaly_type"].values if "anomaly_type" in df.columns else np.full(len(df), "unknown")

    T = len(signals)
    windows, win_labels, win_types = [], [], []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(signals[start:end])
        win_labels.append(int(labels[start:end].max()))  # window is anomalous if any sample is

        # Most frequent anomaly type in window (excluding 'normal')
        types_in_win = anom_types[start:end]
        non_normal = [t for t in types_in_win if t != "normal"]
        win_types.append(non_normal[0] if non_normal else "normal")

    X = np.stack(windows, axis=0)            # (N, T, C)
    y = np.array(win_labels, dtype=np.int32) # (N,)
    anom_type_arr = np.array(win_types)      # (N,)

    anomaly_rate = y.mean()
    logger.info("Extracted {:,} windows | window_size={} | stride={} | anomaly_rate={:.2%}",
                len(X), window_size, stride, anomaly_rate)
    return X, y, anom_type_arr


# ---------------------------------------------------------------------------
# Feature engineering — rolling stats appended to raw channels
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, channel_cols: List[str], window_seconds: int = 10, hz: int = 10) -> pd.DataFrame:
    """
    Append rolling mean, std, and delta features.
    These give the Isolation Forest richer local context.
    The LSTM-AE consumes the raw channels only (learns its own temporal features).
    """
    roll_win = window_seconds * hz
    enriched = df.copy()
    for col in channel_cols:
        enriched[f"{col}__rmean"] = df[col].rolling(roll_win, min_periods=1).mean()
        enriched[f"{col}__rstd"] = df[col].rolling(roll_win, min_periods=1).std().fillna(0)
        enriched[f"{col}__delta"] = df[col].diff().fillna(0)
    return enriched


# ---------------------------------------------------------------------------
# PyTorch Dataset wrappers
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """Dataset for LSTM Autoencoder — returns (window, window) for reconstruction."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.X = torch.from_numpy(X).float()   # (N, T, C)
        self.y = torch.from_numpy(y).long() if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class FeatureDataset(Dataset):
    """Dataset for Isolation Forest — returns flattened feature vectors."""

    def __init__(self, X_flat: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.X = X_flat.astype(np.float32)
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], (self.y[idx] if self.y is not None else -1)


# ---------------------------------------------------------------------------
# High-level dataset builder
# ---------------------------------------------------------------------------
class VehicleDatasetBuilder:
    """
    Orchestrates the full data preparation pipeline:
      1. Accept DataFrame (synthetic or real)
      2. Optional feature engineering
      3. Train/val/test split (temporal — never shuffle time-series!)
      4. Normalise on train-normal only
      5. Return window arrays and fitted normaliser

    Temporal split rationale: random splitting of time-series causes temporal
    leakage — the model sees future context during training. Temporal split
    respects the causal direction and better simulates production deployment.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.normaliser: Optional[SignalNormaliser] = None

    def build(
        self,
        df: pd.DataFrame,
        channel_cols: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns dict with keys 'train', 'val', 'test', each a tuple (X, y, anom_types).
        X shape: (N, T, C) where T=window_size, C=len(channel_cols)
        """
        if channel_cols is None:
            channel_cols = self.cfg["data"]["channels"]

        window_size = self.cfg["data"]["window"]["size"]
        stride = self.cfg["data"]["window"]["stride"]
        train_frac = self.cfg["data"]["window"]["train_split"]
        val_frac = self.cfg["data"]["window"]["val_split"]

        # -- Temporal split on raw df BEFORE windowing
        N = len(df)
        train_end = int(N * train_frac)
        val_end = int(N * (train_frac + val_frac))

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        logger.info("Temporal split — train={:,} val={:,} test={:,} samples",
                    len(df_train), len(df_val), len(df_test))

        # -- Normaliser: fit on train-normal only (strict — no leakage)
        normal_mask = df_train["is_anomaly"] == 0 if "is_anomaly" in df_train.columns else pd.Series(True, index=df_train.index)
        train_normal_signals = df_train.loc[normal_mask, channel_cols].values.astype(np.float32)
        self.normaliser = SignalNormaliser().fit(train_normal_signals)

        splits = {}
        for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            # Apply normalisation
            normalised = split_df.copy()
            normalised[channel_cols] = self.normaliser.transform(
                split_df[channel_cols].values.astype(np.float32)
            )
            X, y, anom_types = extract_windows(normalised, channel_cols, window_size, stride)
            splits[name] = (X, y, anom_types)

        return splits

    def save_normaliser(self, path: str) -> None:
        assert self.normaliser is not None
        self.normaliser.save(path)

    def load_normaliser(self, path: str) -> None:
        self.normaliser = SignalNormaliser.load(path)


# ---------------------------------------------------------------------------
# ROAD dataset loader (plug-in when real data is available)
# ---------------------------------------------------------------------------
def load_road_dataset(data_dir: str, channel_cols: List[str]) -> pd.DataFrame:
    """
    Loads the Real ORNL Automotive Dynamometer (ROAD) dataset.

    Download: https://0xsam.com/road/
    Paper: https://arxiv.org/abs/2012.14600

    The ROAD dataset contains natural (non-injected) driving data and
    three attack categories: fabrication, masquerade, and fuzzing.
    This loader maps those to the same schema as the simulator.

    Expects the directory structure:
        data_dir/
            ambient/          — normal driving recordings (.csv)
            attacks/          — attack recordings
                fabrication/
                masquerade/
                fuzzing/
    """
    import os
    data_dir = Path(data_dir)
    dfs = []

    ambient_dir = data_dir / "ambient"
    if not ambient_dir.exists():
        raise FileNotFoundError(
            f"ROAD dataset not found at {data_dir}.\n"
            "Download from https://0xsam.com/road/ and extract to data/road/"
        )

    # Load normal (ambient) files
    for fpath in sorted(ambient_dir.glob("*.csv")):
        df = pd.read_csv(fpath)
        df["is_anomaly"] = 0
        df["anomaly_type"] = "normal"
        dfs.append(df)
        logger.debug("Loaded ambient: {}", fpath.name)

    # Load attack files
    attack_dirs = {
        "fabrication": data_dir / "attacks" / "fabrication",
        "masquerade": data_dir / "attacks" / "masquerade",
        "fuzzing": data_dir / "attacks" / "fuzzy",
    }
    for attack_type, attack_dir in attack_dirs.items():
        if not attack_dir.exists():
            continue
        for fpath in sorted(attack_dir.glob("*.csv")):
            df = pd.read_csv(fpath)
            # ROAD labels: 'Label' column has 1 for attack samples
            df["is_anomaly"] = df["Label"].astype(int) if "Label" in df.columns else 1
            df["anomaly_type"] = attack_type
            dfs.append(df)
            logger.debug("Loaded attack ({}): {}", attack_type, fpath.name)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("ROAD dataset loaded — {:,} total samples, {:.2%} anomalous",
                len(combined), combined["is_anomaly"].mean())
    return combined
