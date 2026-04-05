"""
src/data/dataset.py

Dataset preparation pipeline with a unified contract:

    INPUT  (identical for synthetic and real ROAD data)
    -------------------------------------------------------
    df_ambient  — all-normal rows  (Label=0 everywhere)
                  source: ambient/ folder  OR  simulator.generate_ambient()

    df_attacks  — mixed rows       (Label ∈ {0,1})
                  source: attacks/ folder  OR  simulator.generate_attacks()

    SPLIT STRATEGY
    -------------------------------------------------------
    df_ambient  → temporal 80/20 → TRAIN (normal) | VAL (normal + used for
                                                    threshold calibration)
    df_attacks  → entirely → TEST  (ground-truth labels from Label col)

    WHY TEMPORAL (not random) SPLIT?
    Time-series shuffling causes temporal leakage — the model sees future
    context during training.  Temporal split preserves causal direction and
    accurately simulates production deployment where the model sees new data
    arriving after the training window.

    NORMALISER
    -------------------------------------------------------
    Fit on TRAIN rows only (all normal, no leakage).
    Applied to TRAIN, VAL, and TEST identically.
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
        self.mean_ = flat.mean(axis=0).astype(np.float32)
        self.std_  = (flat.std(axis=0) + 1e-8).astype(np.float32)
        logger.debug("Normaliser fitted on {:,} samples — {} channels", len(flat), len(self.mean_))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() before transform()"
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X * self.std_ + self.mean_).astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    Slide a fixed-size window over a (time x channels) DataFrame.

    Returns
    -------
    X          (N, T, C)  signal windows
    y          (N,)       window label — 1 if ANY sample in window is anomalous
    anom_types (N,)       most common anomaly type in window ("normal" if clean)
    """
    signals = df[channel_cols].values.astype(np.float32)
    labels  = df["Label"].values  if "Label"  in df.columns else np.zeros(len(df), dtype=np.int32)
    types   = df["anomaly_type"].values if "anomaly_type" in df.columns else np.full(len(df), "unknown")

    T_total = len(signals)
    windows, win_labels, win_types = [], [], []

    for start in range(0, T_total - window_size + 1, stride):
        end = start + window_size
        windows.append(signals[start:end])
        win_labels.append(int(labels[start:end].max()))

        non_normal = [t for t in types[start:end] if t != "normal"]
        win_types.append(non_normal[0] if non_normal else "normal")

    if not windows:
        # Edge case: df shorter than one window
        logger.warning("DataFrame has fewer rows ({}) than window_size ({}). Returning empty arrays.",
                       T_total, window_size)
        C = len(channel_cols)
        return np.empty((0, window_size, C), np.float32), np.empty(0, np.int32), np.empty(0, object)

    X         = np.stack(windows).astype(np.float32)   # (N, T, C)
    y         = np.array(win_labels, dtype=np.int32)   # (N,)
    anom_type = np.array(win_types,  dtype=object)      # (N,)

    logger.info(
        "extract_windows → {:,} windows (size={}, stride={}) | anomaly_rate={:.2%}",
        len(X), window_size, stride, y.mean() if len(y) else 0,
    )
    return X, y, anom_type


# ---------------------------------------------------------------------------
# PyTorch Dataset wrappers
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """For LSTM-AE: returns (window, window) for reconstruction training."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ---------------------------------------------------------------------------
# VehicleDatasetBuilder — unified contract for synthetic AND real data
# ---------------------------------------------------------------------------
class VehicleDatasetBuilder:
    """
    Orchestrates the complete data-preparation pipeline.

    Usage (identical for synthetic and ROAD):
    -----------------------------------------
        builder = VehicleDatasetBuilder(cfg)
        splits  = builder.build(df_ambient, df_attacks)

        splits["train"]  ->  (X_train, y_train, types_train)
        splits["val"]    ->  (X_val,   y_val,   types_val)
        splits["test"]   ->  (X_test,  y_test,  types_test)

    Split logic
    -----------
        df_ambient  (all normal)  ->  temporal 80/20  →  train | val
        df_attacks  (with labels) ->  entirely         →  test

    Normaliser
    ----------
        Fit on TRAIN rows only (guarantees zero label leakage).
        Saved to checkpoints for inference-time reuse.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.normaliser: Optional[SignalNormaliser] = None

    def build(
        self,
        df_ambient: pd.DataFrame,
        df_attacks: pd.DataFrame,
        channel_cols: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        df_ambient : all-normal DataFrame — train + val source
        df_attacks : labelled DataFrame   — test source
        channel_cols : override channel list (defaults to cfg channels)

        Returns
        -------
        dict with keys "train", "val", "test", each a tuple (X, y, anom_types)
        """
        if channel_cols is None:
            channel_cols = self.cfg["data"]["channels"]

        val_frac    = self.cfg["data"]["val_split"]        # e.g. 0.20
        window_size = self.cfg["data"]["window"]["size"]
        stride      = self.cfg["data"]["window"]["stride"]

        # ── 1. Temporal split of ambient → train / val ─────────────────
        N_amb  = len(df_ambient)
        val_start = int(N_amb * (1.0 - val_frac))

        df_train = df_ambient.iloc[:val_start].copy()
        df_val   = df_ambient.iloc[val_start:].copy()
        df_test  = df_attacks.copy()

        logger.info(
            "Split — ambient_total={:,} → train={:,} val={:,} | attacks → test={:,}",
            N_amb, len(df_train), len(df_val), len(df_test),
        )

        # ── 2. Fit normaliser on TRAIN rows only ────────────────────────
        train_signals = df_train[channel_cols].values.astype(np.float32)
        self.normaliser = SignalNormaliser().fit(train_signals)

        # ── 3. Normalise all splits ────────────────────────────────────
        splits: Dict[str, Tuple] = {}
        for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            norm_df = split_df.copy()
            norm_df[channel_cols] = self.normaliser.transform(
                split_df[channel_cols].values.astype(np.float32)
            )
            X, y, anom_types = extract_windows(norm_df, channel_cols, window_size, stride)
            splits[name] = (X, y, anom_types)
            logger.info(
                "  {} → {:,} windows | anomaly_rate={:.2%}",
                name, len(X), y.mean() if len(y) else 0.0,
            )

        return splits

    def save_normaliser(self, path: str) -> None:
        assert self.normaliser is not None, "Call build() first"
        self.normaliser.save(path)

    def load_normaliser(self, path: str) -> None:
        self.normaliser = SignalNormaliser.load(path)


# ---------------------------------------------------------------------------
# ROAD dataset loader — returns (df_ambient, df_attacks)
# ---------------------------------------------------------------------------
ROAD_COLS = ['Label', 'Time', 'ID'] + [f"Signal_{i}_of_ID" for i in range(1, 23)]

def load_road_dataset(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the ROAD dataset and return (df_ambient, df_attacks) using the
    same schema as the simulator so that VehicleDatasetBuilder.build()
    works identically for both.

    Expected directory layout
    -------------------------
    data_dir/
        ambient/
            *.csv               ← normal driving recordings
        attacks/
            **/*.csv            ← attack recordings (any subfolder depth)

    ROAD CSV columns (used here)
    ----------------------------
        Signal_1_of_ID … Signal_22_of_ID   — decoded signal values
        Label                               — 0=normal, 1=attack  (attacks only)

    NaN handling: forward-fill then zero-fill (some signals not present in
    every CAN frame; ROAD convention is to leave those cells empty).

    Reference
    ---------
    Dataset : https://0xsam.com/road/
    Paper   : https://arxiv.org/abs/2012.14600
    """
    data_dir = Path(data_dir)
    ambient_dir = data_dir / "ambient"
    attacks_dir = data_dir / "attacks"

    if not ambient_dir.exists():
        raise FileNotFoundError(
            f"ambient/ folder not found at {data_dir}.\n"
            "Download ROAD dataset from https://0xsam.com/road/ and extract to "
            f"{data_dir}.\nExpected layout:  {data_dir}/ambient/*.csv  and  "
            f"{data_dir}/attacks/**/*.csv"
        )

    # Load normal (ambient) files
    ambient_dfs = []
    for fpath in sorted(ambient_dir.glob("*.csv")):
        df = _load_road_csv(fpath)
        ambient_dfs.append(df)
        logger.debug("Ambient loaded: {} ({:,} rows)", fpath.name, len(df))

    if not ambient_dfs:
        raise FileNotFoundError(f"No .csv files found in {ambient_dir}")

    df_ambient = pd.concat(ambient_dfs, ignore_index=True)
    logger.info("ROAD ambient — {:,} rows total", len(df_ambient))

    # Attacks (test source)
    attack_dfs = []
    if attacks_dir.exists():
        for fpath in sorted(attacks_dir.rglob("*.csv")):   # any subfolder depth
            # Infer attack type from parent folder name
            attack_type = fpath.stem.split('_attack')[0] + '_attack'
            df = _load_road_csv(fpath, attack_type=attack_type)
            attack_dfs.append(df)
            logger.debug("Attack ({}) loaded: {} ({:,} rows)", attack_type, fpath.name, len(df))

    if not attack_dfs:
        logger.warning("No attack CSVs found in {}. Test set will be empty.", attacks_dir)
        df_attacks = pd.DataFrame(columns=ROAD_COLS + ["Label", "anomaly_type", "vehicle_id"])
    else:
        df_attacks = pd.concat(attack_dfs, ignore_index=True)

    logger.info(
        "ROAD attacks — {:,} rows | anomaly_rate={:.2%}",
        len(df_attacks), df_attacks["Label"].mean() if len(df_attacks) else 0,
    )
    return df_ambient, df_attacks


def _load_road_csv(
    fpath: Path,
    attack_type: str = "attack",
) -> pd.DataFrame:
    """
    Load one ROAD CSV, select Signal columns, handle NaN, attach labels.
    """
    raw = pd.read_csv(fpath, low_memory=False)

    # Select only the 22 signal columns that exist in this file
    present_cols = [c for c in ROAD_COLS if c in raw.columns]
    missing_cols = [c for c in ROAD_COLS if c not in raw.columns]

    df = raw.copy()

    # Add any missing signal columns as zeros (some CAN IDs have < 22 signals)
    for c in missing_cols:
        df[c] = 0.0

    df = df[ROAD_COLS]   # enforce consistent column order

    # NaN → forward-fill within recording, then zero-fill remaining
    df = df.ffill().fillna(0.0)
    df = df.astype(np.float32)

    # Labels
    df["anomaly_type"] = np.where(df["Label"] == 1, attack_type, "normal")

    df["vehicle_id"] = fpath.stem   # use filename as vehicle/session ID
    return df
