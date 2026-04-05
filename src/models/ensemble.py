"""
src/models/ensemble.py

Ensemble Anomaly Detector: combines LSTM-AE and Isolation Forest scores.

Design:
  - Both models output normalised scores ∈ [0, 1].
  - Weighted average → final score.
  - Threshold calibrated on validation set to maximise F1 (configurable).
  - Exposes `.explain()` to surface per-channel residuals from the LSTM-AE
    for root-cause attribution downstream.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.isolation_forest import IsolationForestDetector


class EnsembleAnomalyDetector:
    """
    Two-model ensemble for vehicle health monitoring.

    Scoring pipeline:
        raw_window  →  [LSTM-AE score]  ─┐
                                          ├→ weighted avg → final_score → threshold → label
        raw_window  →  [IF score]       ─┘
    """

    def __init__(
        self,
        lstm_ae: LSTMAutoencoder,
        isolation_forest: IsolationForestDetector,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.lstm_ae = lstm_ae.to(device)
        self.isolation_forest = isolation_forest
        self.device = device
        self.cfg = cfg

        ensemble_cfg = cfg["models"]["ensemble"]
        self.ae_weight = ensemble_cfg["lstm_ae_weight"]
        self.if_weight = ensemble_cfg["isolation_forest_weight"]
        self.threshold = ensemble_cfg["decision_threshold"]

        assert abs(self.ae_weight + self.if_weight - 1.0) < 1e-6, \
            "Ensemble weights must sum to 1.0"

    # ------------------------------------------------------------------
    # Primary inference API
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C) normalised windows
        Returns combined anomaly score ∈ [0, 1] per window.
        """
        ae_scores = self._ae_score(X)
        if_scores = self.isolation_forest.score(X)

        combined = self.ae_weight * ae_scores + self.if_weight * if_scores
        return combined.astype(np.float32)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary prediction: 1=anomaly, 0=normal."""
        thr = threshold if threshold is not None else self.threshold
        return (self.score(X) >= thr).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [P(normal), P(anomaly)] per window — compatible with sklearn API."""
        scores = self.score(X)
        return np.stack([1 - scores, scores], axis=1)

    # ------------------------------------------------------------------
    # Explainability — which channels drove the anomaly?
    # ------------------------------------------------------------------
    def explain(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns per-channel mean absolute residuals from the LSTM-AE.
        Shape: (N, C) — higher value → channel contributed more to anomaly score.
        Used by the signal correlator to map anomaly → sub-system.
        """
        X_tensor = torch.from_numpy(X).float().to(self.device)
        residuals = self.lstm_ae.channel_residuals(X_tensor)  # (N, T, C)
        channel_importance = residuals.mean(dim=1).cpu().numpy()  # (N, C)
        return {"channel_residuals": channel_importance}

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------
    def calibrate_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray, strategy: str = "f1_optimal"
    ) -> float:
        """
        Calibrate decision threshold on validation data.

        Strategies
        ----------
        f1_optimal
            Grid-search the threshold that maximises F1 on the val set.
            Requires anomalous examples in val — if none are present
            (e.g. val is all-normal by design), automatically falls back
            to the percentile strategy and logs a warning.

        percentile
            Set threshold at the 95th percentile of normal-window scores.
            Works even when val has no anomalies — correct choice when
            df_ambient is the val source.

        Design note
        -----------
        When the train/val split is derived entirely from normal (ambient)
        data, val contains zero anomalies.  F1-optimal calibration is
        impossible in that case.  The percentile approach is principled:
        it accepts ~5 % false-positive rate on normal traffic, which is a
        common operational target for fleet health systems.
        """
        from sklearn.metrics import f1_score

        scores        = self.score(X_val)
        n_positives   = int(y_val.sum())
        normal_scores = scores[y_val == 0]

        if strategy == "f1_optimal":
            if n_positives == 0:
                logger.warning(
                    "Val set has no anomalous windows (all-normal by design). "
                    "F1-optimal calibration impossible — falling back to 95th-percentile."
                )
                strategy = "percentile"   # fall through to percentile branch

            else:
                best_f1, best_thr = 0.0, float(np.median(scores))
                for thr in np.linspace(scores.min(), scores.max(), 300):
                    preds = (scores >= thr).astype(int)
                    if preds.sum() == 0:
                        continue
                    f1 = f1_score(y_val, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_thr = f1, float(thr)
                self.threshold = best_thr
                logger.info(
                    "Threshold calibrated (F1-optimal) → {:.4f} | val_F1={:.4f}",
                    best_thr, best_f1,
                )
                return self.threshold

        if strategy == "percentile":
            # Use 95th percentile of normal-window scores:
            # ~95 % of normal traffic scores below this → ~5 % FPR target.
            pct = self.cfg.get("evaluation", {}).get("percentile", 95)
            self.threshold = float(np.percentile(normal_scores, pct))
            logger.info(
                "Threshold calibrated ({}th-percentile of normal val scores) → {:.4f}",
                pct, self.threshold,
            )

        return self.threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, checkpoint_dir: str) -> None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.lstm_ae.state_dict(), ckpt_path / "lstm_ae.pt")
        self.isolation_forest.save(str(ckpt_path / "isolation_forest.joblib"))

        meta = {
            "threshold": self.threshold,
            "ae_weight": self.ae_weight,
            "if_weight": self.if_weight,
        }
        with open(ckpt_path / "ensemble_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("Ensemble saved → {}", checkpoint_dir)

    @classmethod
    def load(cls, checkpoint_dir: str, lstm_ae: LSTMAutoencoder,
             cfg: dict, device: torch.device) -> "EnsembleAnomalyDetector":
        ckpt_path = Path(checkpoint_dir)

        lstm_ae.load_state_dict(torch.load(ckpt_path / "lstm_ae.pt", map_location=device))
        logger.info("LSTM-AE weights loaded ← {}", ckpt_path / "lstm_ae.pt")

        if_detector = IsolationForestDetector.load(
            str(ckpt_path / "isolation_forest.joblib"), cfg
        )

        with open(ckpt_path / "ensemble_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        detector = cls(lstm_ae, if_detector, cfg, device)
        detector.threshold = meta["threshold"]
        detector.ae_weight = meta["ae_weight"]
        detector.if_weight = meta["if_weight"]
        logger.info("Ensemble loaded ← {} (threshold={:.4f})", checkpoint_dir, detector.threshold)
        return detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ae_score(self, X: np.ndarray) -> np.ndarray:
        """Normalised LSTM-AE reconstruction error ∈ [0, 1]."""
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            raw_errors = self.lstm_ae.reconstruction_error(X_tensor).cpu().numpy()  # (N,)
        # Normalise per batch (production: use a running percentile / fixed scale from train)
        min_e, max_e = raw_errors.min(), raw_errors.max()
        if max_e > min_e:
            return (raw_errors - min_e) / (max_e - min_e)
        return np.zeros_like(raw_errors)
