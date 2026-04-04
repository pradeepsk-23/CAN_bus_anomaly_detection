"""
src/models/isolation_forest.py

Isolation Forest wrapper for point-anomaly detection on statistical features.

Design notes:
  - IF operates on flattened, hand-engineered feature vectors (not raw windows).
  - Complements the LSTM-AE: IF excels at global outliers in feature space;
    AE excels at contextual / temporal sequence anomalies.
  - Decision function scores are normalised to [0, 1] so they can be blended
    with AE reconstruction errors in the ensemble.
  - Uses joblib for serialisation (sklearn standard).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from loguru import logger


def _extract_window_features(X: np.ndarray) -> np.ndarray:
    """
    Reduce (N, T, C) windows → (N, F) feature vectors for IF.

    Features per channel:
      mean, std, min, max, range, peak-to-peak, skewness proxy (max-mean),
      rate of change (mean absolute delta)
    This gives 8 features × C channels.
    """
    N, T, C = X.shape
    features = []

    for c in range(C):
        ch = X[:, :, c]                          # (N, T)
        feat_mean  = ch.mean(axis=1)
        feat_std   = ch.std(axis=1)
        feat_min   = ch.min(axis=1)
        feat_max   = ch.max(axis=1)
        feat_range = feat_max - feat_min
        feat_p2p   = np.abs(feat_max - feat_mean)   # skewness proxy
        feat_delta = np.abs(np.diff(ch, axis=1)).mean(axis=1)   # mean |Δx|

        features.extend([feat_mean, feat_std, feat_min, feat_max,
                         feat_range, feat_p2p, feat_delta])

    return np.stack(features, axis=1)  # (N, 7*C)


class IsolationForestDetector:
    """
    Scikit-learn Isolation Forest with a standardised scoring interface
    matching the LSTM-AE: higher score → more anomalous.
    """

    def __init__(self, cfg: dict) -> None:
        params = cfg["models"]["isolation_forest"]
        self.model = IsolationForest(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            contamination=params["contamination"],
            max_features=params["max_features"],
            n_jobs=params["n_jobs"],
            random_state=params["random_state"],
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        X: (N, T, C) windows — only normal windows should be passed here.
        """
        features = _extract_window_features(X)
        logger.info("Fitting Isolation Forest on {:,} windows ({} features each)...",
                    len(features), features.shape[1])
        self.model.fit(features)
        self._fitted = True
        logger.info("Isolation Forest fitted — estimators={}", len(self.model.estimators_))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, T, C)
        Returns normalised anomaly score ∈ [0, 1] per window.
        Higher = more anomalous.
        """
        assert self._fitted, "Call fit() first."
        features = _extract_window_features(X)
        # decision_function returns positive values for inliers, negative for outliers
        raw = self.model.decision_function(features)  # (N,) — higher = more normal
        # Flip and normalise to [0, 1] via min-max on this batch
        flipped = -raw
        min_s, max_s = flipped.min(), flipped.max()
        if max_s > min_s:
            normalised = (flipped - min_s) / (max_s - min_s)
        else:
            normalised = np.zeros_like(flipped)
        return normalised.astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary anomaly prediction: 1=anomaly, 0=normal."""
        return (self.score(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("IsolationForest saved → {}", path)

    @classmethod
    def load(cls, path: str, cfg: dict) -> "IsolationForestDetector":
        detector = cls(cfg)
        detector.model = joblib.load(path)
        detector._fitted = True
        logger.info("IsolationForest loaded ← {}", path)
        return detector
