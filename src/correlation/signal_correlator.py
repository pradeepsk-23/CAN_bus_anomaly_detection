"""
src/correlation/signal_correlator.py

Multi-modal signal correlator for root-cause attribution.

When the ensemble flags an anomaly, this module answers: "WHICH sub-system
caused it?" by cross-correlating per-channel residuals from the LSTM-AE with
the temporal structure of detected events.

Methods:
  1. Pearson cross-correlation with lag — identifies leading channels
     (e.g., battery_voltage drops 2s before CAN latency spikes)
  2. Sub-system attribution — maps guilty channels → JD-defined sub-systems
     (powertrain / electrical / thermal / dynamics)
  3. Event timeline builder — produces human-readable root-cause reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
@dataclass
class CorrelationResult:
    """Correlation between a leading channel and a lagging channel."""
    source_channel: str
    target_channel: str
    lag_samples: int           # positive = source leads target
    correlation: float
    p_value: float


@dataclass
class AnomalyRootCause:
    """Attributed root cause for a detected anomaly window."""
    window_idx: int
    ensemble_score: float
    top_channels: List[Tuple[str, float]]    # (channel_name, residual) sorted desc
    primary_subsystem: str
    secondary_subsystem: Optional[str]
    leading_channel: Optional[str]
    lag_seconds: Optional[float]
    correlations: List[CorrelationResult]
    narrative: str                           # human-readable description


# ---------------------------------------------------------------------------
# Core correlator
# ---------------------------------------------------------------------------
class SignalCorrelator:
    """
    Cross-channel root-cause analyser.

    Usage:
        correlator = SignalCorrelator(cfg)
        root_causes = correlator.analyse(
            windows=X_test_anomalous,
            residuals=ensemble.explain(X_test_anomalous)["channel_residuals"],
            scores=ensemble.score(X_test_anomalous),
            channel_names=channel_cols,
            sampling_hz=10,
        )
    """

    def __init__(self, cfg: dict) -> None:
        corr_cfg = cfg["correlator"]
        self.max_lag_samples = int(corr_cfg["max_lag_seconds"] * 10)  # assume 10 Hz default
        self.correlation_threshold = corr_cfg["correlation_threshold"]
        self.subsystems: Dict[str, List[str]] = corr_cfg["subsystems"]

        # Invert: channel → sub-system name
        self._channel_to_subsystem: Dict[str, str] = {}
        for system, channels in self.subsystems.items():
            for ch in channels:
                self._channel_to_subsystem[ch] = system

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyse(
        self,
        windows: np.ndarray,             # (N, T, C)
        residuals: np.ndarray,           # (N, C)  — per-channel mean abs residual
        scores: np.ndarray,              # (N,)
        channel_names: List[str],
        sampling_hz: int = 10,
    ) -> List[AnomalyRootCause]:
        """
        Analyse anomalous windows and return root-cause attribution per window.
        """
        results = []
        for i in range(len(windows)):
            rc = self._analyse_window(
                idx=i,
                window=windows[i],               # (T, C)
                residual_row=residuals[i],        # (C,)
                score=float(scores[i]),
                channel_names=channel_names,
                sampling_hz=sampling_hz,
            )
            results.append(rc)
            logger.debug(
                "Window {:4d} | score={:.3f} | subsystem={} | narrative: {}",
                i, scores[i], rc.primary_subsystem, rc.narrative,
            )

        logger.info("Root-cause analysis complete — {} windows analysed", len(results))
        return results

    def summarise(
        self, root_causes: List[AnomalyRootCause]
    ) -> pd.DataFrame:
        """Flatten root cause list into a summary DataFrame for reporting."""
        rows = []
        for rc in root_causes:
            rows.append({
                "window_idx":         rc.window_idx,
                "ensemble_score":     rc.ensemble_score,
                "primary_subsystem":  rc.primary_subsystem,
                "secondary_subsystem": rc.secondary_subsystem,
                "top_channel":        rc.top_channels[0][0] if rc.top_channels else "N/A",
                "top_channel_residual": rc.top_channels[0][1] if rc.top_channels else 0.0,
                "leading_channel":    rc.leading_channel,
                "lag_seconds":        rc.lag_seconds,
                "narrative":          rc.narrative,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _analyse_window(
        self,
        idx: int,
        window: np.ndarray,      # (T, C)
        residual_row: np.ndarray, # (C,)
        score: float,
        channel_names: List[str],
        sampling_hz: int,
    ) -> AnomalyRootCause:

        # 1) Rank channels by residual magnitude
        ranked = sorted(
            zip(channel_names, residual_row.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        top_channels = ranked[:5]

        # 2) Sub-system attribution
        primary_subsystem = self._get_primary_subsystem([ch for ch, _ in top_channels[:3]])
        secondary_subsystem = self._get_primary_subsystem([ch for ch, _ in top_channels[3:5]])
        if secondary_subsystem == primary_subsystem:
            secondary_subsystem = None

        # 3) Cross-channel lagged correlation (which channel leads?)
        top_channel_names = [ch for ch, _ in top_channels[:4]]
        correlations = self._compute_lagged_correlations(window, channel_names, top_channel_names)

        leading_channel, lag_seconds = self._find_leading_channel(
            correlations, top_channel_names, sampling_hz
        )

        # 4) Build narrative
        narrative = self._build_narrative(
            top_channels=top_channels,
            primary_subsystem=primary_subsystem,
            leading_channel=leading_channel,
            lag_seconds=lag_seconds,
            score=score,
        )

        return AnomalyRootCause(
            window_idx=idx,
            ensemble_score=score,
            top_channels=top_channels,
            primary_subsystem=primary_subsystem,
            secondary_subsystem=secondary_subsystem,
            leading_channel=leading_channel,
            lag_seconds=lag_seconds,
            correlations=correlations,
            narrative=narrative,
        )

    def _get_primary_subsystem(self, channels: List[str]) -> str:
        """Vote: which sub-system do most top channels belong to?"""
        votes: Dict[str, int] = {}
        for ch in channels:
            sys = self._channel_to_subsystem.get(ch, "unknown")
            votes[sys] = votes.get(sys, 0) + 1
        if not votes:
            return "unknown"
        return max(votes, key=votes.get)

    def _compute_lagged_correlations(
        self,
        window: np.ndarray,         # (T, C)
        channel_names: List[str],
        focus_channels: List[str],
    ) -> List[CorrelationResult]:
        """
        Compute Pearson correlation at various lags between focus channels.
        Positive lag_samples → source channel leads target.
        """
        results = []
        ch_idx = {ch: i for i, ch in enumerate(channel_names)}

        for i, src_ch in enumerate(focus_channels):
            for tgt_ch in focus_channels[i + 1:]:
                if src_ch not in ch_idx or tgt_ch not in ch_idx:
                    continue

                src_signal = window[:, ch_idx[src_ch]]
                tgt_signal = window[:, ch_idx[tgt_ch]]

                best_corr, best_lag, best_p = 0.0, 0, 1.0
                for lag in range(-self.max_lag_samples, self.max_lag_samples + 1):
                    if lag > 0:
                        s, t = src_signal[:-lag], tgt_signal[lag:]
                    elif lag < 0:
                        s, t = src_signal[-lag:], tgt_signal[:lag]
                    else:
                        s, t = src_signal, tgt_signal

                    if len(s) < 5:
                        continue
                    try:
                        r, p = stats.pearsonr(s, t)
                    except Exception:
                        continue

                    if abs(r) > abs(best_corr):
                        best_corr, best_lag, best_p = r, lag, p

                if abs(best_corr) >= self.correlation_threshold:
                    results.append(CorrelationResult(
                        source_channel=src_ch,
                        target_channel=tgt_ch,
                        lag_samples=best_lag,
                        correlation=best_corr,
                        p_value=best_p,
                    ))

        return results

    def _find_leading_channel(
        self,
        correlations: List[CorrelationResult],
        focus_channels: List[str],
        sampling_hz: int,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Identify the channel that most consistently LEADS others."""
        lead_count: Dict[str, int] = {ch: 0 for ch in focus_channels}
        lead_lag: Dict[str, float] = {ch: 0.0 for ch in focus_channels}

        for corr in correlations:
            if corr.lag_samples > 0:   # source leads target
                lead_count[corr.source_channel] = lead_count.get(corr.source_channel, 0) + 1
                lead_lag[corr.source_channel] = lead_lag.get(corr.source_channel, 0) + corr.lag_samples
            elif corr.lag_samples < 0:  # target leads source
                lead_count[corr.target_channel] = lead_count.get(corr.target_channel, 0) + 1
                lead_lag[corr.target_channel] = lead_lag.get(corr.target_channel, 0) + abs(corr.lag_samples)

        if not any(v > 0 for v in lead_count.values()):
            return None, None

        leading_ch = max(lead_count, key=lead_count.get)
        if lead_count[leading_ch] == 0:
            return None, None

        avg_lag = lead_lag[leading_ch] / lead_count[leading_ch]
        return leading_ch, round(avg_lag / sampling_hz, 2)

    def _build_narrative(
        self,
        top_channels: List[Tuple[str, float]],
        primary_subsystem: str,
        leading_channel: Optional[str],
        lag_seconds: Optional[float],
        score: float,
    ) -> str:
        top_ch_str = ", ".join(f"{ch} ({v:.3f})" for ch, v in top_channels[:3])
        parts = [
            f"Anomaly detected (score={score:.3f}).",
            f"Primary sub-system: {primary_subsystem.upper()}.",
            f"Top contributing channels: {top_ch_str}.",
        ]
        if leading_channel and lag_seconds is not None:
            parts.append(
                f"'{leading_channel}' appears to be the initiating signal, "
                f"leading correlated channels by ~{lag_seconds}s."
            )
        return " ".join(parts)
