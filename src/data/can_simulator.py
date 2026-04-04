"""
src/data/can_simulator.py

Realistic CAN-bus / vehicle sensor data generator.

Design goals:
  - Mimics the statistical properties of real CAN bus signals (autocorrelated,
    correlated across channels, regime-dependent) so that models trained here
    generalise to real data with minimal fine-tuning.
  - Anomaly injection is physics-aware: a battery sag event causes correlated
    voltage drop AND CAN latency spikes, not independent noise injections.
  - Provides a deterministic seed for reproducible CI/experiment runs.

Anomaly taxonomy (maps directly to JD duties):
  VOLTAGE_SAG     — battery voltage drop → correlates with CAN latency rise
  COOLANT_SPIKE   — thermal runaway precursor
  RPM_DROPOUT     — engine communication loss / CAN frame loss
  LATENCY_STORM   — bus contention / software regression
  CRASH_PRECURSOR — combination: rpm instability + accel anomaly + latency
  SENSOR_DRIFT    — gradual drift on oil pressure (hard to catch without AE)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from loguru import logger


# ---------------------------------------------------------------------------
# Signal definitions — nominal operating envelope per channel
# ---------------------------------------------------------------------------
@dataclass
class SignalSpec:
    name: str
    nominal_mean: float
    nominal_std: float
    min_val: float
    max_val: float
    # AR(1) coefficient — controls temporal autocorrelation (0=white noise, ~1=slow drift)
    ar_coeff: float = 0.95


SIGNAL_SPECS: List[SignalSpec] = [
    SignalSpec("engine_rpm",          mean:=1800, std:=150, min_val=600,  max_val=6500, ar_coeff=0.92),
    SignalSpec("vehicle_speed_kmh",   mean:=65,   std:=10,  min_val=0,    max_val=200,  ar_coeff=0.97),
    SignalSpec("battery_voltage_v",   mean:=14.2, std:=0.3, min_val=9.0,  max_val=16.0, ar_coeff=0.98),
    SignalSpec("oil_pressure_bar",    mean:=3.5,  std:=0.3, min_val=0.5,  max_val=7.0,  ar_coeff=0.96),
    SignalSpec("coolant_temp_c",      mean:=88,   std:=3,   min_val=20,   max_val=130,  ar_coeff=0.99),
    SignalSpec("throttle_position_pct", mean:=25, std:=12,  min_val=0,    max_val=100,  ar_coeff=0.85),
    SignalSpec("can_latency_us",      mean:=120,  std:=20,  min_val=50,   max_val=5000, ar_coeff=0.80),
    SignalSpec("accel_x_g",           mean:=0.05, std:=0.1, min_val=-2.0, max_val=2.0,  ar_coeff=0.70),
    SignalSpec("accel_y_g",           mean:=0.0,  std:=0.1, min_val=-2.0, max_val=2.0,  ar_coeff=0.70),
    SignalSpec("accel_z_g",           mean:=1.0,  std:=0.05,min_val=-2.0, max_val=3.0,  ar_coeff=0.85),
]

# Rework the dataclass — Python walrus in default arg doesn't work cleanly; redefine:
SIGNAL_SPECS = [
    SignalSpec("engine_rpm",            1800, 150,  600,   6500, 0.92),
    SignalSpec("vehicle_speed_kmh",     65,   10,   0,     200,  0.97),
    SignalSpec("battery_voltage_v",     14.2, 0.3,  9.0,   16.0, 0.98),
    SignalSpec("oil_pressure_bar",      3.5,  0.3,  0.5,   7.0,  0.96),
    SignalSpec("coolant_temp_c",        88,   3,    20,    130,  0.99),
    SignalSpec("throttle_position_pct", 25,   12,   0,     100,  0.85),
    SignalSpec("can_latency_us",        120,  20,   50,    5000, 0.80),
    SignalSpec("accel_x_g",             0.05, 0.1,  -2.0,  2.0,  0.70),
    SignalSpec("accel_y_g",             0.0,  0.1,  -2.0,  2.0,  0.70),
    SignalSpec("accel_z_g",             1.0,  0.05, -2.0,  3.0,  0.85),
]

CHANNEL_NAMES = [s.name for s in SIGNAL_SPECS]


# ---------------------------------------------------------------------------
# Anomaly event definitions
# ---------------------------------------------------------------------------
@dataclass
class AnomalyEvent:
    anomaly_type: str
    start_idx: int
    end_idx: int
    severity: float          # 0–1, drives magnitude of perturbation
    affected_channels: List[str]


# Physics-aware anomaly injection functions
# Each function receives the raw signal matrix (T × C) and modifies it in-place.

def _inject_voltage_sag(
    signals: np.ndarray, idx_map: Dict[str, int], start: int, end: int, severity: float
) -> None:
    """Battery sag: voltage drops, CAN latency rises (bus arbitration stress)."""
    duration = end - start
    envelope = np.hanning(duration)
    drop = severity * 3.5 * envelope          # up to 3.5 V drop
    signals[start:end, idx_map["battery_voltage_v"]] -= drop
    latency_rise = severity * 800 * envelope  # up to 800 µs extra latency
    signals[start:end, idx_map["can_latency_us"]] += latency_rise


def _inject_coolant_spike(
    signals: np.ndarray, idx_map: Dict[str, int], start: int, end: int, severity: float
) -> None:
    """Thermal runaway precursor: coolant climbs sharply, oil pressure drops."""
    duration = end - start
    ramp = np.linspace(0, 1, duration)
    signals[start:end, idx_map["coolant_temp_c"]] += severity * 30 * ramp
    signals[start:end, idx_map["oil_pressure_bar"]] -= severity * 1.0 * ramp


def _inject_rpm_dropout(
    signals: np.ndarray, idx_map: Dict[str, int], start: int, end: int, severity: float
) -> None:
    """CAN frame loss: RPM reported as zero intermittently."""
    drop_mask = np.random.random(end - start) < severity * 0.6
    signals[start:end, idx_map["engine_rpm"]][drop_mask] = 0.0
    signals[start:end, idx_map["can_latency_us"]] += severity * 400 * drop_mask


def _inject_latency_storm(
    signals: np.ndarray, idx_map: Dict[str, int], start: int, end: int, severity: float
) -> None:
    """Bus contention / software regression: latency spikes, speed jitter."""
    burst = np.random.exponential(severity * 600, size=end - start)
    signals[start:end, idx_map["can_latency_us"]] += burst
    jitter = np.random.normal(0, severity * 8, size=end - start)
    signals[start:end, idx_map["vehicle_speed_kmh"]] += jitter


def _inject_crash_precursor(
    signals: np.ndarray, idx_map: Dict[str, int], start: int, end: int, severity: float
) -> None:
    """Multi-signal composite: rpm instability + accel anomaly + latency burst."""
    _inject_rpm_dropout(signals, idx_map, start, end, severity * 0.5)
    _inject_latency_storm(signals, idx_map, start, end, severity * 0.7)
    # Unusual acceleration pattern (swerve / emergency brake signature)
    signals[start:end, idx_map["accel_x_g"]] += np.random.normal(0, severity * 0.8, end - start)
    signals[start:end, idx_map["accel_y_g"]] += severity * 1.2 * np.hanning(end - start)


def _inject_sensor_drift(
    signals: np.ndarray, idx_map: Dict[str, int], start: int, end: int, severity: float
) -> None:
    """Gradual sensor drift on oil pressure — subtle, tests AE's temporal memory."""
    drift = np.linspace(0, severity * 2.5, end - start)
    signals[start:end, idx_map["oil_pressure_bar"]] += drift


ANOMALY_INJECTORS = {
    "voltage_sag":      (_inject_voltage_sag,     ["battery_voltage_v", "can_latency_us"]),
    "coolant_spike":    (_inject_coolant_spike,    ["coolant_temp_c", "oil_pressure_bar"]),
    "rpm_dropout":      (_inject_rpm_dropout,      ["engine_rpm", "can_latency_us"]),
    "latency_storm":    (_inject_latency_storm,    ["can_latency_us", "vehicle_speed_kmh"]),
    "crash_precursor":  (_inject_crash_precursor,  ["engine_rpm", "can_latency_us", "accel_x_g", "accel_y_g"]),
    "sensor_drift":     (_inject_sensor_drift,     ["oil_pressure_bar"]),
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class CANBusSimulator:
    """
    Generates multi-channel CAN bus time-series with physics-aware anomalies.

    Usage:
        sim = CANBusSimulator(seed=42)
        df, events = sim.generate(
            duration_seconds=7200,
            sampling_hz=10,
            anomaly_ratio=0.03,
        )
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self._specs = SIGNAL_SPECS
        self._idx_map = {s.name: i for i, s in enumerate(self._specs)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        duration_seconds: int = 7200,
        sampling_hz: int = 10,
        anomaly_ratio: float = 0.03,
        noise_std: float = 0.05,
        vehicle_id: str = "VH-001",
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Returns:
            df      — DataFrame indexed by timestamp, shape (T, C+2)
                      Columns: channels + 'is_anomaly' + 'anomaly_type'
            events  — list of AnomalyEvent objects for evaluation
        """
        T = duration_seconds * sampling_hz
        logger.debug("Simulating {} samples ({} s @ {} Hz) for {}", T, duration_seconds, sampling_hz, vehicle_id)

        signals = self._generate_nominal(T, noise_std)
        events = self._inject_anomalies(signals, T, anomaly_ratio, sampling_hz)

        # Clip to physical bounds
        for i, spec in enumerate(self._specs):
            signals[:, i] = np.clip(signals[:, i], spec.min_val, spec.max_val)

        # Build labels
        is_anomaly = np.zeros(T, dtype=int)
        anomaly_type = np.full(T, "normal", dtype=object)
        for ev in events:
            is_anomaly[ev.start_idx:ev.end_idx] = 1
            anomaly_type[ev.start_idx:ev.end_idx] = ev.anomaly_type

        timestamps = pd.date_range("2024-01-01", periods=T, freq=f"{1000 // sampling_hz}ms")
        df = pd.DataFrame(signals, index=timestamps, columns=CHANNEL_NAMES)
        df["is_anomaly"] = is_anomaly
        df["anomaly_type"] = anomaly_type
        df["vehicle_id"] = vehicle_id

        logger.info(
            "vehicle={} | samples={:,} | anomaly_windows={} | anomaly_rate={:.2%}",
            vehicle_id, T, sum(ev.end_idx - ev.start_idx for ev in events), is_anomaly.mean(),
        )
        return df, events

    def generate_fleet(
        self,
        n_vehicles: int = 5,
        duration_seconds: int = 7200,
        sampling_hz: int = 10,
        anomaly_ratio: float = 0.03,
    ) -> Tuple[pd.DataFrame, Dict[str, List[AnomalyEvent]]]:
        """Generate independent streams for a fleet of vehicles."""
        dfs, all_events = [], {}
        for i in range(n_vehicles):
            vid = f"VH-{i+1:03d}"
            # Different seed per vehicle → independent noise realisations
            sim = CANBusSimulator(seed=self.rng.integers(0, 9999))
            df, events = sim.generate(duration_seconds, sampling_hz, anomaly_ratio, vehicle_id=vid)
            dfs.append(df)
            all_events[vid] = events
        combined = pd.concat(dfs, ignore_index=False)
        return combined, all_events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _generate_nominal(self, T: int, noise_std: float) -> np.ndarray:
        """AR(1) process per channel with cross-channel correlations."""
        n_channels = len(self._specs)
        signals = np.zeros((T, n_channels))

        # Initialise at nominal mean
        signals[0] = [s.nominal_mean for s in self._specs]

        for t in range(1, T):
            for i, spec in enumerate(self._specs):
                # AR(1): x_t = ar * x_{t-1} + (1-ar)*mean + noise
                innovation = self.rng.normal(0, spec.nominal_std * noise_std)
                signals[t, i] = (
                    spec.ar_coeff * signals[t - 1, i]
                    + (1 - spec.ar_coeff) * spec.nominal_mean
                    + innovation
                )

        # Inject a realistic speed-rpm correlation
        rpm_idx = self._idx_map["engine_rpm"]
        spd_idx = self._idx_map["vehicle_speed_kmh"]
        throttle_idx = self._idx_map["throttle_position_pct"]
        signals[:, rpm_idx] += 8 * signals[:, throttle_idx]
        signals[:, spd_idx] += 0.02 * signals[:, rpm_idx]

        return signals

    def _inject_anomalies(
        self, signals: np.ndarray, T: int, anomaly_ratio: float, sampling_hz: int
    ) -> List[AnomalyEvent]:
        """Select random non-overlapping windows and inject typed anomalies."""
        # Target anomaly samples
        target_anomaly_samples = int(T * anomaly_ratio)
        anomaly_types = list(ANOMALY_INJECTORS.keys())
        events: List[AnomalyEvent] = []
        occupied = np.zeros(T, dtype=bool)

        # Minimum gap between events (5 seconds) to avoid bleed-over
        min_gap = 5 * sampling_hz
        max_attempts = 200

        for _ in range(max_attempts):
            if occupied.sum() >= target_anomaly_samples:
                break

            anom_type = self.rng.choice(anomaly_types)
            # Event duration: 3–30 seconds
            duration = int(self.rng.uniform(3, 30) * sampling_hz)
            start = int(self.rng.uniform(min_gap, T - duration - min_gap))
            end = start + duration

            # Check overlap + gap
            check_start = max(0, start - min_gap)
            check_end = min(T, end + min_gap)
            if occupied[check_start:check_end].any():
                continue

            severity = float(self.rng.uniform(0.4, 1.0))
            injector_fn, affected_channels = ANOMALY_INJECTORS[anom_type]
            injector_fn(signals, self._idx_map, start, end, severity)
            occupied[start:end] = True

            events.append(AnomalyEvent(
                anomaly_type=anom_type,
                start_idx=start,
                end_idx=end,
                severity=severity,
                affected_channels=affected_channels,
            ))

        logger.debug("Injected {} anomaly events ({} unique types)", len(events), len({e.anomaly_type for e in events}))
        return events
