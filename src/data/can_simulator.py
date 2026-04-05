"""
src/data/can_simulator.py

Physics-aware CAN-bus simulator that mirrors the ROAD dataset contract:

    generate_ambient()  ->  df_ambient  (all normal, Label=0)
    generate_attacks()  ->  df_attacks  (anomalous portions, Label=1)

COLUMN SCHEMA — identical to the ROAD dataset CSVs:
    Signal_1_of_ID … Signal_22_of_ID   (22 numeric channels)
    Label                          (0 / 1)
    anomaly_type                        ("normal" or attack name)
    vehicle_id                          (string)

This means the same VehicleDatasetBuilder, SignalNormaliser, and
evaluation code runs unchanged on both synthetic and real data.

Physics mapping (22 channels):
    Channels  1- 6  ->  powertrain   (RPM, speed, throttle, gear proxies)
    Channels  7-10  ->  electrical   (battery voltage, CAN latency, bus-load)
    Channels 11-14  ->  thermal      (coolant temp, oil pressure, intake temp)
    Channels 15-22  ->  dynamics     (accel X/Y/Z, yaw, steering, brake, susp)

Attack taxonomy (matches ROAD fabrication / masquerade / fuzzy categories
                 plus domain-relevant sub-types):
    voltage_sag      -> electrical  (battery/alternator fault)
    coolant_spike    -> thermal     (thermostat failure)
    rpm_dropout      -> powertrain  (ECU comms loss / frame drop)
    latency_storm    -> electrical  (bus contention / SW regression)
    crash_precursor  -> composite   (multi-channel instability)
    sensor_drift     -> thermal     (gradual pressure sensor drift)
    fuzzy_flood      -> electrical  (fuzzy injection — random CAN frames)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Channel names — must match ROAD dataset CSV headers exactly
# ---------------------------------------------------------------------------
N_CHANNELS = 22
CHANNEL_NAMES: List[str] = [f"Signal_{i}_of_ID" for i in range(1, N_CHANNELS + 1)]

# Subsystem index slices (0-based)
POWERTRAIN = slice(0, 6)    # signals 1–6
ELECTRICAL = slice(6, 10)   # signals 7–10
THERMAL    = slice(10, 14)  # signals 11–14
DYNAMICS   = slice(14, 22)  # signals 15–22

# ---------------------------------------------------------------------------
# Per-channel physical properties (mean, std, AR coefficient, value range)
# ---------------------------------------------------------------------------
@dataclass
class _ChannelSpec:
    mean: float
    std:  float
    ar:   float        # AR(1) coefficient — higher = smoother / slower drift
    lo:   float        # physical minimum
    hi:   float        # physical maximum

# One entry per channel in CHANNEL_NAMES order
_SPECS: List[_ChannelSpec] = [
    # Powertrain (1–6)
    _ChannelSpec(1800, 150, 0.92, 600,  6500),  # 1 engine RPM
    _ChannelSpec(65,    10, 0.97,  0,    200),  # 2 vehicle speed km/h
    _ChannelSpec(25,    12, 0.85,  0,    100),  # 3 throttle position %
    _ChannelSpec(3.2,  0.3, 0.90,  1,      7),  # 4 gear / trans ratio proxy
    _ChannelSpec(0.85, 0.05,0.93,  0,      1),  # 5 clutch position
    _ChannelSpec(210,  15,  0.88, 100,   500),  # 6 torque proxy Nm
    # Electrical (7–10)
    _ChannelSpec(14.2, 0.3, 0.98,  9,   16.0), # 7 battery voltage V
    _ChannelSpec(120,  20,  0.80, 50,   5000), # 8 CAN latency µs
    _ChannelSpec(42,    5,  0.85, 10,    100), # 9 bus load %
    _ChannelSpec(12.6, 0.2, 0.97,  9,   14.5), # 10 aux voltage V
    # Thermal (11–14)
    _ChannelSpec(88,    3,  0.99, 20,    130), # 11 coolant temp °C
    _ChannelSpec(3.5,  0.3, 0.96,  0.5,  7.0), # 12 oil pressure bar
    _ChannelSpec(35,    4,  0.95, -20,   100), # 13 intake air temp °C
    _ChannelSpec(90,    5,  0.97, 50,    200), # 14 exhaust temp proxy °C
    # Dynamics (15–22)
    _ChannelSpec(0.05, 0.10, 0.70, -3,    3),  # 15 accel X g
    _ChannelSpec(0.00, 0.10, 0.70, -3,    3),  # 16 accel Y g
    _ChannelSpec(1.00, 0.05, 0.85, -2,    4),  # 17 accel Z g
    _ChannelSpec(0.00, 0.02, 0.75, -1,    1),  # 18 yaw rate rad/s
    _ChannelSpec(0.00, 0.15, 0.80, -5,    5),  # 19 steering angle proxy
    _ChannelSpec(0.05, 0.10, 0.88,  0,    1),  # 20 brake pressure proxy
    _ChannelSpec(0.50, 0.08, 0.92,  0,    1),  # 21 front suspension deflect
    _ChannelSpec(0.50, 0.08, 0.92,  0,    1),  # 22 rear  suspension deflect
]
assert len(_SPECS) == N_CHANNELS


# ---------------------------------------------------------------------------
# Anomaly event record
# ---------------------------------------------------------------------------
@dataclass
class AnomalyEvent:
    anomaly_type:      str
    start_idx:         int
    end_idx:           int
    severity:          float           # 0–1
    affected_channels: List[int]       # 0-based channel indices


# ---------------------------------------------------------------------------
# Physics-aware anomaly injectors
# Each takes the signal matrix (T, C) and modifies it in-place.
# ---------------------------------------------------------------------------

def _hann(n: int) -> np.ndarray:
    return np.hanning(n)

def _inj_voltage_sag(s, start, end, sev, **_):
    env = _hann(end - start)
    s[start:end, 6]  -= sev * 3.5  * env   # battery voltage V
    s[start:end, 7]  += sev * 800  * env   # CAN latency µs
    s[start:end, 8]  += sev * 30   * env   # bus load %

def _inj_coolant_spike(s, start, end, sev, **_):
    ramp = np.linspace(0, 1, end - start)
    s[start:end, 10] += sev * 35   * ramp  # coolant temp °C
    s[start:end, 11] -= sev * 1.5  * ramp  # oil pressure bar
    s[start:end, 13] += sev * 40   * ramp  # exhaust temp proxy

def _inj_rpm_dropout(s, start, end, sev, rng, **_):
    mask = rng.random(end - start) < sev * 0.6
    s[start:end, 0][mask]  = 0.0            # engine RPM → 0 (frame lost)
    s[start:end, 7] += sev * 400 * mask.astype(float)  # latency spike

def _inj_latency_storm(s, start, end, sev, rng, **_):
    burst = rng.exponential(sev * 600, size=end - start)
    s[start:end, 7]  += burst                           # CAN latency µs
    s[start:end, 8]  += rng.uniform(0, sev * 40, end - start)  # bus load
    jitter = rng.normal(0, sev * 8, end - start)
    s[start:end, 1]  += jitter                          # speed jitter

def _inj_crash_precursor(s, start, end, sev, rng, **_):
    _inj_rpm_dropout(s, start, end, sev * 0.5, rng=rng)
    _inj_latency_storm(s, start, end, sev * 0.7, rng=rng)
    env = _hann(end - start)
    s[start:end, 14] += rng.normal(0, sev * 0.8, end - start)  # accel X
    s[start:end, 15] += sev * 1.2 * env                        # accel Y
    s[start:end, 17] += sev * 0.4 * env                        # yaw rate

def _inj_sensor_drift(s, start, end, sev, **_):
    drift = np.linspace(0, sev * 2.5, end - start)
    s[start:end, 11] += drift                           # oil pressure drift

def _inj_fuzzy_flood(s, start, end, sev, rng, **_):
    """Simulate fuzzy CAN injection — random values on several channels."""
    for ch in rng.choice(N_CHANNELS, size=max(3, int(sev * 8)), replace=False):
        spec = _SPECS[ch]
        s[start:end, ch] = rng.uniform(
            spec.lo + (spec.hi - spec.lo) * 0.1,
            spec.hi - (spec.hi - spec.lo) * 0.1,
            end - start,
        )
    s[start:end, 7] *= 1 + sev * 3   # latency always spikes under fuzzy flood

# Registry: name → (injector_fn, affected_channel_indices)
_INJECTORS: Dict[str, Tuple] = {
    "voltage_sag":     (_inj_voltage_sag,     [6, 7, 8]),
    "coolant_spike":   (_inj_coolant_spike,   [10, 11, 13]),
    "rpm_dropout":     (_inj_rpm_dropout,     [0, 7]),
    "latency_storm":   (_inj_latency_storm,   [7, 8, 1]),
    "crash_precursor": (_inj_crash_precursor, [0, 7, 14, 15, 17]),
    "sensor_drift":    (_inj_sensor_drift,    [11]),
    "fuzzy_flood":     (_inj_fuzzy_flood,     list(range(N_CHANNELS))),
}


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------
class CANBusSimulator:
    """
    Generates ambient (normal) and attack (anomalous) DataFrames that are
    schema-identical to the ROAD dataset.

    Public API
    ----------
    generate_ambient(n_vehicles, duration_s, sampling_hz) → pd.DataFrame
    generate_attacks(n_vehicles, duration_s, sampling_hz, anomaly_ratio) → pd.DataFrame

    Both methods return DataFrames with columns:
        Signal_1_of_ID … Signal_22_of_ID | Label | anomaly_type | vehicle_id

    Downstream code (VehicleDatasetBuilder, normaliser, evaluator) calls
    these two methods and never needs to know whether it is using synthetic
    or real data.
    """

    def __init__(self, seed: int = 42) -> None:
        self.master_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def generate_ambient(
        self,
        n_vehicles: int = 5,
        duration_s: int = 7200,
        sampling_hz: int = 10,
        noise_std: float = 0.05,
    ) -> pd.DataFrame:
        """
        All-normal driving data → mirrors ambient/ folder of ROAD dataset.
        Label=0 for every row.
        """
        dfs = []
        for i in range(n_vehicles):
            vid = f"AMB-{i+1:03d}"
            seed_i = int(self.master_rng.integers(0, 99999))
            df = self._generate_single_ambient(
                vehicle_id=vid, duration_s=duration_s,
                sampling_hz=sampling_hz, noise_std=noise_std, seed=seed_i,
            )
            dfs.append(df)
        result = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Ambient generated — {} vehicles × {:,}s @ {}Hz = {:,} rows total",
            n_vehicles, duration_s, sampling_hz, len(result),
        )
        return result

    def generate_attacks(
        self,
        n_vehicles: int = 2,
        duration_s: int = 1800,
        sampling_hz: int = 10,
        anomaly_ratio: float = 0.08,
        noise_std: float = 0.05,
    ) -> pd.DataFrame:
        """
        Attack data → mirrors attacks/ folder of ROAD dataset.
        Each row has Label ∈ {0, 1} and anomaly_type label.
        """
        dfs = []
        for i in range(n_vehicles):
            vid = f"ATK-{i+1:03d}"
            seed_i = int(self.master_rng.integers(0, 99999))
            df = self._generate_single_attack(
                vehicle_id=vid, duration_s=duration_s,
                sampling_hz=sampling_hz, anomaly_ratio=anomaly_ratio,
                noise_std=noise_std, seed=seed_i,
            )
            dfs.append(df)
        result = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Attacks generated — {} vehicles × {:,}s @ {}Hz = {:,} rows | anomaly_rate={:.2%}",
            n_vehicles, duration_s, sampling_hz, len(result), result["Label"].mean(),
        )
        return result

    # ------------------------------------------------------------------
    # Private — single-vehicle generation
    # ------------------------------------------------------------------
    def _generate_single_ambient(
        self, vehicle_id: str, duration_s: int,
        sampling_hz: int, noise_std: float, seed: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        T = duration_s * sampling_hz
        signals = self._ar_process(T, noise_std, rng)
        self._apply_cross_channel_correlations(signals)
        self._clip_to_bounds(signals)

        timestamps = pd.date_range("2024-01-01", periods=T,
                                   freq=f"{1000 // sampling_hz}ms")
        df = pd.DataFrame(signals, index=timestamps, columns=CHANNEL_NAMES)
        df["Label"]  = 0
        df["anomaly_type"] = "normal"
        df["vehicle_id"]  = vehicle_id
        logger.debug("Ambient vehicle={} | rows={:,}", vehicle_id, T)
        return df

    def _generate_single_attack(
        self, vehicle_id: str, duration_s: int,
        sampling_hz: int, anomaly_ratio: float,
        noise_std: float, seed: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        T = duration_s * sampling_hz
        signals = self._ar_process(T, noise_std, rng)
        self._apply_cross_channel_correlations(signals)

        events = self._inject_anomaly_events(signals, T, anomaly_ratio, sampling_hz, rng)

        self._clip_to_bounds(signals)

        Label  = np.zeros(T, dtype=np.int32)
        anomaly_type = np.full(T, "normal", dtype=object)
        for ev in events:
            Label[ev.start_idx:ev.end_idx]  = 1
            anomaly_type[ev.start_idx:ev.end_idx] = ev.anomaly_type

        timestamps = pd.date_range("2024-01-01", periods=T,
                                   freq=f"{1000 // sampling_hz}ms")
        df = pd.DataFrame(signals, index=timestamps, columns=CHANNEL_NAMES)
        df["Label"]  = Label
        df["anomaly_type"] = anomaly_type
        df["vehicle_id"]  = vehicle_id
        logger.debug(
            "Attack vehicle={} | rows={:,} | anomaly_rate={:.2%}",
            vehicle_id, T, Label.mean(),
        )
        return df

    # ------------------------------------------------------------------
    # Signal generation helpers
    # ------------------------------------------------------------------
    def _ar_process(self, T: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
        """AR(1) process per channel; shape (T, C)."""
        signals = np.zeros((T, N_CHANNELS), dtype=np.float32)
        signals[0] = [s.mean for s in _SPECS]
        for t in range(1, T):
            for i, spec in enumerate(_SPECS):
                innovation = rng.normal(0, spec.std * noise_std)
                signals[t, i] = (
                    spec.ar * signals[t - 1, i]
                    + (1 - spec.ar) * spec.mean
                    + innovation
                )
        return signals

    def _apply_cross_channel_correlations(self, signals: np.ndarray) -> None:
        """Inject realistic powertrain correlations."""
        # RPM ∝ throttle; speed ∝ RPM
        signals[:, 0] += 8.0  * signals[:, 2]    # RPM ← throttle
        signals[:, 1] += 0.02 * signals[:, 0]    # speed ← RPM
        signals[:, 5] = signals[:, 0] * 0.12     # torque proxy ← RPM

    def _clip_to_bounds(self, signals: np.ndarray) -> None:
        for i, spec in enumerate(_SPECS):
            signals[:, i] = np.clip(signals[:, i], spec.lo, spec.hi)

    def _inject_anomaly_events(
        self, signals: np.ndarray, T: int,
        anomaly_ratio: float, sampling_hz: int,
        rng: np.random.Generator,
    ) -> List[AnomalyEvent]:
        target = int(T * anomaly_ratio)
        min_gap = 5 * sampling_hz
        occupied = np.zeros(T, dtype=bool)
        events: List[AnomalyEvent] = []
        anom_names = list(_INJECTORS.keys())

        for _ in range(300):
            if occupied.sum() >= target:
                break
            atype = rng.choice(anom_names)
            dur   = int(rng.uniform(3, 30) * sampling_hz)
            start = int(rng.uniform(min_gap, T - dur - min_gap))
            end   = start + dur

            check_s = max(0, start - min_gap)
            check_e = min(T, end + min_gap)
            if occupied[check_s:check_e].any():
                continue

            sev = float(rng.uniform(0.4, 1.0))
            fn, affected = _INJECTORS[atype]
            fn(signals, start, end, sev, rng=rng)
            occupied[start:end] = True
            events.append(AnomalyEvent(atype, start, end, sev, affected))

        logger.debug(
            "Injected {} events ({} types) — actual_rate={:.2%}",
            len(events), len({e.anomaly_type for e in events}),
            occupied.sum() / T,
        )
        return events
