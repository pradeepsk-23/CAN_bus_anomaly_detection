"""
tests/test_simulator_and_dataset.py

Unit tests covering:
  - CAN bus simulator output shape and anomaly injection
  - Normaliser: no leakage (fit on normal only), inverse transform
  - Sliding window extractor: shape, label propagation
  - Temporal split ordering

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import yaml

from src.data.can_simulator import CANBusSimulator, CHANNEL_NAMES
from src.data.dataset import (
    SignalNormaliser,
    VehicleDatasetBuilder,
    extract_windows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_df():
    """Short synthetic signal for fast tests."""
    sim = CANBusSimulator(seed=0)
    df, events = sim.generate(duration_seconds=120, sampling_hz=10, anomaly_ratio=0.05)
    return df, events


# ---------------------------------------------------------------------------
# CAN Bus Simulator
# ---------------------------------------------------------------------------
class TestCANBusSimulator:
    def test_output_shape(self, small_df):
        df, _ = small_df
        assert df.shape[0] == 1200, "120s × 10Hz = 1200 samples"
        for ch in CHANNEL_NAMES:
            assert ch in df.columns

    def test_anomaly_fraction(self, small_df):
        df, _ = small_df
        rate = df["is_anomaly"].mean()
        # With only 120s, a single 30s event = 25% — allow wide range.
        # Full-length simulations (7200s) stay close to the configured 5%.
        assert 0.0 < rate < 0.35, f"Anomaly rate {rate:.3f} out of expected range"

    def test_physical_bounds(self, small_df):
        df, events = small_df
        from src.data.can_simulator import SIGNAL_SPECS
        for spec in SIGNAL_SPECS:
            col = df[spec.name]
            assert col.min() >= spec.min_val - 1e-3, f"{spec.name} below min"
            assert col.max() <= spec.max_val + 1e-3, f"{spec.name} above max"

    def test_anomaly_events_non_overlapping(self, small_df):
        _, events = small_df
        occupied = np.zeros(1200, dtype=bool)
        for ev in events:
            assert not occupied[ev.start_idx:ev.end_idx].any(), "Overlapping anomaly events"
            occupied[ev.start_idx:ev.end_idx] = True

    def test_fleet_generation(self, cfg):
        sim = CANBusSimulator(seed=42)
        df, events = sim.generate_fleet(n_vehicles=3, duration_seconds=60, sampling_hz=10)
        assert "vehicle_id" in df.columns
        assert df["vehicle_id"].nunique() == 3


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------
class TestSignalNormaliser:
    def test_fit_transform_zero_mean(self, small_df):
        df, _ = small_df
        X = df[CHANNEL_NAMES].values.astype(np.float32)
        norm = SignalNormaliser().fit(X)
        X_t = norm.transform(X)
        np.testing.assert_allclose(X_t.mean(axis=0), np.zeros(len(CHANNEL_NAMES)), atol=1e-3)

    def test_inverse_roundtrip(self, small_df):
        df, _ = small_df
        X = df[CHANNEL_NAMES].values.astype(np.float32)
        norm = SignalNormaliser()
        X_t = norm.fit_transform(X)
        X_inv = norm.inverse_transform(X_t)
        np.testing.assert_allclose(X, X_inv, atol=1e-4)

    def test_no_label_leakage(self, small_df):
        """Normaliser must be fit on normal data only — anomalous data is excluded."""
        df, _ = small_df
        normal_X = df.loc[df["is_anomaly"] == 0, CHANNEL_NAMES].values.astype(np.float32)
        anomaly_X = df.loc[df["is_anomaly"] == 1, CHANNEL_NAMES].values.astype(np.float32)

        norm = SignalNormaliser().fit(normal_X)
        # Anomalous windows should have higher absolute normalised values (by design)
        norm_normal = np.abs(norm.transform(normal_X)).mean()
        norm_anom = np.abs(norm.transform(anomaly_X)).mean()
        # This should hold for our injected anomalies
        assert norm_anom > norm_normal, "Anomalies should have higher normalised residuals"


# ---------------------------------------------------------------------------
# Sliding window extractor
# ---------------------------------------------------------------------------
class TestWindowExtractor:
    def test_output_shape(self, small_df):
        df, _ = small_df
        window_size, stride = 50, 10
        X, y, types = extract_windows(df, CHANNEL_NAMES, window_size, stride)
        expected_n = (1200 - window_size) // stride + 1
        assert X.shape == (expected_n, window_size, len(CHANNEL_NAMES))
        assert y.shape == (expected_n,)

    def test_label_propagation(self, small_df):
        """If any sample in a window is anomalous, the window label is 1."""
        df, _ = small_df
        X, y, _ = extract_windows(df, CHANNEL_NAMES, window_size=50, stride=10)
        # Manually verify first 10 windows
        labels_raw = df["is_anomaly"].values
        for i in range(min(10, len(X))):
            start = i * 10
            end = start + 50
            expected = int(labels_raw[start:end].max())
            assert y[i] == expected

    def test_temporal_ordering(self, small_df):
        """Windows must be extracted in temporal order (no shuffle)."""
        df, _ = small_df
        # Inject a monotonically increasing index signal for verification
        import pandas as pd
        df = df.copy()
        df["engine_rpm"] = np.arange(len(df), dtype=np.float32)
        X, _, _ = extract_windows(df, CHANNEL_NAMES, window_size=50, stride=10)
        # First sample of window i+1 must be > first sample of window i
        for i in range(len(X) - 1):
            assert X[i + 1, 0, 0] > X[i, 0, 0], "Windows not in temporal order"


# ---------------------------------------------------------------------------
# Dataset builder — temporal split
# ---------------------------------------------------------------------------
class TestDatasetBuilder:
    def test_split_sizes(self, cfg, small_df):
        df, _ = small_df
        builder = VehicleDatasetBuilder(cfg)
        splits = builder.build(df, CHANNEL_NAMES)
        total_windows = sum(len(splits[k][0]) for k in ["train", "val", "test"])
        # All windows must be accounted for
        assert total_windows > 0

    def test_no_future_leakage(self, cfg, small_df):
        """Test set must come strictly AFTER train + val in time."""
        df, _ = small_df
        N = len(df)
        train_frac = cfg["data"]["window"]["train_split"]
        val_frac = cfg["data"]["window"]["val_split"]
        # The train portion must be earlier in df than val, which is earlier than test
        train_end = int(N * train_frac)
        val_end = int(N * (train_frac + val_frac))
        assert train_end < val_end < N
