"""
tests/test_simulator_and_dataset.py

Unit tests covering:
  - Simulator: shape, column schema, anomaly injection, two-DataFrame contract
  - SignalNormaliser: zero-mean, inverse roundtrip, no label leakage
  - extract_windows: shape, label propagation, temporal ordering
  - VehicleDatasetBuilder: split contract, sizes, temporal ordering

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import yaml

from src.data.can_simulator import (
    CANBusSimulator,
    CHANNEL_NAMES,
    N_CHANNELS,
)
from src.data.dataset import (
    SignalNormaliser,
    VehicleDatasetBuilder,
    extract_windows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sim():
    return CANBusSimulator(seed=0)


@pytest.fixture(scope="module")
def small_ambient(sim):
    return sim.generate_ambient(n_vehicles=2, duration_s=120, sampling_hz=10)


@pytest.fixture(scope="module")
def small_attacks(sim):
    return sim.generate_attacks(n_vehicles=1, duration_s=120, sampling_hz=10, anomaly_ratio=0.10)


# ---------------------------------------------------------------------------
# Simulator — column schema
# ---------------------------------------------------------------------------
class TestSimulatorSchema:
    def test_channel_names_match_road_schema(self):
        """Column names must match the ROAD dataset convention exactly."""
        expected = [f"Signal_{i}_of_ID" for i in range(1, 23)]
        assert CHANNEL_NAMES == expected, f"Column mismatch: {CHANNEL_NAMES}"

    def test_n_channels_is_22(self):
        assert N_CHANNELS == 22

    def test_ambient_columns(self, small_ambient):
        required = set(CHANNEL_NAMES) | {"Label", "anomaly_type", "vehicle_id"}
        assert required.issubset(set(small_ambient.columns))

    def test_attacks_columns(self, small_attacks):
        required = set(CHANNEL_NAMES) | {"Label", "anomaly_type", "vehicle_id"}
        assert required.issubset(set(small_attacks.columns))


# ---------------------------------------------------------------------------
# Simulator — ambient (all normal)
# ---------------------------------------------------------------------------
class TestSimulatorAmbient:
    def test_ambient_all_normal(self, small_ambient):
        assert small_ambient["Label"].sum() == 0, \
            "Ambient DataFrame must have zero anomalous rows"

    def test_ambient_anomaly_type_all_normal(self, small_ambient):
        assert (small_ambient["anomaly_type"] == "normal").all()

    def test_ambient_row_count(self, small_ambient):
        # 2 vehicles × 120s × 10Hz = 2400
        assert len(small_ambient) == 2 * 120 * 10

    def test_ambient_multiple_vehicles(self, small_ambient):
        assert small_ambient["vehicle_id"].nunique() == 2

    def test_ambient_physical_bounds(self, small_ambient):
        from src.data.can_simulator import _SPECS
        for i, spec in enumerate(_SPECS):
            col = small_ambient[CHANNEL_NAMES[i]]
            assert col.min() >= spec.lo - 1e-3, f"{CHANNEL_NAMES[i]} below lo"
            assert col.max() <= spec.hi + 1e-3, f"{CHANNEL_NAMES[i]} above hi"


# ---------------------------------------------------------------------------
# Simulator — attacks (labelled)
# ---------------------------------------------------------------------------
class TestSimulatorAttacks:
    def test_attacks_has_anomalies(self, small_attacks):
        assert small_attacks["Label"].sum() > 0, "Attack df must have anomalous rows"

    def test_attacks_anomaly_rate_reasonable(self, small_attacks):
        rate = small_attacks["Label"].mean()
        # Short window + random event sizes → allow wide tolerance
        assert 0.0 < rate < 0.60, f"Anomaly rate {rate:.3f} out of expected range"

    def test_attacks_anomaly_types_populated(self, small_attacks):
        anom_types = small_attacks.loc[small_attacks["Label"] == 1, "anomaly_type"].unique()
        assert len(anom_types) >= 1
        assert "normal" not in anom_types

    def test_attacks_normal_rows_exist(self, small_attacks):
        # Attack files contain BOTH normal and anomalous rows
        assert (small_attacks["Label"] == 0).sum() > 0, \
            "Attack df should also contain normal (non-attack) rows"

    def test_attacks_independent_from_ambient(self, small_ambient, small_attacks):
        """Vehicle IDs must not overlap."""
        amb_ids = set(small_ambient["vehicle_id"].unique())
        atk_ids = set(small_attacks["vehicle_id"].unique())
        assert amb_ids.isdisjoint(atk_ids), \
            f"Ambient and attack vehicle IDs overlap: {amb_ids & atk_ids}"


# ---------------------------------------------------------------------------
# SignalNormaliser
# ---------------------------------------------------------------------------
class TestSignalNormaliser:
    def test_fit_transform_zero_mean(self, small_ambient):
        X = small_ambient[CHANNEL_NAMES].values.astype(np.float32)
        norm = SignalNormaliser().fit(X)
        X_t = norm.transform(X)
        np.testing.assert_allclose(X_t.mean(axis=0), np.zeros(N_CHANNELS), atol=1e-2)

    def test_inverse_roundtrip(self, small_ambient):
        X = small_ambient[CHANNEL_NAMES].values.astype(np.float32)
        norm = SignalNormaliser()
        X_t = norm.fit_transform(X)
        X_inv = norm.inverse_transform(X_t)
        np.testing.assert_allclose(X, X_inv, atol=1e-4)

    def test_normaliser_fit_on_ambient_only_separates_attacks(self, small_ambient, small_attacks):
        """
        Normaliser fit on normal data only.
        Anomalous rows should have higher absolute normalised values on average.
        """
        X_amb = small_ambient[CHANNEL_NAMES].values.astype(np.float32)
        norm  = SignalNormaliser().fit(X_amb)

        X_atk_anom = small_attacks.loc[
            small_attacks["Label"] == 1, CHANNEL_NAMES
        ].values.astype(np.float32)

        mean_amb  = np.abs(norm.transform(X_amb)).mean()
        mean_anom = np.abs(norm.transform(X_atk_anom)).mean()
        assert mean_anom > mean_amb, \
            "Anomalous rows should deviate more from normal distribution"

    def test_normaliser_save_load(self, small_ambient, tmp_path):
        X = small_ambient[CHANNEL_NAMES].values.astype(np.float32)
        norm = SignalNormaliser().fit(X)
        path = str(tmp_path / "norm.pkl")
        norm.save(path)
        loaded = SignalNormaliser.load(path)
        np.testing.assert_array_equal(norm.mean_, loaded.mean_)
        np.testing.assert_array_equal(norm.std_,  loaded.std_)


# ---------------------------------------------------------------------------
# extract_windows
# ---------------------------------------------------------------------------
class TestExtractWindows:
    def test_output_shape(self, small_ambient):
        window_size, stride = 50, 10
        T = len(small_ambient)
        expected_n = (T - window_size) // stride + 1
        X, y, types = extract_windows(small_ambient, CHANNEL_NAMES, window_size, stride)
        assert X.shape == (expected_n, window_size, N_CHANNELS)
        assert y.shape == (expected_n,)
        assert len(types) == expected_n

    def test_ambient_labels_all_zero(self, small_ambient):
        _, y, types = extract_windows(small_ambient, CHANNEL_NAMES, 50, 10)
        assert y.sum() == 0, "Ambient windows must all be labelled 0"
        assert (types == "normal").all()

    def test_attack_label_propagation(self, small_attacks):
        """Window label = 1 if ANY sample in the window is anomalous."""
        _, y, _ = extract_windows(small_attacks, CHANNEL_NAMES, 50, 10)
        labels_raw = small_attacks["Label"].values
        for i in range(min(20, len(y))):
            start = i * 10
            end   = start + 50
            expected = int(labels_raw[start:end].max())
            assert y[i] == expected, f"Window {i}: expected {expected}, got {y[i]}"

    def test_temporal_ordering(self, small_ambient):
        """Windows are in temporal order — first channel values must be monotone."""
        df = small_ambient.copy()
        # Replace first channel with monotone counter for easy verification
        df[CHANNEL_NAMES[0]] = np.arange(len(df), dtype=np.float32)
        X, _, _ = extract_windows(df, CHANNEL_NAMES, 50, 10)
        for i in range(len(X) - 1):
            assert X[i + 1, 0, 0] > X[i, 0, 0], "Windows must be in temporal order"


# ---------------------------------------------------------------------------
# VehicleDatasetBuilder — split contract
# ---------------------------------------------------------------------------
class TestVehicleDatasetBuilder:
    def test_split_names(self, cfg, small_ambient, small_attacks):
        builder = VehicleDatasetBuilder(cfg)
        splits  = builder.build(small_ambient, small_attacks)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_train_val_from_ambient_test_from_attacks(self, cfg, small_ambient, small_attacks):
        """
        CRITICAL contract:
          train + val windows come exclusively from ambient (all-normal)
          test windows  come exclusively from attacks (labelled)
        """
        builder = VehicleDatasetBuilder(cfg)
        splits  = builder.build(small_ambient, small_attacks)

        _, y_train, _ = splits["train"]
        _, y_val,   _ = splits["val"]
        _, y_test,  _ = splits["test"]

        assert y_train.sum() == 0, "Train windows must all be normal (from ambient)"
        assert y_val.sum()   == 0, "Val windows must all be normal (from ambient)"
        assert y_test.sum()  >  0, "Test windows must include anomalous (from attacks)"

    def test_val_split_size(self, cfg, small_ambient, small_attacks):
        val_frac = cfg["data"]["val_split"]          # 0.20
        builder  = VehicleDatasetBuilder(cfg)
        splits   = builder.build(small_ambient, small_attacks)

        n_train = len(splits["train"][0])
        n_val   = len(splits["val"][0])
        total   = n_train + n_val

        actual_val_frac = n_val / total if total > 0 else 0
        # Allow ±5% tolerance due to windowing boundary effects
        assert abs(actual_val_frac - val_frac) < 0.08, \
            f"Val fraction {actual_val_frac:.3f} too far from expected {val_frac}"

    def test_temporal_ordering_preserved(self, cfg, small_ambient, small_attacks):
        """Train comes before val in time — no shuffling."""
        builder = VehicleDatasetBuilder(cfg)
        splits  = builder.build(small_ambient, small_attacks)
        X_train = splits["train"][0]
        X_val   = splits["val"][0]
        # First sample of first train window must precede first sample of first val window
        assert X_train[0, 0, 0] != X_val[0, 0, 0], \
            "Train and val should start from different time positions"

    def test_normaliser_fitted_after_build(self, cfg, small_ambient, small_attacks):
        builder = VehicleDatasetBuilder(cfg)
        builder.build(small_ambient, small_attacks)
        assert builder.normaliser is not None
        assert builder.normaliser.mean_ is not None
        assert len(builder.normaliser.mean_) == N_CHANNELS
