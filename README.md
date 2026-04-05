# Vehicle Edge AI — Unsupervised Anomaly Detection

End-to-end production-grade pipeline for detecting and attributing vehicle health
anomalies from multi-channel CAN bus sensor streams.

---

## Architecture

```
vehicle-anomaly-detection/
├── configs/
│   └── config.yaml                  <- All tunables — no magic numbers in code
├── src/
│   ├── data/
│   │   ├── can_simulator.py         <- Physics-aware CAN bus simulator
│   │   │                               generate_ambient() / generate_attacks()
│   │   │                               22-channel Signal_N_of_ID schema
│   │   └── dataset.py               <- SignalNormaliser, extract_windows,
│   │                                   VehicleDatasetBuilder, load_road_dataset
│   ├── models/
│   │   ├── lstm_autoencoder.py      <- Causal LSTM-AE (unidirectional)
│   │   ├── isolation_forest.py      <- IF on statistical window features
│   │   └── ensemble.py              <- Weighted ensemble + threshold calibration
│   ├── training/
│   │   └── trainer.py               <- Early stopping, cosine LR, TensorBoard
│   ├── evaluation/
│   │   └── metrics.py               <- PR curve, ROC-AUC, per-type F1, plots
│   ├── correlation/
│   │   └── signal_correlator.py     <- Cross-channel lag correlator → root cause
│   └── utils/
│       └── logging.py               <- Structured logging (loguru)
├── train.py                         <- python train.py
├── evaluate.py                      <- python evaluate.py --explain
├── requirements.txt
└── tests/
    └── test_simulator_and_dataset.py   <- 27 unit tests
```

---

## Dataset

### Option A — ROAD Dataset (recommended for real results)

Download from **https://0xsam.com/road/** and extract to:

```
data/road/
    ambient/
        ambient_dyno_drive_basic_long.csv
        ambient_dyno_drive_basic_short.csv
        ... (all ambient CSVs here)
    attacks/
        correlated_signal_attack_1.csv
        ... (all attack CSVs here)
```

Then set in `configs/config.yaml`:
```yaml
data:
  mode: road
  road_dataset_path: data/road/signal_extractions
```

### Option B — Synthetic (works out of the box, no download)

```yaml
data:
  mode: synthetic
```

The simulator produces two DataFrames with **identical schema** to the ROAD CSVs:

| DataFrame | Source | `Label` | Use |
|---|---|---|---|
| `df_ambient` | `generate_ambient()` | always 0 | Train + Val |
| `df_attacks` | `generate_attacks()` | 0 or 1 | Test only |

---

## Column Schema

Both the simulator and the ROAD loader produce identical columns:

```
Signal_1_of_ID  …  Signal_22_of_ID  |  Label  |  anomaly_type  |  vehicle_id
```

**Subsystem mapping (synthetic physics / ROAD correlation):**

| Channels | Subsystem | Signals |
|---|---|---|
| 1–6 | powertrain | RPM, speed, throttle, gear, clutch, torque |
| 7–10 | electrical | battery voltage, CAN latency, bus load, aux voltage |
| 11–14 | thermal | coolant temp, oil pressure, intake temp, exhaust temp |
| 15–22 | dynamics | accel X/Y/Z, yaw, steering, brake, front/rear suspension |

---

## Data Split Strategy

```
df_ambient  ──── temporal 80 / 20 ────►  TRAIN (all normal)
                                          VAL   (all normal — used for threshold calibration)

df_attacks  -------------------------->  TEST  (labelled, ground truth)
```

**Why temporal split?** Shuffling time-series causes temporal leakage — the model
sees future context during training. Temporal split preserves causal direction and
accurately simulates production deployment.

**Why val is all-normal?** The val set comes from ambient data so threshold
calibration uses the 95th-percentile of normal-window scores (≈5 % FPR target).
On real ROAD data where you have labelled attacks, use `--calibrate` at evaluation
time with attack samples included in your validation fold.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run unit tests (27 tests, ~35s)
pytest tests/ -v

# 3. Train
python train.py                          # synthetic data, all defaults
python train.py --data-mode road         # real ROAD dataset
python train.py --epochs 50 --lr 5e-4   # override hyperparameters

# 4. Evaluate
python evaluate.py                       # metrics + plots
python evaluate.py --explain             # + root-cause attribution CSV
python evaluate.py --calibrate           # re-calibrate threshold first
```

**Outputs after training + evaluation:**
```
artifacts/
    checkpoints/
        best_model.pt
        isolation_forest.joblib
        normaliser.pkl
        ensemble_meta.pkl
        config_snapshot.yaml
    reports/
        metrics.json
        pr_curve.png
        roc_curve.png
        score_distribution.png
        confusion_matrix.png
        per_type_f1.png
        training_history.png
        root_cause_summary.csv     (with --explain)
```

---

## Model Architecture

### LSTM Autoencoder
- **Why unidirectional?** Bidirectional LSTM requires future frames — illegal at
  real-time edge inference. Causal model only.
- **Training:** MSE reconstruction loss on **normal windows only** (unsupervised).
- **Inference:** Anomaly score = mean MSE reconstruction error per window.
  High error → model failed to reconstruct → anomaly.
- **Parameters:** 248,782 (22 channels, hidden=96, latent=24, layers=2)

### Isolation Forest
- **Why complement LSTM-AE?** IF detects global point outliers in feature space;
  LSTM-AE detects contextual/temporal sequence anomalies. Complementary failure modes.
- **Features:** 7 statistics × 22 channels = 154 features per window
  (mean, std, min, max, range, skewness proxy, mean |Δx|)
- **Training:** Normal windows only.

### Ensemble
- Combined score = 0.60 × AE_score + 0.40 × IF_score
- Threshold calibrated on val set: 95th-percentile of normal-window scores

### Signal Correlator
- Cross-channel Pearson correlation with lag sweep (±5 seconds)
- Identifies which channel leads an anomaly event (root cause)
- Maps guilty channels -> powertrain / electrical / thermal / dynamics sub-system

---

## Attack Taxonomy (Simulator)

| Type | Channels | Real-world cause |
|---|---|---|
| `voltage_sag` | battery_v, CAN latency, bus load | Alternator fault / parasitic drain |
| `coolant_spike` | coolant temp, oil pressure, exhaust | Thermostat failure / coolant leak |
| `rpm_dropout` | RPM, CAN latency | ECU comms loss / CAN frame drop |
| `latency_storm` | CAN latency, bus load, speed | Bus contention / software regression |
| `crash_precursor` | RPM, latency, accel X/Y, yaw | Multi-system composite fault |
| `sensor_drift` | oil pressure | Gradual sensor degradation |
| `fuzzy_flood` | random multi-channel | Fuzzy CAN injection attack |

---

## Key Engineering Decisions

1. **Unidirectional LSTM** — causal, real-time safe; bidirectional requires future frames
2. **Normaliser fit on normal train data only** — fitting on anomalous data leaks
   anomaly scale into the normaliser; covered by unit test `test_normaliser_fit_on_ambient_only_separates_attacks`
3. **Temporal split, never shuffle** — shuffling time-series = temporal leakage
4. **Train on normal windows only** — LSTM-AE learns the normal manifold;
   anomalies fall off it at inference
5. **IF on engineered features, AE on raw** — complementary: IF catches point outliers,
   AE catches temporal/contextual patterns
6. **Physics-aware anomaly injection** — correlated cross-channel events, not
   independent noise (battery sag → CAN latency spike)
7. **Percentile threshold calibration** — principled even when val is all-normal;
   targets a fixed FPR on normal traffic
8. **Unified (df_ambient, df_attacks) contract** — identical code path for synthetic
   and real ROAD data; validated by 27 unit tests

---

## Switching to the ROAD Dataset

```bash
# 1. Download from https://0xsam.com/road/
# 2. Extract to data/road/ following the layout above

# 3. In configs/config.yaml:
#    data:
#      mode: road
#      road_dataset_path: data/road/

python train.py --data-mode road
python evaluate.py --explain --calibrate
```
