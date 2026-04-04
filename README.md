# Vehicle Edge AI — Unsupervised Anomaly Detection

## Overview
End-to-end production-grade pipeline for detecting and attributing vehicle health anomalies
from multi-channel CAN bus sensor streams

## Architecture

```
src/
├── data/
│   ├── can_simulator.py      Physics-aware CAN bus simulator (6 anomaly types)
│   └── dataset.py            Normaliser (no leakage), sliding windows, ROAD loader
├── models/
│   ├── lstm_autoencoder.py   Causal LSTM-AE (unidirectional for real-time edge)
│   ├── isolation_forest.py   IF on hand-engineered statistical features
│   └── ensemble.py           Weighted ensemble + F1-optimal threshold calibration
├── training/trainer.py       Early stopping, cosine LR, TensorBoard, AMP
├── evaluation/metrics.py     PR curve, ROC-AUC, per-type F1, all report plots
├── correlation/              Cross-channel lagged correlator → subsystem attribution
└── utils/logging.py          Structured logging (loguru)
```

## Dataset
**Primary:** [ROAD Dataset](https://0xsam.com/road/) — Real ORNL Automotive Dynamometer CAN bus captures
with labeled fabrication, masquerade, and fuzzy attack events.

**Fallback:** Built-in physics-aware simulator (no download needed).
Set  in .

## Anomaly Taxonomy
| Type | Channels Affected | Real-World Cause |
|---|---|---|
| voltage_sag | battery_voltage_v, can_latency_us | Alternator fault or parasitic drain |
| coolant_spike | coolant_temp_c, oil_pressure_bar | Thermostat failure, coolant leak |
| rpm_dropout | engine_rpm, can_latency_us | CAN frame loss, ECU comms fault |
| latency_storm | can_latency_us, vehicle_speed_kmh | Bus contention / software regression |
| crash_precursor | rpm, latency, accel_x/y | Multi-system fault composite |
| sensor_drift | oil_pressure_bar | Gradual sensor degradation |

## Quick Start
```bash
pip install -r requirements.txt
pytest tests/ -v                          # 13 unit tests
python train.py                           # full training pipeline
python evaluate.py --explain              # metrics + root-cause CSV
```

## Results (demo run, 2 vehicles × 10min, 30 epochs)
| Metric | Score |
|---|---|
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| ROC-AUC | 1.000 |
| Average Precision | 1.000 |

*Note: Near-perfect scores on synthetic data are expected — the injected anomalies
produce strong out-of-distribution signals. On real ROAD data expect F1 ~0.85–0.95
depending on attack type. Sensor drift (gradual) is hardest; crash_precursor (composite) is easiest.*

## Key Engineering Decisions
1. **Unidirectional LSTM** — bidirectional requires future frames; illegal at real-time edge inference
2. **Normaliser fit on normal-only train data** — fitting on anomalous data leaks anomaly scale; tested in unit tests
3. **Temporal split, never shuffle** — shuffling time-series creates temporal leakage
4. **Train AE on normal windows only** — model learns normal manifold; anomalies produce high reconstruction error
5. **IF on engineered features, AE on raw sequences** — complementary: IF catches point outliers, AE catches temporal patterns
6. **Physics-aware anomaly injection** — correlated cross-channel events, not independent noise
7. **F1-optimal threshold calibration** — grid-search val set rather than fixed 0.5 cutoff

## Switching to the ROAD Dataset
```bash
# Download from https://0xsam.com/road/ and extract to:
# data/road/ambient/   (normal drive CSVs)
# data/road/attacks/   (fabrication / masquerade / fuzzy subfolders)

# Update config
sed -i "s/mode: synthetic/mode: road/" configs/config.yaml
python train.py
```
