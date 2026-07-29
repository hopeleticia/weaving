# WAVE: PLC-Based Industrial Anomaly Detection Benchmark

## Overview

WAVE is a benchmark framework for industrial anomaly detection using PLC sensor logs collected from a weaving machine. The project evaluates multiple baseline anomaly detection models under the same preprocessing pipeline and experimental protocol.

The objective is to establish strong baseline results before developing a novel anomaly detection framework.

---

## Dataset

The dataset consists of eight daily PLC log files collected from a weaving machine.

```
dataset/
├── plc_log_20260708.csv
├── plc_log_20260709.csv
├── plc_log_20260710.csv
├── plc_log_20260711.csv
├── plc_log_20260712.csv
├── plc_log_20260713.csv
├── plc_log_20260714.csv
└── plc_log_20260715.csv
```

Each file contains approximately 850k PLC records with timestamps and multiple industrial sensor measurements.

### Main Features

- Infeed Speed
- Motor RPM
- Loadcell Sensors
- Dancer Pressure
- Torque
- Temperature
- Humidity
- Weaving Status

Alarm_Code is used as a weak anomaly label for evaluation only.

---

## Data Split

Chronological splitting is used to prevent temporal data leakage.

| Dataset | Files |
|----------|-------|
| Training | 20260708 – 20260713 |
| Validation | 20260714 |
| Testing | 20260715 |

---

## Models

The following baseline models are evaluated.

### 1. Chronos-Bolt Tiny

- Zero-shot pretrained forecasting model
- Forecasting error based anomaly detection

---

### 2. LSTM Autoencoder

- Sequence reconstruction
- Reconstruction error based anomaly detection

---

### 3. Tabular MLP Autoencoder

- Row-wise reconstruction
- Reconstruction error based anomaly detection

---

### Future Baselines

- Transformer Autoencoder
- TCN Autoencoder
- Isolation Forest
- One-Class SVM
- XGBoost
- Proposed Model

---

## Project Structure

```
WAVE/
│
├── dataset/
│
├── models/
│
├── results/
│   ├── chronos_single_signal/
│   ├── lstm_ae_full_dataset/
│   └── mlp_ae_full_dataset/
│
├── scripts/
│   ├── 01_chronos_single_signal.py
│   ├── 02_lstm_ae_full_dataset.py
│   └── 03_mlp_ae_full_dataset.py
│
└── README.md
```

---

## Installation

Create a virtual environment

```bash
python -m venv .wave
```

Activate

Windows

```bash
.wave\Scripts\activate
```

Linux

```bash
source .wave/bin/activate
```

Install dependencies

```bash
pip install torch pandas numpy matplotlib scikit-learn tqdm joblib transformers chronos-forecasting
```

---

## Running Experiments

### Chronos Baseline

```bash
python scripts/01_chronos_single_signal.py
```

---

### LSTM Autoencoder

```bash
python scripts/02_lstm_ae_full_dataset.py
```

---

### Tabular MLP Autoencoder

```bash
python scripts/03_mlp_ae_full_dataset.py
```

---

## Evaluation Metrics

The following metrics are computed.

- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- False Positive Rate
- False Negative Rate
- Balanced Accuracy
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix
- Training Time
- Inference Time
- Throughput
- Model Size

---

## Outputs

Each experiment produces

- Model checkpoint
- Training history
- Reconstruction error
- Threshold
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Summary CSV
- Summary JSON

---

## Current Findings

### Chronos

- Extremely fast zero-shot inference.
- Unable to model abrupt PLC transitions.
- Useful as a forecasting baseline.

### LSTM Autoencoder

- Stable training convergence.
- Good modeling of normal operating behavior.
- Poor anomaly recall.

### MLP Autoencoder

- Excellent reconstruction of normal samples.
- Very low false positive rate.
- Failed to detect most weakly labeled anomalies.

---

## Future Work

The benchmark will be extended with

- Transformer Autoencoder
- TCN Autoencoder
- Isolation Forest
- One-Class SVM
- XGBoost
- Hybrid Deep Learning Model
- Adaptive Thresholding
- Multivariate Anomaly Scoring

The final objective is to develop a robust industrial anomaly detection framework that significantly outperforms conventional reconstruction-based methods.

---

## Citation

If you use this benchmark, please cite the corresponding publication when available.
