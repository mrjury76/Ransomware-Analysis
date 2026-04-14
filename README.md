# Ransomware Memory Forensics — Stage Classification Pipeline

An end-to-end pipeline for extracting forensic features from ransomware memory snapshots and training a family-agnostic stage classifier using machine learning.

> **Note:** This is an academic research project. The pipeline is designed to run in a controlled, isolated lab environment and is not intended for deployment.

---

## Overview

1. **Feature Extraction** — Per-plugin Volatility CSVs are aggregated into a single ML-ready feature matrix (`features.csv`)
2. **Training** — A classifier predicts the ransomware execution stage from memory forensics alone, without using family identity as a feature

The core research question is whether behavioral indicators extracted from memory can identify the stage of ransomware execution — and whether that generalizes to families the model has never seen.

---

## Families

Four ransomware families and a benign baseline:

- **WannaCry** — crypto-ransomware with SMB worm propagation
- **Cerber** — multi-stage crypto-ransomware
- **Jigsaw** — screen locker with incremental file deletion
- **Dharma** — traditional file-encrypting ransomware
- **Benign** — randomly selected user applications (Notepad, Firefox, Word, etc.) used as a negative-class baseline

### Stage labels (`stage_hint`)

| Value | Meaning |
|---|---|
| 0 | Benign (baseline — no malware present) |
| 1 | Pre-launch (malware loaded, not yet active) |
| 2 | Pre-encryption (malware active, no encryption yet) |
| 3 | Encrypting |
| 4 | Post-encryption |

---

## Setup

### Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install pandas scikit-learn xgboost joblib matplotlib
  ```

### Dataset

The dataset is a **separate git repo**. Clone it alongside this repo, then point `--scan-dir` at it when running the pipeline. The dataset contains pre-generated Volatility plugin CSVs — you do **not** need Volatility 3 or WSL installed.

Each snapshot folder in the dataset has:
- `meta.json` — metadata (`family`, `stage_hint`, timing info)
- Per-plugin CSVs — `windows.pslist.csv`, `windows.dlllist.csv`, etc.

```
dataset_root/
  WannaCry_20260326_140000/
    T015_rep01/
      meta.json
      windows.pslist.csv
      windows.dlllist.csv
      ...
    T030_rep01/
      ...
  Cerber_20260326_150000/
    ...
  Benign_20260409_020100/
    T015_rep01/
      meta.json
      windows.pslist.csv
      ...
```

---

## Running the Pipeline

The main entry point is `run_pipeline.py`. It runs feature extraction and model training in sequence.

### Quick start

```bash
# Extract features + train model on all families
python3 run_pipeline.py --scan-dir /path/to/dataset --skip-analysis

# Train only WannaCry vs Benign
python3 run_pipeline.py --scan-dir /path/to/dataset --skip-analysis --family WannaCry

# Extract features only (no training)
python3 run_pipeline.py --scan-dir /path/to/dataset --skip-analysis --skip-training

# Skip leave-one-out evaluation (faster)
python3 run_pipeline.py --scan-dir /path/to/dataset --skip-analysis --no-loo
```

> **Important:** Always pass `--skip-analysis` — the Volatility step (Step 1) requires a WSL + Volatility 3 setup that is specific to the lab machine. The plugin CSVs in the dataset are already generated.

### All flags

| Flag | Required | Description |
|---|---|---|
| `--scan-dir` | Yes | Root of the dataset directory (the cloned dataset repo) |
| `--skip-analysis` | **Yes** for teammates | Skips the Volatility step (plugin CSVs already exist in the dataset) |
| `--family` | No | Run only this family + Benign (e.g. `--family WannaCry`). Omit to use all families |
| `--out` | No | Output path for `features.csv`. Default: `<repo>/output/features.csv` |
| `--model-out` | No | Output directory for model artifacts. Default: `<repo>/model_results/runNN` (auto-increments) |
| `--skip-training` | No | Stop after writing `features.csv` — don't train models |
| `--no-loo` | No | Skip leave-one-family-out evaluation (faster, but no generalization metrics) |

### What each step produces

| Step | Script | Input | Output |
|---|---|---|---|
| Features | `extract_features.py` | Plugin CSVs + `meta.json` | `features.csv` — one row per snapshot, ~110 feature columns |
| Training | `train_stage_model.py` | `features.csv` | Trained models, confusion matrices, LOO reports in `model_results/runNN/` |

### Output structure

```
model_results/
  run01/
    stage_hint/
      cm_standard.png          # confusion matrix (80/20 split)
      report_standard.csv      # classification report
      cm_loo_WannaCry.png      # LOO confusion matrix per family
      report_loo_WannaCry.csv
      ...
    behaviour_binary/
      ...
    behavior_stage/
      ...
```

---

## Running scripts individually

Each script can be run standalone if you don't want to use the pipeline:

```bash
# Feature extraction only
python3 extract_features.py --scan-dir /path/to/dataset --out features.csv

# Model training only (on a pre-built features.csv)
python3 train_stage_model.py --features features.csv --out model_output/

# Model training — specific label type
python3 train_stage_model.py --features features.csv --label behavior_stage

# Model training — skip LOO
python3 train_stage_model.py --features features.csv --no-loo
```

### `train_stage_model.py` flags

| Flag | Required | Description |
|---|---|---|
| `--features` | Yes | Path to `features.csv` |
| `--out` | No | Output directory. Default: `model_output/` |
| `--no-loo` | No | Skip leave-one-family-out evaluation |
| `--label` | No | Which label to train on: `stage_hint`, `behaviour_binary`, `behavior_stage`, or `all` (default: `all`) |

---

## Data Collection (Lab Only)

> This section is only relevant if you are collecting new memory snapshots. Teammates running the pipeline on an existing dataset can ignore this.

The collection script `wannacry_automate_v3.ps1` runs on the Windows host and requires VMware Workstation + an isolated Windows 10 guest VM. It automates VM revert, malware/benign launch, memory snapshot capture, and vmem extraction.

```powershell
.\wannacry_automate_v3.ps1 -FamilyArg WannaCry -NumRunsArg 5
.\wannacry_automate_v3.ps1 -FamilyArg Benign -NumRunsArg 5
.\wannacry_automate_v3.ps1 -FamilyArg ALL -NumRunsArg 5
```
