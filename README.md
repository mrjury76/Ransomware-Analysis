# Ransomware Memory Forensics — Stage Classification Pipeline

An end-to-end pipeline for collecting ransomware memory snapshots, extracting forensic features via Volatility 3, and training a family-agnostic stage classifier using machine learning.

---

## Overview

This project automates the full workflow:

1. **Collection** — VMware Workstation executes ransomware samples in an isolated VM and captures memory snapshots at timed intervals
2. **Analysis** — Volatility 3 runs 12 memory forensics plugins on each snapshot
3. **Feature Extraction** — Per-plugin CSVs are aggregated into a single ML-ready feature matrix
4. **Training** — An XGBoost/RandomForest classifier predicts the ransomware execution stage from memory forensics alone, without using family identity

The goal is to determine whether behavioral indicators extracted from memory can identify what stage of execution ransomware is in — and whether that generalizes to ransomware families the model has never seen before.

---

## Ransomware Families

| Family | Type | Notes |
|--------|------|-------|
| WannaCry | Crypto + Worm | Spreads via EternalBlue/SMB |
| Cerber | Crypto | VoIP ransom note, multi-stage |
| Jigsaw | Screen locker | Deletes files incrementally |
| Dharma | Crypto | Appends contact email to extension |

---

## Execution Stages

Snapshots are captured at T+15s, T+30s, T+60s, T+120s, T+180s, T+240s after malware launch. Each snapshot is assigned a stage label based on elapsed time:

| Stage | Label | Time | Description |
|-------|-------|------|-------------|
| 0 | Pre-launch | < 20s | Baseline — malware not yet active |
| 1 | Pre-encryption | 20–50s | Malware executing, no encryption yet |
| 2 | Encrypting | 50–150s | Active file encryption |
| 3 | Post-encryption | > 150s | Encryption complete |

---

## Pipeline Components

### `wannacry_automate_v3.ps1`
PowerShell automation script. For each family and time offset:
- Reverts VM to clean snapshot (`CleanFamily4`)
- Boots VM and launches malware via `vmrun runProgramInGuest`
- Takes a VMware snapshot at the target offset
- Copies the `.vmem` file to `D:\Patrick\VMSnapshots\<Family>_<timestamp>\T<offset>_rep<N>\`
- Writes `meta.json` with family, stage hint, rep, and timing metadata
- After all families complete, calls `run_pipeline.py` in WSL

**Usage:**
```powershell
# Single family
.\wannacry_automate_v3.ps1 -FamilyArg Dharma -NumRunsArg 5

# All families
.\wannacry_automate_v3.ps1 -FamilyArg ALL -NumRunsArg 5
```

---

### `autovol4_new.py`
Volatility 3 wrapper. Runs 12 plugins on a `.vmem` file, filters output by malware PID tree, and saves per-plugin CSVs alongside a combined `vol3_combined.csv`.

Plugins run: `pslist`, `psscan`, `cmdline`, `dlllist`, `ldrmodules`, `malfind`, `vadinfo`, `handles`, `filescan`, `svcscan`, `privileges`, `netstat`

Plugins run unfiltered (no PID scope): `malfind`, `filescan`, `svcscan`, `netstat`

**Usage (WSL):**
```bash
# Single vmem
python3 autovol4.py --family WannaCry --vmem /mnt/d/.../snapshot.vmem --output-dir /mnt/d/.../output

# Batch — processes all unfinished vmem files under a directory
python3 autovol4.py --batch-dir /mnt/d/Patrick/VMSnapshots
```

---

### `extract_features.py`
Walks the VMSnapshots directory, reads per-plugin CSVs from each snapshot folder, and outputs a single `features.csv` with one row per snapshot and ~40 numeric features.

Feature groups extracted:

| Group | Features |
|-------|---------|
| pslist | process count, avg threads, avg handles, wow64, exited |
| psscan | hidden process count (delta from pslist) |
| cmdline | suspicious argument count |
| dlllist | total DLLs, non-system DLLs, avg per process |
| ldrmodules | hidden module count, not-in-load/init/mem counts |
| vadinfo | executable VAD regions, private executable regions |
| malfind | injected regions, MZ header count, private count |
| handles | file/registry/mutex/process handles, encrypted file handles |
| filescan | total files, encrypted file count |
| svcscan | running/stopped service count |
| privileges | enabled privileges, SeDebugPrivilege count |
| netstat | total connections, established, listening, unique IPs |

**Usage (WSL):**
```bash
python3 extract_features.py --scan-dir /mnt/d/Patrick/VMSnapshots --out features.csv
```

---

### `train_stage_model.py`
Trains a family-agnostic ransomware stage classifier on `features.csv`. Tests two evaluation scenarios:

- **Standard split** — random 80/20, all families mixed. Saves `stage_model.joblib`, confusion matrix, and feature importance plots
- **Leave-one-family-out (LOO)** — trains on N-1 families, tests on the held-out family. Repeated for each family. Tests true generalization to unseen ransomware

**Usage (WSL):**
```bash
python3 train_stage_model.py --features features.csv --out model_output/
```

---

### `run_pipeline.py`
Master pipeline — calls autovol4, extract_features, and train_stage_model in sequence. Safe to re-run: autovol4 skips snapshots that already have `vol3_combined.csv`.

**Usage (WSL):**
```bash
# Full pipeline
python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots

# Skip Volatility analysis (use existing CSVs)
python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --skip-analysis

# Skip model training
python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --skip-training
```

---

## Output Structure

```
D:\Patrick\VMSnapshots\
  WannaCry_20260329_120000\
    T015_rep01\
      WannaCry_T015_rep01.vmem     # raw memory image
      meta.json                    # family, stage, timing metadata
      windows.pslist.csv           # raw Volatility output
      windows.malfind.csv
      ... (12 plugin CSVs total)
      vol3_combined.csv            # PID-filtered combined output
    T030_rep01\
    ...
  Cerber_20260329_140000\
  ...
  features.csv                     # ML feature matrix (all families)
  model_output\
    stage_model.joblib             # trained classifier
    feature_importance.csv
    cm_standard.png                # confusion matrix — standard split
    cm_loo_WannaCry.png            # confusion matrix — LOO per family
    loo_results.csv                # LOO accuracy summary
```

---

## Results (Preliminary)

Trained on WannaCry, Cerber, Jigsaw with 5 reps × 6 offsets per family.

| Evaluation | Accuracy |
|-----------|---------|
| Standard 80/20 split | 94.4% |
| LOO mean | 83.2% |

**LOO breakdown:**

| Held-out family | Accuracy | Test samples |
|----------------|---------|-------------|
| Cerber | 92.9% | 28 |
| WannaCry | 83.3% | 30 |
| Jigsaw | 73.3% | 30 |

Jigsaw's lower LOO accuracy reflects its behaviorally distinct execution pattern (incremental file deletion vs. bulk encryption), suggesting time-based stage labels don't align uniformly across families. Behavior-based relabeling is planned as a next step.

---

## Requirements

**Host (Windows):**
- VMware Workstation
- PowerShell 5+
- WSL2 with Ubuntu

**WSL / Python:**
```bash
pip install volatility3 pandas scikit-learn xgboost joblib matplotlib
```

---

## Setup

1. Create a clean Windows 10 VM snapshot named `CleanFamily4`
2. Stage malware executables inside the VM at the paths in `$FAMILIES`
3. Disable VM network adapter or set to host-only
4. Edit config variables at the top of `wannacry_automate_v3.ps1`
5. Sync autovol4 to WSL:
   ```bash
   cp /mnt/c/Users/Patrick/Desktop/MusfiqFinalProject/Ransomware-Analysis/autovol4_new.py \
      /home/patrick/tools/volatility3/autovol4.py
   ```
6. Run collection and pipeline as shown above
