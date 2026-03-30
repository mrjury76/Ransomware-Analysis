# Ransomware Memory Forensics — Stage Classification Pipeline

An end-to-end pipeline for collecting ransomware memory snapshots, extracting forensic features via Volatility 3, and training a family-agnostic stage classifier using machine learning.

> **Note:** This is an academic research project. The pipeline is designed to run in a controlled, isolated lab environment and is not intended for deployment.

---

## Overview

This project automates the full workflow:

1. **Collection** — VMware Workstation executes ransomware samples in an isolated VM and captures memory snapshots at timed intervals post-launch
2. **Analysis** — Volatility 3 runs memory forensics plugins on each snapshot, filtering output by malware process tree
3. **Feature Extraction** — Per-plugin CSVs are aggregated into a single ML-ready feature matrix
4. **Training** — A classifier predicts the ransomware execution stage from memory forensics alone, without using family identity as a feature

The core research question is whether behavioral indicators extracted from memory can identify the stage of ransomware execution — and whether that generalizes to families the model has never seen.

---

## Ransomware Families

Four families are included, chosen to represent distinct behavioral archetypes:

- **WannaCry** — crypto-ransomware with SMB worm propagation
- **Cerber** — multi-stage crypto-ransomware
- **Jigsaw** — screen locker with incremental file deletion
- **Dharma** — traditional file-encrypting ransomware

---

## Pipeline Components

### `wannacry_automate_v3.ps1`
PowerShell collection script. Automates VM revert, malware launch, memory snapshot capture, and vmem extraction for each family and time offset. Writes `meta.json` metadata alongside each vmem. Calls the analysis pipeline in WSL after collection completes.

### `autovol4_new.py`
Volatility 3 wrapper. Runs memory forensics plugins on a vmem file, identifies the malware process tree by PID, filters plugin output accordingly, and saves per-plugin CSVs. Supports single-file and batch modes. Safe to re-run — skips snapshots already processed.

### `extract_features.py`
Reads per-plugin CSVs from each snapshot directory and outputs a single `features.csv` with one row per snapshot. Features are derived from process, DLL, memory, handle, file, service, privilege, and network artifacts.

### `train_stage_model.py`
Trains a family-agnostic stage classifier on `features.csv`. Evaluates using a standard 80/20 split and a leave-one-family-out (LOO) cross-validation to test generalization to unseen ransomware families.

### `run_pipeline.py`
Master script that calls autovol4, extract_features, and train_stage_model in sequence. Safe to re-run at any point.

---

## Lab Environment

- VMware Workstation with an isolated Windows 10 x64 guest
- VM network adapter disconnected before malware execution
- Clean snapshot restored between every run
- All output written to a dedicated drive outside the VM

---

## Requirements

**WSL / Python:**
```bash
pip install volatility3 pandas scikit-learn xgboost joblib matplotlib
```
