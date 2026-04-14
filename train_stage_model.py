"""
train_stage_model.py
--------------------
Trains a family-agnostic ransomware stage classifier from features.csv.

Primary label:  behavior_stage  (0=benign/dormant, 1=pre-enc active, 2=active encryption, 3=post-encryption)
                Grounded in memory evidence -- not collection-time clock labels.
Secondary label: stage_binary   (0=early/benign, 1=active/post-enc) -- collapsed behavior label.
Reference only:  stage_hint     (time-based clock label -- kept in CSV but NOT trained on)

Tests three scenarios:
  1. Standard split      -- random 80/20, all families in both train and test
  2. Leave-one-out (LOO) -- train on N-1 families, test on the held-out family
  3. LOIO                -- leave-one-instance-out per family

Usage:
    python3 train_stage_model.py --features features.csv
    python3 train_stage_model.py --features features.csv --out model_output/
    python3 train_stage_model.py --features features.csv --cv-mode both
"""

import argparse
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
# from sklearn.svm import SVC               # unused -- removed
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier  # unused -- removed
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, balanced_accuracy_score,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold  # unused -- removed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder       # unused -- removed
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

STAGE_NAMES_TIME = {
    0: "Benign (baseline)",
    1: "Pre-launch",
    2: "Pre-encryption",
    3: "Encrypting",
    4: "Post-encryption",
}

STAGE_NAMES_BEHAVIOR = {
    0: "Benign/Dormant",
    1: "Pre-enc Active",
    2: "Active Encryption",
    3: "Post-Encryption",
}

STAGE_NAMES_EARLY_LATE = {
    0: "Early (benign/pre-launch/pre-encryption)",
    1: "Late (encrypting/post-encryption)",
}

# Columns to drop -- not forensic features
DROP_COLS = {
    "family",           # don't let the model use family identity
    "stage_hint",       # time-based label
    "stage_binary",     # collapsed time-based label (0+1 -> 0, 2+3 -> 1)
    "behavior_stage",   # behavior-based label
    "actual_offset_s",  # time since launch -- not available in real detection
    "target_offset_s",
    "rep",
    "run",
    "snap_name",
    "snap_dir",
}

# Extra columns to drop when using behavior labels to avoid data leakage
# (these features directly define the behavior_stage label)
BEHAVIOR_LEAKAGE_COLS = {
    "filescan_encrypted",
    "filescan_ransom_notes",
    "filescan_wncryt",
    "filescan_fun",
    "filescan_jigsaw_notes",
    "pslist_ransom_procs",
    "handle_encrypted_files",
    "handle_wncryt_count",
    "handle_eky_count",
    "handle_fun_count",
    "filescan_encrypted_ratio",
}

# Features that act as shortcuts to the ransomware stage (late-stage indicators).
# Dropped during stage_hint / stage_binary training to avoid trivial leakage.
STAGE_SHORTCUT_COLS = {
    "filescan_encrypted",
    "filescan_ransom_notes",
    "filescan_wncryt",
    "filescan_fun",
    "filescan_jigsaw_notes",
    "handle_encrypted_files",
    "handle_wncryt_count",
    "handle_eky_count",
    "handle_fun_count",
    "filescan_encrypted_ratio",
}

# Features that contributed zero importance across all models, or that have
# high between-family variance relative to between-stage variance (family-specific
# noise that hurts cross-family generalization).
DEAD_FEATURES = {
    # pslist -- Handles/CreateTime/ExitTime cols missing from vol3 CSV
    "pslist_avg_handles",
    "pslist_max_runtimes",
    "pslist_avg_runtimes",
    # cmdline -- patterns never fire on current data
    "cmdline_sus_args_count",
    "cmdline_script_exec_count",
    "cmdline_ransom_indicators",
    "cmdline_encoded_count",
    "cmdline_script_exec_ratio",
    "cmdline_encoded_ratio",
    # ldrmodules -- hidden detection not working in vol3
    "ldrmodules_hidden_count",
    "ldrmodules_hidden_ratio",
    # VAD -- Size col missing from vol3 vadinfo CSV
    "vad_avg_total_mem_per_process",
    "vad_max_total_mem_per_process",
    "vad_avg_max_region_size_per_process",
    # malfind -- disasm/hexdump parsing not yielding signal
    "malfind_shellcode_regions",
    "malfind_mz_regions",
    # network -- old columns removed from feat_netstat
    "netstat_suspicious_port_ratio",
    "netstat_suspicious_port_hit_count",
    "netstat_pid_count",
    "netstat_tcp_pid_count",
    "netstat_udp_pid_count",
    "netstat_established_pid_count",
    "netstat_listening_pid_count",
    "netstat_avg_connections_per_process",
    "netstat_max_connections_per_process",
    "netstat_avg_unique_ips_per_process",
    "netstat_outbound_pid_count",
    # dlllist -- high family-CV, low stage-CV (family-specific noise)
    "dlllist_crypto_dlls",
    "dlllist_crypto_pid_count",
    "dlllist_total",
    "dlllist_non_system",
    "dlllist_unique_dlls",
    # cmdline -- family-specific text patterns, not stage-discriminative
    "cmdline_avg_length",
    "cmdline_max_length",
}



# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data(features_csv):
    df = pd.read_csv(features_csv, low_memory=False)

    required = {"behavior_stage", "family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features.csv is missing columns: {missing}")

    print(f"[+] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"    Families : {sorted(df['family'].unique())}")
    print(f"    Behavior stages : {sorted(df['behavior_stage'].unique())}")
    print(f"    Distribution:\n{df.groupby(['family','behavior_stage']).size().to_string()}\n")

    # stage_hint kept as reference column only -- not used for training
    if "stage_hint" in df.columns:
        print(f"    Stage hint (reference): {sorted(df['stage_hint'].unique())}")

    # stage_binary: collapsed behavior label -- 0 (benign/pre-enc) vs 1 (active/post-enc)
    df["stage_binary"] = (df["behavior_stage"].astype(int) >= 2).astype(int)

    return df


def load_top_features(csv_path, top_n=20):
    """Read a feature_importance.csv and return the top N feature names."""
    fi = pd.read_csv(csv_path)
    return list(fi["feature"].head(top_n))


def prepare_xy(df, label_col="stage_hint", selected_features=None):
    """Split into feature matrix X and label vector y.

    Remaps labels to 0..N-1 so XGBoost works with non-contiguous classes.
    Returns (X, y, feat_cols, label_map) where label_map converts idx->original.

    If selected_features is provided, only those columns are used.
    """
    drop = set(DROP_COLS) | DEAD_FEATURES
    if label_col == "behavior_stage":
        drop |= BEHAVIOR_LEAKAGE_COLS
    if label_col in {"stage_hint", "stage_binary"}:
        drop |= STAGE_SHORTCUT_COLS
    if selected_features:
        feat_cols = [c for c in selected_features if c in df.columns and c not in drop]
    else:
        feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].copy()
    y_raw = df[label_col].astype(int)

    # Remap to 0..N-1
    unique_sorted = sorted(y_raw.unique())
    to_idx = {v: i for i, v in enumerate(unique_sorted)}
    y = y_raw.map(to_idx)
    label_map = {i: v for v, i in to_idx.items()}

    # Coerce everything to numeric, fill non-numeric with NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, feat_cols, label_map


# -----------------------------------------------------------------------------
# Model zoo
# -----------------------------------------------------------------------------

def get_models(only=None):
    """Return dict of name -> classifier for multi-model comparison.

    If `only` is provided (list/set of names), restrict to those models.
    """
    # Tree depth reduced 12->8, min_samples_leaf raised 2->5 to reduce the
    # 100% train-accuracy overfitting seen in run10.
    # RF and ExtraTrees are wrapped in isotonic calibration so their
    # predict_proba outputs are reliable (run10 showed RF mean confidence
    # of only 72% despite 94% accuracy -- a sign of poor calibration).
    models = {
        "RandomForest": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        ),
        "ExtraTrees": CalibratedClassifierCV(
            ExtraTreesClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, min_child_weight=5,   # tighter regularisation
            random_state=42, verbosity=0,
        )
    return models


def make_pipeline(clf=None):
    """Wrap a classifier in an imputer+scaler pipeline."""
    if clf is None:
        clf = list(get_models().values())[0]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   clf),
    ])


# -----------------------------------------------------------------------------
# Scenario 1: standard stratified split
# -----------------------------------------------------------------------------

def run_standard_split(X, y, feat_cols, out_dir, stage_names=None, label_map=None):
    stage_names = stage_names or STAGE_NAMES_TIME
    label_map = label_map or {}
    print("=" * 60)
    print(" SCENARIO 1: Standard 80/20 split (all families mixed)")
    print("=" * 60)

    if len(pd.Series(y).unique()) < 2:
        print(f"[!] y has only 1 class -- skipping standard split.")
        return 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if len(pd.Series(y_train).unique()) < 2:
        print(f"[!] y_train has only 1 class after split -- skipping standard split.")
        return 0

    print(f"    Train: {len(y_train)} rows  |  Test: {len(y_test)} rows")

    present_stages = sorted(y.unique())
    target_names   = [stage_names.get(label_map.get(s, s), f"Stage {s}") for s in present_stages]

    models = get_models()
    comparison = []
    best_score, best_name, best_pipe = 0, None, None

    for name, clf in tqdm(models.items(), desc="Standard split", total=len(models), unit="model"):
        try:
            pipe = make_pipeline(clf)
            pipe.fit(X_train, y_train)
            preds_train = pipe.predict(X_train)
            preds       = pipe.predict(X_test)
        except Exception as e:
            print(f"\n  [!] {name} failed: {type(e).__name__}: {e} -- skipping")
            continue

        m         = _metrics(y_test, preds)
        m_train   = _metrics(y_train, preds_train)
        ci        = _bootstrap_ci(y_test, preds)
        conf      = _prediction_confidence(pipe, X_test) or {}
        overfit_gap = round(m_train["accuracy"] - m["accuracy"], 4)

        row = {"model": name, "n_train": len(y_train), "n_test": len(y_test),
               **m,
               "train_accuracy": m_train["accuracy"],
               "overfit_gap":    overfit_gap,
               **ci,
               **conf}
        comparison.append(row)

        print(f"\n  {name}:")
        print(f"    test  -- acc: {m['accuracy']:.4f}  [{ci['acc_ci_lo']:.4f}–{ci['acc_ci_hi']:.4f} 95% CI]"
              f"  bal_acc: {m['balanced_acc']:.4f}")
        print(f"    macro_f1: {m['macro_f1']:.4f}  [{ci['f1_ci_lo']:.4f}–{ci['f1_ci_hi']:.4f} 95% CI]"
              f"  weighted_f1: {m['weighted_f1']:.4f}")
        print(f"    train -- acc: {m_train['accuracy']:.4f}  overfit gap: {overfit_gap:+.4f}")
        if conf:
            print(f"    confidence -- mean: {conf['conf_mean']:.4f}  min: {conf['conf_min']:.4f}"
                  f"  >=90%: {conf['conf_pct_90']:.1%}  >=70%: {conf['conf_pct_70']:.1%}")
        print(classification_report(y_test, preds, labels=present_stages,
                                    target_names=target_names, zero_division=0))

        _save_confusion_matrix(y_test, preds, present_stages, target_names,
                               os.path.join(out_dir, f"cm_standard_{name}.png"),
                               title=f"Standard split -- {name}", normalize=True)

        if m["macro_f1"] > best_score:
            best_score, best_name, best_pipe = m["macro_f1"], name, pipe

    if not comparison:
        print("  [!] No model succeeded.")
        return 0

    comp_df = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    comp_df.to_csv(os.path.join(out_dir, "standard_results.csv"), index=False)
    print(f"\n{'=' * 40}")
    print(f" SCENARIO 1 -- Standard 80/20 split -- Model comparison:")
    print(f"{'=' * 40}")
    cols_display = ["model", "accuracy", "acc_ci_lo", "acc_ci_hi",
                    "macro_f1", "f1_ci_lo", "f1_ci_hi",
                    "balanced_acc", "weighted_f1",
                    "train_accuracy", "overfit_gap",
                    "conf_mean", "conf_pct_90", "conf_pct_70",
                    "n_train", "n_test"]
    print(comp_df[[c for c in cols_display if c in comp_df.columns]].to_string(index=False))

    if best_pipe is None:
        return 0

    print(f"\n  Best (macro F1): {best_name} ({best_score:.4f})")
    _save_feature_importance(best_pipe, feat_cols,
                             os.path.join(out_dir, "feature_importance.csv"),
                             os.path.join(out_dir, "feature_importance.png"))

    model_path = os.path.join(out_dir, "stage_model.joblib")
    joblib.dump({"pipeline": best_pipe, "feature_cols": feat_cols,
                 "stage_names": stage_names, "model_name": best_name}, model_path)
    print(f"  [done] Model saved: {model_path}")

    return comp_df.iloc[0]["accuracy"]


# -----------------------------------------------------------------------------
# Scenario 2: leave-one-family-out
# -----------------------------------------------------------------------------

def run_loo(df, feat_cols, out_dir, label_col="stage_hint", stage_names=None, label_map=None,
            selected_features=None, benign_family="Benign", benign_always_train=True):
    """
    benign_always_train=True (default): Benign samples are ALWAYS included in the
    training fold -- LOO folds only cycle over ransomware families. After all
    ransomware folds, a dedicated 80/20 Benign holdout measures real FPR without
    the structural impossibility of predicting stage-0 when the model was never
    trained on it.
    """
    stage_names = stage_names or STAGE_NAMES_TIME
    label_map = label_map or {}
    print("\n" + "=" * 60)
    print(" SCENARIO 2: Leave-one-family-out (generalization test)")
    print(" Train on N-1 families, test on the held-out family.")
    print(f" Benign '{benign_family}' always in train set -- dedicated holdout measures FPR.")
    print("=" * 60)

    all_families    = sorted(df["family"].unique())
    ransom_families = [f for f in all_families if f != benign_family]
    X_all, y_all, _, _ = prepare_xy(df, label_col=label_col, selected_features=selected_features)
    models = get_models()

    # all_results[model_name] = list of per-family metric dicts (ransomware folds only)
    all_results = {name: [] for name in models}

    for held_out in tqdm(ransom_families, desc="LOO folds", unit="family"):
        # Benign always stays in training; only ransomware families are held out
        train_mask = df["family"] != held_out
        test_mask  = df["family"] == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]

        if len(X_test) == 0 or len(y_test.unique()) < 1:
            continue
        if len(y_train.unique()) < 2:
            print(f"\n  --- Held-out: {held_out} -- skipping (training set has only 1 class) ---")
            continue

        # Remap train labels to 0..N-1 for XGBoost
        unique_train = sorted(int(x) for x in y_train.unique())
        to_idx = {v: i for i, v in enumerate(unique_train)}
        from_idx = {i: v for v, i in to_idx.items()}
        y_train_enc = np.array([to_idx[int(v)] for v in y_train])

        present = sorted(y_test.unique())
        names   = [stage_names.get(label_map.get(s, s), f"Stage {s}") for s in present]

        print(f"\n  --- Held-out: {held_out} ({len(y_test)} rows, {len(y_train)} train) ---")

        best_macro_f1, best_name, best_preds = -1, None, None
        fold_preds = {}   # model_name -> preds array, saved for per-model outputs

        for model_name, clf in models.items():
            try:
                pipe = make_pipeline(clf)
                pipe.fit(X_train, y_train_enc)
                preds_enc = pipe.predict(X_test)
            except Exception as e:
                print(f"    {model_name:20s}  [!] failed: {type(e).__name__}: {e}")
                continue
            preds = np.array([from_idx.get(int(p), int(p)) for p in preds_enc])
            fold_preds[model_name] = preds

            m = _metrics(y_test, preds)
            print(f"    {model_name:20s}  acc: {m['accuracy']:.4f}  "
                  f"bal_acc: {m['balanced_acc']:.4f}  macro_f1: {m['macro_f1']:.4f}  "
                  f"weighted_f1: {m['weighted_f1']:.4f}")
            all_results[model_name].append({
                "held_out_family": held_out,
                "n_test":          len(y_test),
                **m,
            })
            if m["macro_f1"] > best_macro_f1:
                best_macro_f1, best_name, best_preds = m["macro_f1"], model_name, preds

        if not fold_preds:
            print(f"    [!] All models failed for {held_out} -- skipping outputs.")
            continue

        # Best-model confusion matrix only (one PNG per family)
        if best_name and best_preds is not None:
            _save_confusion_matrix(
                y_test, best_preds, present, names,
                os.path.join(out_dir, f"loo_cm_{held_out}.png"),
                title=f"LOO -- {held_out} -- {best_name} [best]",
                normalize=True,
            )

    # ── Benign holdout: dedicated 80/20 split to measure real FPR ────────────────
    benign_fpr_by_model = {}
    benign_df = df[df["family"] == benign_family]
    if not benign_df.empty:
        print(f"\n  --- Benign FPR holdout ({len(benign_df)} rows, trained on ransomware only) ---")
        X_b, y_b, _, _ = prepare_xy(benign_df, label_col=label_col,
                                     selected_features=selected_features)
        # Train on ALL ransomware (no benign in this training set -- realistic FPR scenario)
        ransom_mask = df["family"] != benign_family
        X_r, y_r, _, _ = prepare_xy(df[ransom_mask], label_col=label_col,
                                     selected_features=selected_features)
        X_bv = X_b   # all benign samples are unseen -- no need to split
        # Remap ransomware labels for XGBoost
        unique_r = sorted(int(x) for x in y_r.unique())
        to_r = {v: i for i, v in enumerate(unique_r)}
        from_r = {i: v for v, i in to_r.items()}
        y_r_enc = np.array([to_r[int(v)] for v in y_r])

        benign_label = int(y_b.iloc[0]) if not y_b.empty else 0  # stage-0

        for model_name, clf in models.items():
            try:
                pipe = make_pipeline(clf)
                pipe.fit(X_r, y_r_enc)
                preds_enc = pipe.predict(X_bv)
                preds = np.array([from_r.get(int(p), int(p)) for p in preds_enc])
            except Exception as e:
                print(f"    {model_name:20s}  [!] failed: {type(e).__name__}: {e}")
                continue
            fpr = float((preds != benign_label).mean())
            specificity = 1.0 - fpr
            benign_fpr_by_model[model_name] = round(fpr, 4)
            print(f"    {model_name:20s}  specificity: {specificity:.4f}  "
                  f"FPR (flagged as ransomware): {fpr:.4f}")

    # ── Build LOO summary (ransomware families only) ──────────────────────────
    loo_summary = []
    for model_name, results in all_results.items():
        if not results:
            continue
        rdf = pd.DataFrame(results)

        if rdf.empty:
            continue

        total_n = rdf["n_test"].sum()
        n_folds  = len(rdf)
        sem_acc  = rdf["accuracy"].std() / np.sqrt(n_folds) if n_folds > 1 else 0.0
        sem_f1   = rdf["macro_f1"].std()  / np.sqrt(n_folds) if n_folds > 1 else 0.0

        row = {
            "model":             model_name,
            "mean_accuracy":     round(rdf["accuracy"].mean(), 4),
            "std_accuracy":      round(rdf["accuracy"].std(), 4),
            "acc_ci95_lo":       round(rdf["accuracy"].mean() - 1.96 * sem_acc, 4),
            "acc_ci95_hi":       round(rdf["accuracy"].mean() + 1.96 * sem_acc, 4),
            "wtd_accuracy":      round((rdf["accuracy"] * rdf["n_test"]).sum() / total_n, 4),
            "mean_balanced_acc": round(rdf["balanced_acc"].mean(), 4),
            "mean_macro_f1":     round(rdf["macro_f1"].mean(), 4),
            "std_macro_f1":      round(rdf["macro_f1"].std(), 4),
            "f1_ci95_lo":        round(rdf["macro_f1"].mean() - 1.96 * sem_f1, 4),
            "f1_ci95_hi":        round(rdf["macro_f1"].mean() + 1.96 * sem_f1, 4),
            "wtd_macro_f1":      round((rdf["macro_f1"] * rdf["n_test"]).sum() / total_n, 4),
            "mean_weighted_f1":  round(rdf["weighted_f1"].mean(), 4),
            "n_folds":           n_folds,
            "benign_fpr":        benign_fpr_by_model.get(model_name),
        }
        for _, r in rdf.iterrows():
            row[r["held_out_family"]] = round(r["macro_f1"], 4)
        loo_summary.append(row)

    if loo_summary:
        loo_df = pd.DataFrame(loo_summary).sort_values("mean_macro_f1", ascending=False)
        print(f"\n{'=' * 60}")
        print(f" SCENARIO 2 -- LOO -- Model Comparison (sorted by mean macro F1):")
        print(f" benign_fpr = fraction of benign samples flagged as ransomware.")
        print(f"{'=' * 60}")
        summary_cols = ["model", "mean_accuracy", "acc_ci95_lo", "acc_ci95_hi",
                        "wtd_accuracy", "std_accuracy", "mean_balanced_acc",
                        "mean_macro_f1", "f1_ci95_lo", "f1_ci95_hi",
                        "wtd_macro_f1", "std_macro_f1", "mean_weighted_f1",
                        "n_folds", "benign_fpr"]
        print(loo_df[[c for c in summary_cols if c in loo_df.columns]].to_string(index=False))
        loo_df.to_csv(os.path.join(out_dir, "loo_results.csv"), index=False)


# -----------------------------------------------------------------------------
# Scenario 3: leave-one-instance-out (leave one full (family, rep) group out)
# -----------------------------------------------------------------------------

def run_loio(df, feat_cols, out_dir, label_col="stage_hint", stage_names=None, label_map=None,
             selected_features=None):
    """
    Leave-one-instance-out CV.
    An "instance" = one full collection run of a family, identified by (family, rep).
    Each instance contains snapshots at every timed offset (all stages).
    Trains on all families' remaining reps, tests on the held-out instance.
    """
    stage_names = stage_names or STAGE_NAMES_TIME
    label_map = label_map or {}
    print("\n" + "=" * 60)
    print(" SCENARIO 3: Leave-one-instance-out")
    print(" Hold out one (family, rep) group; train on everything else.")
    print("=" * 60)

    if "rep" not in df.columns:
        print("[!] 'rep' column not in features.csv -- cannot run LOIO")
        return

    # Group key = (family, rep). Each group = one full timed collection run.
    df_work = df.copy()
    df_work["_instance"] = df_work["family"].astype(str) + "::rep" + df_work["rep"].astype(str)
    instances = sorted(df_work["_instance"].unique())

    X_all, y_all, _, _ = prepare_xy(df_work, label_col=label_col, selected_features=selected_features)
    models = get_models()
    all_results = {name: [] for name in models}
    # Track best preds per family for per-family confusion matrices
    # best_by_family[family] = (best_macro_f1, best_model_name, y_test_concat, preds_concat)
    best_by_family = {}

    print(f"[+] {len(instances)} instances (family × rep groups)\n")

    for held_out in tqdm(instances, desc="LOIO folds", unit="inst"):
        train_mask = df_work["_instance"] != held_out
        test_mask  = df_work["_instance"] == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]

        if len(X_test) == 0:
            continue
        if len(y_train.unique()) < 2:
            print(f"  --- Held-out: {held_out} -- skipping (training set has only 1 class) ---")
            continue

        unique_train = sorted(int(x) for x in y_train.unique())
        to_idx = {v: i for i, v in enumerate(unique_train)}
        from_idx = {i: v for v, i in to_idx.items()}
        y_train_enc = np.array([to_idx[int(v)] for v in y_train])

        fam_label = held_out.split("::")[0]

        for model_name, clf in models.items():
            try:
                pipe = make_pipeline(clf)
                pipe.fit(X_train, y_train_enc)
                preds_enc = pipe.predict(X_test)
            except Exception as e:
                print(f"  [!] {model_name} failed on {held_out}: {type(e).__name__}: {e}")
                continue
            preds = np.array([from_idx.get(int(p), int(p)) for p in preds_enc])
            m = _metrics(y_test, preds)
            all_results[model_name].append({
                "held_out": held_out, "family": fam_label,
                "n_test":   len(y_test),
                **m,
            })
            # Accumulate best-model predictions per family for aggregated confusion matrix
            prev = best_by_family.get((fam_label, model_name))
            if prev is None:
                best_by_family[(fam_label, model_name)] = (
                    np.array(y_test), np.array(preds))
            else:
                best_by_family[(fam_label, model_name)] = (
                    np.concatenate([prev[0], y_test]),
                    np.concatenate([prev[1], preds]))

    # Build LOIO summary: unweighted and n_test-weighted means per model
    loio_summary = []
    for model_name, results in all_results.items():
        if not results:
            continue
        rdf = pd.DataFrame(results)
        total_n = rdf["n_test"].sum()
        n_inst  = len(rdf)
        sem_acc = rdf["accuracy"].std() / np.sqrt(n_inst) if n_inst > 1 else 0.0
        sem_f1  = rdf["macro_f1"].std()  / np.sqrt(n_inst) if n_inst > 1 else 0.0
        row = {
            "model":             model_name,
            "mean_accuracy":     round(rdf["accuracy"].mean(), 4),
            "std_accuracy":      round(rdf["accuracy"].std() if n_inst > 1 else 0.0, 4),
            "acc_ci95_lo":       round(rdf["accuracy"].mean() - 1.96 * sem_acc, 4),
            "acc_ci95_hi":       round(rdf["accuracy"].mean() + 1.96 * sem_acc, 4),
            "wtd_accuracy":      round((rdf["accuracy"] * rdf["n_test"]).sum() / total_n, 4),
            "mean_balanced_acc": round(rdf["balanced_acc"].mean(), 4),
            "mean_macro_f1":     round(rdf["macro_f1"].mean(), 4),
            "std_macro_f1":      round(rdf["macro_f1"].std() if n_inst > 1 else 0.0, 4),
            "f1_ci95_lo":        round(rdf["macro_f1"].mean() - 1.96 * sem_f1, 4),
            "f1_ci95_hi":        round(rdf["macro_f1"].mean() + 1.96 * sem_f1, 4),
            "wtd_macro_f1":      round((rdf["macro_f1"] * rdf["n_test"]).sum() / total_n, 4),
            "mean_weighted_f1":  round(rdf["weighted_f1"].mean(), 4),
            "n_instances":       n_inst,
        }
        # Per-family macro F1 breakdown
        for fam, fam_df in rdf.groupby("family"):
            row[f"{fam}_macro_f1"] = round(fam_df["macro_f1"].mean(), 4)
        loio_summary.append(row)

    if loio_summary:
        loio_df = pd.DataFrame(loio_summary).sort_values("mean_macro_f1", ascending=False)
        print(f"\n{'=' * 60}")
        print(f" SCENARIO 3 -- LOIO -- Model Comparison (sorted by mean macro F1):")
        print(f"{'=' * 60}")
        summary_cols = ["model", "mean_accuracy", "acc_ci95_lo", "acc_ci95_hi",
                        "wtd_accuracy", "std_accuracy",
                        "mean_balanced_acc", "mean_macro_f1", "f1_ci95_lo", "f1_ci95_hi",
                        "wtd_macro_f1", "std_macro_f1", "mean_weighted_f1", "n_instances"]
        print(loio_df[[c for c in summary_cols if c in loio_df.columns]].to_string(index=False))
        loio_df.to_csv(os.path.join(out_dir, "loio_results.csv"), index=False)

        # Per-instance detail CSV for every model
        for model_name, results in all_results.items():
            if results:
                pd.DataFrame(results).to_csv(
                    os.path.join(out_dir, f"loio_detail_{model_name}.csv"), index=False
                )

        # Per-family aggregated confusion matrix + report CSV (all instances of a family combined)
        # Use the best model per family (highest mean macro F1 in loio_summary)
        best_model_overall = loio_df.iloc[0]["model"] if not loio_df.empty else None
        all_families_loio  = sorted({k[0] for k in best_by_family})
        for fam in all_families_loio:
            key = (fam, best_model_overall)
            if key not in best_by_family:
                continue
            yt, yp = best_by_family[key]
            present = sorted(np.unique(np.concatenate([yt, yp])))
            names   = [stage_names.get(label_map.get(s, s), f"Stage {s}") for s in present]
            _save_confusion_matrix(
                yt, yp, present, names,
                os.path.join(out_dir, f"loio_cm_{fam}.png"),
                title=f"LOIO -- {fam} (all reps, {best_model_overall})",
                normalize=True,
            )

        print(f"  [done] Per-family confusion matrices + reports saved (model: {best_model_overall}).")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def write_run_log(out_dir, df, label_cols, models_used, scenarios_run, extra=None):
    """Write a human-readable run_log.txt summarising the training run."""
    path = os.path.join(out_dir, "run_log.txt")
    families = sorted(df["family"].unique())
    n_rows   = len(df)
    n_feats  = sum(1 for c in df.columns if c not in DROP_COLS | BEHAVIOR_LEAKAGE_COLS | STAGE_SHORTCUT_COLS)

    lines = [
        "=" * 60,
        " TRAINING RUN LOG",
        "=" * 60,
        f"  Timestamp  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Output dir : {out_dir}",
        "",
        f"  Snapshots  : {n_rows} rows",
        f"  Families   : {len(families)}",
    ]
    for fam in families:
        fam_df = df[df["family"] == fam]
        lines.append(f"      {fam:<25s}  {len(fam_df)} rows")

    lines += [
        "",
        f"  Feature cols (approx): {n_feats}",
        "",
        "  Labels trained:",
    ]
    for lc in label_cols:
        if lc in df.columns:
            dist = df[lc].value_counts().sort_index().to_dict()
            lines.append(f"      {lc:<20s}  {dist}")

    lines += [
        "",
        "  Models:",
    ]
    for m in models_used:
        lines.append(f"      {m}")

    lines += [
        "",
        "  Scenarios run:",
    ]
    for s in scenarios_run:
        lines.append(f"      {s}")

    if extra:
        lines += ["", "  Notes:"]
        for k, v in extra.items():
            lines.append(f"      {k}: {v}")

    lines.append("=" * 60)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [done] Run log: {path}")


def write_master_summary(run_dir, label_cols, df=None):
    """
    Reads standard_results.csv, loo_results.csv, loio_results.csv from each
    label sub-folder and writes:
      master_summary.csv  -- flat table: one row per (label, scenario, model)
      master_summary.txt  -- full human-readable report with per-family breakdown
    """
    SCENARIO_FILES = {
        "Standard 80/20": "standard_results.csv",
        "LOO":            "loo_results.csv",
        "LOIO":           "loio_results.csv",
    }
    # Rename standard's single-split column names to the common mean_* schema
    STD_RENAME = {
        "accuracy":     "mean_accuracy",
        "balanced_acc": "mean_balanced_acc",
        "macro_f1":     "mean_macro_f1",
        "weighted_f1":  "mean_weighted_f1",
        "acc_ci_lo":    "acc_ci95_lo",
        "acc_ci_hi":    "acc_ci95_hi",
        "f1_ci_lo":     "f1_ci95_lo",
        "f1_ci_hi":     "f1_ci95_hi",
    }
    CORE = ["model", "mean_accuracy", "acc_ci95_lo", "acc_ci95_hi",
            "mean_balanced_acc", "mean_macro_f1", "f1_ci95_lo", "f1_ci95_hi",
            "mean_weighted_f1"]
    # Extra cols kept per scenario
    STD_EXTRA  = ["train_accuracy", "overfit_gap", "conf_mean", "conf_pct_90",
                  "n_train", "n_test"]
    LOO_EXTRA  = ["benign_fpr", "wtd_macro_f1", "n_folds"]
    LOIO_EXTRA = ["wtd_macro_f1", "n_instances"]

    # ── Load all CSVs ────────────────────────────────────────────────────────
    # raw_data[(label, scenario)] = DataFrame (full, unrenamed for per-family)
    raw_data = {}
    all_rows = []

    for label_col in label_cols:
        label_dir = os.path.join(run_dir, label_col)
        if not os.path.isdir(label_dir):
            continue
        for scenario, fname in SCENARIO_FILES.items():
            fpath = os.path.join(label_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                sdf = pd.read_csv(fpath)
            except Exception:
                continue
            raw_data[(label_col, scenario)] = sdf

            # Build normalised flat row for master CSV
            sdf_n = sdf.rename(columns=STD_RENAME).copy()
            extra = (STD_EXTRA if scenario == "Standard 80/20"
                     else LOO_EXTRA if scenario == "LOO"
                     else LOIO_EXTRA)
            keep = [c for c in CORE + extra if c in sdf_n.columns]
            sdf_n = sdf_n[keep].copy()
            sdf_n.insert(0, "label",    label_col)
            sdf_n.insert(1, "scenario", scenario)
            all_rows.append(sdf_n)

    if not all_rows:
        print("  [!] No result CSVs found -- skipping master summary.")
        return

    master = pd.concat(all_rows, ignore_index=True, sort=False)
    if "mean_macro_f1" in master.columns:
        master = master.sort_values("mean_macro_f1", ascending=False)
    master.to_csv(os.path.join(run_dir, "master_summary.csv"), index=False)

    # ── Helper: best row for a (label, scenario) ────────────────────────────
    def best_row(label_col, scenario):
        key = (label_col, scenario)
        if key not in raw_data:
            return None
        sdf = raw_data[key].rename(columns=STD_RENAME)
        col = "mean_macro_f1" if "mean_macro_f1" in sdf.columns else "macro_f1"
        if col not in sdf.columns:
            return None
        return sdf.sort_values(col, ascending=False).iloc[0]

    def fmt_ci(lo, hi):
        try:
            return f"[{float(lo):.4f}-{float(hi):.4f}]"
        except (TypeError, ValueError):
            return ""

    def fmt_val(v, dec=4):
        try:
            return f"{float(v):.{dec}f}"
        except (TypeError, ValueError):
            return str(v) if v is not None else "--"

    # ── Detect families from data ────────────────────────────────────────────
    known_families = []
    if df is not None:
        known_families = sorted(df["family"].unique())
    else:
        # Infer from LOO columns (bare family names) or LOIO (_macro_f1 suffix)
        for (lc, sc), sdf in raw_data.items():
            if sc == "LOO":
                skip = set(sdf.rename(columns=STD_RENAME).columns) & set(
                    CORE + LOO_EXTRA + ["std_accuracy","std_macro_f1","wtd_accuracy",
                                        "mean_weighted_f1","n_folds","benign_fpr"])
                known_families = [c for c in sdf.columns if c not in skip]
                break
            if sc == "LOIO":
                known_families = sorted(
                    c.replace("_macro_f1", "") for c in sdf.columns
                    if c.endswith("_macro_f1"))
                break

    # ── Build text report ────────────────────────────────────────────────────
    W = 74
    lines = [
        "=" * W,
        " MASTER RESULTS SUMMARY",
        f" Run     : {run_dir}",
        f" Created : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * W,
    ]

    if df is not None:
        dist = df["behavior_stage"].value_counts().sort_index().to_dict() \
               if "behavior_stage" in df.columns else {}
        lines += [
            "",
            f"  Snapshots : {len(df)}   Families : {len(known_families)}",
            f"  Families  : {', '.join(known_families)}",
            f"  Labels    : {', '.join(label_cols)}",
        ]

    scenario_order = list(SCENARIO_FILES.keys())
    label_order    = label_cols

    # ── Section 1: per-scenario best model ───────────────────────────────────
    lines += ["", "=" * W,
              " SECTION 1 -- BEST MODEL PER SCENARIO (all labels)",
              "=" * W]

    for scenario in scenario_order:
        lines += ["", f"  {scenario}",  "  " + "-" * (W - 2)]
        any_found = False
        for label_col in label_order:
            b = best_row(label_col, scenario)
            if b is None:
                continue
            any_found = True
            f1    = fmt_val(b.get("mean_macro_f1"))
            ci    = fmt_ci(b.get("f1_ci95_lo"), b.get("f1_ci95_hi"))
            acc   = fmt_val(b.get("mean_accuracy"))
            bal   = fmt_val(b.get("mean_balanced_acc"))
            wf1   = fmt_val(b.get("mean_weighted_f1"))
            model = str(b.get("model", "?"))

            line = (f"  {label_col:<18}  {model:<22}"
                    f"  acc={acc}  macro_f1={f1} {ci}"
                    f"  bal={bal}  wtd_f1={wf1}")

            # Scenario-specific extras
            if scenario == "Standard 80/20":
                gap  = fmt_val(b.get("overfit_gap"), 4)
                conf = fmt_val(b.get("conf_mean"), 4)
                line += f"  overfit={gap}  conf={conf}"
            elif scenario == "LOO":
                fpr = fmt_val(b.get("benign_fpr"), 4)
                line += f"  benign_fpr={fpr}"
            lines.append(line)
        if not any_found:
            lines.append("  (no results)")

    # ── Section 2: per-label full breakdown ──────────────────────────────────
    lines += ["", "=" * W,
              " SECTION 2 -- FULL BREAKDOWN PER LABEL",
              "=" * W]

    for label_col in label_order:
        lines += ["", f"  [{label_col}]", "  " + "=" * (W - 2)]

        for scenario in scenario_order:
            key = (label_col, scenario)
            if key not in raw_data:
                continue
            sdf = raw_data[key].rename(columns=STD_RENAME)
            col = "mean_macro_f1" if "mean_macro_f1" in sdf.columns else "macro_f1"
            if col not in sdf.columns:
                continue
            sdf_s = sdf.sort_values(col, ascending=False)

            lines += ["", f"  -- {scenario} --"]

            # All-model metric table
            hdr_cols = ["model", "mean_accuracy", "acc_ci95_lo", "acc_ci95_hi",
                        "mean_balanced_acc", "mean_macro_f1", "f1_ci95_lo",
                        "f1_ci95_hi", "mean_weighted_f1"]
            if scenario == "Standard 80/20":
                hdr_cols += ["train_accuracy", "overfit_gap", "conf_mean", "conf_pct_90"]
            elif scenario == "LOO":
                hdr_cols += ["benign_fpr", "n_folds"]
            elif scenario == "LOIO":
                hdr_cols += ["n_instances"]
            hdr_cols = [c for c in hdr_cols if c in sdf_s.columns]
            lines.append(sdf_s[hdr_cols].to_string(index=False))

            # Per-family breakdown (LOO and LOIO only)
            if scenario == "LOO" and known_families:
                fam_cols = [f for f in known_families
                            if f in sdf_s.columns and f not in ("Benign",)]
                if fam_cols:
                    lines += ["", f"  Per-family macro F1 (best model = {sdf_s.iloc[0]['model']}):"]
                    best = sdf_s.iloc[0]
                    for fam in fam_cols:
                        val = fmt_val(best.get(fam))
                        lines.append(f"    {fam:<14}  {val}")
                    fpr = fmt_val(best.get("benign_fpr"))
                    lines.append(f"    {'Benign FPR':<14}  {fpr}")

            elif scenario == "LOIO" and known_families:
                fam_f1_cols = {f: f"{f}_macro_f1" for f in known_families
                               if f"{f}_macro_f1" in sdf_s.columns}
                if fam_f1_cols:
                    lines += ["", f"  Per-family macro F1 (best model = {sdf_s.iloc[0]['model']}):"]
                    best = sdf_s.iloc[0]
                    for fam, col in fam_f1_cols.items():
                        val = fmt_val(best.get(col))
                        lines.append(f"    {fam:<14}  {val}")

    # ── Section 3: overall single best ───────────────────────────────────────
    lines += ["", "=" * W,
              " SECTION 3 -- OVERALL BEST (highest mean macro F1)",
              "=" * W, ""]

    ob = master.sort_values("mean_macro_f1", ascending=False).iloc[0]
    lines += [
        f"  Label    : {ob['label']}",
        f"  Scenario : {ob['scenario']}",
        f"  Model    : {ob['model']}",
        f"  Accuracy : {fmt_val(ob.get('mean_accuracy'))}  "
        f"{fmt_ci(ob.get('acc_ci95_lo'), ob.get('acc_ci95_hi'))}",
        f"  Macro F1 : {fmt_val(ob.get('mean_macro_f1'))}  "
        f"{fmt_ci(ob.get('f1_ci95_lo'), ob.get('f1_ci95_hi'))}",
        f"  Bal Acc  : {fmt_val(ob.get('mean_balanced_acc'))}",
        f"  Wtd F1   : {fmt_val(ob.get('mean_weighted_f1'))}",
        "",
        "=" * W,
    ]

    txt_path = os.path.join(run_dir, "master_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [+] Master summary : {txt_path}")
    print(f"  [+] Master CSV     : {os.path.join(run_dir, 'master_summary.csv')}")


def _metrics(y_true, y_pred):
    """Return a dict of accuracy, balanced_acc, macro_f1, weighted_f1."""
    return {
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "balanced_acc":     round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1":         round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "weighted_f1":      round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def _bootstrap_ci(y_true, y_pred, n_boot=1000, ci=95, random_state=42):
    """Bootstrap confidence interval on accuracy and macro F1 over the test set.

    Resamples (y_true, y_pred) pairs with replacement n_boot times.
    Returns a dict with lower/upper bounds for each metric.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    acc_boot, f1_boot = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        acc_boot.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1_boot.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))

    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "acc_ci_lo":   round(float(np.percentile(acc_boot, lo)), 4),
        "acc_ci_hi":   round(float(np.percentile(acc_boot, hi)), 4),
        "f1_ci_lo":    round(float(np.percentile(f1_boot,  lo)), 4),
        "f1_ci_hi":    round(float(np.percentile(f1_boot,  hi)), 4),
        "ci_pct":      ci,
        "n_boot":      n_boot,
    }


def _prediction_confidence(pipe, X_test):
    """Return confidence stats from predict_proba if available.

    Confidence = max class probability for each sample.
    Returns a dict with mean, min, and fraction above common thresholds.
    Returns None if the model does not support predict_proba.
    """
    model = pipe.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = pipe.predict_proba(X_test)
    except Exception:
        return None

    conf = proba.max(axis=1)          # highest class probability per sample
    return {
        "conf_mean":   round(float(conf.mean()), 4),
        "conf_min":    round(float(conf.min()),  4),
        "conf_pct_90": round(float((conf >= 0.90).mean()), 4),
        "conf_pct_70": round(float((conf >= 0.70).mean()), 4),
        "conf_pct_50": round(float((conf >= 0.50).mean()), 4),
    }



def _save_confusion_matrix(y_true, y_pred, labels, names, path, title="", normalize=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) if normalize else plt.subplots(figsize=(8, 6))
    axes = list(axes) if normalize else [axes]

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm_raw, display_labels=names).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(f"{title} -- counts")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha="right")

    if normalize:
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        ConfusionMatrixDisplay(
            np.round(cm_norm, 2), display_labels=names
        ).plot(ax=axes[1], colorbar=False, cmap="Blues")
        axes[1].set_title(f"{title} -- recall (row-normalized)")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [done] Saved: {path}")


def _save_feature_importance(pipe, feat_cols, csv_path, plot_path, top_n=20):
    model = pipe.named_steps["model"]
    # Unwrap CalibratedClassifierCV -- average importances across its cv folds
    if isinstance(model, CalibratedClassifierCV) and hasattr(model, "calibrated_classifiers_"):
        base_models = [c.estimator for c in model.calibrated_classifiers_
                       if hasattr(c, "estimator") and hasattr(c.estimator, "feature_importances_")]
        if not base_models:
            return
        importances = np.mean([m.feature_importances_ for m in base_models], axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    fi = pd.DataFrame({
        "feature":    feat_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    fi.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    fi.head(top_n).plot.barh(x="feature", y="importance", ax=ax, legend=False)
    ax.set_title(f"Top {top_n} features")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print(f"\n  Top 10 features:")
    print(fi.head(10).to_string(index=False))
    print(f"  [done] Saved: {csv_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train family-agnostic ransomware stage classifier")
    parser.add_argument("--features", required=True, help="Path to features.csv from extract_features.py")
    parser.add_argument("--out",      default="model_output", help="Output directory (default: model_output)")
    parser.add_argument("--no-loo",   action="store_true",    help="Skip leave-one-family-out evaluation")
    parser.add_argument("--cv-mode",  default="family",
                        choices=["family", "instance", "both", "none"],
                        help="Cross-validation mode: 'family' (LOO), 'instance' (LOIO), 'both', or 'none'")
    parser.add_argument("--label",    default="all",
                        choices=["stage_hint", "stage_binary", "behavior_stage", "all"],
                        help="Label column to use (default: all)")
    parser.add_argument("--top-features", default=None,
                        help="Path to a feature_importance.csv -- restrict training to its top N features")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top features to use when --top-features is set (default: 20)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.features)

    selected_features = None
    if args.top_features:
        selected_features = load_top_features(args.top_features, top_n=args.top_n)
        print(f"[+] Restricting to top {args.top_n} features from {args.top_features}")
        print(f"    {selected_features}\n")

    labels_to_run = []
    if args.label == "stage_hint":
        print("[~] stage_hint is a reference label only -- skipping training")
    if args.label in ("stage_binary", "all"):
        labels_to_run.append(("stage_binary", STAGE_NAMES_EARLY_LATE))
    if args.label in ("behavior_stage", "all"):
        if "behavior_stage" in df.columns:
            labels_to_run.append(("behavior_stage", STAGE_NAMES_BEHAVIOR))
        else:
            print("[!] behavior_stage not found in features.csv -- skipping")

    models_used   = list(get_models().keys())
    label_cols    = [lc for lc, _ in labels_to_run]
    cv_mode       = "none" if args.no_loo else args.cv_mode
    scenarios_run = ["Standard 80/20 split"]
    if cv_mode in ("family", "both"):
        scenarios_run.append("LOO (leave-one-family-out)")
    if cv_mode in ("instance", "both"):
        scenarios_run.append("LOIO (leave-one-instance-out)")

    # Write top-level run log before training starts
    write_run_log(args.out, df, label_cols, models_used, scenarios_run,
                  extra={"cv_mode": cv_mode,
                         "top_features": args.top_features or "all",
                         "top_n": args.top_n if args.top_features else "--"})

    for label_col, stage_names in labels_to_run:
        label_dir = os.path.join(args.out, label_col) if len(labels_to_run) > 1 else args.out
        os.makedirs(label_dir, exist_ok=True)

        print("\n" + "#" * 60)
        print(f" LABEL: {label_col}")
        print(f" Stages: {stage_names}")
        print("#" * 60)

        print(f"\n    Distribution:\n{df.groupby(['family', label_col]).size().to_string()}\n")

        X, y, feat_cols, lm = prepare_xy(df, label_col=label_col, selected_features=selected_features)

        print(f"[+] Feature columns: {len(feat_cols)}")
        print(f"[+] Models: {', '.join(models_used)}\n")

        acc = run_standard_split(X, y, feat_cols, label_dir, stage_names=stage_names, label_map=lm)

        if cv_mode in ("family", "both") and len(df["family"].unique()) > 1:
            run_loo(df, feat_cols, label_dir, label_col=label_col, stage_names=stage_names, label_map=lm,
                    selected_features=selected_features)

        if cv_mode in ("instance", "both"):
            run_loio(df, feat_cols, label_dir, label_col=label_col, stage_names=stage_names, label_map=lm,
                     selected_features=selected_features)

        # Per-label log so each sub-folder is self-contained
        write_run_log(label_dir, df, [label_col], models_used, scenarios_run,
                      extra={"standard_accuracy": acc, "cv_mode": cv_mode})

        print("\n" + "=" * 60)
        print(f" Done ({label_col}).  Standard accuracy: {acc:.4f}")
        print(f" Output directory: {label_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
