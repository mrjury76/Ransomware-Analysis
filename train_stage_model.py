"""
train_stage_model.py
--------------------
Trains a family-agnostic ransomware stage classifier from features.csv.

Label:   stage_hint  (0=benign, 1=pre-launch, 2=pre-encryption, 3=encrypting, 4=post-encryption)
Goal:    detect the stage of ransomware execution from memory forensics alone,
         regardless of which ransomware family produced the snapshot.

Tests two scenarios:
  1. Standard split      — random 80/20, all families in both train and test
  2. Leave-one-out (LOO) — train on N-1 families, test on the held-out family
                           repeated for every family. Tests true generalization
                           to unseen ransomware.

Usage:
    python3 train_stage_model.py --features features.csv
    python3 train_stage_model.py --features features.csv --out model_output/
    python3 train_stage_model.py --features features.csv --no-loo
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, balanced_accuracy_score,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    0: "Pre-encryption",
    1: "Encryption-active",
}

STAGE_NAMES_EARLY_LATE = {
    0: "Early (benign/pre-launch/pre-encryption)",
    1: "Late (encrypting/post-encryption)",
}

# Columns to drop — not forensic features
DROP_COLS = {
    "family",           # don't let the model use family identity
    "stage_hint",       # time-based label
    "stage_binary",     # collapsed time-based label (0+1 -> 0, 2+3 -> 1)
    "behavior_stage",   # behavior-based label
    "actual_offset_s",  # time since launch — not available in real detection
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
    "pslist_ransom_procs",
    "handle_encrypted_files",
    "filescan_encrypted_ratio",
}

    # Features that act as shortcuts to the ransomware stage (late-stage indicators).
# Dropped during stage_hint / stage_binary training to avoid trivial leakage.
STAGE_SHORTCUT_COLS = {
    "filescan_encrypted",
    "filescan_ransom_notes",
    "handle_encrypted_files",
    "filescan_encrypted_ratio",
}



# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data(features_csv):
    df = pd.read_csv(features_csv, low_memory=False)

    required = {"stage_hint", "family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features.csv is missing columns: {missing}")

    print(f"[+] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"    Families : {sorted(df['family'].unique())}")
    print(f"    Stages   : {sorted(df['stage_hint'].unique())}")
    print(f"    Distribution:\n{df.groupby(['family','stage_hint']).size().to_string()}\n")

    # Collapsed binary label: 0+1+2 -> 0 (early/benign), 3+4 -> 1 (late/active)
    df["stage_binary"] = (df["stage_hint"].astype(int) >= 3).astype(int)

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
    # from extract_features import STAGE_SHORTCUT_COLS
    drop = set(DROP_COLS)
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
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        ),
        # "SVM": SVC(
        #     kernel="rbf", C=1.0, class_weight="balanced",
        #     random_state=42,
        # ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        # "KNN": KNeighborsClassifier(
        #     n_neighbors=5, n_jobs=-1,
        # ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.2,
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
        print(f"[!] y has only 1 class — skipping standard split.")
        return 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if len(pd.Series(y_train).unique()) < 2:
        print(f"[!] y_train has only 1 class after split — skipping standard split.")
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
            print(f"\n  [!] {name} failed: {type(e).__name__}: {e} — skipping")
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
        print(f"    test  — acc: {m['accuracy']:.4f}  [{ci['acc_ci_lo']:.4f}–{ci['acc_ci_hi']:.4f} 95% CI]"
              f"  bal_acc: {m['balanced_acc']:.4f}")
        print(f"    macro_f1: {m['macro_f1']:.4f}  [{ci['f1_ci_lo']:.4f}–{ci['f1_ci_hi']:.4f} 95% CI]"
              f"  weighted_f1: {m['weighted_f1']:.4f}")
        print(f"    train — acc: {m_train['accuracy']:.4f}  overfit gap: {overfit_gap:+.4f}")
        if conf:
            print(f"    confidence — mean: {conf['conf_mean']:.4f}  min: {conf['conf_min']:.4f}"
                  f"  ≥90%: {conf['conf_pct_90']:.1%}  ≥70%: {conf['conf_pct_70']:.1%}")
        print(classification_report(y_test, preds, labels=present_stages,
                                    target_names=target_names, zero_division=0))

        report_df = pd.DataFrame(
            classification_report(y_test, preds, labels=present_stages,
                                  target_names=target_names,
                                  output_dict=True, zero_division=0)
        ).T
        report_df.to_csv(os.path.join(out_dir, f"report_standard_{name}.csv"))

        _save_confusion_matrix(y_test, preds, present_stages, target_names,
                               os.path.join(out_dir, f"cm_standard_{name}.png"),
                               title=f"Standard split — {name}", normalize=True)

        _save_calibration_plot(pipe, X_test, y_test, len(present_stages),
                               os.path.join(out_dir, f"calibration_{name}.png"))

        if m["macro_f1"] > best_score:
            best_score, best_name, best_pipe = m["macro_f1"], name, pipe

    if not comparison:
        print("  [!] No model succeeded.")
        return 0

    comp_df = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False)
    comp_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    print(f"\n{'=' * 40}")
    print(f" Model comparison (standard split):")
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
    print(f"  [✓] Model saved: {model_path}")

    return comp_df.iloc[0]["accuracy"]


# -----------------------------------------------------------------------------
# Scenario 2: leave-one-family-out
# -----------------------------------------------------------------------------

def run_loo(df, feat_cols, out_dir, label_col="stage_hint", stage_names=None, label_map=None,
            selected_features=None, benign_family="Benign"):
    """
    benign_family: the family name used for the control/benign group.
      When this family is held out the model was never trained on stage-0 (benign)
      samples, so accuracy is always 0% — not because it detected ransomware, but
      because the label can't appear in predictions.  We instead report
      *specificity* = fraction of benign samples the model did NOT flag as a
      ransomware stage, which inverts the metric so that 100% = no false positives.
      Benign is excluded from the ransomware-family mean so it doesn't drag it down.
    """
    stage_names = stage_names or STAGE_NAMES_TIME
    label_map = label_map or {}
    print("\n" + "=" * 60)
    print(" SCENARIO 2: Leave-one-family-out (generalization test)")
    print(" Train on N-1 families, test on the held-out family.")
    print(f" Benign control family: '{benign_family}' (tracked separately, inverted metric)")
    print("=" * 60)

    families  = sorted(df["family"].unique())
    X_all, y_all, _, _ = prepare_xy(df, label_col=label_col, selected_features=selected_features)
    models = get_models()

    # all_results[model_name] = list of per-family metric dicts
    all_results = {name: [] for name in models}

    # Reverse label_map: remapped-idx → original stage value (e.g. {0: 0, 1: 1, ...})
    # Used to identify which remapped label corresponds to the benign/stage-0 class.
    benign_original_label = 0          # stage 0 = benign in all label schemas
    # The global remap from prepare_xy maps original labels to 0..N-1.
    # Find which remapped index corresponds to the benign original label.
    benign_remapped = None
    for idx, orig in label_map.items():
        if orig == benign_original_label:
            benign_remapped = idx
            break

    for held_out in tqdm(families, desc="LOO folds", unit="family"):
        is_benign_fold = (held_out == benign_family)
        train_mask = df["family"] != held_out
        test_mask  = df["family"] == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]

        if len(X_test) == 0 or len(y_test.unique()) < 1:
            continue
        if len(y_train.unique()) < 2:
            print(f"\n  --- Held-out: {held_out} — skipping (training set has only 1 class) ---")
            continue

        # Remap train labels to 0..N-1 for XGBoost
        unique_train = sorted(int(x) for x in y_train.unique())
        to_idx = {v: i for i, v in enumerate(unique_train)}
        from_idx = {i: v for v, i in to_idx.items()}
        y_train_enc = np.array([to_idx[int(v)] for v in y_train])

        present = sorted(y_test.unique())
        names   = [stage_names.get(label_map.get(s, s), f"Stage {s}") for s in present]

        if is_benign_fold:
            print(f"\n  --- Held-out: {held_out} [CONTROL] ({len(y_test)} rows) ---")
            print(f"      Metric: specificity (1 - FPR). 100% = no benign samples")
            print(f"      flagged as ransomware. Excluded from ransomware mean.")
        else:
            print(f"\n  --- Held-out: {held_out} ({len(y_test)} rows, {len(y_train)} train) ---")

        best_macro_f1, best_name, best_preds = -1, None, None

        for model_name, clf in models.items():
            try:
                pipe = make_pipeline(clf)
                pipe.fit(X_train, y_train_enc)
                preds_enc = pipe.predict(X_test)
            except Exception as e:
                print(f"    {model_name:20s}  [!] failed: {type(e).__name__}: {e}")
                continue
            preds = np.array([from_idx.get(int(p), int(p)) for p in preds_enc])

            if is_benign_fold:
                # Specificity: fraction of benign samples predicted as the benign label.
                # Since the model was trained without benign, it can never output the benign
                # label → all predictions are ransomware stages → FPR = 1.0.
                # specificity = fraction predicted as benign_remapped (the "benign" label idx).
                # This will typically be 0.0, inverted to show FPR = 100%.
                # Fraction of benign test samples NOT predicted as any ransomware stage
                # = fraction predicted as the benign label (index benign_remapped).
                if benign_remapped is not None:
                    specificity = float((preds == benign_remapped).mean())
                else:
                    specificity = 0.0
                fpr = 1.0 - specificity   # fraction incorrectly flagged as ransomware
                print(f"    {model_name:20s}  specificity: {specificity:.4f}  "
                      f"FPR (flagged as ransomware): {fpr:.4f}")
                all_results[model_name].append({
                    "held_out_family": held_out,
                    "n_test":          len(y_test),
                    "is_benign":       True,
                    "specificity":     round(specificity, 4),
                    "fpr":             round(fpr, 4),
                    # Fill standard metric keys so DataFrame columns are consistent.
                    "accuracy":        round(specificity, 4),
                    "balanced_acc":    round(specificity, 4),
                    "macro_f1":        round(specificity, 4),
                    "weighted_f1":     round(specificity, 4),
                })
                if specificity > best_macro_f1:
                    best_macro_f1, best_name, best_preds = specificity, model_name, preds
            else:
                m = _metrics(y_test, preds)
                print(f"    {model_name:20s}  acc: {m['accuracy']:.4f}  "
                      f"bal_acc: {m['balanced_acc']:.4f}  macro_f1: {m['macro_f1']:.4f}  "
                      f"weighted_f1: {m['weighted_f1']:.4f}")
                all_results[model_name].append({
                    "held_out_family": held_out,
                    "n_test":          len(y_test),
                    "is_benign":       False,
                    **m,
                })
                if m["macro_f1"] > best_macro_f1:
                    best_macro_f1, best_name, best_preds = m["macro_f1"], model_name, preds

        if best_preds is None:
            print(f"    [!] All models failed for {held_out} — skipping outputs.")
            continue

        cm_title = (f"LOO — {held_out} specificity={best_macro_f1:.2f} ({best_name})"
                    if is_benign_fold else
                    f"LOO — {held_out} (best macro F1: {best_name})")
        _save_confusion_matrix(
            y_test, best_preds, present, names,
            os.path.join(out_dir, f"cm_loo_{held_out}.png"),
            title=cm_title,
            normalize=True,
        )
        report_df = pd.DataFrame(
            classification_report(y_test, best_preds, labels=present,
                                  target_names=names, output_dict=True, zero_division=0)
        ).T
        report_df.to_csv(os.path.join(out_dir, f"report_loo_{held_out}.csv"))

    # Build LOO summary:
    # Ransomware families → mean/std/CI on macro F1 (higher = better).
    # Benign family      → mean specificity (higher = fewer false positives).
    loo_summary = []
    for model_name, results in all_results.items():
        if not results:
            continue
        rdf = pd.DataFrame(results)

        ransom_rdf = rdf[~rdf["is_benign"]]
        benign_rdf = rdf[rdf["is_benign"]]

        if ransom_rdf.empty:
            continue

        total_n_r = ransom_rdf["n_test"].sum()
        n_folds   = len(ransom_rdf)
        sem_acc   = ransom_rdf["accuracy"].std() / np.sqrt(n_folds) if n_folds > 1 else 0.0
        sem_f1    = ransom_rdf["macro_f1"].std()  / np.sqrt(n_folds) if n_folds > 1 else 0.0

        row = {
            "model":                  model_name,
            # Ransomware-only stats
            "mean_accuracy":          round(ransom_rdf["accuracy"].mean(), 4),
            "std_accuracy":           round(ransom_rdf["accuracy"].std(), 4),
            "acc_ci95_lo":            round(ransom_rdf["accuracy"].mean() - 1.96 * sem_acc, 4),
            "acc_ci95_hi":            round(ransom_rdf["accuracy"].mean() + 1.96 * sem_acc, 4),
            "wtd_accuracy":           round((ransom_rdf["accuracy"] * ransom_rdf["n_test"]).sum() / total_n_r, 4),
            "mean_balanced_acc":      round(ransom_rdf["balanced_acc"].mean(), 4),
            "mean_macro_f1":          round(ransom_rdf["macro_f1"].mean(), 4),
            "std_macro_f1":           round(ransom_rdf["macro_f1"].std(), 4),
            "f1_ci95_lo":             round(ransom_rdf["macro_f1"].mean() - 1.96 * sem_f1, 4),
            "f1_ci95_hi":             round(ransom_rdf["macro_f1"].mean() + 1.96 * sem_f1, 4),
            "wtd_macro_f1":           round((ransom_rdf["macro_f1"] * ransom_rdf["n_test"]).sum() / total_n_r, 4),
            "mean_weighted_f1":       round(ransom_rdf["weighted_f1"].mean(), 4),
            "n_ransom_folds":         n_folds,
            # Benign control stats (separate, inverted metric)
            "benign_specificity":     round(benign_rdf["specificity"].mean(), 4) if not benign_rdf.empty else None,
            "benign_fpr":             round(benign_rdf["fpr"].mean(), 4) if not benign_rdf.empty else None,
        }
        # Per-family column: ransomware families show macro_f1, Benign shows specificity
        for _, r in rdf.iterrows():
            fam = r["held_out_family"]
            row[fam] = round(r["specificity"], 4) if r["is_benign"] else round(r["macro_f1"], 4)
        loo_summary.append(row)

    if loo_summary:
        loo_df = pd.DataFrame(loo_summary).sort_values("mean_macro_f1", ascending=False)
        print(f"\n{'=' * 60}")
        print(f" LOO Model Comparison (ransomware families only, sorted by mean macro F1):")
        print(f" Benign column = specificity (1-FPR). 100% = zero false positives.")
        print(f"{'=' * 60}")
        summary_cols = ["model", "mean_accuracy", "acc_ci95_lo", "acc_ci95_hi",
                        "wtd_accuracy", "std_accuracy",
                        "mean_balanced_acc", "mean_macro_f1", "f1_ci95_lo", "f1_ci95_hi",
                        "wtd_macro_f1", "std_macro_f1", "mean_weighted_f1",
                        "n_ransom_folds", "benign_specificity", "benign_fpr"]
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
        print("[!] 'rep' column not in features.csv — cannot run LOIO")
        return

    # Group key = (family, rep). Each group = one full timed collection run.
    df_work = df.copy()
    df_work["_instance"] = df_work["family"].astype(str) + "::rep" + df_work["rep"].astype(str)
    instances = sorted(df_work["_instance"].unique())

    X_all, y_all, _, _ = prepare_xy(df_work, label_col=label_col, selected_features=selected_features)
    models = get_models()
    all_results = {name: [] for name in models}

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
            print(f"  --- Held-out: {held_out} — skipping (training set has only 1 class) ---")
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
        print(f" LOIO Model Comparison (sorted by mean macro F1):")
        print(f"{'=' * 60}")
        summary_cols = ["model", "mean_accuracy", "acc_ci95_lo", "acc_ci95_hi",
                        "wtd_accuracy", "std_accuracy",
                        "mean_balanced_acc", "mean_macro_f1", "f1_ci95_lo", "f1_ci95_hi",
                        "wtd_macro_f1", "std_macro_f1", "mean_weighted_f1", "n_instances"]
        print(loio_df[[c for c in summary_cols if c in loio_df.columns]].to_string(index=False))
        loio_df.to_csv(os.path.join(out_dir, "loio_results.csv"), index=False)

        # Save per-instance detail for every model so nothing is lost
        for model_name, results in all_results.items():
            if results:
                pd.DataFrame(results).to_csv(
                    os.path.join(out_dir, f"loio_detail_{model_name}.csv"), index=False
                )
        print(f"  [✓] Per-instance detail CSVs saved for all models.")


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
    print(f"  [✓] Run log: {path}")


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


def _save_calibration_plot(pipe, X_test, y_test, n_classes, out_path):
    """One-vs-rest calibration curves: predicted probability vs actual frequency.

    Only drawn for models that support predict_proba.
    """
    model = pipe.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        return
    try:
        proba = pipe.predict_proba(X_test)
    except Exception:
        return

    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for cls_idx in range(min(n_classes, proba.shape[1])):
        binary_y = (np.asarray(y_test) == cls_idx).astype(int)
        if binary_y.sum() == 0:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(binary_y, proba[:, cls_idx], n_bins=10)
            ax.plot(mean_pred, frac_pos, marker="o", label=f"Class {cls_idx}")
        except Exception:
            continue

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves (one-vs-rest)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  [✓] Saved: {out_path}")


def _save_confusion_matrix(y_true, y_pred, labels, names, path, title="", normalize=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) if normalize else plt.subplots(figsize=(8, 6))
    axes = list(axes) if normalize else [axes]

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm_raw, display_labels=names).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(f"{title} — counts")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha="right")

    if normalize:
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        ConfusionMatrixDisplay(
            np.round(cm_norm, 2), display_labels=names
        ).plot(ax=axes[1], colorbar=False, cmap="Blues")
        axes[1].set_title(f"{title} — recall (row-normalized)")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [✓] Saved: {path}")


def _save_feature_importance(pipe, feat_cols, csv_path, plot_path, top_n=20):
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    fi = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.feature_importances_,
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
    print(f"  [✓] Saved: {csv_path}")


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
                        help="Path to a feature_importance.csv — restrict training to its top N features")
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
    if args.label in ("stage_hint", "all"):
        labels_to_run.append(("stage_hint", STAGE_NAMES_TIME))
    if args.label in ("stage_binary", "all"):
        labels_to_run.append(("stage_binary", STAGE_NAMES_EARLY_LATE))
    if args.label in ("behavior_stage", "all"):
        if "behavior_stage" in df.columns:
            labels_to_run.append(("behavior_stage", STAGE_NAMES_BEHAVIOR))
        else:
            print("[!] behavior_stage not found in features.csv — skipping")

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
                         "top_n": args.top_n if args.top_features else "—"})

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
