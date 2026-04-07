"""
train_stage_model.py
--------------------
Trains a family-agnostic ransomware stage classifier from features.csv.

Label:   stage_hint  (0=pre-launch, 1=pre-encryption, 2=encrypting, 3=post-encryption)
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
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

STAGE_NAMES_TIME = {
    0: "Pre-launch (baseline)",
    1: "Pre-encryption",
    2: "Encrypting",
    3: "Post-encryption",
}

STAGE_NAMES_BEHAVIOR = {
    0: "Pre-encryption",
    1: "Encryption-active",
}

# Columns to drop — not forensic features
DROP_COLS = {
    "family",           # don't let the model use family identity
    "stage_hint",       # time-based label
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

    return df


def prepare_xy(df, label_col="stage_hint"):
    """Split into feature matrix X and label vector y.

    Remaps labels to 0..N-1 so XGBoost works with non-contiguous classes.
    Returns (X, y, feat_cols, label_map) where label_map converts idx->original.
    """
    drop = set(DROP_COLS)
    if label_col == "behavior_stage":
        drop |= BEHAVIOR_LEAKAGE_COLS
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

def get_models():
    """Return dict of name -> classifier for multi-model comparison."""
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
        "SVM": SVC(
            kernel="rbf", C=1.0, class_weight="balanced",
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1,
        ),
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    present_stages = sorted(y.unique())
    target_names   = [stage_names.get(label_map.get(s, s), f"Stage {s}") for s in present_stages]

    models = get_models()
    comparison = []
    best_acc, best_name, best_pipe = 0, None, None

    for name, clf in models.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        comparison.append({"model": name, "accuracy": round(acc, 4)})

        report = classification_report(y_test, preds,
                                       labels=present_stages,
                                       target_names=target_names,
                                       zero_division=0)
        print(f"\n  {name}: {acc:.4f}")
        print(report)

        # Save per-model classification report
        report_df = pd.DataFrame(
            classification_report(y_test, preds,
                                  labels=present_stages,
                                  target_names=target_names,
                                  output_dict=True,
                                  zero_division=0)
        ).T
        report_df.to_csv(os.path.join(out_dir, f"report_standard_{name}.csv"))

        _save_confusion_matrix(y_test, preds, present_stages, target_names,
                               os.path.join(out_dir, f"cm_standard_{name}.png"),
                               title=f"Standard split — {name}")

        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    # Save comparison table
    comp_df = pd.DataFrame(comparison).sort_values("accuracy", ascending=False)
    comp_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    print(f"\n{'=' * 40}")
    print(f" Model comparison (standard split):")
    print(f"{'=' * 40}")
    print(comp_df.to_string(index=False))

    # Save best model
    print(f"\n  Best: {best_name} ({best_acc:.4f})")
    _save_feature_importance(best_pipe, feat_cols,
                             os.path.join(out_dir, "feature_importance.csv"),
                             os.path.join(out_dir, "feature_importance.png"))

    model_path = os.path.join(out_dir, "stage_model.joblib")
    joblib.dump({"pipeline": best_pipe, "feature_cols": feat_cols,
                 "stage_names": stage_names, "model_name": best_name}, model_path)
    print(f"  [✓] Model saved: {model_path}")

    return best_acc


# -----------------------------------------------------------------------------
# Scenario 2: leave-one-family-out
# -----------------------------------------------------------------------------

def run_loo(df, feat_cols, out_dir, label_col="stage_hint", stage_names=None, label_map=None):
    stage_names = stage_names or STAGE_NAMES_TIME
    label_map = label_map or {}
    print("\n" + "=" * 60)
    print(" SCENARIO 2: Leave-one-family-out (generalization test)")
    print(" Train on N-1 families, test on the held-out family.")
    print("=" * 60)

    families  = sorted(df["family"].unique())
    X_all, y_all, _, _ = prepare_xy(df, label_col=label_col)
    models = get_models()

    # results[model_name] = list of per-family dicts
    all_results = {name: [] for name in models}

    for held_out in families:
        train_mask = df["family"] != held_out
        test_mask  = df["family"] == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]

        if len(X_test) == 0 or len(y_test.unique()) < 1:
            continue

        # Remap train labels to 0..N-1 for XGBoost
        unique_train = sorted(int(x) for x in y_train.unique())
        to_idx = {v: i for i, v in enumerate(unique_train)}
        from_idx = {i: v for v, i in to_idx.items()}
        y_train_enc = np.array([to_idx[int(v)] for v in y_train])

        present = sorted(y_test.unique())
        names   = [stage_names.get(label_map.get(s, s), f"Stage {s}") for s in present]

        print(f"\n  --- Held-out: {held_out} ({len(y_test)} rows) ---")

        best_acc, best_name, best_preds = 0, None, None

        for model_name, clf in models.items():
            pipe = make_pipeline(clf)
            pipe.fit(X_train, y_train_enc)
            preds_enc = pipe.predict(X_test)
            preds = np.array([from_idx.get(int(p), int(p)) for p in preds_enc])

            acc = accuracy_score(y_test, preds)
            print(f"    {model_name:20s}  accuracy: {acc:.4f}")
            all_results[model_name].append({
                "held_out_family": held_out, "accuracy": acc,
                "n_test": len(y_test),
            })

            if acc > best_acc:
                best_acc, best_name, best_preds = acc, model_name, preds

        # Save confusion matrix and classification report for best model per family
        _save_confusion_matrix(
            y_test, best_preds, present, names,
            os.path.join(out_dir, f"cm_loo_{held_out}.png"),
            title=f"LOO — {held_out} (best: {best_name})"
        )
        report_df = pd.DataFrame(
            classification_report(y_test, best_preds,
                                  labels=present,
                                  target_names=names,
                                  output_dict=True,
                                  zero_division=0)
        ).T
        report_df.to_csv(os.path.join(out_dir, f"report_loo_{held_out}.csv"))

    # Build LOO comparison table
    loo_summary = []
    for model_name, results in all_results.items():
        if not results:
            continue
        rdf = pd.DataFrame(results)
        mean_acc = rdf["accuracy"].mean()
        row = {"model": model_name, "mean_accuracy": round(mean_acc, 4)}
        for _, r in rdf.iterrows():
            row[r["held_out_family"]] = round(r["accuracy"], 4)
        loo_summary.append(row)

    if loo_summary:
        loo_df = pd.DataFrame(loo_summary).sort_values("mean_accuracy", ascending=False)
        print(f"\n{'=' * 60}")
        print(f" LOO Model Comparison:")
        print(f"{'=' * 60}")
        print(loo_df.to_string(index=False))
        loo_df.to_csv(os.path.join(out_dir, "loo_results.csv"), index=False)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_pred, labels, names, path, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
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
    parser.add_argument("--label",    default="both",
                        choices=["stage_hint", "behavior_stage", "both"],
                        help="Label column to use (default: both)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.features)

    labels_to_run = []
    if args.label in ("stage_hint", "both"):
        labels_to_run.append(("stage_hint", STAGE_NAMES_TIME))
    if args.label in ("behavior_stage", "both"):
        if "behavior_stage" in df.columns:
            labels_to_run.append(("behavior_stage", STAGE_NAMES_BEHAVIOR))
        else:
            print("[!] behavior_stage not found in features.csv — skipping")

    for label_col, stage_names in labels_to_run:
        label_dir = os.path.join(args.out, label_col) if len(labels_to_run) > 1 else args.out
        os.makedirs(label_dir, exist_ok=True)

        print("\n" + "#" * 60)
        print(f" LABEL: {label_col}")
        print(f" Stages: {stage_names}")
        print("#" * 60)

        print(f"\n    Distribution:\n{df.groupby(['family', label_col]).size().to_string()}\n")

        X, y, feat_cols, lm = prepare_xy(df, label_col=label_col)

        print(f"[+] Feature columns: {len(feat_cols)}")
        print(f"[+] Models: {', '.join(get_models().keys())}\n")

        acc = run_standard_split(X, y, feat_cols, label_dir, stage_names=stage_names, label_map=lm)

        if not args.no_loo and len(df["family"].unique()) > 1:
            run_loo(df, feat_cols, label_dir, label_col=label_col, stage_names=stage_names, label_map=lm)

        print("\n" + "=" * 60)
        print(f" Done ({label_col}).  Standard accuracy: {acc:.4f}")
        print(f" Output directory: {label_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
