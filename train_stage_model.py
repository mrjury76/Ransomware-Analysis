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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

STAGE_NAMES = {
    0: "Pre-launch (baseline)",
    1: "Pre-encryption",
    2: "Encrypting",
    3: "Post-encryption",
}

# Columns to drop — not forensic features
DROP_COLS = {
    "family",           # don't let the model use family identity
    "stage_hint",       # label
    "actual_offset_s",  # time since launch — not available in real detection
    "target_offset_s",
    "rep",
    "run",
    "snap_name",
    "snap_dir",
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


def prepare_xy(df):
    """Split into feature matrix X and label vector y."""
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()
    y = df["stage_hint"].astype(int)

    # Coerce everything to numeric, fill non-numeric with NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, feat_cols


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def make_pipeline():
    if HAS_XGBOOST:
        clf = XGBClassifier(
            objective="multi:softprob",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.2,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   clf),
    ])


# -----------------------------------------------------------------------------
# Scenario 1: standard stratified split
# -----------------------------------------------------------------------------

def run_standard_split(X, y, feat_cols, out_dir):
    print("=" * 60)
    print(" SCENARIO 1: Standard 80/20 split (all families mixed)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    present_stages = sorted(y.unique())
    target_names   = [STAGE_NAMES[s] for s in present_stages]

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds,
                                labels=present_stages,
                                target_names=target_names,
                                zero_division=0))

    _save_confusion_matrix(y_test, preds, present_stages, target_names,
                           os.path.join(out_dir, "cm_standard.png"),
                           title="Standard split confusion matrix")

    _save_feature_importance(pipe, feat_cols,
                             os.path.join(out_dir, "feature_importance.csv"),
                             os.path.join(out_dir, "feature_importance.png"))

    # Save model
    model_path = os.path.join(out_dir, "stage_model.joblib")
    joblib.dump({"pipeline": pipe, "feature_cols": feat_cols,
                 "stage_names": STAGE_NAMES}, model_path)
    print(f"\n[✓] Model saved: {model_path}")

    return acc


# -----------------------------------------------------------------------------
# Scenario 2: leave-one-family-out
# -----------------------------------------------------------------------------

def run_loo(df, feat_cols, out_dir):
    print("\n" + "=" * 60)
    print(" SCENARIO 2: Leave-one-family-out (generalization test)")
    print(" Train on N-1 families, test on the held-out family.")
    print("=" * 60)

    families  = sorted(df["family"].unique())
    X_all, y_all, _ = prepare_xy(df)
    results = []

    for held_out in families:
        train_mask = df["family"] != held_out
        test_mask  = df["family"] == held_out

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]

        if len(X_test) == 0 or len(y_test.unique()) < 1:
            continue

        pipe = make_pipeline()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"\n  Held-out: {held_out:12s}  |  test rows: {len(y_test):4d}  |  accuracy: {acc:.4f}")

        present = sorted(y_test.unique())
        names   = [STAGE_NAMES[s] for s in present]
        report = classification_report(y_test, preds,
                                       labels=present,
                                       target_names=names,
                                       zero_division=0)
        print("\n".join("    " + l for l in report.splitlines()))

        _save_confusion_matrix(
            y_test, preds, present, names,
            os.path.join(out_dir, f"cm_loo_{held_out}.png"),
            title=f"LOO confusion matrix — held-out: {held_out}"
        )

        results.append({"held_out_family": held_out, "accuracy": acc,
                        "n_test": len(y_test)})

    if results:
        loo_df = pd.DataFrame(results)
        print(f"\n  LOO summary:")
        print(loo_df.to_string(index=False))
        print(f"\n  Mean LOO accuracy: {loo_df['accuracy'].mean():.4f}")
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
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.features)
    X, y, feat_cols = prepare_xy(df)

    print(f"[+] Feature columns: {len(feat_cols)}")
    print(f"[+] Using {'XGBoost' if HAS_XGBOOST else 'RandomForest'}\n")

    acc = run_standard_split(X, y, feat_cols, args.out)

    if not args.no_loo and len(df["family"].unique()) > 1:
        run_loo(df, feat_cols, args.out)

    print("\n" + "=" * 60)
    print(f" Done.  Standard accuracy: {acc:.4f}")
    print(f" Output directory: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
