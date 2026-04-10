"""
train_binary_detector.py
------------------------
Simple binary classifier: Benign vs Ransomware.

Takes features.csv, collapses all ransomware families into a single "Ransomware"
class, and evaluates whether memory forensics can distinguish benign from
malicious activity.

Usage:
    python3 train_binary_detector.py --features output/features.csv
    python3 train_binary_detector.py --features output/features.csv --out binary_results/
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Columns to drop — metadata, not forensic features
DROP_COLS = {
    "family", "label",
    "stage_hint", "stage_binary", "behavior_stage",
    "actual_offset_s", "target_offset_s",
    "rep", "run", "snap_name", "snap_dir",
}

CLASS_NAMES = ["Benign", "Ransomware"]


def get_models():
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


def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   clf),
    ])


def main():
    parser = argparse.ArgumentParser(description="Binary classifier: Benign vs Ransomware")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--out", default="binary_results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load and label ───────────────────────────────────────────────────────
    df = pd.read_csv(args.features, low_memory=False)
    df["label"] = (df["family"] != "Benign").astype(int)  # 0=Benign, 1=Ransomware

    n_benign = (df["label"] == 0).sum()
    n_ransom = (df["label"] == 1).sum()
    families = sorted(df.loc[df["label"] == 1, "family"].unique())

    print("=" * 60)
    print(" Binary Detector: Benign vs Ransomware")
    print("=" * 60)
    print(f"  Benign samples    : {n_benign}")
    print(f"  Ransomware samples: {n_ransom}  ({', '.join(families)})")
    print(f"  Total             : {len(df)}")
    print()

    # ── Feature matrix ───────────────────────────────────────────────────────
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].values

    # ── 1) Stratified 80/20 split ────────────────────────────────────────────
    print("=" * 60)
    print(" SCENARIO 1: Standard 80/20 split")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = get_models()
    results = []

    for name, clf in tqdm(models.items(), desc="Standard split", total=len(models), unit="model"):
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append({"model": name, "accuracy": round(acc, 4)})

        print(f"\n  {name}: {acc:.4f}")
        report = classification_report(y_test, preds, target_names=CLASS_NAMES,
                                       zero_division=0)
        print(report)

        # Save confusion matrix
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Benign vs Ransomware — {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"cm_{name}.png"), dpi=150)
        plt.close(fig)

        # Save classification report
        report_df = pd.DataFrame(
            classification_report(y_test, preds, target_names=CLASS_NAMES,
                                  output_dict=True, zero_division=0)
        ).T
        report_df.to_csv(os.path.join(args.out, f"report_{name}.csv"))

    # Model comparison table
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    print("\n" + "=" * 60)
    print(" Model comparison:")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # ── 2) 5-fold cross-validation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SCENARIO 2: 5-fold stratified cross-validation")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for name, clf in tqdm(models.items(), desc="5-fold CV", total=len(models), unit="model"):
        pipe = make_pipeline(clf)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        mean, std = scores.mean(), scores.std()
        cv_results.append({"model": name, "mean_accuracy": round(mean, 4),
                           "std": round(std, 4)})
        tqdm.write(f"  {name:20s}  {mean:.4f} +/- {std:.4f}")

    cv_df = pd.DataFrame(cv_results).sort_values("mean_accuracy", ascending=False)
    cv_df.to_csv(os.path.join(args.out, "cv_results.csv"), index=False)

    # ── 3) Feature importance (best model from split) ────────────────────────
    best_name = results_df.iloc[0]["model"]
    best_clf = models[best_name]
    pipe = make_pipeline(best_clf)
    pipe.fit(X, y)

    if hasattr(pipe["model"], "feature_importances_"):
        importances = pipe["model"].feature_importances_
        fi = pd.DataFrame({"feature": feat_cols, "importance": importances})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(args.out, "feature_importance.csv"), index=False)

        # Plot top 20
        top = fi.head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(top)), top["importance"].values, color="steelblue")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 20 Features — Benign vs Ransomware ({best_name})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "feature_importance.png"), dpi=150)
        plt.close(fig)
        print(f"\n  Top 10 features ({best_name}):")
        for _, row in fi.head(10).iterrows():
            print(f"    {row['feature']:40s} {row['importance']:.4f}")

    print(f"\n{'=' * 60}")
    print(f" Results saved to: {args.out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
