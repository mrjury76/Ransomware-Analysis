"""
run_pipeline.py
---------------
Master pipeline:
  Step 1 -- autovol4    : run Volatility on all vmem files
  Step 2 -- features    : extract ML feature matrix -> features.csv
  Step 3 -- train model : train family-agnostic stage classifier

Usage:
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --skip-analysis
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --skip-analysis --skip-training
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --model-out /mnt/d/Patrick/model
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --family WannaCry --skip-analysis
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --skip-analysis --label behavior_stage
"""

import argparse
import json
import os
import subprocess
import sys

# -- locate sibling scripts ---------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
AUTOVOL4_PATH = "/home/patrick/tools/volatility3/autovol4_new.py"

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(AUTOVOL4_PATH))


def main():
    parser = argparse.ArgumentParser(description="autovol4 + feature extraction + model training pipeline")
    parser.add_argument("--scan-dir",       required=True,
                        help="Root directory containing snapshot folders")
    parser.add_argument("--out",            default=None,
                        help="Output features CSV path (default: <scan-dir>/features.csv)")
    parser.add_argument("--model-out",      default=None,
                        help="Output directory for trained model (default: <scan-dir>/model_output)")
    parser.add_argument("--skip-analysis",  action="store_true",
                        help="Skip autovol4 -- use existing plugin CSVs")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip model training -- stop after features.csv")
    parser.add_argument("--no-loo",         action="store_true",
                        help="Skip leave-one-family-out evaluation during training")
    parser.add_argument("--cv-mode",        default="family",
                        choices=["family", "instance", "both", "none"],
                        help="CV mode: 'family' (LOO), 'instance' (LOIO hold one family×rep), 'both', 'none'")
    parser.add_argument("--family",        default=None,
                        help="Run only this family + Benign (e.g. --family WannaCry)")
    parser.add_argument("--cache",         action="store_true",
                        help="Use cached feature extraction results (features_cache.json) when available")
    parser.add_argument("--top-features",  default=None,
                        help="Path to a feature_importance.csv -- restrict stage training to its top N features")
    parser.add_argument("--top-n",         type=int, default=20,
                        help="Number of top features to use when --top-features is set (default: 20)")
    parser.add_argument("--label",         default="all",
                        choices=["stage_hint", "stage_binary", "behavior_stage", "all"],
                        help="Which label(s) to train on (default: all)")
    args = parser.parse_args()

    scan_dir   = os.path.abspath(args.scan_dir)
    output_base = os.path.join(SCRIPT_DIR, "output")

    # --family filter: run only this family + Benign
    only_families = None
    if args.family:
        only_families = {args.family, "Benign"}
        print(f"[+] Family filter: {sorted(only_families)}")

    if args.model_out:
        model_out = args.model_out
    else:
        base = os.path.join(SCRIPT_DIR, "model_results", "run")
        n = 1
        while os.path.isdir(f"{base}{n:02d}"):
            n += 1
        model_out = f"{base}{n:02d}"

    out_csv = args.out or os.path.join(output_base, "features.csv")

    os.makedirs(output_base, exist_ok=True)

    if not os.path.isdir(scan_dir):
        print(f"[-] Directory not found: {scan_dir}")
        sys.exit(1)

    # ── Step 1: autovol4 batch analysis ───────────────────────────────────────
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print(" STEP 1: autovol4 -- Volatility analysis on all vmem files")
        print("=" * 60)

        # Import batch_mode from wherever autovol4.py lives
        import importlib.util
        spec    = importlib.util.spec_from_file_location("autovol4", AUTOVOL4_PATH)
        autovol4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(autovol4)

        autovol4.batch_mode(scan_dir, only_families=only_families)

        # Auto-commit and push new CSVs to dataset repo
    #     print("\n[+] Pushing new plugin CSVs to dataset repo...")
    #     try:
    #         git = ["git", "-C", scan_dir]
    #         subprocess.run(git + ["add", "-A"], check=True)
    #         result = subprocess.run(git + ["diff", "--cached", "--quiet"])
    #         if result.returncode != 0:
    #             subprocess.run(git + ["commit", "-m",
    #                            "Add volatility plugin CSVs from pipeline run"], check=True)
    #             subprocess.run(git + ["push"], check=True)
    #             print("[+] Dataset repo pushed.")
    #         else:
    #             print("[~] No new CSVs to commit.")
    #     except subprocess.CalledProcessError as e:
    #         print(f"[!] Git push failed: {e} -- continuing pipeline")
    # else:
    #     print("\n[~] Skipping autovol4 analysis (--skip-analysis)")

    # ── Step 2: feature extraction ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 2: extract_features -- building ML feature matrix")
    print("=" * 60)

    import extract_features

    snap_dirs = []
    for root, _, files in os.walk(scan_dir):
        if "meta.json" in files:
            if only_families:
                try:
                    with open(os.path.join(root, "meta.json")) as f:
                        fam = json.load(f).get("family", "")
                except Exception:
                    fam = ""
                if fam not in only_families:
                    continue
            snap_dirs.append(root)

    if not snap_dirs:
        print(f"[-] No snapshot directories found under {scan_dir}")
        sys.exit(1)

    print(f"[+] Found {len(snap_dirs)} snapshot(s)")

    rows = []
    for i, snap_dir_path in enumerate(sorted(snap_dirs), 1):
        print(f"  [{i}/{len(snap_dirs)}] {snap_dir_path}")
        row = extract_features.process_snapshot(snap_dir_path, use_cache=args.cache)
        if row:
            rows.append(row)

    if not rows:
        print("[-] No feature rows extracted.")
        sys.exit(1)

    import csv
    meta_cols = ["family", "stage_hint", "behavior_stage", "actual_offset_s",
                 "target_offset_s", "rep", "run", "snap_name", "snap_dir"]
    feat_cols  = [k for k in rows[0] if k not in meta_cols]
    fieldnames = meta_cols + feat_cols

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[[done]] Features saved: {out_csv} ({len(rows)} rows, {len(feat_cols)} features)")

    if args.skip_training:
        print("\n[~] Skipping model training (--skip-training)")
        print(f"\n{'=' * 60}\n Pipeline complete\n{'=' * 60}")
        return

    # ── Step 3: train stage model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 3: train_stage_model -- family-agnostic stage classifier")
    print("=" * 60)

    import train_stage_model

    os.makedirs(model_out, exist_ok=True)
    df = train_stage_model.load_data(out_csv)

    selected_features = None
    if args.top_features:
        selected_features = train_stage_model.load_top_features(args.top_features, top_n=args.top_n)
        print(f"[+] Restricting stage training to top {args.top_n} features from {args.top_features}")
        print(f"    {selected_features}\n")

    cv_mode      = "none" if args.no_loo else args.cv_mode
    models_used  = list(train_stage_model.get_models().keys())
    all_label_pairs = [("stage_hint",    train_stage_model.STAGE_NAMES_TIME),
                       ("stage_binary",  train_stage_model.STAGE_NAMES_EARLY_LATE),
                       ("behavior_stage",train_stage_model.STAGE_NAMES_BEHAVIOR)]
    label_pairs  = [(lc, sn) for lc, sn in all_label_pairs
                    if (args.label == "all" or args.label == lc) and lc in df.columns]
    label_cols   = [lc for lc, _ in label_pairs]
    scenarios    = ["Standard 80/20 split"]
    if cv_mode in ("family", "both"):
        scenarios.append("LOO (leave-one-family-out)")
    if cv_mode in ("instance", "both"):
        scenarios.append("LOIO (leave-one-instance-out)")

    # Top-level run log for the whole model_out directory
    train_stage_model.write_run_log(
        model_out, df, label_cols, models_used, scenarios,
        extra={"features_csv":  out_csv,
               "scan_dir":      scan_dir,
               "family_filter": sorted(only_families) if only_families else "all",
               "cv_mode":       cv_mode,
               "top_features":  args.top_features or "all",
               "top_n":         args.top_n if args.top_features else "--"},
    )

    # Train with behavior-based labels only.
    # stage_hint is kept in features.csv as a reference/baseline column but is
    # NOT used for training -- it reflects collection-time clock labels which are
    # inaccurate for fast families (Dharma, Cerber) where behavior_stage is
    # grounded in actual memory evidence.
    for label_col, stage_names in label_pairs:
        if label_col not in df.columns:
            continue
        if label_col == "stage_hint":
            print(f"\n[~] Skipping stage_hint training (reference label only)")
            continue

        label_dir = os.path.join(model_out, label_col)
        os.makedirs(label_dir, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f" LABEL: {label_col}")
        print(f"{'#' * 60}")

        X, y, fc, lm = train_stage_model.prepare_xy(df, label_col=label_col,
                                                    selected_features=selected_features)
        print(f"[+] Models: {', '.join(models_used)}\n")
        acc = train_stage_model.run_standard_split(X, y, fc, label_dir, stage_names=stage_names, label_map=lm)

        if cv_mode in ("family", "both") and len(df["family"].unique()) > 1:
            train_stage_model.run_loo(df, fc, label_dir, label_col=label_col,
                                      stage_names=stage_names, label_map=lm,
                                      selected_features=selected_features)

        if cv_mode in ("instance", "both"):
            train_stage_model.run_loio(df, fc, label_dir, label_col=label_col,
                                       stage_names=stage_names, label_map=lm,
                                       selected_features=selected_features)

        # Per-label log inside each label sub-folder
        train_stage_model.write_run_log(
            label_dir, df, [label_col], models_used, scenarios,
            extra={"standard_accuracy": acc, "cv_mode": cv_mode},
        )

    # ── Master summary across all labels and scenarios ───────────────────────
    print("\n" + "=" * 60)
    print(" Writing master summary...")
    print("=" * 60)
    train_stage_model.write_master_summary(model_out, label_cols, df=df)

    print(f"\n{'=' * 60}")
    print(f" Pipeline complete")
    print(f" Snapshots : {len(rows)}")
    print(f" Features  : {len(feat_cols)}")
    print(f" CSV       : {out_csv}")
    print(f" Model     : {model_out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
