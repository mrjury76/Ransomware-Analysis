"""
run_pipeline.py
---------------
Master pipeline:
  Step 1 — autovol4    : run Volatility on all vmem files
  Step 2 — features    : extract ML feature matrix -> features.csv
  Step 3 — train model : train family-agnostic stage classifier

Usage:
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --skip-analysis
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --skip-analysis --skip-training
    python3 run_pipeline.py --scan-dir /mnt/d/Patrick/VMSnapshots --model-out /mnt/d/Patrick/model
    └─$ python3 /mnt/c/Users/Patrick/Desktop/MusfiqFinalProject/Ransomware-Analysis/run_pipeline.py   --scan-dir /mnt/d/Patrick/VMSnapshots --skip-analysis
"""

import argparse
import os
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
                        help="Skip autovol4 — use existing plugin CSVs")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip model training — stop after features.csv")
    parser.add_argument("--no-loo",         action="store_true",
                        help="Skip leave-one-family-out evaluation during training")
    args = parser.parse_args()

    scan_dir   = os.path.abspath(args.scan_dir)
    output_base = os.path.join(SCRIPT_DIR, "output")

    if args.model_out:
        model_out = args.model_out
    else:
        base = os.path.join(scan_dir, "model_output")
        n = 1
        while os.path.isdir(f"{base}_run{n:02d}"):
            n += 1
        model_out = f"{base}_run{n:02d}"

    out_csv = args.out or os.path.join(output_base, "features.csv")

    os.makedirs(output_base, exist_ok=True)

    if not os.path.isdir(scan_dir):
        print(f"[-] Directory not found: {scan_dir}")
        sys.exit(1)

    # ── Step 1: autovol4 batch analysis ───────────────────────────────────────
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print(" STEP 1: autovol4 — Volatility analysis on all vmem files")
        print("=" * 60)

        # Import batch_mode from wherever autovol4.py lives
        import importlib.util
        spec    = importlib.util.spec_from_file_location("autovol4", AUTOVOL4_PATH)
        autovol4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(autovol4)

        autovol4.batch_mode(scan_dir)
    else:
        print("\n[~] Skipping autovol4 analysis (--skip-analysis)")

    # ── Step 2: feature extraction ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 2: extract_features — building ML feature matrix")
    print("=" * 60)

    import extract_features

    snap_dirs = []
    for root, _, files in os.walk(scan_dir):
        if "meta.json" in files:
            snap_dirs.append(root)

    if not snap_dirs:
        print(f"[-] No snapshot directories found under {scan_dir}")
        sys.exit(1)

    print(f"[+] Found {len(snap_dirs)} snapshot(s)")

    rows = []
    for i, snap_dir_path in enumerate(sorted(snap_dirs), 1):
        print(f"  [{i}/{len(snap_dirs)}] {snap_dir_path}")
        row = extract_features.process_snapshot(snap_dir_path)
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

    print(f"\n[✓] Features saved: {out_csv} ({len(rows)} rows, {len(feat_cols)} features)")

    if args.skip_training:
        print("\n[~] Skipping model training (--skip-training)")
        print(f"\n{'=' * 60}\n Pipeline complete\n{'=' * 60}")
        return

    # ── Step 3: train stage model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 3: train_stage_model — family-agnostic stage classifier")
    print("=" * 60)

    import train_stage_model

    os.makedirs(model_out, exist_ok=True)
    df = train_stage_model.load_data(out_csv)

    # Train with all label types: time-based (4-class), binary time-based, and behavior-based
    for label_col, stage_names in [("stage_hint", train_stage_model.STAGE_NAMES_TIME),
                                   ("stage_binary", train_stage_model.STAGE_NAMES_EARLY_LATE),
                                   ("behavior_stage", train_stage_model.STAGE_NAMES_BEHAVIOR)]:
        if label_col not in df.columns:
            continue

        label_dir = os.path.join(model_out, label_col)
        os.makedirs(label_dir, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f" LABEL: {label_col}")
        print(f"{'#' * 60}")

        X, y, fc, lm = train_stage_model.prepare_xy(df, label_col=label_col)
        print(f"[+] Models: {', '.join(train_stage_model.get_models().keys())}\n")
        train_stage_model.run_standard_split(X, y, fc, label_dir, stage_names=stage_names, label_map=lm)

        if not args.no_loo and len(df["family"].unique()) > 1:
            train_stage_model.run_loo(df, fc, label_dir, label_col=label_col, stage_names=stage_names, label_map=lm)

    print(f"\n{'=' * 60}")
    print(f" Pipeline complete")
    print(f" Snapshots : {len(rows)}")
    print(f" Features  : {len(feat_cols)}")
    print(f" CSV       : {out_csv}")
    print(f" Model     : {model_out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
