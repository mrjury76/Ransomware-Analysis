"""
analyze_family_behavior.py
--------------------------
Per-family behavioral profiling across ransomware lifecycle stages.

For each family, compute mean feature values per stage_hint (0-3) and
identify which signals are:
  - Consistent across ALL ransomware families (generalizable)
  - Shared by some families (partial overlap)
  - Family-specific (fingerprints)

Outputs (all in analysis_output/):
  family_profiles/          per-family CSVs (mean per feature per stage)
  heatmaps/                 per-family feature heatmaps
  cross_family_heatmap.png  side-by-side comparison at peak stage (stage 2)
  signal_consistency.csv    per-feature consistency score across families
  similarity_matrix.png     cosine similarity between family stage-2 profiles
  report.txt                human-readable summary of cross-family findings

Usage:
    python analyze_family_behavior.py
    python analyze_family_behavior.py --csv output/features.csv --out analysis_output
    python analyze_family_behavior.py --stage-col stage_hint   # or behavior_stage
"""

import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")

# ── defaults ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "output", "features.csv")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "analysis_output")

STAGE_NAMES = {
    0: "Early",
    1: "Recon/Evasion",
    2: "Active Enc",
    3: "Post-Enc",
}

# Features to focus on in heatmaps (high-importance, interpretable signals)
FOCUS_FEATURES = [
    # ldrmodules
    "ldrmodules_not_in_load", "ldrmodules_not_in_init", "ldrmodules_not_in_mem",
    "ldrmodules_hidden_ratio", "ldrmodules_suspicious_path_ratio",
    "ldrmodules_avg_per_process", "ldrmodules_max_per_process",
    # VAD / memory
    "vad_rwx_ratio", "vad_exec_ratio", "vad_priv_exec_ratio",
    "vad_rwx_region_count", "vad_private_exec",
    # malfind
    "malfind_count", "malfind_mz_regions", "malfind_rwx_region_count",
    "malfind_private_exec", "malfind_shellcode_regions",
    # processes
    "pslist_count", "pslist_wow64_count", "pslist_ransom_procs",
    "pslist_exited_count", "pslist_parent_missing_count", "hidden_process_count",
    # handles
    "handle_mutex_count", "handle_mutex_ratio", "handle_event_count",
    "handle_section_count", "handle_token_count",
    # services
    "svcscan_security_stopped", "svcscan_stopped", "svcscan_running",
    # dlls
    "dlllist_unusual_path_ratio", "dlllist_crypto_ratio",
    "dlllist_unusual_path_hit_count",
    # cmdline
    "cmdline_suspicious_count", "cmdline_unusual_dir_ratio",
    # filescan
    "filescan_encrypted", "filescan_ransom_notes",
    # priv
    "priv_total_enabled",
    # network
    "netstat_outbound_ratio", "netstat_suspicious_port_ratio",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[+] Loaded {len(df)} rows, {df.shape[1]} columns from {csv_path}")
    return df


def get_feat_cols(df):
    meta = {"family", "stage_hint", "behavior_stage", "actual_offset_s",
            "target_offset_s", "rep", "run", "snap_name", "snap_dir",
            "stage_binary"}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {os.path.relpath(path, SCRIPT_DIR)}")


# ── per-family profile ────────────────────────────────────────────────────────

def build_family_profiles(df, stage_col, feat_cols, out_dir):
    """
    For each family, produce a DataFrame: index=stage, columns=features
    Values are means across all snapshots at that stage.
    """
    profiles = {}
    profile_dir = os.path.join(out_dir, "family_profiles")
    os.makedirs(profile_dir, exist_ok=True)

    families = sorted(df["family"].dropna().unique())
    stages   = sorted(df[stage_col].dropna().unique().astype(int))

    for fam in families:
        fdf = df[df["family"] == fam].copy()
        fdf[stage_col] = fdf[stage_col].astype(int)

        rows = {}
        for s in stages:
            sdf = fdf[fdf[stage_col] == s]
            if len(sdf) == 0:
                rows[s] = {f: np.nan for f in feat_cols}
            else:
                rows[s] = sdf[feat_cols].mean().to_dict()

        profile = pd.DataFrame(rows).T  # index=stage, cols=features
        profile.index.name = "stage"
        profile.index = [STAGE_NAMES.get(s, str(s)) for s in profile.index]

        profiles[fam] = profile
        csv_path = os.path.join(profile_dir, f"{fam}_profile.csv")
        profile.to_csv(csv_path)
        print(f"  [profile] {fam}: {len(fdf)} snapshots across stages {list(fdf[stage_col].unique())}")

    return profiles, families, stages


# ── per-family heatmap ────────────────────────────────────────────────────────

def plot_family_heatmap(profile, fam, out_dir, focus_feats):
    """Heatmap of feature means across stages for one family."""
    # Filter to features that actually exist in profile
    feats = [f for f in focus_feats if f in profile.columns]
    data  = profile[feats].copy()

    # Normalise each feature 0-1 across stages so colour encodes relative change
    norm = data.copy()
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        if hi > lo:
            norm[col] = (norm[col] - lo) / (hi - lo)
        else:
            norm[col] = 0.0

    fig, ax = plt.subplots(figsize=(max(10, len(feats) * 0.28), 4))
    im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(norm.index)))
    ax.set_yticklabels(norm.index, fontsize=9)
    ax.set_title(f"{fam} — feature intensity by stage (0=min, 1=max within feature)", fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    path = os.path.join(out_dir, "heatmaps", f"{fam}_heatmap.png")
    save_fig(fig, path)


# ── cross-family comparison heatmap ──────────────────────────────────────────

def plot_cross_family_heatmap(profiles, families, out_dir, focus_feats, stage_label="Active Enc"):
    """
    For each family, extract the row matching stage_label, then plot a
    family x feature heatmap (raw means, column-normalised).
    """
    feats = [f for f in focus_feats if all(f in profiles[fam].columns for fam in families)]

    rows = {}
    for fam in families:
        if stage_label in profiles[fam].index:
            rows[fam] = profiles[fam].loc[stage_label, feats]
        else:
            # Use highest available stage
            rows[fam] = profiles[fam][feats].iloc[-1]

    mat = pd.DataFrame(rows).T  # family x feature

    # Column-normalise
    norm = mat.copy().astype(float)
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        if hi > lo:
            norm[col] = (norm[col] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(max(12, len(feats) * 0.3), max(3, len(families) * 0.5 + 1)))
    im = ax.imshow(norm.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=9)
    ax.set_title(f"Cross-family feature comparison @ stage '{stage_label}' (column-normalised)", fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)

    path = os.path.join(out_dir, "cross_family_heatmap.png")
    save_fig(fig, path)

    return mat  # raw means


# ── signal consistency analysis ──────────────────────────────────────────────

def compute_signal_consistency(profiles, families, feat_cols, out_dir, peak_stage="Active Enc"):
    """
    For each feature, measure how consistently it rises from stage 0 -> peak across families.

    Consistency score = fraction of (ransomware) families where the feature value
    at peak_stage is higher than at early stage.

    Also computes mean_peak_value, cv (coefficient of variation) across families
    to flag noisy vs stable signals.
    """
    ransom_fams = [f for f in families if f != "Benign"]
    early_label = STAGE_NAMES.get(0, "Early")

    records = []
    for feat in feat_cols:
        rises = 0
        peak_vals = []
        early_vals = []
        for fam in ransom_fams:
            prof = profiles[fam]
            if feat not in prof.columns:
                continue
            peak_row  = prof.loc[peak_stage, feat] if peak_stage in prof.index else np.nan
            # Find the earliest stage with data
            early_row = None
            for stage in prof.index:
                val = prof.loc[stage, feat]
                if pd.notna(val):
                    early_row = val
                    break
            if pd.notna(peak_row) and early_row is not None:
                if peak_row > early_row:
                    rises += 1
                peak_vals.append(peak_row)
                early_vals.append(early_row)

        n = len(ransom_fams)
        consistency = rises / n if n > 0 else 0.0
        mean_peak   = np.mean(peak_vals) if peak_vals else 0.0
        cv_peak     = (np.std(peak_vals) / mean_peak) if mean_peak > 0 and len(peak_vals) > 1 else 0.0
        delta       = np.mean(np.array(peak_vals) - np.array(early_vals)) if peak_vals else 0.0

        records.append({
            "feature":      feat,
            "consistency":  round(consistency, 3),
            "mean_peak":    round(mean_peak, 3),
            "mean_delta":   round(delta, 3),
            "cv_peak":      round(cv_peak, 3),
            "n_families":   len(peak_vals),
        })

    sig_df = pd.DataFrame(records).sort_values("consistency", ascending=False)
    sig_df.to_csv(os.path.join(out_dir, "signal_consistency.csv"), index=False)
    print(f"[+] Signal consistency saved: {len(sig_df)} features analysed")
    return sig_df


# ── similarity matrix ─────────────────────────────────────────────────────────

def plot_similarity_matrix(profiles, families, feat_cols, out_dir, stage_label="Active Enc"):
    """Cosine similarity between family profiles at peak stage."""
    vecs = {}
    for fam in families:
        prof = profiles[fam]
        feats = [f for f in feat_cols if f in prof.columns]
        if stage_label in prof.index:
            v = prof.loc[stage_label, feats].fillna(0).values.astype(float)
        else:
            v = prof[feats].iloc[-1].fillna(0).values.astype(float)
        vecs[fam] = v

    n = len(families)
    sim = np.zeros((n, n))
    for i, fa in enumerate(families):
        for j, fb in enumerate(families):
            va, vb = vecs[fa], vecs[fb]
            if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
                sim[i, j] = 0.0
            else:
                sim[i, j] = 1 - cosine(va, vb)

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(families, fontsize=9)
    ax.set_title(f"Family profile cosine similarity @ '{stage_label}'", fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if sim[i,j] < 0.7 else "white")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = os.path.join(out_dir, "similarity_matrix.png")
    save_fig(fig, path)

    return pd.DataFrame(sim, index=families, columns=families)


# ── stage trajectory plot ─────────────────────────────────────────────────────

def plot_stage_trajectories(profiles, families, out_dir, signals):
    """
    For a curated list of signals, plot mean value across stages for all families
    on the same axes — reveals which signals track lifecycle consistently.
    """
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v"]

    for feat in signals:
        fig, ax = plt.subplots(figsize=(7, 4))
        for idx, fam in enumerate(families):
            prof = profiles[fam]
            if feat not in prof.columns:
                continue
            vals = prof[feat].values
            stages = list(prof.index)
            ax.plot(stages, vals, marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)], label=fam, linewidth=1.8,
                    markersize=6)

        ax.set_title(f"Feature trajectory: {feat}", fontsize=10)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Mean value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        path = os.path.join(traj_dir, f"traj_{feat}.png")
        save_fig(fig, path)


# ── text report ───────────────────────────────────────────────────────────────

def write_report(sig_df, sim_df, families, out_dir, stage_col, peak_stage="Active Enc"):
    ransom_fams = [f for f in families if f != "Benign"]
    n_ransom = len(ransom_fams)

    lines = []
    lines.append("=" * 70)
    lines.append(" CROSS-FAMILY BEHAVIORAL ANALYSIS REPORT")
    lines.append(f" Stage column : {stage_col}")
    lines.append(f" Families     : {', '.join(families)}")
    lines.append(f" Peak stage   : {peak_stage}")
    lines.append("=" * 70)

    # Universal signals (all ransomware families show increase)
    universal = sig_df[sig_df["consistency"] == 1.0].head(20)
    lines.append(f"\n[UNIVERSAL SIGNALS] Rise in ALL {n_ransom} ransomware families at '{peak_stage}':")
    lines.append(f"  {'Feature':<45} {'MeanPeak':>10} {'MeanDelta':>11} {'CV':>6}")
    lines.append("  " + "-" * 74)
    for _, r in universal.iterrows():
        lines.append(f"  {r['feature']:<45} {r['mean_peak']:>10.2f} {r['mean_delta']:>11.2f} {r['cv_peak']:>6.2f}")

    # Majority signals (>= 75% of families)
    threshold = max(0.75, (n_ransom - 1) / n_ransom)
    majority = sig_df[(sig_df["consistency"] >= threshold) & (sig_df["consistency"] < 1.0)].head(15)
    lines.append(f"\n[MAJORITY SIGNALS] Rise in >= {threshold:.0%} of ransomware families:")
    lines.append(f"  {'Feature':<45} {'Consistency':>12} {'MeanDelta':>11}")
    lines.append("  " + "-" * 70)
    for _, r in majority.iterrows():
        lines.append(f"  {r['feature']:<45} {r['consistency']:>12.2f} {r['mean_delta']:>11.2f}")

    # Low-consistency signals (potential fingerprints)
    fingerprints = sig_df[sig_df["consistency"] <= 0.25].sort_values("mean_peak", ascending=False).head(15)
    lines.append(f"\n[FAMILY FINGERPRINTS] Low consistency (<= 25%), high mean peak:")
    lines.append(f"  {'Feature':<45} {'Consistency':>12} {'MeanPeak':>10}")
    lines.append("  " + "-" * 68)
    for _, r in fingerprints.iterrows():
        lines.append(f"  {r['feature']:<45} {r['consistency']:>12.2f} {r['mean_peak']:>10.2f}")

    # Similarity summary
    if sim_df is not None and not sim_df.empty:
        lines.append(f"\n[SIMILARITY MATRIX] Cosine similarity at '{peak_stage}':")
        lines.append("  " + sim_df.round(3).to_string().replace("\n", "\n  "))

        # Find most/least similar pair
        pairs = []
        fam_list = list(sim_df.index)
        for i, fa in enumerate(fam_list):
            for j, fb in enumerate(fam_list):
                if j > i:
                    pairs.append((fa, fb, sim_df.loc[fa, fb]))
        if pairs:
            pairs.sort(key=lambda x: x[2], reverse=True)
            most_sim  = pairs[0]
            least_sim = pairs[-1]
            lines.append(f"\n  Most similar  : {most_sim[0]} <-> {most_sim[1]}  ({most_sim[2]:.3f})")
            lines.append(f"  Least similar : {least_sim[0]} <-> {least_sim[1]}  ({least_sim[2]:.3f})")

    lines.append("\n" + "=" * 70)

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Report written: {os.path.relpath(report_path, SCRIPT_DIR)}")
    print("\n" + "\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-family behavioral profiling and cross-family similarity")
    parser.add_argument("--csv",       default=DEFAULT_CSV, help="Path to features.csv")
    parser.add_argument("--out",       default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--stage-col", default="stage_hint",
                        choices=["stage_hint", "behavior_stage"],
                        help="Which stage column to use for grouping")
    parser.add_argument("--peak",      default=2, type=int,
                        help="Stage index to treat as 'peak' (default: 2 = Active Encryption)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.csv)

    if args.stage_col not in df.columns:
        print(f"[-] Column '{args.stage_col}' not found in CSV")
        return

    feat_cols = get_feat_cols(df)
    print(f"[+] Feature columns: {len(feat_cols)}")
    print(f"[+] Stage column   : {args.stage_col}")

    # ── build per-family profiles ──────────────────────────────────────────
    print("\n[1] Building per-family profiles...")
    profiles, families, stages = build_family_profiles(df, args.stage_col, feat_cols, args.out)

    peak_stage_label = STAGE_NAMES.get(args.peak, str(args.peak))
    print(f"[+] Peak stage: {args.peak} = '{peak_stage_label}'")

    # ── per-family heatmaps ────────────────────────────────────────────────
    print("\n[2] Plotting per-family heatmaps...")
    focus = [f for f in FOCUS_FEATURES if f in feat_cols]
    for fam in families:
        plot_family_heatmap(profiles[fam], fam, args.out, focus)

    # ── cross-family heatmap ───────────────────────────────────────────────
    print("\n[3] Plotting cross-family comparison heatmap...")
    cross_mat = plot_cross_family_heatmap(profiles, families, args.out, focus, peak_stage_label)
    cross_mat.to_csv(os.path.join(args.out, "cross_family_peak_values.csv"))

    # ── signal consistency ─────────────────────────────────────────────────
    print("\n[4] Computing signal consistency...")
    sig_df = compute_signal_consistency(profiles, families, feat_cols, args.out, peak_stage_label)

    # ── similarity matrix ──────────────────────────────────────────────────
    print("\n[5] Computing family similarity matrix...")
    sim_df = plot_similarity_matrix(profiles, families, feat_cols, args.out, peak_stage_label)
    sim_df.to_csv(os.path.join(args.out, "similarity_matrix.csv"))

    # ── stage trajectory plots for top universal signals ──────────────────
    print("\n[6] Plotting stage trajectory curves...")
    top_signals = sig_df[sig_df["consistency"] >= 0.75]["feature"].tolist()[:20]
    # Always include the key known signals
    for must_have in ["ldrmodules_not_in_load", "filescan_encrypted",
                      "malfind_count", "vad_rwx_ratio", "svcscan_security_stopped",
                      "handle_mutex_count", "priv_total_enabled"]:
        if must_have in feat_cols and must_have not in top_signals:
            top_signals.append(must_have)
    plot_stage_trajectories(profiles, families, args.out, top_signals)

    # ── write report ───────────────────────────────────────────────────────
    print("\n[7] Writing text report...")
    write_report(sig_df, sim_df, families, args.out, args.stage_col, peak_stage_label)

    print(f"\n{'=' * 60}")
    print(f" Analysis complete")
    print(f" Output: {args.out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
