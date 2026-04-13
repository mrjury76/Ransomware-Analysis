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
  benign_separability.csv   Cohen's d between Benign and ransomware at peak
  universal_feature_set.csv final ranked universal feature set for detectors
  report.txt                human-readable summary of cross-family findings

Usage:
    python analyze_family_behavior.py
    python analyze_family_behavior.py --csv output/features.csv --out analysis_output
    python analyze_family_behavior.py --stage-col stage_hint   # or behavior_stage
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Features known to always be zero / broken in the dataset — skip them
# (annotated in extract_features.py with "always 0")
DEAD_FEATURES = {
    "pslist_avg_handles",
    "pslist_avg_runtimes",
    "pslist_max_runtimes",
    "cmdline_sus_args_count",
    "cmdline_script_exec_count",
    "cmdline_encoded_count",
    "cmdline_ransom_indicators",
    "cmdline_encoded_ratio",
    "cmdline_script_exec_ratio",
    "ldrmodules_hidden_count",
    "ldrmodules_hidden_ratio",
    "netstat_suspicious_port_ratio",
    "netstat_suspicious_port_hit_count",
    "malfind_mz_regions",       # always 0 in practice
    "malfind_shellcode_regions", # always 0 in practice
}

# Features to focus on in heatmaps (high-importance, interpretable signals)
FOCUS_FEATURES = [
    # ldrmodules
    "ldrmodules_not_in_load", "ldrmodules_not_in_init", "ldrmodules_not_in_mem",
    "ldrmodules_suspicious_path_ratio",
    "ldrmodules_avg_per_process", "ldrmodules_max_per_process",
    # VAD / memory
    "vad_rwx_ratio", "vad_exec_ratio", "vad_priv_exec_ratio",
    "vad_rwx_region_count", "vad_private_exec",
    # malfind
    "malfind_count", "malfind_rwx_region_count",
    "malfind_private_exec", "malfind_avg_regions_per_process",
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
    "netstat_outbound_ratio",
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
    cols = [c for c in df.columns
            if c not in meta
            and c not in DEAD_FEATURES
            and pd.api.types.is_numeric_dtype(df[c])]
    return cols


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
    Excludes Benign from the comparison since it has no ransomware activity.
    """
    ransom_fams = [f for f in families if f != "Benign"]
    feats = [f for f in focus_feats if all(f in profiles[fam].columns for fam in ransom_fams)]

    rows = {}
    for fam in ransom_fams:
        if stage_label in profiles[fam].index:
            rows[fam] = profiles[fam].loc[stage_label, feats]
        else:
            rows[fam] = profiles[fam][feats].iloc[-1]

    mat = pd.DataFrame(rows).T  # family x feature

    # Column-normalise
    norm = mat.copy().astype(float)
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        if hi > lo:
            norm[col] = (norm[col] - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(max(12, len(feats) * 0.3), max(3, len(ransom_fams) * 0.5 + 1)))
    im = ax.imshow(norm.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(ransom_fams)))
    ax.set_yticklabels(ransom_fams, fontsize=9)
    ax.set_title(f"Cross-family feature comparison @ stage '{stage_label}' (column-normalised)", fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)

    path = os.path.join(out_dir, "cross_family_heatmap.png")
    save_fig(fig, path)

    return mat  # raw means


# ── signal consistency analysis ──────────────────────────────────────────────

def find_per_family_peak(profiles, feat_cols, ransom_fams):
    """
    For each family, find which stage label has the highest mean value
    across the top features (argmax of row-sum rather than fixed stage 2).
    Returns dict: fam -> peak_stage_label
    """
    peak_stages = {}
    for fam in ransom_fams:
        prof = profiles[fam]
        # drop NaN rows (e.g. Early has no data)
        valid = prof.dropna(how="all")
        if valid.empty:
            peak_stages[fam] = list(prof.index)[-1]
            continue
        feats = [f for f in feat_cols if f in valid.columns]
        row_sums = valid[feats].sum(axis=1)
        peak_stages[fam] = row_sums.idxmax()
    return peak_stages


def compute_signal_consistency(profiles, families, feat_cols, out_dir, peak_stage="Active Enc"):
    """
    For each feature, measure how consistently it rises from earliest stage -> peak.

    Improvements over v1:
      - Per-family peak: uses each family's actual argmax stage rather than fixed stage 2
      - Weighted consistency: multiplies rise fraction by mean relative delta,
        so features that barely tick up don't score the same as ones that explode
      - Cohen's d: effect size of the rise (mean_delta / pooled_std)
      - CV gating: separates stable vs noisy universal signals

    Consistency score = fraction of (ransomware) families where peak > early
    Weighted score    = consistency * mean_relative_delta (0 if no rise)
    """
    ransom_fams = [f for f in families if f != "Benign"]
    n = len(ransom_fams)

    # Per-family true peak stage
    per_fam_peak = find_per_family_peak(profiles, feat_cols, ransom_fams)
    print(f"  [peak stages] { {f: per_fam_peak[f] for f in ransom_fams} }")

    records = []
    for feat in feat_cols:
        rises = 0
        peak_vals  = []
        early_vals = []
        rel_deltas = []

        for fam in ransom_fams:
            prof = profiles[fam]
            if feat not in prof.columns:
                continue
            peak_label = per_fam_peak[fam]
            peak_val = prof.loc[peak_label, feat] if peak_label in prof.index else np.nan

            # Find earliest stage with non-NaN data
            early_val = None
            for stage_label in prof.index:
                v = prof.loc[stage_label, feat]
                if pd.notna(v):
                    early_val = v
                    break

            if pd.notna(peak_val) and early_val is not None:
                delta = peak_val - early_val
                if delta > 0:
                    rises += 1
                    rel_delta = delta / (early_val + 1e-9)  # avoid div/0
                    rel_deltas.append(rel_delta)
                peak_vals.append(peak_val)
                early_vals.append(early_val)

        consistency  = rises / n if n > 0 else 0.0
        mean_peak    = np.mean(peak_vals)  if peak_vals  else 0.0
        mean_delta   = np.mean(np.array(peak_vals) - np.array(early_vals)) if peak_vals else 0.0
        cv_peak      = (np.std(peak_vals) / mean_peak) if mean_peak > 0 and len(peak_vals) > 1 else 0.0
        mean_rel_delta = np.mean(rel_deltas) if rel_deltas else 0.0
        weighted_score = consistency * min(mean_rel_delta, 10.0)  # cap at 10x to avoid outlier dominance

        # Cohen's d for the rise (mean_delta / pooled std of peak & early)
        if len(peak_vals) > 1 and len(early_vals) > 1:
            pooled_std = np.sqrt((np.std(peak_vals)**2 + np.std(early_vals)**2) / 2 + 1e-9)
            cohens_d = mean_delta / pooled_std
        else:
            cohens_d = 0.0

        records.append({
            "feature":        feat,
            "consistency":    round(consistency, 3),
            "weighted_score": round(weighted_score, 3),
            "mean_peak":      round(mean_peak, 3),
            "mean_delta":     round(mean_delta, 3),
            "mean_rel_delta": round(mean_rel_delta, 3),
            "cohens_d":       round(cohens_d, 3),
            "cv_peak":        round(cv_peak, 3),
            "n_families":     len(peak_vals),
        })

    # Sort by weighted score descending
    sig_df = pd.DataFrame(records).sort_values("weighted_score", ascending=False)
    sig_df.to_csv(os.path.join(out_dir, "signal_consistency.csv"), index=False)
    print(f"[+] Signal consistency saved: {len(sig_df)} features analysed")
    return sig_df, per_fam_peak


# ── Benign separability ───────────────────────────────────────────────────────

def compute_benign_separability(profiles, families, feat_cols, out_dir, per_fam_peak):
    """
    For each feature, compute Cohen's d between Benign mean and ransomware mean
    at each family's peak stage.  High d = good separator from Benign = useful
    for avoiding the LOO Benign FPR problem (currently 94-98%).

    Also computes:
      - benign_mean: mean value in Benign profile (at its latest non-NaN stage)
      - ransom_mean: mean across ransomware families at their peak stages
      - direction: 'ransomware_higher' or 'benign_higher'
    """
    ransom_fams = [f for f in families if f != "Benign"]

    # Benign peak: last non-NaN stage
    benign_profile = profiles.get("Benign")
    if benign_profile is not None:
        valid_benign = benign_profile.dropna(how="all")
        benign_stage = valid_benign.index[-1] if not valid_benign.empty else None
    else:
        benign_stage = None

    records = []
    for feat in feat_cols:
        ransom_peak_vals = []
        for fam in ransom_fams:
            prof = profiles[fam]
            if feat not in prof.columns:
                continue
            pk = per_fam_peak.get(fam)
            if pk and pk in prof.index:
                v = prof.loc[pk, feat]
                if pd.notna(v):
                    ransom_peak_vals.append(v)

        if not ransom_peak_vals:
            continue

        ransom_mean = np.mean(ransom_peak_vals)
        ransom_std  = np.std(ransom_peak_vals) if len(ransom_peak_vals) > 1 else 0.0

        # Benign value at its reference stage
        benign_mean = 0.0
        if benign_profile is not None and benign_stage and feat in benign_profile.columns:
            bv = benign_profile.loc[benign_stage, feat]
            benign_mean = bv if pd.notna(bv) else 0.0

        # Cohen's d (ransomware vs benign)
        pooled_std = np.sqrt((ransom_std**2 + 1e-9))  # benign std unavailable (single mean)
        cohens_d_sep = abs(ransom_mean - benign_mean) / (pooled_std + 1e-9)

        direction = "ransomware_higher" if ransom_mean >= benign_mean else "benign_higher"
        records.append({
            "feature":       feat,
            "ransom_mean":   round(ransom_mean, 3),
            "benign_mean":   round(benign_mean, 3),
            "abs_diff":      round(abs(ransom_mean - benign_mean), 3),
            "cohens_d_sep":  round(cohens_d_sep, 3),
            "direction":     direction,
        })

    sep_df = pd.DataFrame(records).sort_values("cohens_d_sep", ascending=False)
    sep_df.to_csv(os.path.join(out_dir, "benign_separability.csv"), index=False)
    print(f"[+] Benign separability saved: {len(sep_df)} features")
    return sep_df


# ── universal feature set ─────────────────────────────────────────────────────

def build_universal_feature_set(sig_df, sep_df, out_dir):
    """
    Produces a ranked universal feature set by combining:
      1. Cross-family consistency (weighted_score)   — cross-family generalization
      2. Benign separability (cohens_d_sep)          — avoids FP on benign
      3. Signal stability (low CV)                   — reliable in new families

    Final score = weighted_score * cohens_d_sep_norm * stability
    where stability = 1 / (1 + cv_peak)

    Splits the final table into three tiers:
      - Tier 1: consistency=1.0 AND cv < 0.5 AND good benign sep  (gold)
      - Tier 2: consistency>=0.75 OR good sep                      (silver)
      - Tier 3: the rest with consistency>0                         (bronze)
    """
    merged = sig_df.merge(sep_df[["feature", "cohens_d_sep", "benign_mean",
                                   "ransom_mean", "direction"]],
                          on="feature", how="left").fillna(0)

    # Normalise cohens_d_sep to 0-1 range
    max_d = merged["cohens_d_sep"].max()
    merged["cohens_d_sep_norm"] = merged["cohens_d_sep"] / (max_d + 1e-9)

    merged["stability"] = 1.0 / (1.0 + merged["cv_peak"])
    merged["universal_score"] = (
        merged["weighted_score"] *
        merged["cohens_d_sep_norm"] *
        merged["stability"]
    )
    merged = merged.sort_values("universal_score", ascending=False)

    # Tier assignment
    def assign_tier(row):
        if row["consistency"] == 1.0 and row["cv_peak"] < 0.5 and row["cohens_d_sep"] > 1.0:
            return 1
        elif row["consistency"] >= 0.75 or row["cohens_d_sep"] > 2.0:
            return 2
        elif row["consistency"] > 0:
            return 3
        else:
            return 4  # no rise anywhere — low value

    merged["tier"] = merged.apply(assign_tier, axis=1)
    merged = merged[merged["tier"] <= 3]  # drop features that never rise

    path = os.path.join(out_dir, "universal_feature_set.csv")
    merged.to_csv(path, index=False)
    print(f"[+] Universal feature set saved: {len(merged)} features "
          f"({len(merged[merged.tier==1])} T1, "
          f"{len(merged[merged.tier==2])} T2, "
          f"{len(merged[merged.tier==3])} T3)")
    return merged


# ── similarity matrix ─────────────────────────────────────────────────────────

def plot_similarity_matrix(profiles, families, feat_cols, out_dir, per_fam_peak):
    """
    Cosine similarity between family profiles at each family's own peak stage.
    Benign is excluded — its zero-norm vector produces undefined cosine similarity
    and makes the matrix misleading.
    """
    ransom_fams = [f for f in families if f != "Benign"]
    vecs = {}
    for fam in ransom_fams:
        prof = profiles[fam]
        feats = [f for f in feat_cols if f in prof.columns]
        pk = per_fam_peak.get(fam)
        if pk and pk in prof.index:
            v = prof.loc[pk, feats].fillna(0).values.astype(float)
        else:
            v = prof[feats].iloc[-1].fillna(0).values.astype(float)
        vecs[fam] = v

    n = len(ransom_fams)
    sim = np.zeros((n, n))
    for i, fa in enumerate(ransom_fams):
        for j, fb in enumerate(ransom_fams):
            va, vb = vecs[fa], vecs[fb]
            if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
                sim[i, j] = np.nan
            else:
                sim[i, j] = 1 - cosine(va, vb)

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    # Use masked array so NaN cells render as grey
    masked = np.ma.masked_invalid(sim)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(ransom_fams, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ransom_fams, fontsize=9)
    ax.set_title("Family profile cosine similarity (per-family peak stage, Benign excluded)", fontsize=10)

    for i in range(n):
        for j in range(n):
            val = sim[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if val < 0.7 else "white")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = os.path.join(out_dir, "similarity_matrix.png")
    save_fig(fig, path)

    sim_df = pd.DataFrame(sim, index=ransom_fams, columns=ransom_fams)
    return sim_df


# ── stage trajectory plot ─────────────────────────────────────────────────────

def plot_stage_trajectories(profiles, families, out_dir, signals):
    """
    For a curated list of signals, plot mean value across stages for all families
    on the same axes — reveals which signals track lifecycle consistently.
    Benign is included for comparison reference.
    """
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P"]

    for feat in signals:
        fig, ax = plt.subplots(figsize=(7, 4))
        for idx, fam in enumerate(families):
            prof = profiles[fam]
            if feat not in prof.columns:
                continue
            valid = prof[feat].dropna()
            if valid.empty:
                continue
            stages = list(valid.index)
            vals   = list(valid.values)
            ls = "--" if fam == "Benign" else "-"
            ax.plot(stages, vals, marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)], label=fam, linewidth=1.8,
                    markersize=6, linestyle=ls)

        ax.set_title(f"Feature trajectory: {feat}", fontsize=10)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Mean value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        path = os.path.join(traj_dir, f"traj_{feat}.png")
        save_fig(fig, path)


# ── behavior_stage alignment check ───────────────────────────────────────────

def check_behavior_stage_alignment(df, out_dir):
    """
    Compare behavior_stage vs stage_hint per snapshot.
    Reports: exact match rate, confusion matrix, and common disagreements.

    Helps validate whether the heuristic behavior_stage in extract_features.py
    correctly reflects the ground-truth stage_hint label.
    """
    if "behavior_stage" not in df.columns or "stage_hint" not in df.columns:
        print("  [skip] behavior_stage or stage_hint column missing")
        return

    sub = df[["family", "stage_hint", "behavior_stage"]].dropna()
    sub = sub.astype({"stage_hint": int, "behavior_stage": int})

    # Overall match
    match = (sub["stage_hint"] == sub["behavior_stage"]).mean()
    print(f"\n  [behavior_stage alignment]")
    print(f"  Overall exact match: {match:.1%}  ({int(match*len(sub))}/{len(sub)} snapshots)")

    # Per-family match
    for fam, grp in sub.groupby("family"):
        fmatch = (grp["stage_hint"] == grp["behavior_stage"]).mean()
        print(f"    {fam:<12}: {fmatch:.1%}  (n={len(grp)})")

    # Confusion matrix
    from collections import Counter
    confusion = Counter(zip(sub["stage_hint"], sub["behavior_stage"]))
    stages = sorted(set(sub["stage_hint"]) | set(sub["behavior_stage"]))
    cm = pd.DataFrame(0, index=stages, columns=stages)
    for (true, pred), cnt in confusion.items():
        cm.loc[true, pred] = cnt
    cm.index.name   = "stage_hint \\ behavior_stage"
    cm.columns.name = ""

    cm_path = os.path.join(out_dir, "behavior_stage_alignment.csv")
    cm.to_csv(cm_path)
    print(f"\n  Confusion matrix (rows=stage_hint, cols=behavior_stage):")
    print("  " + cm.to_string().replace("\n", "\n  "))
    print(f"  [saved] {os.path.relpath(cm_path, SCRIPT_DIR)}")

    # Biggest disagreements
    disagreements = sub[sub["stage_hint"] != sub["behavior_stage"]].copy()
    if not disagreements.empty:
        top = (disagreements.groupby(["family", "stage_hint", "behavior_stage"])
               .size().reset_index(name="count")
               .sort_values("count", ascending=False).head(10))
        print(f"\n  Top disagreement patterns:")
        print("  " + top.to_string(index=False).replace("\n", "\n  "))


# ── text report ───────────────────────────────────────────────────────────────

def write_report(sig_df, sim_df, sep_df, univ_df, families, out_dir,
                 stage_col, per_fam_peak):
    ransom_fams = [f for f in families if f != "Benign"]
    n_ransom = len(ransom_fams)

    lines = []
    lines.append("=" * 70)
    lines.append(" CROSS-FAMILY BEHAVIORAL ANALYSIS REPORT  (v2)")
    lines.append(f" Stage column : {stage_col}")
    lines.append(f" Families     : {', '.join(families)}")
    lines.append(f" Per-family peak stages:")
    for fam, pk in per_fam_peak.items():
        lines.append(f"   {fam:<12}: {pk}")
    lines.append("=" * 70)

    # ── Universal signals (consistency=1.0, low CV) ────────────────────────
    stable_univ = sig_df[(sig_df["consistency"] == 1.0) & (sig_df["cv_peak"] < 0.5)]
    noisy_univ  = sig_df[(sig_df["consistency"] == 1.0) & (sig_df["cv_peak"] >= 0.5)]

    lines.append(f"\n[UNIVERSAL STABLE SIGNALS]  consistency=1.0, CV<0.5  ({len(stable_univ)} features)")
    lines.append(f"  {'Feature':<45} {'WtdScore':>9} {'MeanPeak':>10} {'MeanDelta':>11} {'CohenD':>7} {'CV':>6}")
    lines.append("  " + "-" * 90)
    for _, r in stable_univ.sort_values("weighted_score", ascending=False).iterrows():
        lines.append(f"  {r['feature']:<45} {r['weighted_score']:>9.3f} {r['mean_peak']:>10.2f}"
                     f" {r['mean_delta']:>11.2f} {r['cohens_d']:>7.2f} {r['cv_peak']:>6.2f}")

    lines.append(f"\n[UNIVERSAL NOISY SIGNALS]  consistency=1.0, CV>=0.5  ({len(noisy_univ)} features)")
    lines.append(f"  (Rise in all families but magnitude varies — less reliable as general detectors)")
    lines.append(f"  {'Feature':<45} {'WtdScore':>9} {'MeanPeak':>10} {'MeanDelta':>11} {'CohenD':>7} {'CV':>6}")
    lines.append("  " + "-" * 90)
    for _, r in noisy_univ.sort_values("weighted_score", ascending=False).iterrows():
        lines.append(f"  {r['feature']:<45} {r['weighted_score']:>9.3f} {r['mean_peak']:>10.2f}"
                     f" {r['mean_delta']:>11.2f} {r['cohens_d']:>7.2f} {r['cv_peak']:>6.2f}")

    # ── Majority signals ───────────────────────────────────────────────────
    threshold = max(0.75, (n_ransom - 1) / n_ransom)
    majority = sig_df[(sig_df["consistency"] >= threshold) & (sig_df["consistency"] < 1.0)].head(15)
    lines.append(f"\n[MAJORITY SIGNALS] consistency >= {threshold:.0%}, not universal  ({len(majority)} shown)")
    lines.append(f"  {'Feature':<45} {'Consistency':>12} {'WtdScore':>9} {'MeanDelta':>11}")
    lines.append("  " + "-" * 79)
    for _, r in majority.iterrows():
        lines.append(f"  {r['feature']:<45} {r['consistency']:>12.2f} {r['weighted_score']:>9.3f}"
                     f" {r['mean_delta']:>11.2f}")

    # ── Family fingerprints ────────────────────────────────────────────────
    fingerprints = sig_df[sig_df["consistency"] <= 0.25].sort_values("mean_peak", ascending=False).head(15)
    lines.append(f"\n[FAMILY FINGERPRINTS] consistency<=25%, high mean peak  ({len(fingerprints)} shown)")
    lines.append(f"  {'Feature':<45} {'Consistency':>12} {'MeanPeak':>10}")
    lines.append("  " + "-" * 68)
    for _, r in fingerprints.iterrows():
        lines.append(f"  {r['feature']:<45} {r['consistency']:>12.2f} {r['mean_peak']:>10.2f}")

    # ── Benign separability ────────────────────────────────────────────────
    top_sep = sep_df.head(20)
    lines.append(f"\n[BENIGN SEPARABILITY] Top features by Cohen's d (ransomware vs Benign):")
    lines.append(f"  (High d = good at separating ransomware from benign → reduces Benign FPR)")
    lines.append(f"  {'Feature':<45} {'CohenD_sep':>11} {'RansomMean':>11} {'BenignMean':>11} {'Direction':>20}")
    lines.append("  " + "-" * 100)
    for _, r in top_sep.iterrows():
        lines.append(f"  {r['feature']:<45} {r['cohens_d_sep']:>11.2f} {r['ransom_mean']:>11.2f}"
                     f" {r['benign_mean']:>11.2f} {r['direction']:>20}")

    # ── Universal feature set summary ─────────────────────────────────────
    t1 = univ_df[univ_df["tier"] == 1]
    t2 = univ_df[univ_df["tier"] == 2]
    lines.append(f"\n[UNIVERSAL FEATURE SET SUMMARY]")
    lines.append(f"  Tier 1 (gold)   — {len(t1)} features: consistent, stable, benign-separable")
    lines.append(f"  Tier 2 (silver) — {len(t2)} features: majority or high benign sep")
    lines.append(f"\n  Tier 1 features (recommended for any new-family detector):")
    for _, r in t1.sort_values("universal_score", ascending=False).iterrows():
        lines.append(f"    {r['feature']:<45}  score={r['universal_score']:.3f}  d_sep={r['cohens_d_sep']:.2f}")

    # ── Similarity matrix ──────────────────────────────────────────────────
    if sim_df is not None and not sim_df.empty:
        lines.append(f"\n[SIMILARITY MATRIX] Cosine similarity (per-family peaks, Benign excluded):")
        lines.append("  " + sim_df.round(3).to_string().replace("\n", "\n  "))

        pairs = []
        fam_list = list(sim_df.index)
        for i, fa in enumerate(fam_list):
            for j, fb in enumerate(fam_list):
                if j > i:
                    val = sim_df.loc[fa, fb]
                    if not np.isnan(val):
                        pairs.append((fa, fb, val))
        if pairs:
            pairs.sort(key=lambda x: x[2], reverse=True)
            lines.append(f"\n  Most similar  : {pairs[0][0]} <-> {pairs[0][1]}  ({pairs[0][2]:.3f})")
            lines.append(f"  Least similar : {pairs[-1][0]} <-> {pairs[-1][1]}  ({pairs[-1][2]:.3f})")

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
                        help="Fallback peak stage index if per-family argmax fails (default: 2)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.csv)

    if args.stage_col not in df.columns:
        print(f"[-] Column '{args.stage_col}' not found in CSV")
        return

    feat_cols = get_feat_cols(df)
    print(f"[+] Feature columns: {len(feat_cols)}  (dead features excluded)")
    print(f"[+] Stage column   : {args.stage_col}")

    # ── behavior_stage alignment check ─────────────────────────────────────
    print("\n[0] Checking behavior_stage vs stage_hint alignment...")
    check_behavior_stage_alignment(df, args.out)

    # ── build per-family profiles ──────────────────────────────────────────
    print("\n[1] Building per-family profiles...")
    profiles, families, _ = build_family_profiles(df, args.stage_col, feat_cols, args.out)

    # ── per-family heatmaps ────────────────────────────────────────────────
    print("\n[2] Plotting per-family heatmaps...")
    focus = [f for f in FOCUS_FEATURES if f in feat_cols]
    for fam in families:
        plot_family_heatmap(profiles[fam], fam, args.out, focus)

    # ── signal consistency (with per-family peak) ──────────────────────────
    print("\n[3] Computing signal consistency (per-family peak stages)...")
    sig_df, per_fam_peak = compute_signal_consistency(profiles, families, feat_cols, args.out)

    # ── cross-family heatmap ───────────────────────────────────────────────
    print("\n[4] Plotting cross-family comparison heatmap...")
    peak_stage_label = STAGE_NAMES.get(args.peak, str(args.peak))
    cross_mat = plot_cross_family_heatmap(profiles, families, args.out, focus, peak_stage_label)
    cross_mat.to_csv(os.path.join(args.out, "cross_family_peak_values.csv"))

    # ── Benign separability ────────────────────────────────────────────────
    print("\n[5] Computing Benign separability (to address LOO FPR)...")
    sep_df = compute_benign_separability(profiles, families, feat_cols, args.out, per_fam_peak)

    # ── universal feature set ──────────────────────────────────────────────
    print("\n[6] Building universal feature set...")
    univ_df = build_universal_feature_set(sig_df, sep_df, args.out)

    # ── similarity matrix ──────────────────────────────────────────────────
    print("\n[7] Computing family similarity matrix...")
    sim_df = plot_similarity_matrix(profiles, families, feat_cols, args.out, per_fam_peak)
    sim_df.to_csv(os.path.join(args.out, "similarity_matrix.csv"))

    # ── stage trajectory plots for top universal signals ──────────────────
    print("\n[8] Plotting stage trajectory curves...")
    top_signals = univ_df[univ_df["tier"] <= 2]["feature"].tolist()[:25]
    # Always include the key known signals
    for must_have in ["ldrmodules_not_in_load", "filescan_encrypted",
                      "malfind_count", "vad_rwx_ratio", "svcscan_security_stopped",
                      "handle_mutex_count", "priv_total_enabled"]:
        if must_have in feat_cols and must_have not in top_signals:
            top_signals.append(must_have)
    plot_stage_trajectories(profiles, families, args.out, top_signals)

    # ── write report ───────────────────────────────────────────────────────
    print("\n[9] Writing text report...")
    write_report(sig_df, sim_df, sep_df, univ_df, families, args.out,
                 args.stage_col, per_fam_peak)

    print(f"\n{'=' * 60}")
    print(f" Analysis complete")
    print(f" Output: {args.out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
