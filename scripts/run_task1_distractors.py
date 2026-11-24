"""
Task 1: Distractors-Only Re-evaluation with Alpha-centric metrics.
No new API calls; re-evaluates existing Phase 1 LLM predictions after removing
the 90 correct answers (keeping 270 distractors only).

Primary metric: Krippendorff's Alpha (ordinal)
Secondary: Cohen's Kappa, bias, distance, self-consistency
"""

import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
HUMAN_CSV = DATA / "processed" / "katerina_annotations.csv"
RESULTS = DATA / "results"
OUT_DIR = RESULTS / "phase1_distractors_only"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "C1_zero_shot": RESULTS / "zero_shot" / "zero_shot_results.csv",
    "C2_rubric_guided": RESULTS / "rubric_guided" / "rubric_guided_results.csv",
    "C3_few_shot": RESULTS / "few_shot" / "few_shot_results.csv",
}

DIMS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility"]
SUBJECTS = ["sociology", "history", "anthropology", "lifespan"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def alpha_ordinal(rater1, rater2):
    """Krippendorff's Alpha (ordinal) for two raters."""
    r1, r2 = np.array(rater1, dtype=float), np.array(rater2, dtype=float)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    if len(r1) < 2:
        return np.nan
    data = np.array([r1, r2])
    try:
        return krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    except ValueError:
        return np.nan


def alpha_multi_rater(ratings_list):
    """Krippendorff's Alpha for N raters. ratings_list = list of arrays."""
    data = np.array(ratings_list, dtype=float)
    if data.shape[1] < 2:
        return np.nan
    try:
        return krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    except ValueError:
        return np.nan


def kappa_weighted(h, l):
    """Cohen's Kappa (quadratic weighted)."""
    h, l = np.array(h, dtype=int), np.array(l, dtype=int)
    if len(h) < 2:
        return np.nan
    return cohen_kappa_score(h, l, weights="quadratic", labels=[1, 2, 3, 4])


def parse_all_runs(series):
    """Parse the *_all_runs column (string repr of list) into list of ints."""
    def _parse(val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []
    return series.apply(_parse)


# ── 1. Load & Filter ────────────────────────────────────────────────────────

print("=" * 70)
print("TASK 1: Distractors-Only Re-evaluation")
print("=" * 70)

df_human = pd.read_csv(HUMAN_CSV)
df_human["is_correct"] = df_human["option_choice"] == df_human["answer"]
n_correct = df_human["is_correct"].sum()
n_distractors = (~df_human["is_correct"]).sum()
print(f"Total items: {len(df_human)} | Correct: {n_correct} | Distractors: {n_distractors}")
assert n_distractors == 270, f"Expected 270 distractors, got {n_distractors}"

# Build merge key
df_human["merge_key"] = df_human["question_uuid"] + "_" + df_human["item_index"].astype(str) if "item_index" in df_human.columns else df_human["question_uuid"] + "_" + df_human.index.astype(str)


def load_and_merge(cond_path, df_h):
    """Load LLM results and merge with human data on shared keys."""
    df_l = pd.read_csv(cond_path)
    df_l["merge_key"] = df_l["question_uuid"] + "_" + df_l["item_index"].astype(str)
    merged = df_h.merge(df_l, on="merge_key", suffixes=("_human", "_llm"))
    return merged


# ── 2. Compute metrics per condition × dimension ────────────────────────────

all_results = {}
comparison_rows = []
# Cache merged dataframes for later use
merged_cache = {}

for cond_name, cond_path in CONDITIONS.items():
    print(f"\n--- {cond_name} ---")
    merged = load_and_merge(cond_path, df_human)
    merged_all = merged.copy()  # all items that matched
    merged_dist = merged[~merged["is_correct"]].copy()  # distractors only
    merged_cache[cond_name] = {"all": merged_all, "dist": merged_dist}
    print(f"  Matched items: {len(merged_all)} | Distractors: {len(merged_dist)}")

    cond_results = {"dimensions": {}, "aggregate": {}}
    kappas, alphas = [], []

    for dim in DIMS:
        h_col = f"{dim}_human"
        l_col = f"{dim}_llm"

        h_all = merged_all[h_col].values.astype(float)
        l_all = merged_all[l_col].values.astype(float)

        h_dist = merged_dist[h_col].values.astype(float)
        l_dist = merged_dist[l_col].values.astype(float)

        # Alpha (all matched items)
        alpha_all = alpha_ordinal(h_all, l_all)
        # Alpha (distractors only)
        alpha_dist = alpha_ordinal(h_dist, l_dist)
        # Kappa (distractors)
        kappa_dist = kappa_weighted(h_dist[~np.isnan(h_dist) & ~np.isnan(l_dist)],
                                    l_dist[~np.isnan(h_dist) & ~np.isnan(l_dist)])
        # Bias
        signed_diff = l_dist - h_dist
        mean_bias = float(np.nanmean(signed_diff))
        mean_abs_dist = float(np.nanmean(np.abs(signed_diff)))
        # Exact match & within-1
        valid = ~(np.isnan(h_dist) | np.isnan(l_dist))
        exact_pct = float(np.mean(h_dist[valid] == l_dist[valid]) * 100) if valid.sum() > 0 else np.nan
        within1_pct = float(np.mean(np.abs(h_dist[valid] - l_dist[valid]) <= 1) * 100) if valid.sum() > 0 else np.nan

        # Self-consistency from all_runs column
        runs_col = f"{dim}_all_runs"
        if runs_col in merged_dist.columns:
            runs = parse_all_runs(merged_dist[runs_col])
            consistencies = runs.apply(
                lambda r: len(set(r)) == 1 if len(r) > 0 else np.nan
            )
            self_consistency = float(consistencies.mean())
        else:
            self_consistency = np.nan

        dim_metrics = {
            "alpha_all": round(alpha_all, 4) if not np.isnan(alpha_all) else None,
            "alpha_distractors_270": round(alpha_dist, 4) if not np.isnan(alpha_dist) else None,
            "alpha_delta": round(alpha_dist - alpha_all, 4) if not (np.isnan(alpha_all) or np.isnan(alpha_dist)) else None,
            "kappa_distractors": round(kappa_dist, 4) if not np.isnan(kappa_dist) else None,
            "mean_bias": round(mean_bias, 4),
            "mean_abs_distance": round(mean_abs_dist, 4),
            "exact_match_pct": round(exact_pct, 2) if not np.isnan(exact_pct) else None,
            "within_one_pct": round(within1_pct, 2) if not np.isnan(within1_pct) else None,
            "self_consistency": round(self_consistency, 4) if not np.isnan(self_consistency) else None,
            "n_distractors": int(valid.sum()),
            "n_all": int(len(merged_all)),
        }
        cond_results["dimensions"][dim] = dim_metrics
        sc_str = f"{self_consistency:.3f}" if not np.isnan(self_consistency) else "N/A"
        print(f"  {dim:25s} | Alpha(dist)={dim_metrics['alpha_distractors_270']} | Kappa={dim_metrics['kappa_distractors']} | Bias={mean_bias:+.3f} | SC={sc_str}")

        if dim_metrics["alpha_distractors_270"] is not None:
            alphas.append(dim_metrics["alpha_distractors_270"])
        if dim_metrics["kappa_distractors"] is not None:
            kappas.append(dim_metrics["kappa_distractors"])

        comparison_rows.append({
            "condition": cond_name,
            "dimension": dim,
            "alpha_all": dim_metrics["alpha_all"],
            "alpha_distractors_270": dim_metrics["alpha_distractors_270"],
            "alpha_delta": dim_metrics["alpha_delta"],
            "kappa_distractors": dim_metrics["kappa_distractors"],
            "mean_bias": dim_metrics["mean_bias"],
            "mean_abs_distance": dim_metrics["mean_abs_distance"],
            "self_consistency": dim_metrics["self_consistency"],
            "n": dim_metrics["n_distractors"],
        })

    cond_results["aggregate"] = {
        "mean_alpha": round(np.mean(alphas), 4) if alphas else None,
        "mean_kappa": round(np.mean(kappas), 4) if kappas else None,
    }
    all_results[cond_name] = cond_results


# ── 3. Descriptive Alpha Breakdown ──────────────────────────────────────────

print("\n\n--- Descriptive Alpha Breakdown ---")
alpha_breakdown = {"by_dimension": {}, "by_subject": {}, "by_score_level": {}}

# 3a. By subject domain
for cond_name in CONDITIONS:
    m_dist = merged_cache[cond_name]["dist"]
    # textID is in the human side; check both possible column names after merge
    tid_col = "textID_human" if "textID_human" in m_dist.columns else "textID"

    for subj in SUBJECTS:
        subj_mask = m_dist[tid_col].str.contains(subj, case=False, na=False)
        if subj_mask.sum() == 0:
            continue
        for dim in DIMS:
            h_sub = m_dist.loc[subj_mask, f"{dim}_human"].values.astype(float)
            l_sub = m_dist.loc[subj_mask, f"{dim}_llm"].values.astype(float)
            a = alpha_ordinal(h_sub, l_sub)
            key = f"{cond_name}|{subj}|{dim}"
            alpha_breakdown["by_subject"][key] = {
                "alpha": round(a, 4) if not np.isnan(a) else None,
                "n": int(subj_mask.sum()),
            }

    # 3b. By human score level
    for dim in DIMS:
        for score_level in [1, 2, 3, 4]:
            score_mask = m_dist[f"{dim}_human"] == score_level
            if score_mask.sum() < 3:
                continue
            h_sub = m_dist.loc[score_mask, f"{dim}_human"].values.astype(float)
            l_sub = m_dist.loc[score_mask, f"{dim}_llm"].values.astype(float)
            mean_abs = float(np.mean(np.abs(l_sub - h_sub)))
            mean_bias_val = float(np.mean(l_sub - h_sub))
            key = f"{cond_name}|score_{score_level}|{dim}"
            alpha_breakdown["by_score_level"][key] = {
                "human_score": score_level,
                "mean_llm_score": round(float(np.mean(l_sub)), 2),
                "mean_bias": round(mean_bias_val, 3),
                "mean_abs_distance": round(mean_abs, 3),
                "n": int(score_mask.sum()),
            }
            print(f"  {cond_name} | {dim} | H={score_level} | LLM_mean={np.mean(l_sub):.2f} | bias={mean_bias_val:+.3f} | n={score_mask.sum()}")


# ── 4. Within-condition run Alpha ────────────────────────────────────────────

print("\n\n--- Within-condition Run Alpha (3 runs as 3 raters) ---")
run_alpha_results = {}

for cond_name in CONDITIONS:
    m_dist = merged_cache[cond_name]["dist"]

    for dim in DIMS:
        runs_col = f"{dim}_all_runs"
        if runs_col not in m_dist.columns:
            continue
        runs = parse_all_runs(m_dist[runs_col])
        run1 = runs.apply(lambda r: r[0] if len(r) >= 1 else np.nan).values.astype(float)
        run2 = runs.apply(lambda r: r[1] if len(r) >= 2 else np.nan).values.astype(float)
        run3 = runs.apply(lambda r: r[2] if len(r) >= 3 else np.nan).values.astype(float)
        a = alpha_multi_rater([run1, run2, run3])
        key = f"{cond_name}|{dim}"
        run_alpha_results[key] = round(a, 4) if not np.isnan(a) else None
        print(f"  {cond_name} | {dim} | Run Alpha = {run_alpha_results[key]}")


# ── 5. Save results ─────────────────────────────────────────────────────────

# Main evaluation JSON
with open(OUT_DIR / "distractors_only_evaluation.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Alpha breakdown
alpha_breakdown["within_condition_run_alpha"] = run_alpha_results
with open(OUT_DIR / "distractors_only_alpha_breakdown.json", "w") as f:
    json.dump(alpha_breakdown, f, indent=2, default=str)

# Comparison table
df_comp = pd.DataFrame(comparison_rows)
comp_table = df_comp.to_string(index=False)
with open(OUT_DIR / "distractors_only_comparison_table.txt", "w") as f:
    f.write(comp_table)
print(f"\n\nComparison table:\n{comp_table}")


# ── 6. Figures ───────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.1)
COND_LABELS = {"C1_zero_shot": "C1: Zero-shot", "C2_rubric_guided": "C2: Rubric-guided", "C3_few_shot": "C3: Few-shot"}

# Fig 1: Alpha by dimension (distractors only)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(DIMS))
width = 0.22
for i, (cond, label) in enumerate(COND_LABELS.items()):
    vals = [all_results[cond]["dimensions"][d]["alpha_distractors_270"] or 0 for d in DIMS]
    ax.bar(x + i * width, vals, width, label=label, zorder=3)
ax.set_xticks(x + width)
ax.set_xticklabels([d.replace("_", "\n") for d in DIMS])
ax.set_ylabel("Krippendorff's Alpha (ordinal)")
ax.set_title("Alpha by Dimension: Distractors Only (n=270)")
ax.legend()
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_ylim(-0.4, 0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_alpha_by_dimension_distractors.png", dpi=150)
plt.close()
print("Saved fig1_alpha_by_dimension_distractors.png")

# Fig 2: Bias histogram per dimension
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, dim in enumerate(DIMS):
    ax = axes[idx // 2][idx % 2]
    for cond_name in CONDITIONS:
        m = merged_cache[cond_name]["dist"]
        h = m[f"{dim}_human"].values.astype(float)
        l = m[f"{dim}_llm"].values.astype(float)
        diff = l - h
        ax.hist(diff, bins=np.arange(-3.5, 4.5, 1), alpha=0.4, label=COND_LABELS[cond_name], edgecolor="black")
    ax.set_title(dim.replace("_", " "))
    ax.set_xlabel("LLM Score − Human Score")
    ax.set_ylabel("Count")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax.legend(fontsize=8)
fig.suptitle("Score Difference Distribution: Distractors Only", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_bias_histogram_distractors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_bias_histogram_distractors.png")

# Fig 3: Mean LLM score by human score level (reveals agreeableness bias)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, dim in enumerate(DIMS):
    ax = axes[idx // 2][idx % 2]
    for cond_name in CONDITIONS:
        md = merged_cache[cond_name]["dist"]
        h = md[f"{dim}_human"].values.astype(float)
        l = md[f"{dim}_llm"].values.astype(float)
        means, levels = [], []
        for sv in [1, 2, 3, 4]:
            sv_mask = h == sv
            if sv_mask.sum() > 0:
                means.append(np.mean(l[sv_mask]))
                levels.append(sv)
        ax.plot(levels, means, "o-", label=COND_LABELS[cond_name], markersize=6)
    ax.plot([1, 4], [1, 4], "k--", alpha=0.5, label="Perfect agreement")
    ax.set_title(dim.replace("_", " "))
    ax.set_xlabel("Human Score")
    ax.set_ylabel("Mean LLM Score")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.legend(fontsize=7)
fig.suptitle("Mean LLM Score by Human Score Level: Distractors Only", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_alpha_by_score_level_distractors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_alpha_by_score_level_distractors.png")

# Fig 4: Alpha heatmap (condition × dimension)
heatmap_data = []
for cond in CONDITIONS:
    row = [(all_results[cond]["dimensions"][d]["alpha_distractors_270"] or 0) for d in DIMS]
    heatmap_data.append(row)
heatmap_df = pd.DataFrame(heatmap_data, index=list(COND_LABELS.values()), columns=[d.replace("_", " ") for d in DIMS])

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="RdYlGn", center=0, vmin=-0.3, vmax=0.5, ax=ax, linewidths=0.5)
ax.set_title("Krippendorff's Alpha Heatmap: Distractors Only")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig8_alpha_heatmap_distractors.png", dpi=150)
plt.close()
print("Saved fig8_alpha_heatmap_distractors.png")


# ── 7. Summary markdown ─────────────────────────────────────────────────────

summary_lines = [
    "# Task 1: Distractors-Only Re-evaluation: Summary",
    "",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    f"**Items:** {n_distractors} distractors (correct answers removed: {n_correct})",
    f"**Primary metric:** Krippendorff's Alpha (ordinal)",
    "",
    "## Comparison: All Items vs Distractors-Only",
    "",
    "| Condition | Dimension | Alpha (all) | Alpha (dist) | Δ | Kappa (dist) | Bias | Abs Dist | SC | n |",
    "|-----------|-----------|------------|-------------|---|-------------|------|----------|----|---||",
]

for _, row in df_comp.iterrows():
    bias_str = f"{row['mean_bias']:+.3f}" if pd.notna(row['mean_bias']) else "N/A"
    dist_str = f"{row['mean_abs_distance']:.3f}" if pd.notna(row['mean_abs_distance']) else "N/A"
    summary_lines.append(
        f"| {row['condition']} | {row['dimension']} | "
        f"{row['alpha_all']} | {row['alpha_distractors_270']} | "
        f"{row['alpha_delta']} | {row['kappa_distractors']} | "
        f"{bias_str} | {dist_str} | "
        f"{row['self_consistency']} | {row['n']} |"
    )

summary_lines.extend([
    "",
    "## Aggregate",
    "",
    "| Condition | Mean Alpha (dist) | Mean Kappa (dist) |",
    "|-----------|------------------|------------------|",
])
for cond in CONDITIONS:
    agg = all_results[cond]["aggregate"]
    summary_lines.append(f"| {cond} | {agg['mean_alpha']} | {agg['mean_kappa']} |")

summary_lines.extend([
    "",
    "## Within-condition Run Alpha (3 runs as 3 raters)",
    "",
    "| Condition | Dimension | Run Alpha |",
    "|-----------|-----------|-----------|",
])
for key, val in run_alpha_results.items():
    cond, dim = key.split("|")
    summary_lines.append(f"| {cond} | {dim} | {val} |")

summary_lines.extend([
    "",
    "## Figures",
    "- `figures/fig1_alpha_by_dimension_distractors.png`",
    "- `figures/fig_bias_histogram_distractors.png`",
    "- `figures/fig_alpha_by_score_level_distractors.png`",
    "- `figures/fig8_alpha_heatmap_distractors.png`",
])

with open(OUT_DIR / "distractors_only_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"\n\nAll outputs saved to: {OUT_DIR}")
print("Done!")
