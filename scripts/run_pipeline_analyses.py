"""
Run the missing pipeline analyses: Active Selection + Fairness Audit + Validity.

No API calls needed; operates on cached experiment results.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.active_selector import (
    rank_items_by_uncertainty,
    simulate_active_selection,
    compute_optimal_budget,
)
from src.pipeline.fairness_audit import run_full_fairness_audit, run_fairness_audit
from src.pipeline.dimension_router import ALL_PIPELINE_DIMENSIONS

DIMS = list(ALL_PIPELINE_DIMENSIONS)
RESULTS_DIR = ROOT / "Data" / "results" / "neurips_experiments"
FIGURES_DIR = ROOT / "Paper" / "figures"
HUMAN_DIM_MAP = {
    "Relevance": "Relevance",
    "Clarity": "Clarity",
    "Linguistic_Feature": "Linguistic_Feature",
    "Semantic_Plausibility": "Semantic_Plausibility",
    "Semantic_Uniqueness": "Semantic_Unique",
}
K_SUFFIX = "_KC"
L_SUFFIX = "_LH"


def load_data():
    human = pd.read_csv(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")
    for dim, hd in HUMAN_DIM_MAP.items():
        human[f"{dim}_consensus"] = (
            (human[f"{hd}{K_SUFFIX}"] + human[f"{hd}{L_SUFFIX}"]) / 2
        ).round().astype(int)
    return human


def load_best_scores():
    """Load best-config raw scores per Tier A dimension."""
    best = {
        "Linguistic_Feature": "gemini_flash_P0_baseline",
        "Relevance": "gpt54_P1_manual",
    }
    all_scores = {}
    for dim, cell_name in best.items():
        path = RESULTS_DIR / cell_name / "raw_scores.json"
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            scores = {}
            for uuid, dims in raw.items():
                scores[uuid] = {
                    d: [float(v) if v is not None else np.nan for v in vs]
                    for d, vs in dims.items()
                }
            all_scores[dim] = (cell_name, scores)
    return all_scores


def load_all_scores_for_cell(cell_name: str) -> dict:
    path = RESULTS_DIR / cell_name / "raw_scores.json"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {
        uuid: {d: [float(v) if v is not None else np.nan for v in vs] for d, vs in dims.items()}
        for uuid, dims in raw.items()
    }


# ── 1. Active Learning Simulation ─────────────────────────────────────

def run_active_learning(human):
    print("=== ACTIVE LEARNING SIMULATION ===\n")
    best_scores = load_best_scores()

    all_sim_results = {}
    for dim, (cell_name, scores) in best_scores.items():
        print(f"  {dim} ({cell_name}):")

        # Build consensus lookup
        consensus = {}
        for _, row in human.iterrows():
            uuid = str(row["question_uuid"])
            consensus[uuid] = float(row[f"{dim}_consensus"])

        # Rank by uncertainty
        ranked = rank_items_by_uncertainty(scores, dim)
        n_unanimous = (ranked["agreement_level"] == "unanimous").sum()
        n_majority = (ranked["agreement_level"] == "majority").sum()
        n_split = (ranked["agreement_level"] == "split").sum()
        print(f"    Unanimous: {n_unanimous}, Majority: {n_majority}, Split: {n_split}")

        # Simulate
        sim = simulate_active_selection(scores, consensus, dim)
        all_sim_results[dim] = sim
        print(sim.to_string(index=False))

        # Find optimal budget
        opt = compute_optimal_budget(sim, target_alpha=0.50)
        print(f"    Optimal budget for alpha>=0.50: {opt}")
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (dim, sim) in enumerate(all_sim_results.items()):
        ax = axes[i]
        ax.plot(sim["budget_pct"], sim["alpha_active"], "o-", color="#8C1D40",
                linewidth=2, markersize=6, label="Active (uncertainty)")
        ax.plot(sim["budget_pct"], sim["alpha_random"], "s--", color="#999",
                linewidth=1.5, markersize=5, label="Random")
        ax.axhline(y=0.50, color="green", linestyle=":", alpha=0.7, label="Tier A threshold")
        ax.set_xlabel("Human Review Budget (%)", fontsize=11)
        ax.set_ylabel("Krippendorff Alpha", fontsize=11)
        ax.set_title(f"{dim.replace('_', ' ')}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.0)

    plt.suptitle("Active Selection vs Random: Alpha by Human Review Budget", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "fig9_active_learning.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Save results
    for dim, sim in all_sim_results.items():
        sim.to_csv(RESULTS_DIR / f"active_learning_{dim}.csv", index=False)


# ── 2. Fairness Audit ─────────────────────────────────────────────────

def run_fairness(human):
    print("\n=== FAIRNESS AUDIT ===\n")

    # Use best config for each dimension
    configs = {
        "gemini_flash_P0_baseline": ["Linguistic_Feature", "Semantic_Uniqueness"],
        "gpt54_P1_manual": ["Relevance"],
        "gpt54_P0_baseline": ["Semantic_Plausibility", "Clarity"],
    }

    all_results = []
    for cell_name, dims_for_cell in configs.items():
        scores = load_all_scores_for_cell(cell_name)
        if not scores:
            print(f"  Skipping {cell_name} (no data)")
            continue

        for dim in dims_for_cell:
            result = run_fairness_audit(scores, human, dim)
            if "error" in result:
                print(f"  {dim}: {result['error']}")
                continue

            print(f"  {dim} ({cell_name}):")
            for analysis_name in ["by_subject", "by_difficulty", "by_human_agreement", "by_text_length"]:
                analysis = result.get(analysis_name, {})
                if "subgroups" not in analysis:
                    continue
                fair_str = "FAIR" if analysis.get("fair", True) else "UNFAIR"
                gap = analysis.get("max_gap", 0)
                print(f"    {analysis_name:25s}: gap={gap:.3f} [{fair_str}]")
                for sg_name, sg in analysis["subgroups"].items():
                    print(f"      {sg_name:20s}: alpha={sg['alpha']:.4f}, bias={sg.get('bias', 0):+.3f}, n={sg['n']}")

            if result.get("flags"):
                print(f"    FLAGS: {result['flags']}")
            print()

    # Run full audit for summary table
    scores_all = load_all_scores_for_cell("gemini_flash_P0_baseline")
    if scores_all:
        summary = run_full_fairness_audit(scores_all, human)
        summary.to_csv(RESULTS_DIR / "fairness_audit.csv", index=False)
        print(f"  Full audit saved to fairness_audit.csv ({len(summary)} rows)")

    # Plot fairness by subject
    fig, axes = plt.subplots(1, len(DIMS), figsize=(18, 5), sharey=True)
    subjects = ["sociology", "history", "anthropology", "lifespan"]
    colors = {"sociology": "#8C1D40", "history": "#FFC627", "anthropology": "#4A90D9", "lifespan": "#2E7D32"}

    for i, dim in enumerate(DIMS):
        ax = axes[i]
        for cell_name in configs:
            scores = load_all_scores_for_cell(cell_name)
            if not scores:
                continue
            if dim not in configs.get(cell_name, []):
                continue
            result = run_fairness_audit(scores, human, dim)
            if "error" in result or "by_subject" not in result:
                continue

            sgs = result["by_subject"]["subgroups"]
            x_vals = []
            y_vals = []
            c_vals = []
            for subj in subjects:
                if subj in sgs and not np.isnan(sgs[subj]["alpha"]):
                    x_vals.append(subj[:4])
                    y_vals.append(sgs[subj]["alpha"])
                    c_vals.append(colors.get(subj, "gray"))

            ax.bar(x_vals, y_vals, color=c_vals, alpha=0.8)
        ax.axhline(y=0.50, color="green", linestyle=":", alpha=0.5)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(dim.replace("_", "\n"), fontsize=9)
        ax.set_ylim(-0.6, 0.8)
        if i == 0:
            ax.set_ylabel("Alpha", fontsize=11)

    plt.suptitle("Fairness Audit: Alpha by Textbook Subject and Dimension", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "fig10_fairness_by_subject.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── 3. Construct Validity ──────────────────────────────────────────────

def run_validity(human):
    print("\n=== CONSTRUCT VALIDITY ANALYSIS ===\n")

    scores = load_all_scores_for_cell("gemini_flash_P0_baseline")
    if not scores:
        print("  No data available")
        return

    print("  Testing: Do LLM scores correlate with item difficulty indicators?")
    print("  (Item difficulty = human consensus mean; harder items = lower scores)\n")

    for dim in DIMS:
        hd = HUMAN_DIM_MAP[dim]
        llm_vals, consensus_vals, k_vals, l_vals = [], [], [], []
        human_disagree = []

        for _, row in human.iterrows():
            uuid = str(row["question_uuid"])
            if uuid not in scores or dim not in scores[uuid]:
                continue
            runs = [s for s in scores[uuid][dim] if not (isinstance(s, float) and np.isnan(s))]
            if not runs:
                continue

            k = float(row.get(f"{hd}{K_SUFFIX}", np.nan))
            l = float(row.get(f"{hd}{L_SUFFIX}", np.nan))
            if np.isnan(k) or np.isnan(l):
                continue

            llm_vals.append(float(np.median(runs)))
            consensus_vals.append(round((k + l) / 2))
            k_vals.append(k)
            l_vals.append(l)
            human_disagree.append(abs(k - l))

        if len(llm_vals) < 10:
            continue

        llm_arr = np.array(llm_vals)
        cons_arr = np.array(consensus_vals)
        disagree_arr = np.array(human_disagree)

        # Spearman correlation: LLM vs consensus
        from scipy.stats import spearmanr
        rho_consensus, p_consensus = spearmanr(llm_arr, cons_arr)

        # Does LLM error increase on items where humans disagree?
        llm_error = np.abs(llm_arr - cons_arr)
        rho_disagree, p_disagree = spearmanr(disagree_arr, llm_error)

        # Score distribution similarity (Jensen-Shannon divergence proxy)
        from collections import Counter
        llm_dist = Counter(llm_arr.astype(int))
        human_dist = Counter(cons_arr.astype(int))
        dist_diff = sum(abs(llm_dist.get(s, 0) - human_dist.get(s, 0)) for s in range(1, 5)) / (2 * len(llm_arr))

        print(f"  {dim:25s}:")
        print(f"    Spearman rho (LLM vs consensus): {rho_consensus:+.4f} (p={p_consensus:.4f})")
        print(f"    Error ~ disagreement corr:       {rho_disagree:+.4f} (p={p_disagree:.4f})")
        print(f"    Distribution difference (TVD):    {dist_diff:.4f}")
        print()

    print("  Validity interpretation:")
    print("    - High rho = LLM tracks construct (scores aren't random)")
    print("    - Positive error~disagree corr = LLM fails on ambiguous items (expected)")
    print("    - Low TVD = LLM captures score distribution shape")


def main():
    print("Loading data...")
    human = load_data()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    run_active_learning(human)
    run_fairness(human)
    run_validity(human)

    print("\n=== ALL ANALYSES COMPLETE ===")


if __name__ == "__main__":
    main()
