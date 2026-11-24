"""
Sensitivity Analysis: threshold sweep and power analysis.

Computes:
1. Workload reduction vs quality as tier-A threshold varies (0.2 to 0.7)
2. Required sample size per dimension for CI to not cross tier boundary
3. Actual workload reduction from item-level routing (not just dim-level)

No API calls; operates on cached experiment results.
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

from src.pipeline.metrics import compute_krippendorff_alpha_multi
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
    master = pd.read_csv(RESULTS_DIR / "master_results.csv")
    human = pd.read_csv(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")
    for dim, hd in HUMAN_DIM_MAP.items():
        human[f"{dim}_consensus"] = (
            (human[f"{hd}{K_SUFFIX}"] + human[f"{hd}{L_SUFFIX}"]) / 2
        ).round().astype(int)
    return master, human


def load_all_raw_scores():
    """Load raw per-run scores for all experiment cells."""
    all_scores = {}
    for cell_dir in RESULTS_DIR.iterdir():
        if not cell_dir.is_dir():
            continue
        raw_path = cell_dir / "raw_scores.json"
        if raw_path.exists():
            with open(raw_path) as f:
                data = json.load(f)
            scores = {}
            for uuid, dims in data.items():
                scores[uuid] = {
                    d: [float(v) if v is not None else np.nan for v in vs]
                    for d, vs in dims.items()
                }
            all_scores[cell_dir.name] = scores
    return all_scores


# ── 1. Threshold Sensitivity Analysis ─────────────────────────────────

def threshold_sweep(master: pd.DataFrame):
    """Sweep tier-A threshold from 0.20 to 0.70 and compute outcomes."""
    thresholds = np.arange(0.20, 0.71, 0.05)
    results = []

    for dim in DIMS:
        dim_data = master[master["dimension"] == dim]
        best_alpha = dim_data["alpha"].max()
        best_row = dim_data.loc[dim_data["alpha"].idxmax()] if not np.isnan(best_alpha) else None

        for thresh in thresholds:
            is_tier_a = best_alpha >= thresh
            results.append({
                "threshold": round(thresh, 2),
                "dimension": dim,
                "best_alpha": round(best_alpha, 4),
                "tier_a": is_tier_a,
                "best_config": f"{best_row['model']}/{best_row['prompt']}" if best_row is not None else "N/A",
            })

    df = pd.DataFrame(results)

    # Compute workload reduction per threshold
    summary = []
    for thresh in thresholds:
        sub = df[df["threshold"] == round(thresh, 2)]
        n_tier_a = sub["tier_a"].sum()
        n_total = len(sub)
        workload_reduction = n_tier_a / n_total * 100
        tier_a_dims = sub[sub["tier_a"]]["dimension"].tolist()
        mean_alpha = sub[sub["tier_a"]]["best_alpha"].mean() if n_tier_a > 0 else 0
        summary.append({
            "threshold": round(thresh, 2),
            "n_tier_a": int(n_tier_a),
            "n_total": n_total,
            "workload_reduction_pct": round(workload_reduction, 1),
            "tier_a_dimensions": tier_a_dims,
            "mean_tier_a_alpha": round(mean_alpha, 4) if not np.isnan(mean_alpha) else 0,
        })

    return pd.DataFrame(summary), df


def plot_threshold_sensitivity(summary_df: pd.DataFrame):
    """Plot workload reduction vs threshold with quality overlay."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#8C1D40"
    color2 = "#FFC627"

    ax1.bar(summary_df["threshold"], summary_df["workload_reduction_pct"],
            width=0.04, color=color1, alpha=0.8, label="Workload Reduction (%)")
    ax1.set_xlabel("Tier A Threshold (Krippendorff Alpha)", fontsize=12)
    ax1.set_ylabel("Workload Reduction (%)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(summary_df["threshold"], summary_df["mean_tier_a_alpha"],
             "o-", color=color2, linewidth=2, markersize=8, label="Mean Tier A Alpha")
    ax2.set_ylabel("Mean Alpha of Tier A Dimensions", fontsize=12, color="#B8860B")
    ax2.tick_params(axis="y", labelcolor="#B8860B")
    ax2.set_ylim(0, 0.8)

    ax1.axvline(x=0.50, color="gray", linestyle="--", alpha=0.5, label="Default threshold (0.50)")

    for _, row in summary_df.iterrows():
        if row["n_tier_a"] > 0:
            ax1.annotate(
                f"{int(row['n_tier_a'])}/5",
                (row["threshold"], row["workload_reduction_pct"] + 2),
                ha="center", fontsize=9, color=color1,
            )

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=10)
    plt.title("Sensitivity Analysis: Tier A Threshold vs Workload Reduction",
              fontsize=14, pad=20)
    plt.tight_layout()

    out = FIGURES_DIR / "fig7_threshold_sensitivity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


# ── 2. Item-Level Routing Analysis ────────────────────────────────────

def item_level_routing_analysis(all_scores: dict, human: pd.DataFrame):
    """Compute REAL workload reduction from inter-run agreement, not just dim count."""
    results = []

    for cell_name, scores in all_scores.items():
        parts = cell_name.split("_", 1)
        if len(parts) < 2:
            continue
        # Parse model and prompt from directory name
        if "P0" in cell_name:
            prompt = "P0_baseline"
            model = cell_name.replace("_P0_baseline", "")
        elif "P1" in cell_name:
            prompt = "P1_manual"
            model = cell_name.replace("_P1_manual", "")
        elif "P2" in cell_name:
            prompt = "P2_evidently"
            model = cell_name.replace("_P2_evidently", "")
        else:
            continue

        for dim in DIMS:
            n_3of3 = 0
            n_2of3 = 0
            n_split = 0
            n_total = 0

            for uuid, dim_scores in scores.items():
                runs = dim_scores.get(dim, [])
                valid = [int(s) for s in runs if not (isinstance(s, float) and np.isnan(s))]
                if len(valid) < 2:
                    continue
                n_total += 1
                if len(valid) == 3 and valid[0] == valid[1] == valid[2]:
                    n_3of3 += 1
                elif len(valid) >= 2:
                    from collections import Counter
                    counts = Counter(valid)
                    if counts.most_common(1)[0][1] >= 2:
                        n_2of3 += 1
                    else:
                        n_split += 1

            if n_total > 0:
                results.append({
                    "model": model,
                    "prompt": prompt,
                    "dimension": dim,
                    "n_total": n_total,
                    "pct_3of3": round(n_3of3 / n_total * 100, 1),
                    "pct_2of3": round(n_2of3 / n_total * 100, 1),
                    "pct_split": round(n_split / n_total * 100, 1),
                    "pct_auto_accept": round((n_3of3 + n_2of3) / n_total * 100, 1),
                })

    return pd.DataFrame(results)


# ── 3. Power Analysis ─────────────────────────────────────────────────

def bootstrap_power_analysis(
    all_scores: dict,
    human: pd.DataFrame,
    target_ci_width: float = 0.20,
    n_bootstrap: int = 1000,
):
    """Estimate required sample size for each dimension to achieve target CI width."""
    results = []

    for dim in DIMS:
        hd = HUMAN_DIM_MAP[dim]

        # Get best cell for this dimension
        best_cell = None
        best_alpha = -999
        for cell_name, scores in all_scores.items():
            llm_vals, cons_vals = [], []
            for _, row in human.iterrows():
                uuid = str(row["question_uuid"])
                if uuid in scores and dim in scores[uuid]:
                    runs = [s for s in scores[uuid][dim] if not (isinstance(s, float) and np.isnan(s))]
                    if runs:
                        llm_vals.append(float(np.median(runs)))
                        cons_vals.append(float(row[f"{dim}_consensus"]))

            if len(llm_vals) >= 10:
                alpha = compute_krippendorff_alpha_multi(
                    np.array(llm_vals), np.array(cons_vals)
                )["alpha"]
                if alpha > best_alpha:
                    best_alpha = alpha
                    best_cell = (llm_vals, cons_vals, cell_name)

        if best_cell is None:
            results.append({
                "dimension": dim, "best_alpha": np.nan,
                "ci_width_at_100": np.nan, "ci_width_at_200": np.nan,
                "required_n_for_target": np.nan, "best_config": "N/A",
            })
            continue

        llm_vals, cons_vals, cell_name = best_cell
        llm_arr = np.array(llm_vals)
        cons_arr = np.array(cons_vals)
        n_full = len(llm_arr)

        # Bootstrap at different sample sizes
        sample_sizes = [30, 50, 75, 100, 150, 200, 300, 500]
        ci_widths = {}

        for n in sample_sizes:
            if n > n_full:
                # Extrapolate: CI width scales roughly as 1/sqrt(n)
                ratio = np.sqrt(n_full / n)
                ci_widths[n] = ci_widths.get(n_full, 0.3) * ratio
                continue

            widths = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(n_full, size=n, replace=True)
                try:
                    a = compute_krippendorff_alpha_multi(
                        llm_arr[idx], cons_arr[idx]
                    )["alpha"]
                    widths.append(a)
                except Exception:
                    pass

            if widths:
                ci_lower = np.percentile(widths, 2.5)
                ci_upper = np.percentile(widths, 97.5)
                ci_widths[n] = ci_upper - ci_lower

        # Find required n for target CI width
        required_n = np.nan
        for n in sorted(ci_widths.keys()):
            if ci_widths[n] <= target_ci_width:
                required_n = n
                break

        results.append({
            "dimension": dim,
            "best_alpha": round(best_alpha, 4),
            "best_config": cell_name,
            "ci_width_at_100": round(ci_widths.get(100, np.nan), 4),
            "ci_width_at_200": round(ci_widths.get(200, np.nan), 4),
            "ci_width_at_300": round(ci_widths.get(300, np.nan), 4),
            "required_n_for_target": int(required_n) if not np.isnan(required_n) else ">500",
        })

    return pd.DataFrame(results)


def plot_power_analysis(power_df: pd.DataFrame):
    """Plot required sample sizes per dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dims = power_df["dimension"].tolist()
    alphas = power_df["best_alpha"].tolist()
    ci_100 = power_df["ci_width_at_100"].tolist()
    ci_200 = power_df["ci_width_at_200"].tolist()
    ci_300 = power_df["ci_width_at_300"].tolist()

    x = np.arange(len(dims))
    w = 0.25

    bars1 = ax.bar(x - w, ci_100, w, label="n=100", color="#8C1D40", alpha=0.8)
    bars2 = ax.bar(x, ci_200, w, label="n=200", color="#FFC627", alpha=0.8)
    bars3 = ax.bar(x + w, ci_300, w, label="n=300", color="#4A90D9", alpha=0.8)

    ax.axhline(y=0.20, color="green", linestyle="--", alpha=0.7, label="Target CI width (0.20)")
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("95% CI Width", fontsize=12)
    ax.set_title("Power Analysis: CI Width by Sample Size and Dimension", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIGURES_DIR / "fig8_power_analysis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    master, human = load_data()
    all_scores = load_all_raw_scores()
    print(f"  Master: {len(master)} rows, {len(all_scores)} experiment cells")

    print("\n=== 1. Threshold Sensitivity Analysis ===")
    summary, detail = threshold_sweep(master)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "threshold_sensitivity.csv", index=False)
    plot_threshold_sensitivity(summary)

    print("\n=== 2. Item-Level Routing Analysis ===")
    routing = item_level_routing_analysis(all_scores, human)
    if not routing.empty:
        # Show best config per dimension
        for dim in DIMS:
            dim_data = routing[routing["dimension"] == dim]
            if len(dim_data) > 0:
                best = dim_data.loc[dim_data["pct_auto_accept"].idxmax()]
                print(f"  {dim:25s}: {best['pct_auto_accept']:5.1f}% auto-accept "
                      f"({best['pct_3of3']:.0f}% 3/3 + {best['pct_2of3']:.0f}% 2/3) "
                      f"[{best['model']}/{best['prompt']}]")
        routing.to_csv(RESULTS_DIR / "item_level_routing.csv", index=False)

    # Compute REAL workload reduction with Tier A = LF + Relevance
    print("\n  --- Real Workload Reduction (Tier A: LF + Relevance) ---")
    tier_a_dims = ["Linguistic_Feature", "Relevance"]
    best_configs = {
        "Linguistic_Feature": "gemini_flash_P0_baseline",
        "Relevance": "gpt54_P1_manual",
    }
    total_items_x_dims = 0
    total_auto = 0
    for dim in DIMS:
        if dim in tier_a_dims:
            cell = best_configs[dim]
            dim_routing = routing[(routing["dimension"] == dim) &
                                  (routing["model"] + "_" + routing["prompt"] == cell)]
            if len(dim_routing) > 0:
                row = dim_routing.iloc[0]
                auto = row["pct_auto_accept"]
                n = row["n_total"]
            else:
                auto = 0
                n = 100
        else:
            auto = 0
            n = 100
        total_items_x_dims += n
        total_auto += n * auto / 100
        print(f"    {dim:25s}: {auto:5.1f}% auto-accepted")

    real_reduction = total_auto / total_items_x_dims * 100 if total_items_x_dims > 0 else 0
    print(f"    {'TOTAL':25s}: {real_reduction:.1f}% workload reduction")

    print("\n=== 3. Power Analysis ===")
    np.random.seed(42)
    power = bootstrap_power_analysis(all_scores, human,
                                     target_ci_width=0.20, n_bootstrap=500)
    print(power.to_string(index=False))
    power.to_csv(RESULTS_DIR / "power_analysis.csv", index=False)
    plot_power_analysis(power)

    print("\nDone. Results saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
