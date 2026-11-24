"""
Generate paper figures from experiment results.

Usage:
    python scripts/generate_paper_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Data" / "results" / "neurips_experiments"
FIGURES_DIR = ROOT / "Paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility", "Semantic_Uniqueness"]
DIM_SHORT = {"Relevance": "Rel", "Clarity": "Cla", "Linguistic_Feature": "LF",
             "Semantic_Plausibility": "SP", "Semantic_Uniqueness": "SU"}

# Paper figure style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_results():
    csv_path = RESULTS_DIR / "master_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run experiments first.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} result rows from {csv_path}")
    return df


def fig1_alpha_heatmap(df):
    """Heatmap: Models × Dimensions, best Alpha across prompts."""
    # Get best alpha per model × dimension
    pivot = df.groupby(["model", "dimension"])["alpha"].max().reset_index()
    pivot_table = pivot.pivot(index="model", columns="dimension", values="alpha")
    pivot_table = pivot_table.reindex(columns=DIMS)

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot_table) * 0.7 + 1)))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap=cmap, center=0,
                vmin=-0.5, vmax=0.8, linewidths=0.5, ax=ax,
                xticklabels=[DIM_SHORT.get(d, d) for d in DIMS])
    ax.set_title("Krippendorff α (ordinal): LLM vs human consensus")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_alpha_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig1_alpha_heatmap.png")
    plt.close(fig)
    print("  [OK] Fig 1: Alpha heatmap")


def fig2_bias_heatmap(df):
    """Heatmap: bias per model × dimension (best prompt)."""
    # For each model×dim, get the bias of the prompt with best alpha
    rows = []
    for (model, dim), grp in df.groupby(["model", "dimension"]):
        best_idx = grp["alpha"].idxmax()
        rows.append({"model": model, "dimension": dim, "bias": grp.loc[best_idx, "bias"]})
    bias_df = pd.DataFrame(rows)
    pivot = bias_df.pivot(index="model", columns="dimension", values="bias")
    pivot = pivot.reindex(columns=DIMS)

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.7 + 1)))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt="+.2f", cmap=cmap, center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                xticklabels=[DIM_SHORT.get(d, d) for d in DIMS])
    ax.set_title("Mean Signed Bias (LLM − Human)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_bias_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig2_bias_heatmap.png")
    plt.close(fig)
    print("  [OK] Fig 2: Bias heatmap")


def fig3_prompt_effect(df):
    """Grouped bar chart: Alpha by dimension, grouped by prompt strategy."""
    # Average alpha across models for each prompt × dimension
    avg = df.groupby(["prompt", "dimension"])["alpha"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    prompts = sorted(df["prompt"].unique())
    x = np.arange(len(DIMS))
    width = 0.25
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    for i, prompt in enumerate(prompts):
        vals = []
        for dim in DIMS:
            row = avg[(avg["prompt"] == prompt) & (avg["dimension"] == dim)]
            vals.append(row["alpha"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, label=prompt, color=colors[i % len(colors)])

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Krippendorff α")
    ax.set_title("Effect of Prompt Strategy on Agreement (averaged across models)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DIM_SHORT.get(d, d) for d in DIMS])
    ax.legend()
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Tier A threshold")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_prompt_effect.pdf")
    fig.savefig(FIGURES_DIR / "fig3_prompt_effect.png")
    plt.close(fig)
    print("  [OK] Fig 3: Prompt effect")


def fig4_radar_chart(df):
    """Radar/spider chart: per-model profile across 5 dimensions (best prompt)."""
    models = df["model"].unique()
    best = df.loc[df.groupby(["model", "dimension"])["alpha"].idxmax()]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for i, model in enumerate(models):
        vals = []
        for dim in DIMS:
            row = best[(best["model"] == model) & (best["dimension"] == dim)]
            vals.append(max(0, row["alpha"].values[0]) if len(row) > 0 else 0)
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=model, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIM_SHORT.get(d, d) for d in DIMS])
    ax.set_ylim(0, 1)
    ax.set_title("Per-Model Dimension Profile (best α per dimension)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_radar.pdf")
    fig.savefig(FIGURES_DIR / "fig4_radar.png")
    plt.close(fig)
    print("  [OK] Fig 4: Radar chart")


def fig5_cost_vs_quality(df):
    """Scatter: best Alpha vs cost per model."""
    # Load cost data from call logs
    cost_data = {}
    for cell_dir in RESULTS_DIR.iterdir():
        if not cell_dir.is_dir():
            continue
        log_path = cell_dir / "call_log.jsonl"
        if not log_path.exists():
            continue
        total_cost = 0
        model_key = None
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("_type") == "cell_metadata":
                    model_key = entry.get("model_key")
                elif entry.get("_type") == "cell_summary":
                    total_cost = entry.get("total_cost_usd", 0)
        if model_key:
            cost_data[model_key] = cost_data.get(model_key, 0) + total_cost

    if not cost_data:
        print("  [WARN] Fig 5: no cost data (rerun the experiments to log cost)")
        return

    # Best alpha per model
    best_alpha = df.groupby("model")["alpha"].max()

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in best_alpha.index:
        cost = cost_data.get(model, 0)
        alpha = best_alpha[model]
        ax.scatter(cost, alpha, s=120, zorder=5)
        ax.annotate(model, (cost, alpha), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Best Krippendorff α")
    ax.set_title("Cost vs Quality Pareto Frontier")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Tier A threshold")
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_cost_quality.pdf")
    fig.savefig(FIGURES_DIR / "fig5_cost_quality.png")
    plt.close(fig)
    print("  [OK] Fig 5: Cost vs quality")


def fig6_consistency_bar(df):
    """Bar chart: inter-run consistency per model × dimension."""
    if "consistency" not in df.columns:
        print("  [WARN] Fig 6: no consistency data")
        return

    best = df.loc[df.groupby(["model", "dimension"])["alpha"].idxmax()]
    pivot = best.pivot(index="model", columns="dimension", values="consistency")
    pivot = pivot.reindex(columns=DIMS)

    fig, ax = plt.subplots(figsize=(10, max(3, len(pivot) * 0.7 + 1)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGn",
                vmin=0, vmax=100, linewidths=0.5, ax=ax,
                xticklabels=[DIM_SHORT.get(d, d) for d in DIMS])
    ax.set_title("Inter-Run Consistency (% items with 3/3 agreement)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_consistency.pdf")
    fig.savefig(FIGURES_DIR / "fig6_consistency.png")
    plt.close(fig)
    print("  [OK] Fig 6: Consistency heatmap")


def table_main_results(df):
    """Generate LaTeX table for main results."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Krippendorff $\\alpha$ (ordinal) per model $\\times$ prompt $\\times$ dimension.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll" + "c" * len(DIMS) + "}")
    lines.append("\\toprule")
    header = "Model & Prompt & " + " & ".join(DIM_SHORT[d] for d in DIMS) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for model in df["model"].unique():
        for prompt in sorted(df["prompt"].unique()):
            vals = []
            for dim in DIMS:
                row = df[(df["model"] == model) & (df["prompt"] == prompt) & (df["dimension"] == dim)]
                if len(row) > 0:
                    a = row.iloc[0]["alpha"]
                    s = f"{a:.3f}" if not np.isnan(a) else "--"
                    if a >= 0.5:
                        s = f"\\textbf{{{s}}}"
                    vals.append(s)
                else:
                    vals.append("--")
            lines.append(f"{model} & {prompt} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = FIGURES_DIR / "table_main_results.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Table: main results LaTeX -> {tex_path}")


def main():
    print("Generating paper figures...")
    df = load_results()

    fig1_alpha_heatmap(df)
    fig2_bias_heatmap(df)
    fig3_prompt_effect(df)
    fig4_radar_chart(df)
    fig5_cost_vs_quality(df)
    fig6_consistency_bar(df)
    table_main_results(df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
