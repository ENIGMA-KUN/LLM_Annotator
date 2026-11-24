"""
Prompt Engineering Experiments: 3 conditions × 100 items × 3 runs each.

Conditions:
    (A) Baseline: Phase 1 zero-shot results (reused, $0 cost)
    (B) best_prompt_v1: Debiased prompt with calibration notes (~$1.50)
    (C) pairwise_sem_plaus_v1: Pairwise reformulation for Semantic Plausibility (~$1.50)

Success criterion: If any dimension crosses 0.5 Alpha vs human consensus,
we have a new Tier A candidate and a paper contribution.

Usage:
    python scripts/run_prompt_experiments.py [--dry-run] [--n-items 100]
"""
import argparse
import ast
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.pipeline import AnnotationEngine, DimensionRouter, ConfidenceGate
from src.pipeline.metrics import compute_krippendorff_alpha_multi
from src.pipeline.dimension_router import ALL_PIPELINE_DIMENSIONS

DIMS = list(ALL_PIPELINE_DIMENSIONS)
HUMAN_DIM_MAP = {
    "Relevance": "Relevance",
    "Clarity": "Clarity",
    "Linguistic_Feature": "Linguistic_Feature",
    "Semantic_Plausibility": "Semantic_Plausibility",
    "Semantic_Uniqueness": "Semantic_Unique",
}
K_SUFFIX = "_KC"
L_SUFFIX = "_LH"

OUTPUT_DIR = ROOT / "Data" / "results" / "prompt_experiments"


def load_human_data():
    """Load human scores and compute consensus."""
    df = pd.read_csv(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")
    for dim, hd in HUMAN_DIM_MAP.items():
        k_col = f"{hd}{K_SUFFIX}"
        l_col = f"{hd}{L_SUFFIX}"
        df[f"{dim}_consensus"] = ((df[k_col] + df[l_col]) / 2).round().astype(int)
    return df


def load_phase1_baseline(n_items: int) -> tuple[list[dict], pd.DataFrame]:
    """Load Phase 1 zero-shot results as baseline (no API cost)."""
    csv_path = ROOT / "Data" / "results" / "phase1_question_level" / "zero_shot_qlevel" / "zero_shot_qlevel_results.csv"
    df = pd.read_csv(csv_path)

    # Use first n_items
    df = df.head(n_items)
    engine_items = []
    for _, row in df.iterrows():
        uuid = str(row["question_uuid"])
        runs = []
        for ri in range(3):
            scores = {}
            for dim in DIMS:
                col = f"{dim}_all_runs"
                if col in df.columns:
                    try:
                        rl = ast.literal_eval(str(row[col]))
                        scores[dim] = rl[ri] if ri < len(rl) else np.nan
                    except:
                        scores[dim] = np.nan
                else:
                    scores[dim] = row.get(dim, np.nan)
            runs.append({"scores": scores})
        engine_items.append({"question_uuid": uuid, "runs": runs})

    return engine_items, df


def run_experiment_b(items_df: pd.DataFrame, n_items: int) -> list[dict]:
    """Run Experiment B: best_prompt_v1 on n_items."""
    prompt_path = ROOT / "prompts" / "best_prompt_v1.txt"
    prompt = prompt_path.read_text(encoding="utf-8")

    engine = AnnotationEngine(prompt_version="best_prompt_v1")

    # Build item list
    items = []
    for _, row in items_df.head(n_items).iterrows():
        items.append({
            "question_uuid": str(row["question_uuid"]),
            "text": str(row.get("text", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        })

    results = engine.annotate_batch(items, system_prompt=prompt, condition_name="best_prompt_v1")
    cost = engine.get_cost_summary()
    print(f"\n  Experiment B cost: ${cost['total_cost_usd']:.4f} | Tokens: {cost['total_tokens']:,}")
    return results


def run_experiment_c(items_df: pd.DataFrame, n_items: int) -> list[dict]:
    """
    Run Experiment C: pairwise Semantic Plausibility.

    This uses the pairwise prompt for Sem.Plausibility only.
    Other dimensions come from Experiment B (or baseline).
    """
    prompt_path = ROOT / "prompts" / "pairwise_sem_plaus_v1.txt"
    prompt = prompt_path.read_text(encoding="utf-8")

    engine = AnnotationEngine(prompt_version="pairwise_sem_plaus_v1")

    items = []
    for _, row in items_df.head(n_items).iterrows():
        items.append({
            "question_uuid": str(row["question_uuid"]),
            "text": str(row.get("text", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        })

    results = engine.annotate_batch(items, system_prompt=prompt, condition_name="pairwise_sem_plaus_v1")
    cost = engine.get_cost_summary()
    print(f"\n  Experiment C cost: ${cost['total_cost_usd']:.4f} | Tokens: {cost['total_tokens']:,}")
    return results


def extract_scores(engine_items: list[dict], dim: str) -> dict:
    """Extract per-UUID median score from engine items for a given dimension."""
    scores = {}
    for item in engine_items:
        uuid = item["question_uuid"]
        run_scores = []
        for run in item["runs"]:
            s = run.get("scores", {}).get(dim, np.nan)
            if not (isinstance(s, float) and np.isnan(s)):
                run_scores.append(s)
        if run_scores:
            scores[uuid] = float(np.median(run_scores))
    return scores


def compute_alpha_table(
    conditions: dict[str, list[dict]],
    df_human: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-dimension Alpha for each condition vs human consensus.

    Returns DataFrame: condition × dimension with Alpha values.
    """
    rows = []
    for cond_name, engine_items in conditions.items():
        for dim in DIMS:
            llm_scores_map = extract_scores(engine_items, dim)
            hd = HUMAN_DIM_MAP[dim]
            k_col = f"{hd}{K_SUFFIX}"
            l_col = f"{hd}{L_SUFFIX}"

            llm_vals, k_vals, l_vals, cons_vals = [], [], [], []
            for _, row in df_human.iterrows():
                uuid = str(row["question_uuid"])
                if uuid in llm_scores_map:
                    llm_vals.append(llm_scores_map[uuid])
                    k_vals.append(float(row[k_col]))
                    l_vals.append(float(row[l_col]))
                    cons_vals.append(float(row[f"{dim}_consensus"]))

            if len(llm_vals) < 5:
                rows.append({"condition": cond_name, "dimension": dim, 
                            "alpha_llm_vs_consensus": np.nan, "alpha_llm_vs_K": np.nan,
                            "alpha_llm_vs_L": np.nan, "alpha_KL": np.nan,
                            "exact_match": np.nan, "bias": np.nan, "n": len(llm_vals)})
                continue

            llm_arr = np.array(llm_vals)
            cons_arr = np.array(cons_vals)
            k_arr = np.array(k_vals)
            l_arr = np.array(l_vals)

            alpha_cons = compute_krippendorff_alpha_multi(llm_arr, cons_arr)["alpha"]
            alpha_k = compute_krippendorff_alpha_multi(llm_arr, k_arr)["alpha"]
            alpha_l = compute_krippendorff_alpha_multi(llm_arr, l_arr)["alpha"]
            alpha_kl = compute_krippendorff_alpha_multi(k_arr, l_arr)["alpha"]
            exact = (llm_arr == cons_arr).mean() * 100
            bias = (llm_arr - cons_arr).mean()

            rows.append({
                "condition": cond_name,
                "dimension": dim,
                "alpha_llm_vs_consensus": round(alpha_cons, 4),
                "alpha_llm_vs_K": round(alpha_k, 4),
                "alpha_llm_vs_L": round(alpha_l, 4),
                "alpha_KL": round(alpha_kl, 4),
                "exact_match": round(exact, 1),
                "bias": round(bias, 3),
                "n": len(llm_vals),
            })

    return pd.DataFrame(rows)


def report_results(alpha_df: pd.DataFrame, output_dir: Path):
    """Print and save the results report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("PROMPT EXPERIMENT RESULTS: Alpha per Dimension per Condition")
    print("=" * 90)

    # Pivot for comparison
    for dim in DIMS:
        dim_data = alpha_df[alpha_df["dimension"] == dim]
        print(f"\n--- {dim} ---")
        print(f"  {'Condition':<25} {'alpha(LLM,cons)':>12} {'alpha(LLM,K)':>10} {'alpha(LLM,L)':>10} {'alpha(K,L)':>8} {'Exact%':>7} {'Bias':>7} {'N':>4}")
        print(f"  {'-'*83}")
        for _, row in dim_data.iterrows():
            flag = " *** NEW TIER A? ***" if row["alpha_llm_vs_consensus"] >= 0.5 else ""
            print(
                f"  {row['condition']:<25} {row['alpha_llm_vs_consensus']:>12.4f} "
                f"{row['alpha_llm_vs_K']:>10.4f} {row['alpha_llm_vs_L']:>10.4f} "
                f"{row['alpha_KL']:>8.4f} {row['exact_match']:>6.1f}% {row['bias']:>+6.3f} {row['n']:>4}{flag}"
            )

    # Delta report
    print("\n" + "=" * 90)
    print("ALPHA DELTA vs BASELINE (positive = improvement)")
    print("=" * 90)

    baseline = alpha_df[alpha_df["condition"] == "A_baseline"].set_index("dimension")
    new_tier_a = []

    for cond in alpha_df["condition"].unique():
        if cond == "A_baseline":
            continue
        cond_data = alpha_df[alpha_df["condition"] == cond].set_index("dimension")
        print(f"\n{cond}:")
        for dim in DIMS:
            if dim in baseline.index and dim in cond_data.index:
                base_alpha = baseline.loc[dim, "alpha_llm_vs_consensus"]
                new_alpha = cond_data.loc[dim, "alpha_llm_vs_consensus"]
                delta = new_alpha - base_alpha
                crossed = new_alpha >= 0.5 and base_alpha < 0.5
                flag = " <<< CROSSED 0.5 THRESHOLD >>>" if crossed else ""
                print(f"  {dim:<25} {base_alpha:.4f} -> {new_alpha:.4f} (delta={delta:+.4f}){flag}")
                if crossed:
                    new_tier_a.append((cond, dim, new_alpha))

    # Final verdict
    print("\n" + "=" * 90)
    if new_tier_a:
        print("NEW TIER A CANDIDATES FOUND!")
        for cond, dim, alpha in new_tier_a:
            print(f"  -> {dim} (alpha={alpha:.4f}) via {cond}")
        print("\nThis is a paper contribution. Expand TIER_A_DIMENSIONS and re-run pipeline.")
    else:
        print("NO NEW TIER A CANDIDATES.")
        print("Prompt engineering alone doesn't cross 0.5 threshold.")
        print("Next step: fine-tuning experiments.")
    print("=" * 90)

    # Save
    alpha_df.to_csv(output_dir / "alpha_results.csv", index=False)
    
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conditions": list(alpha_df["condition"].unique()),
        "n_items_per_condition": int(alpha_df["n"].max()),
        "new_tier_a_candidates": [
            {"condition": c, "dimension": d, "alpha": a} for c, d, a in new_tier_a
        ],
    }
    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run prompt engineering experiments")
    parser.add_argument("--dry-run", action="store_true", help="Only run baseline (no API calls)")
    parser.add_argument("--n-items", type=int, default=100, help="Number of items per experiment")
    args = parser.parse_args()

    n_items = args.n_items

    # Load human data
    df_human = load_human_data()
    print(f"Human data: {len(df_human)} items")

    # Also load the upload CSV for text/question/answer
    upload_path = ROOT / "Data" / "processed" / "pipeline_upload_200items.csv"
    items_df = pd.read_csv(upload_path)
    print(f"Upload data: {len(items_df)} items")

    # Experiment A: Baseline (free, reuses Phase 1)
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT A: Phase 1 Zero-Shot Baseline ({n_items} items, $0)")
    print(f"{'='*60}")
    baseline_items, baseline_df = load_phase1_baseline(n_items)
    print(f"  Loaded {len(baseline_items)} items from Phase 1")

    conditions = {"A_baseline": baseline_items}

    if not args.dry_run:
        # Experiment B: best_prompt_v1
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT B: best_prompt_v1 ({n_items} items, ~$1.50)")
        print(f"{'='*60}")
        exp_b_results = run_experiment_b(items_df, n_items)
        conditions["B_best_prompt_v1"] = exp_b_results

        # Experiment C: Pairwise Semantic Plausibility
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT C: Pairwise Sem.Plausibility ({n_items} items, ~$1.50)")
        print(f"{'='*60}")
        exp_c_results = run_experiment_c(items_df, n_items)

        # For Experiment C, merge: use baseline scores for all dims EXCEPT Sem.Plausibility
        merged_c = []
        for base_item, exp_item in zip(baseline_items[:n_items], exp_c_results):
            merged_runs = []
            for ri in range(3):
                if ri < len(base_item["runs"]) and ri < len(exp_item["runs"]):
                    base_scores = base_item["runs"][ri].get("scores", {})
                    exp_scores = exp_item["runs"][ri].get("scores", {})
                    merged_scores = dict(base_scores)
                    # Override Semantic_Plausibility from experiment C
                    if "Semantic_Plausibility" in exp_scores:
                        merged_scores["Semantic_Plausibility"] = exp_scores["Semantic_Plausibility"]
                    merged_runs.append({"scores": merged_scores})
                elif ri < len(base_item["runs"]):
                    merged_runs.append(base_item["runs"][ri])
            merged_c.append({
                "question_uuid": base_item["question_uuid"],
                "runs": merged_runs,
            })
        conditions["C_pairwise_sem_plaus"] = merged_c
    else:
        print("\n  [DRY RUN] Skipping Experiments B and C (no API calls)")

    # Compute Alpha table
    alpha_df = compute_alpha_table(conditions, df_human)

    # Report
    report_results(alpha_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
