"""
Phase 1 Main Runner - LLM Annotation Replication Study (AIME 3-Condition Framework).

This script executes the full Phase 1 pipeline with three experimental conditions:
1. Load and explore data
2. Compute human baseline (Katerina vs Linh agreement)
3. Condition 1: Zero-shot LLM annotation (baseline)
4. Condition 2: Rubric-guided LLM annotation (detailed rubric, no examples)
5. Condition 3: Rubric + examples LLM annotation (detailed rubric + few-shot examples)
6. Evaluate all conditions & cross-condition comparison
7. Run Dawid-Skene aggregation pilot
8. Compute calibration metrics
9. Generate comparison tables and plots

Usage:
    python run_phase1.py                        # Full pipeline (all 3 conditions)
    python run_phase1.py --step explore          # Data exploration only
    python run_phase1.py --step baseline         # Human baseline only
    python run_phase1.py --step zero_shot        # Condition 1: Zero-shot
    python run_phase1.py --step rubric_guided    # Condition 2: Rubric-guided
    python run_phase1.py --step few_shot         # Condition 3: Rubric + examples
    python run_phase1.py --step evaluate         # Evaluate from saved results
    python run_phase1.py --step aggregate        # Dawid-Skene aggregation
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    load_annotator_csv,
    load_full_scores_overlap,
    load_reliability_statistics,
    setup_logging,
    ensure_dirs,
    save_results,
    ANNOTATION_DIMENSIONS,
    RESULTS_DIR,
    DATA_DIR,
)
from src.annotator import LLMAnnotator
from src.evaluator import (
    evaluate_full,
    evaluate_human_agreement,
    print_comparison_table,
)
from src.aggregator import run_dawid_skene, run_majority_vote, compare_aggregation_methods
from src.calibrator import (
    compute_calibration_curve,
    plot_reliability_diagram,
    analyze_self_consistency,
    plot_dimension_comparison,
)
from src.example_selector import select_few_shot_examples, validate_examples
from src.metadata import get_metadata_logger

logger = setup_logging()


# ---------------------------------------------------------------------------
# Step 1: Data Exploration
# ---------------------------------------------------------------------------

def step_explore() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, and explore annotation data."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA EXPLORATION")
    logger.info("=" * 60)

    df_k = load_annotator_csv("katerina")
    df_l = load_annotator_csv("linh")

    logger.info(f"Katerina: {len(df_k)} rows, {df_k.columns.tolist()}")
    logger.info(f"Linh:     {len(df_l)} rows, {df_l.columns.tolist()}")

    # Summary statistics per dimension
    for name, df in [("Katerina", df_k), ("Linh", df_l)]:
        logger.info(f"\n--- {name} Annotation Summary ---")
        for dim in ANNOTATION_DIMENSIONS:
            if dim in df.columns:
                valid = df[dim].dropna()
                logger.info(
                    f"  {dim:25s} | n={len(valid):3d} | "
                    f"mean={valid.mean():.2f} | std={valid.std():.2f} | "
                    f"dist={valid.value_counts().sort_index().to_dict()}"
                )

    # Check alignment
    logger.info(f"\nData alignment check:")
    logger.info(f"  Katerina rows: {len(df_k)}")
    logger.info(f"  Linh rows:     {len(df_l)}")

    # Unique questions/texts
    for name, df in [("Katerina", df_k), ("Linh", df_l)]:
        if "question_uuid" in df.columns:
            logger.info(f"  {name} unique questions: {df['question_uuid'].nunique()}")
        if "textID" in df.columns:
            logger.info(f"  {name} unique texts: {df['textID'].nunique()}")
        if "Source" in df.columns:
            logger.info(f"  {name} sources: {df['Source'].value_counts().to_dict()}")

    # Save exploration summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "katerina_rows": len(df_k),
        "linh_rows": len(df_l),
        "dimensions": {},
    }
    for dim in ANNOTATION_DIMENSIONS:
        summary["dimensions"][dim] = {
            "katerina_valid": int(df_k[dim].dropna().shape[0]) if dim in df_k.columns else 0,
            "linh_valid": int(df_l[dim].dropna().shape[0]) if dim in df_l.columns else 0,
        }
    save_results([summary], "data_exploration_summary")

    return df_k, df_l


# ---------------------------------------------------------------------------
# Step 2: Human Baseline
# ---------------------------------------------------------------------------

def step_baseline(df_k: pd.DataFrame, df_l: pd.DataFrame) -> dict:
    """Compute inter-rater agreement between human annotators.

    IMPORTANT: The K and L CSVs contain DISJOINT question sets (90 each,
    0 overlap). The real human baseline must come from the 'Full Scores'
    sheet (200 questions, question-level, both annotators scored the same
    items) and/or the pre-computed 'Reliability Statistics' sheet.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: HUMAN BASELINE (Full Scores Overlap, 200 Questions)")
    logger.info("=" * 60)

    # ── Load the overlap data (Full Scores sheet, question-level) ──
    df_full = load_full_scores_overlap()
    logger.info(f"Full Scores overlap: {len(df_full)} questions, both annotators")

    # Build two DataFrames with matching column names for evaluate_full()
    dims = ANNOTATION_DIMENSIONS
    df_kc = pd.DataFrame({dim: df_full[f"{dim}_KC"] for dim in dims})
    df_lh = pd.DataFrame({dim: df_full[f"{dim}_LH"] for dim in dims})

    human_results = evaluate_human_agreement(df_kc, df_lh)
    human_results["data_source"] = "Full Scores sheet (question-level, 200 items)"
    human_results["note"] = (
        "K and L CSVs are DISJOINT (90 different questions each). "
        "This baseline uses the Full Scores overlap sheet where both "
        "annotators scored the same 200 questions at question level."
    )

    # ── Also load pre-computed Reliability Statistics ──
    try:
        df_rel = load_reliability_statistics()
        # Extract Full Set row (the final, post-discussion values)
        rel_data = {}
        for idx, row in df_rel.iterrows():
            label = str(row.iloc[0]).strip()
            if label == "Full Set":
                # Next 3 rows: Percent Agreement, Unweighted Kappa, Weighted Kappa
                for offset in range(1, 4):
                    if idx + offset < len(df_rel):
                        metric_row = df_rel.iloc[idx + offset]
                        metric_name = str(metric_row.iloc[0]).strip()
                        vals = {}
                        for col in df_rel.columns[1:]:
                            v = metric_row[col]
                            if pd.notna(v):
                                vals[col] = float(v)
                        rel_data[metric_name] = vals
                break
        human_results["precomputed_reliability_statistics"] = rel_data
        if rel_data:
            logger.info("\nPre-computed Reliability Statistics (Full Set):")
            for metric, vals in rel_data.items():
                logger.info(f"  {metric}: {vals}")
    except Exception as e:
        logger.warning(f"Could not load Reliability Statistics: {e}")

    save_results([human_results], "human_baseline_results")

    logger.info("\nHuman baseline establishes the CEILING for LLM performance.")
    logger.info("LLM should aim to match or approach human inter-rater agreement.")

    return human_results


# ---------------------------------------------------------------------------
# Step 3: Condition 1 - Zero-Shot Experiment
# ---------------------------------------------------------------------------

def step_zero_shot(df_k: pd.DataFrame) -> pd.DataFrame:
    """Condition 1: Zero-shot LLM annotation (baseline).
    Brief rubric description, no examples."""
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION 1: ZERO-SHOT LLM ANNOTATION")
    logger.info("=" * 60)

    annotator = LLMAnnotator()
    df_llm = annotator.annotate_zero_shot(df_k, num_runs=3, save_tag="zero_shot")

    # Save cost summary
    cost = annotator.get_cost_summary()
    logger.info(f"Cost summary: {json.dumps(cost, indent=2)}")
    save_results([cost], "zero_shot_cost", subdir="zero_shot")

    return df_llm


# ---------------------------------------------------------------------------
# Step 4: Condition 2 - Rubric-Guided Experiment
# ---------------------------------------------------------------------------

def step_rubric_guided(df_k: pd.DataFrame) -> pd.DataFrame:
    """Condition 2: Rubric-guided LLM annotation.
    Detailed rubric with evaluation process, no examples."""
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION 2: RUBRIC-GUIDED LLM ANNOTATION")
    logger.info("=" * 60)

    annotator = LLMAnnotator()
    df_llm = annotator.annotate_rubric_guided(df_k, num_runs=3, save_tag="rubric_guided")

    cost = annotator.get_cost_summary()
    logger.info(f"Cost summary: {json.dumps(cost, indent=2)}")
    save_results([cost], "rubric_guided_cost", subdir="rubric_guided")

    return df_llm


# ---------------------------------------------------------------------------
# Step 5: Condition 3 - Rubric + Examples (Few-Shot) Experiment
# ---------------------------------------------------------------------------

def step_few_shot(df_k: pd.DataFrame, df_l: pd.DataFrame) -> pd.DataFrame:
    """Condition 3: Rubric + examples (few-shot) LLM annotation.
    Detailed rubric + stratified human-annotated examples."""
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION 3: RUBRIC + EXAMPLES (FEW-SHOT) LLM ANNOTATION")
    logger.info("=" * 60)

    # Select examples
    examples = select_few_shot_examples(df_k, df_l, n_examples=8, seed=42)
    validate_examples(examples)

    # Save selected examples for reproducibility
    save_results(examples, "few_shot_selected_examples", subdir="few_shot")

    # Get indices to exclude from evaluation (avoid data leakage)
    example_indices = {ex["index"] for ex in examples}
    df_eval = df_k.drop(index=[i for i in example_indices if i in df_k.index])
    logger.info(
        f"Evaluating on {len(df_eval)} items "
        f"(excluded {len(example_indices)} few-shot examples)"
    )

    annotator = LLMAnnotator()
    df_llm = annotator.annotate_few_shot(
        df_eval, examples=examples, num_runs=3, save_tag="few_shot"
    )

    cost = annotator.get_cost_summary()
    logger.info(f"Cost summary: {json.dumps(cost, indent=2)}")
    save_results([cost], "few_shot_cost", subdir="few_shot")

    return df_llm


# ---------------------------------------------------------------------------
# Step 6: Evaluation (All 3 Conditions)
# ---------------------------------------------------------------------------

def step_evaluate(
    df_k: pd.DataFrame,
    df_l: pd.DataFrame,
    df_zero_shot: pd.DataFrame,
    df_rubric_guided: pd.DataFrame,
    df_few_shot: pd.DataFrame,
    human_results: dict,
) -> list[dict]:
    """Evaluate all three conditions and generate cross-condition comparison."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: FULL EVALUATION (3-CONDITION COMPARISON)")
    logger.info("=" * 60)

    all_results = [human_results]

    # NOTE: K and L annotated DISJOINT question sets (90 each, 0 overlap).
    # LLM was run on Katerina's 360 items, so only LLM vs Katerina is valid.
    # LLM vs Linh comparisons are REMOVED (they compared misaligned data).

    # Condition 1: Zero-shot vs Katerina
    if df_zero_shot is not None and len(df_zero_shot) > 0:
        zs_vs_k = evaluate_full(
            df_k, df_zero_shot,
            condition_name="C1_zero_shot_vs_katerina",
            confidences=df_zero_shot.get("confidence"),
        )
        all_results.append(zs_vs_k)

    # Condition 2: Rubric-guided vs Katerina
    if df_rubric_guided is not None and len(df_rubric_guided) > 0:
        rg_vs_k = evaluate_full(
            df_k, df_rubric_guided,
            condition_name="C2_rubric_guided_vs_katerina",
            confidences=df_rubric_guided.get("confidence"),
        )
        all_results.append(rg_vs_k)

    # Condition 3: Rubric + examples (few-shot) vs Katerina
    if df_few_shot is not None and len(df_few_shot) > 0:
        fs_vs_k = evaluate_full(
            df_k, df_few_shot,
            condition_name="C3_rubric_examples_vs_katerina",
            confidences=df_few_shot.get("confidence"),
        )
        all_results.append(fs_vs_k)

    # Print comparison table
    table = print_comparison_table(all_results)

    # Cross-condition comparison summary
    condition_summary = _build_condition_comparison(
        all_results, df_zero_shot, df_rubric_guided, df_few_shot
    )
    save_results([condition_summary], "condition_comparison_summary")

    # Save full results
    save_results(all_results, "phase1_full_evaluation")

    # Save table as text
    table_path = RESULTS_DIR / "phase1_comparison_table.txt"
    table_path.write_text(table, encoding="utf-8")
    logger.info(f"Saved comparison table: {table_path}")

    return all_results


def _build_condition_comparison(
    all_results: list[dict],
    df_zero_shot: pd.DataFrame,
    df_rubric_guided: pd.DataFrame,
    df_few_shot: pd.DataFrame,
) -> dict:
    """Build a cross-condition comparison summary for the AIME paper."""
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-CONDITION COMPARISON (AIME Framework)")
    logger.info("=" * 60)

    conditions = {
        "C1_zero_shot": {"df": df_zero_shot, "prefix": "C1_zero_shot"},
        "C2_rubric_guided": {"df": df_rubric_guided, "prefix": "C2_rubric_guided"},
        "C3_rubric_examples": {"df": df_few_shot, "prefix": "C3_rubric_examples"},
    }

    summary = {"timestamp": datetime.now().isoformat(), "conditions": {}}

    for cond_name, cond_info in conditions.items():
        # Find matching results (vs_katerina as primary reference)
        matching = [
            r for r in all_results
            if r.get("condition", "").startswith(cond_info["prefix"])
            and "katerina" in r.get("condition", "")
        ]
        if not matching:
            continue

        result = matching[0]
        agg = result.get("aggregate", {})

        cond_summary = {
            "mean_kappa": agg.get("mean_kappa"),
            "mean_alpha": agg.get("mean_alpha"),
            "mean_exact_match_pct": agg.get("mean_exact_match_pct"),
            "dimensions": {},
        }

        # Per-dimension breakdown
        for dim, metrics in result.get("dimensions", {}).items():
            cond_summary["dimensions"][dim] = {
                "kappa": metrics.get("cohens_kappa", {}).get("kappa"),
                "alpha": metrics.get("krippendorffs_alpha", {}).get("alpha"),
                "exact_match_pct": metrics.get("exact_match", {}).get("exact_match_pct"),
            }

        # Self-consistency (from LLM output)
        df_llm = cond_info["df"]
        if df_llm is not None and "overall_consistency" in df_llm.columns:
            cond_summary["mean_self_consistency"] = round(
                float(df_llm["overall_consistency"].mean()), 4
            )

        summary["conditions"][cond_name] = cond_summary

    # Log the comparison
    logger.info("\n{:<25s} | {:>10s} | {:>10s} | {:>10s}".format(
        "Condition", "Kappa", "Alpha", "Exact%"
    ))
    logger.info("-" * 65)
    for cond_name, cond_data in summary["conditions"].items():
        logger.info("{:<25s} | {:>10.4f} | {:>10.4f} | {:>9.1f}%".format(
            cond_name,
            cond_data.get("mean_kappa", float("nan")),
            cond_data.get("mean_alpha", float("nan")),
            cond_data.get("mean_exact_match_pct", float("nan")),
        ))

    return summary


# ---------------------------------------------------------------------------
# Step 7: Dawid-Skene Aggregation
# ---------------------------------------------------------------------------

def step_aggregate(
    df_k: pd.DataFrame,
    df_l: pd.DataFrame,
    df_llm: pd.DataFrame,
) -> dict:
    """Run Dawid-Skene aggregation pilot."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: DAWID-SKENE AGGREGATION PILOT")
    logger.info("=" * 60)

    ds_results = run_dawid_skene(df_k, df_l, df_llm)
    mv_results = run_majority_vote(df_k, df_l, df_llm)
    comparison = compare_aggregation_methods(ds_results, mv_results)

    save_results(
        [{"dawid_skene": ds_results, "majority_vote": mv_results, "comparison": comparison}],
        "aggregation_results",
        subdir="aggregation",
    )

    return {"dawid_skene": ds_results, "comparison": comparison}


# ---------------------------------------------------------------------------
# Step 8: Calibration & Visualization
# ---------------------------------------------------------------------------

def step_calibrate(
    df_k: pd.DataFrame,
    df_llm: pd.DataFrame,
    all_results: list[dict],
) -> None:
    """Compute calibration metrics and generate plots."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: CALIBRATION & VISUALIZATION")
    logger.info("=" * 60)

    # Self-consistency analysis
    if df_llm is not None:
        consistency = analyze_self_consistency(df_llm)
        logger.info(f"Self-consistency: {json.dumps(consistency, indent=2, default=str)}")
        save_results([consistency], "self_consistency_analysis")

        # Calibration curves per dimension
        if "confidence" in df_llm.columns:
            for dim in ANNOTATION_DIMENSIONS:
                if dim in df_k.columns and dim in df_llm.columns:
                    human_vals = df_k[dim].dropna().values[:len(df_llm)]
                    llm_vals = df_llm[dim].dropna().values
                    conf_vals = df_llm["confidence"].values

                    min_len = min(len(human_vals), len(llm_vals), len(conf_vals))
                    if min_len > 10:
                        cal_data = compute_calibration_curve(
                            human_vals[:min_len].astype(int),
                            llm_vals[:min_len].astype(int),
                            conf_vals[:min_len],
                        )
                        plot_reliability_diagram(cal_data, dim)

    # Dimension comparison plots
    if all_results:
        plot_dimension_comparison(all_results, metric="kappa")
        plot_dimension_comparison(all_results, metric="exact_match")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: LLM Annotation Replication Study")
    parser.add_argument(
        "--step",
        choices=["explore", "baseline", "zero_shot", "rubric_guided", "few_shot", "evaluate", "aggregate", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Phase 1 started at {timestamp}")
    logger.info(f"Running step: {args.step}")

    # Initialize metadata logger
    meta = get_metadata_logger(run_id=timestamp)
    meta.log_environment()
    meta.log_provenance("phase", "phase_1")
    meta.log_provenance("step", args.step)
    meta.log_provenance("framework", "AIME 3-Condition")
    meta.log_provenance("conditions", {
        "C1": "Zero-shot (brief rubric, no examples)",
        "C2": "Rubric-guided (detailed rubric, no examples)",
        "C3": "Rubric + examples (detailed rubric + few-shot examples)",
    })
    meta.start_timer("phase1_total")

    # Always load data first
    meta.start_timer("step_explore")
    df_k, df_l = step_explore()
    meta.stop_timer("step_explore")
    meta.log_data("katerina_raw", df_k)
    meta.log_data("linh_raw", df_l)

    if args.step == "explore":
        return

    # Human baseline
    meta.start_timer("step_baseline")
    human_results = step_baseline(df_k, df_l)
    meta.stop_timer("step_baseline")
    if args.step == "baseline":
        meta.save()
        return

    # Condition 1: Zero-shot
    df_zero_shot = None
    if args.step in ("zero_shot", "all"):
        df_zero_shot = step_zero_shot(df_k)
        if args.step == "zero_shot":
            meta.save()
            return

    # Condition 2: Rubric-guided
    df_rubric_guided = None
    if args.step in ("rubric_guided", "all"):
        df_rubric_guided = step_rubric_guided(df_k)
        if args.step == "rubric_guided":
            meta.save()
            return

    # Condition 3: Rubric + examples (few-shot)
    df_few_shot = None
    if args.step in ("few_shot", "all"):
        df_few_shot = step_few_shot(df_k, df_l)
        if args.step == "few_shot":
            meta.save()
            return

    # Load from saved results if running evaluate/aggregate standalone
    if args.step in ("evaluate", "aggregate"):
        zs_path = RESULTS_DIR / "zero_shot" / "zero_shot_results.csv"
        rg_path = RESULTS_DIR / "rubric_guided" / "rubric_guided_results.csv"
        fs_path = RESULTS_DIR / "few_shot" / "few_shot_results.csv"
        if zs_path.exists() and df_zero_shot is None:
            df_zero_shot = pd.read_csv(zs_path)
        if rg_path.exists() and df_rubric_guided is None:
            df_rubric_guided = pd.read_csv(rg_path)
        if fs_path.exists() and df_few_shot is None:
            df_few_shot = pd.read_csv(fs_path)

    # Evaluate all 3 conditions
    all_results = step_evaluate(
        df_k, df_l, df_zero_shot, df_rubric_guided, df_few_shot, human_results
    )

    if args.step == "evaluate":
        meta.save()
        return

    # Aggregation (use best-performing condition's results)
    df_llm_for_agg = None
    if df_few_shot is not None:
        df_llm_for_agg = df_few_shot
    elif df_rubric_guided is not None:
        df_llm_for_agg = df_rubric_guided
    elif df_zero_shot is not None:
        df_llm_for_agg = df_zero_shot
    
    if df_llm_for_agg is not None and args.step in ("aggregate", "all"):
        step_aggregate(df_k, df_l, df_llm_for_agg)

    # Calibration & plots
    if args.step == "all":
        step_calibrate(df_k, df_llm_for_agg, all_results)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("Conditions: C1=Zero-shot, C2=Rubric-guided, C3=Rubric+Examples")

    # Save all metadata
    meta.stop_timer("phase1_total")
    meta_path = meta.save()
    logger.info(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
