"""
Dawid-Skene Aggregation Module.
Treats LLM and human annotators as peers with individual bias matrices.
Uses crowd-kit library for aggregation.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import ANNOTATION_DIMENSIONS, setup_logging, save_results
from src.metadata import get_metadata_logger

logger = setup_logging()


def prepare_crowd_labels(
    df_katerina: pd.DataFrame,
    df_linh: pd.DataFrame,
    df_llm: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """
    Prepare data in crowd-kit format:
    DataFrame with columns: [worker, task, label]

    Each annotator (Katerina, Linh, LLM) is a 'worker'.
    Each item (row) is a 'task'.
    """
    records = []

    for idx in range(len(df_katerina)):
        task_id = idx

        # Katerina
        val_k = df_katerina.iloc[idx].get(dimension)
        if pd.notna(val_k) and val_k not in (999, "NA"):
            records.append({
                "worker": "katerina",
                "task": task_id,
                "label": int(float(val_k)),
            })

        # Linh
        if idx < len(df_linh):
            val_l = df_linh.iloc[idx].get(dimension)
            if pd.notna(val_l) and val_l not in (999, "NA"):
                records.append({
                    "worker": "linh",
                    "task": task_id,
                    "label": int(float(val_l)),
                })

        # LLM
        if idx < len(df_llm):
            val_llm = df_llm.iloc[idx].get(dimension)
            if pd.notna(val_llm):
                records.append({
                    "worker": "llm",
                    "task": task_id,
                    "label": int(float(val_llm)),
                })

    return pd.DataFrame(records)


def run_dawid_skene(
    df_katerina: pd.DataFrame,
    df_linh: pd.DataFrame,
    df_llm: pd.DataFrame,
    dimensions: Optional[list[str]] = None,
    n_iter: int = 100,
) -> dict:
    """
    Run Dawid-Skene aggregation across specified dimensions.

    Returns dict with:
    - aggregated labels per dimension
    - error matrices per annotator
    - comparison with individual annotators
    """
    try:
        from crowdkit.aggregation import DawidSkene
    except ImportError:
        logger.error(
            "crowd-kit not installed. Run: pip install crowd-kit"
        )
        return {"error": "crowd-kit not installed"}

    if dimensions is None:
        dimensions = ANNOTATION_DIMENSIONS

    # Log aggregation metadata
    meta = get_metadata_logger()
    meta.start_timer("aggregation_dawid_skene")
    meta.log_provenance("aggregation_method", "Dawid-Skene")
    meta.log_provenance("aggregation_n_iter", n_iter)
    meta.log_provenance("aggregation_workers", ["katerina", "linh", "llm"])

    results = {}

    for dim in dimensions:
        logger.info(f"Running Dawid-Skene for dimension: {dim}")

        crowd_df = prepare_crowd_labels(df_katerina, df_linh, df_llm, dim)

        if crowd_df.empty:
            logger.warning(f"No valid labels for {dim}, skipping.")
            results[dim] = {"error": "no valid labels"}
            continue

        # Run Dawid-Skene
        ds = DawidSkene(n_iter=n_iter)
        ds.fit(crowd_df)

        # Get aggregated labels
        agg_labels = ds.labels_

        # Get error matrices per worker
        error_matrices = {}
        if hasattr(ds, "errors_"):
            for worker in ["katerina", "linh", "llm"]:
                if worker in ds.errors_:
                    error_matrices[worker] = ds.errors_[worker].to_dict()

        # Compare each annotator against Dawid-Skene consensus
        agreements = {}
        for worker_name, df_worker in [
            ("katerina", df_katerina),
            ("linh", df_linh),
            ("llm", df_llm),
        ]:
            worker_labels = crowd_df[crowd_df["worker"] == worker_name].set_index("task")["label"]
            common_tasks = agg_labels.index.intersection(worker_labels.index)
            if len(common_tasks) > 0:
                agree = (
                    agg_labels.loc[common_tasks] == worker_labels.loc[common_tasks]
                ).mean()
                agreements[worker_name] = round(float(agree), 4)
            else:
                agreements[worker_name] = np.nan

        results[dim] = {
            "aggregated_labels": agg_labels.to_dict(),
            "error_matrices": error_matrices,
            "agreement_with_consensus": agreements,
            "num_tasks": len(agg_labels),
            "num_records": len(crowd_df),
        }

        logger.info(
            f"  {dim}: Consensus tasks={len(agg_labels)} | "
            f"Agreement: K={agreements.get('katerina', 'N/A')}, "
            f"L={agreements.get('linh', 'N/A')}, "
            f"LLM={agreements.get('llm', 'N/A')}"
        )

    meta.stop_timer("aggregation_dawid_skene")
    return results


def run_majority_vote(
    df_katerina: pd.DataFrame,
    df_linh: pd.DataFrame,
    df_llm: pd.DataFrame,
    dimensions: Optional[list[str]] = None,
) -> dict:
    """
    Simple majority vote aggregation for comparison with Dawid-Skene.
    """
    try:
        from crowdkit.aggregation import MajorityVote
    except ImportError:
        logger.warning("crowd-kit not installed, using manual majority vote.")
        return _manual_majority_vote(df_katerina, df_linh, df_llm, dimensions)

    if dimensions is None:
        dimensions = ANNOTATION_DIMENSIONS

    results = {}

    for dim in dimensions:
        crowd_df = prepare_crowd_labels(df_katerina, df_linh, df_llm, dim)
        if crowd_df.empty:
            continue

        mv = MajorityVote()
        mv.fit(crowd_df)
        agg_labels = mv.labels_

        results[dim] = {
            "aggregated_labels": agg_labels.to_dict(),
            "num_tasks": len(agg_labels),
        }

    return results


def _manual_majority_vote(
    df_katerina: pd.DataFrame,
    df_linh: pd.DataFrame,
    df_llm: pd.DataFrame,
    dimensions: Optional[list[str]] = None,
) -> dict:
    """Fallback majority vote without crowd-kit."""
    if dimensions is None:
        dimensions = ANNOTATION_DIMENSIONS

    results = {}
    for dim in dimensions:
        labels = {}
        for idx in range(len(df_katerina)):
            votes = []
            for df in [df_katerina, df_linh, df_llm]:
                if idx < len(df):
                    val = df.iloc[idx].get(dim)
                    if pd.notna(val) and val not in (999, "NA"):
                        votes.append(int(float(val)))
            if votes:
                from statistics import mode, StatisticsError
                try:
                    labels[idx] = mode(votes)
                except StatisticsError:
                    labels[idx] = round(np.mean(votes))

        results[dim] = {
            "aggregated_labels": labels,
            "num_tasks": len(labels),
        }

    return results


def compare_aggregation_methods(
    ds_results: dict,
    mv_results: dict,
    gold_standard: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compare Dawid-Skene vs Majority Vote vs individual annotators.
    If gold standard is provided, compute accuracy against it.
    """
    comparison = {}

    for dim in ANNOTATION_DIMENSIONS:
        if dim not in ds_results or dim not in mv_results:
            continue

        ds_labels = ds_results[dim].get("aggregated_labels", {})
        mv_labels = mv_results[dim].get("aggregated_labels", {})

        # Agreement between DS and MV
        common = set(ds_labels.keys()) & set(mv_labels.keys())
        if common:
            agree = sum(
                1 for t in common if ds_labels.get(t) == mv_labels.get(t)
            ) / len(common)
        else:
            agree = np.nan

        comparison[dim] = {
            "ds_mv_agreement": round(float(agree), 4) if not np.isnan(agree) else None,
            "num_common_tasks": len(common),
        }

        # If gold standard provided
        if gold_standard is not None and dim in gold_standard.columns:
            for method_name, method_labels in [("dawid_skene", ds_labels), ("majority_vote", mv_labels)]:
                correct = 0
                total = 0
                for task_id, label in method_labels.items():
                    if task_id < len(gold_standard):
                        gold_val = gold_standard.iloc[task_id].get(dim)
                        if pd.notna(gold_val) and gold_val not in (999, "NA"):
                            total += 1
                            if int(float(gold_val)) == label:
                                correct += 1
                if total > 0:
                    comparison[dim][f"{method_name}_accuracy"] = round(correct / total, 4)

    return comparison
