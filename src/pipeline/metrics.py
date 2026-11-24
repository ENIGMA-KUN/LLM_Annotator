"""
Metrics: core inter-rater agreement and evaluation functions.

Canonical location for all metric computations.  The legacy
``src/evaluator.py`` re-exports from here for backward compatibility.

Functions:
    compute_cohens_kappa          Weighted Cohen's Kappa
    compute_krippendorff_alpha    Krippendorff Alpha (ordinal)
    compute_exact_match           Exact-match percentage
    compute_within_one            Within ±1 percentage
    compute_per_dimension_f1      Macro + per-class F1
    compute_confusion_matrix      4×4 confusion matrix
    compute_expected_calibration_error  ECE
    evaluate_full                 Full suite across all dimensions
    evaluate_human_agreement      Human–human baseline
    print_comparison_table        Multi-condition comparison table
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import krippendorff

from src.utils import (
    ANNOTATION_DIMENSIONS,
    get_valid_pairs,
    save_results,
    setup_logging,
)
from src.metadata import get_metadata_logger

logger = setup_logging()


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    human_scores: pd.Series,
    llm_scores: pd.Series,
    weights: str = "quadratic",
) -> dict:
    """
    Compute Cohen's Kappa (weighted) between human and LLM annotations.

    Args:
        weights: 'linear', 'quadratic', or None for unweighted

    Returns:
        dict with kappa, interpretation, and sample size
    """
    h = human_scores.dropna().astype(int)
    l = llm_scores.dropna().astype(int)

    # Align
    mask = h.index.intersection(l.index)
    h, l = h.loc[mask], l.loc[mask]

    if len(h) < 2:
        return {"kappa": np.nan, "interpretation": "insufficient data", "n": len(h)}

    kappa = cohen_kappa_score(h, l, weights=weights, labels=[1, 2, 3, 4])

    return {
        "kappa": round(kappa, 4),
        "interpretation": _interpret_kappa(kappa),
        "n": len(h),
        "weights": weights,
    }


def compute_krippendorff_alpha(
    human_scores: pd.Series,
    llm_scores: pd.Series,
    level: str = "ordinal",
) -> dict:
    """
    Compute Krippendorff's alpha for inter-rater reliability.
    """
    h = human_scores.dropna().values
    l = llm_scores.dropna().values
    min_len = min(len(h), len(l))
    h, l = h[:min_len], l[:min_len]

    if min_len < 2:
        return {"alpha": np.nan, "n": min_len}

    # krippendorff expects a reliability data matrix (raters x items)
    reliability_data = np.array([h, l], dtype=float)
    # Replace any remaining NaN
    reliability_data[np.isnan(reliability_data)] = np.nan

    alpha = krippendorff.alpha(
        reliability_data=reliability_data,
        level_of_measurement=level,
    )

    return {
        "alpha": round(alpha, 4),
        "interpretation": _interpret_kappa(alpha),  # same scale
        "n": min_len,
        "level": level,
    }


def compute_krippendorff_alpha_multi(
    *rater_scores: np.ndarray,
    level: str = "ordinal",
) -> dict:
    """
    Compute Krippendorff Alpha for 2+ raters.

    Handles NaN correctly: raters with NaN on an item are treated as
    missing data (not as a category).  This is critical for the
    three-rater pipeline where Rater 3 (LLM) only has values for
    Tier A dimensions.

    Args:
        *rater_scores: Variable number of 1-D arrays (one per rater).
        level: 'ordinal', 'nominal', 'interval', 'ratio'.

    Returns:
        dict with alpha, n_items, n_raters.
    """
    if len(rater_scores) < 2:
        return {"alpha": np.nan, "n_items": 0, "n_raters": len(rater_scores)}

    reliability_data = np.array(rater_scores, dtype=float)
    # Items where ALL raters are NaN are useless
    valid_cols = ~np.all(np.isnan(reliability_data), axis=0)
    reliability_data = reliability_data[:, valid_cols]

    n_items = reliability_data.shape[1] if reliability_data.ndim == 2 else 0
    if n_items < 2:
        return {"alpha": np.nan, "n_items": n_items, "n_raters": len(rater_scores)}

    alpha = krippendorff.alpha(
        reliability_data=reliability_data,
        level_of_measurement=level,
    )

    return {
        "alpha": round(alpha, 4),
        "interpretation": _interpret_kappa(alpha),
        "n_items": n_items,
        "n_raters": len(rater_scores),
        "level": level,
    }


def compute_exact_match(
    human_scores: pd.Series,
    llm_scores: pd.Series,
) -> dict:
    """Compute percentage of exact matches."""
    h = human_scores.dropna().astype(int)
    l = llm_scores.dropna().astype(int)
    mask = h.index.intersection(l.index)
    h, l = h.loc[mask], l.loc[mask]

    if len(h) == 0:
        return {"exact_match_pct": np.nan, "n": 0}

    matches = (h == l).sum()
    return {
        "exact_match_pct": round(matches / len(h) * 100, 2),
        "exact_matches": int(matches),
        "n": len(h),
    }


def compute_within_one(
    human_scores: pd.Series,
    llm_scores: pd.Series,
) -> dict:
    """Compute percentage of predictions within +/- 1 of ground truth."""
    h = human_scores.dropna().astype(int)
    l = llm_scores.dropna().astype(int)
    mask = h.index.intersection(l.index)
    h, l = h.loc[mask], l.loc[mask]

    if len(h) == 0:
        return {"within_one_pct": np.nan, "n": 0}

    within = (abs(h - l) <= 1).sum()
    return {
        "within_one_pct": round(within / len(h) * 100, 2),
        "n": len(h),
    }


def compute_per_dimension_f1(
    human_scores: pd.Series,
    llm_scores: pd.Series,
) -> dict:
    """Compute macro and per-class F1 scores."""
    h = human_scores.dropna().astype(int)
    l = llm_scores.dropna().astype(int)
    mask = h.index.intersection(l.index)
    h, l = h.loc[mask], l.loc[mask]

    if len(h) == 0:
        return {"macro_f1": np.nan, "n": 0}

    labels = [1, 2, 3, 4]
    macro_f1 = f1_score(h, l, labels=labels, average="macro", zero_division=0)
    per_class = f1_score(h, l, labels=labels, average=None, zero_division=0)

    return {
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {str(label): round(f, 4) for label, f in zip(labels, per_class)},
        "n": len(h),
    }


def compute_confusion_matrix(
    human_scores: pd.Series,
    llm_scores: pd.Series,
) -> np.ndarray:
    """Compute confusion matrix (human x LLM)."""
    h = human_scores.dropna().astype(int)
    l = llm_scores.dropna().astype(int)
    mask = h.index.intersection(l.index)
    h, l = h.loc[mask], l.loc[mask]
    return confusion_matrix(h, l, labels=[1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Confidence Calibration
# ---------------------------------------------------------------------------

def compute_expected_calibration_error(
    human_scores: pd.Series,
    llm_scores: pd.Series,
    confidences: pd.Series,
    n_bins: int = 10,
) -> dict:
    """
    Compute Expected Calibration Error (ECE).
    Measures how well LLM confidence predicts actual accuracy.
    """
    h = human_scores.dropna().astype(int).values
    l = llm_scores.dropna().astype(int).values
    c = confidences.values / 100.0  # normalize to 0-1

    min_len = min(len(h), len(l), len(c))
    h, l, c = h[:min_len], l[:min_len], c[:min_len]

    if min_len == 0:
        return {"ece": np.nan, "n": 0}

    correct = (h == l).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        mask = (c >= bin_boundaries[i]) & (c < bin_boundaries[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = (c >= bin_boundaries[i]) & (c <= bin_boundaries[i + 1])

        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = c[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / min_len) * abs(bin_acc - bin_conf)
            bin_details.append({
                "bin": f"{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}",
                "accuracy": round(float(bin_acc), 4),
                "confidence": round(float(bin_conf), 4),
                "count": int(bin_size),
            })

    return {
        "ece": round(float(ece), 4),
        "n": min_len,
        "bin_details": bin_details,
    }


# ---------------------------------------------------------------------------
# Full Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_full(
    df_human: pd.DataFrame,
    df_llm: pd.DataFrame,
    condition_name: str = "experiment",
    confidences: Optional[pd.Series] = None,
) -> dict:
    """
    Run full evaluation suite across all dimensions.

    Returns a dict of metrics per dimension.
    """
    results = {
        "condition": condition_name,
        "dimensions": {},
        "aggregate": {},
    }

    all_kappas = []
    all_alphas = []
    all_exact = []

    for dim in ANNOTATION_DIMENSIONS:
        if dim not in df_human.columns or dim not in df_llm.columns:
            logger.warning(f"Dimension {dim} not found in data, skipping.")
            continue

        human_scores, llm_scores = get_valid_pairs(df_human, df_llm, dim)

        if len(human_scores) == 0:
            logger.warning(f"No valid pairs for {dim}, skipping.")
            continue

        kappa = compute_cohens_kappa(human_scores, llm_scores)
        alpha = compute_krippendorff_alpha(human_scores, llm_scores)
        exact = compute_exact_match(human_scores, llm_scores)
        within1 = compute_within_one(human_scores, llm_scores)
        f1 = compute_per_dimension_f1(human_scores, llm_scores)
        cm = compute_confusion_matrix(human_scores, llm_scores)

        dim_result = {
            "cohens_kappa": kappa,
            "krippendorffs_alpha": alpha,
            "exact_match": exact,
            "within_one": within1,
            "f1_scores": f1,
            "confusion_matrix": cm.tolist(),
        }

        # Add ECE if confidences provided
        if confidences is not None:
            ece = compute_expected_calibration_error(
                human_scores, llm_scores, confidences
            )
            dim_result["calibration"] = ece

        results["dimensions"][dim] = dim_result

        if not np.isnan(kappa["kappa"]):
            all_kappas.append(kappa["kappa"])
        if not np.isnan(alpha["alpha"]):
            all_alphas.append(alpha["alpha"])
        if not np.isnan(exact["exact_match_pct"]):
            all_exact.append(exact["exact_match_pct"])

    # Aggregate metrics
    results["aggregate"] = {
        "mean_kappa": round(np.mean(all_kappas), 4) if all_kappas else np.nan,
        "mean_alpha": round(np.mean(all_alphas), 4) if all_alphas else np.nan,
        "mean_exact_match_pct": round(np.mean(all_exact), 2) if all_exact else np.nan,
        "num_dimensions_evaluated": len(all_kappas),
    }

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION RESULTS: {condition_name}")
    logger.info(f"{'='*60}")
    for dim, metrics in results["dimensions"].items():
        k = metrics["cohens_kappa"]["kappa"]
        em = metrics["exact_match"]["exact_match_pct"]
        n = metrics["cohens_kappa"]["n"]
        logger.info(f"  {dim:25s} | Kappa={k:.4f} | Exact={em:.1f}% | n={n}")
    logger.info(f"{'='*60}")
    logger.info(f"  AGGREGATE Mean Kappa: {results['aggregate']['mean_kappa']}")
    logger.info(f"  AGGREGATE Mean Exact Match: {results['aggregate']['mean_exact_match_pct']}%")

    # Log evaluation metadata
    meta = get_metadata_logger()
    # Infer human annotator name from condition_name
    human_annotator = "unknown"
    for name in ["katerina", "linh", "human_baseline"]:
        if name in condition_name.lower():
            human_annotator = name
            break
    n_items = max(
        (m["cohens_kappa"]["n"] for m in results["dimensions"].values()),
        default=0,
    )
    meta.log_evaluation(
        condition_name=condition_name,
        human_annotator=human_annotator,
        n_items=n_items,
        metrics_summary=results["aggregate"],
        extra={
            "dimensions_evaluated": list(results["dimensions"].keys()),
            "per_dimension_kappa": {
                dim: m["cohens_kappa"]["kappa"]
                for dim, m in results["dimensions"].items()
            },
        },
    )

    return results


def evaluate_human_agreement(
    df_katerina: pd.DataFrame,
    df_linh: pd.DataFrame,
) -> dict:
    """
    Compute inter-rater agreement between the two human annotators.
    This establishes the baseline ceiling for LLM performance.
    """
    return evaluate_full(df_katerina, df_linh, condition_name="human_baseline")


def print_comparison_table(results_list: list[dict]) -> str:
    """
    Format multiple evaluation results into a comparison table.
    """
    lines = []
    header = f"{'Condition':<20} | {'Dimension':<25} | {'Kappa':>8} | {'Alpha':>8} | {'Exact%':>8} | {'n':>5}"
    lines.append(header)
    lines.append("-" * len(header))

    for result in results_list:
        cond = result["condition"]
        for dim, metrics in result["dimensions"].items():
            k = metrics["cohens_kappa"]["kappa"]
            a = metrics["krippendorffs_alpha"]["alpha"]
            em = metrics["exact_match"]["exact_match_pct"]
            n = metrics["cohens_kappa"]["n"]
            lines.append(
                f"{cond:<20} | {dim:<25} | {k:>8.4f} | {a:>8.4f} | {em:>7.1f}% | {n:>5}"
            )
        # Aggregate row
        agg = result["aggregate"]
        lines.append(
            f"{cond:<20} | {'** AGGREGATE **':<25} | "
            f"{agg['mean_kappa']:>8.4f} | {agg['mean_alpha']:>8.4f} | "
            f"{agg['mean_exact_match_pct']:>7.1f}% |"
        )
        lines.append("")

    table = "\n".join(lines)
    logger.info(f"\n{table}")
    return table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value using Landis & Koch scale."""
    if np.isnan(kappa):
        return "undefined"
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
