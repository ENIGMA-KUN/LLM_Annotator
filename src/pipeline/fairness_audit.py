"""
Fairness Audit: subgroup analysis of LLM annotation quality.

Tests whether LLM annotation quality varies systematically across:
    1. Textbook source (Sociology, History, Anthropology, Lifespan)
    2. Item difficulty (easy vs hard based on human consensus)
    3. Human disagreement level (agreed vs disputed items)
    4. Score level (items scored 1-2 vs 3-4 by humans)

Flags dimensions where subgroup alpha differs by > 0.15 as a fairness concern.
"""

import numpy as np
import pandas as pd

from src.pipeline.metrics import compute_krippendorff_alpha_multi
from src.pipeline.dimension_router import ALL_PIPELINE_DIMENSIONS
from src.utils import setup_logging

logger = setup_logging()

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

FAIRNESS_THRESHOLD = 0.15


def run_fairness_audit(
    scores: dict,
    human_df: pd.DataFrame,
    dimension: str,
) -> dict:
    """
    Run complete fairness audit for one dimension.

    Args:
        scores: dict of uuid -> {dim: [run1, run2, run3]}
        human_df: DataFrame with human scores and metadata
        dimension: which dimension to audit

    Returns:
        dict with per-subgroup alpha, flags, and summary
    """
    hd = HUMAN_DIM_MAP[dimension]
    results = {}

    # Build aligned arrays
    items = []
    for _, row in human_df.iterrows():
        uuid = str(row["question_uuid"])
        if uuid not in scores or dimension not in scores[uuid]:
            continue
        runs = [s for s in scores[uuid][dimension] if not (isinstance(s, float) and np.isnan(s))]
        if not runs:
            continue

        k_score = row.get(f"{hd}{K_SUFFIX}")
        l_score = row.get(f"{hd}{L_SUFFIX}")
        if pd.isna(k_score) or pd.isna(l_score):
            continue

        consensus = round((float(k_score) + float(l_score)) / 2)
        items.append({
            "uuid": uuid,
            "llm_median": float(np.median(runs)),
            "consensus": consensus,
            "k_score": float(k_score),
            "l_score": float(l_score),
            "human_agree": int(k_score) == int(l_score),
            "subject": str(row.get("textID", "")).split("_")[0] if "textID" in row else "unknown",
            "text_len": len(str(row.get("text", ""))),
        })

    if len(items) < 10:
        return {"error": "insufficient_data", "n": len(items)}

    df = pd.DataFrame(items)

    # 1. By textbook subject
    results["by_subject"] = _subgroup_alpha(df, "subject", dimension)

    # 2. By item difficulty (consensus score level)
    df["difficulty"] = df["consensus"].apply(lambda x: "easy (3-4)" if x >= 3 else "hard (1-2)")
    results["by_difficulty"] = _subgroup_alpha(df, "difficulty", dimension)

    # 3. By human agreement
    df["human_agreement"] = df["human_agree"].apply(lambda x: "agreed" if x else "disputed")
    results["by_human_agreement"] = _subgroup_alpha(df, "human_agreement", dimension)

    # 4. By text length (short vs long)
    median_len = df["text_len"].median()
    df["text_length_group"] = df["text_len"].apply(lambda x: "short" if x < median_len else "long")
    results["by_text_length"] = _subgroup_alpha(df, "text_length_group", dimension)

    # Fairness flags
    flags = []
    for analysis_name, analysis in results.items():
        if isinstance(analysis, dict) and "subgroups" in analysis:
            alphas = [sg["alpha"] for sg in analysis["subgroups"].values() if not np.isnan(sg["alpha"])]
            if len(alphas) >= 2:
                gap = max(alphas) - min(alphas)
                if gap > FAIRNESS_THRESHOLD:
                    flags.append({
                        "analysis": analysis_name,
                        "gap": round(gap, 4),
                        "severity": "HIGH" if gap > 2 * FAIRNESS_THRESHOLD else "MEDIUM",
                    })

    results["flags"] = flags
    results["dimension"] = dimension
    results["n_items"] = len(df)

    return results


def _subgroup_alpha(df: pd.DataFrame, group_col: str, dimension: str) -> dict:
    """Compute alpha per subgroup."""
    subgroups = {}
    for group_name, group_df in df.groupby(group_col):
        n = len(group_df)
        if n < 5:
            subgroups[str(group_name)] = {"alpha": np.nan, "n": n, "bias": np.nan}
            continue

        llm_arr = group_df["llm_median"].values
        cons_arr = group_df["consensus"].values
        alpha = compute_krippendorff_alpha_multi(llm_arr, cons_arr)["alpha"]
        bias = float((llm_arr - cons_arr).mean())

        subgroups[str(group_name)] = {
            "alpha": round(alpha, 4),
            "n": n,
            "bias": round(bias, 3),
            "mean_llm": round(float(llm_arr.mean()), 3),
            "mean_human": round(float(cons_arr.mean()), 3),
        }

    overall_alphas = [sg["alpha"] for sg in subgroups.values() if not np.isnan(sg["alpha"])]
    return {
        "subgroups": subgroups,
        "max_gap": round(max(overall_alphas) - min(overall_alphas), 4) if len(overall_alphas) >= 2 else 0,
        "fair": (max(overall_alphas) - min(overall_alphas)) <= FAIRNESS_THRESHOLD if len(overall_alphas) >= 2 else True,
    }


def run_full_fairness_audit(
    scores: dict,
    human_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run fairness audit across all dimensions. Returns summary DataFrame."""
    rows = []
    for dim in DIMS:
        result = run_fairness_audit(scores, human_df, dim)
        if "error" in result:
            continue

        for analysis_name in ["by_subject", "by_difficulty", "by_human_agreement", "by_text_length"]:
            analysis = result.get(analysis_name, {})
            if "subgroups" not in analysis:
                continue
            for group_name, metrics in analysis["subgroups"].items():
                rows.append({
                    "dimension": dim,
                    "analysis": analysis_name,
                    "subgroup": group_name,
                    "alpha": metrics["alpha"],
                    "n": metrics["n"],
                    "bias": metrics.get("bias", np.nan),
                    "fair": analysis.get("fair", True),
                    "max_gap": analysis.get("max_gap", 0),
                })

    return pd.DataFrame(rows)
