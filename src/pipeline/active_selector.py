"""
Active Selector: uncertainty-based item selection for human review.

Uses inter-run disagreement as the uncertainty signal to prioritize
which items should be routed to human annotators. This transforms the
hybrid routing from a static threshold into an active learning pipeline.

Key insight: Items where the LLM is UNCERTAIN (inter-run disagreement)
are the same items where human review adds the most value. Items where
the LLM is confident (3/3 agreement) are safe to auto-accept.

This module implements:
    1. Per-item uncertainty scoring from 3-run agreement patterns
    2. Uncertainty-ranked item selection for human review
    3. Simulation: show that routing uncertain items to humans improves alpha
    4. Budget-constrained selection: maximize quality under annotation budget
"""

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import setup_logging

logger = setup_logging()


def compute_item_uncertainty(runs: list[float]) -> dict:
    """
    Compute uncertainty metrics for a single item from its 3 runs.

    Returns:
        dict with: entropy, spread, agreement_level, uncertainty_score
    """
    valid = [int(s) for s in runs if not (isinstance(s, float) and np.isnan(s))]

    if len(valid) == 0:
        return {
            "entropy": 1.0,
            "spread": 4,
            "agreement_level": "no_data",
            "uncertainty_score": 1.0,
        }

    counts = Counter(valid)
    probs = np.array([c / len(valid) for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(max(len(counts), 2))

    spread = max(valid) - min(valid)
    most_common_count = counts.most_common(1)[0][1]

    if len(valid) == 3 and most_common_count == 3:
        agreement = "unanimous"
        uncertainty = 0.0
    elif most_common_count >= 2:
        agreement = "majority"
        uncertainty = 0.3 + 0.2 * spread
    else:
        agreement = "split"
        uncertainty = 0.7 + 0.3 * (spread / 3.0)

    return {
        "entropy": round(float(entropy), 4),
        "spread": spread,
        "agreement_level": agreement,
        "uncertainty_score": round(min(1.0, uncertainty), 4),
    }


def rank_items_by_uncertainty(
    scores: dict,
    dimension: str,
) -> pd.DataFrame:
    """
    Rank all items by uncertainty for a given dimension.

    Args:
        scores: dict of uuid -> {dim: [run1, run2, run3]}
        dimension: which dimension to rank on

    Returns:
        DataFrame sorted by uncertainty (highest first), with columns:
        uuid, median_score, uncertainty_score, entropy, spread, agreement_level
    """
    rows = []
    for uuid, dim_scores in scores.items():
        runs = dim_scores.get(dimension, [])
        valid = [s for s in runs if not (isinstance(s, float) and np.isnan(s))]
        unc = compute_item_uncertainty(runs)
        rows.append({
            "uuid": uuid,
            "dimension": dimension,
            "median_score": float(np.median(valid)) if valid else np.nan,
            "scores": valid,
            **unc,
        })

    df = pd.DataFrame(rows).sort_values("uncertainty_score", ascending=False)
    return df


def simulate_active_selection(
    scores: dict,
    human_consensus: dict,
    dimension: str,
    budget_fractions: list[float] = None,
) -> pd.DataFrame:
    """
    Simulate: what if we only send the top-K% most uncertain items to humans?

    For each budget level, compute:
    - alpha using LLM scores for auto-accepted items + human for reviewed items
    - comparison with random selection at the same budget

    Args:
        scores: dict of uuid -> {dim: [run1, run2, run3]}
        human_consensus: dict of uuid -> consensus score
        dimension: dimension to evaluate
        budget_fractions: list of fractions (0.0 to 1.0) of items to route to human

    Returns:
        DataFrame with columns: budget_pct, alpha_active, alpha_random, alpha_gain
    """
    from src.pipeline.metrics import compute_krippendorff_alpha_multi

    if budget_fractions is None:
        budget_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    ranked = rank_items_by_uncertainty(scores, dimension)
    uuids = ranked["uuid"].tolist()

    results = []
    for frac in budget_fractions:
        n_human = int(len(uuids) * frac)

        # Active selection: route most uncertain items to human
        human_uuids_active = set(uuids[:n_human])
        llm_vals_a, gold_vals_a = [], []
        for uuid in uuids:
            gold = human_consensus.get(uuid)
            if gold is None or np.isnan(gold):
                continue
            if uuid in human_uuids_active:
                llm_vals_a.append(gold)
            else:
                runs = scores[uuid].get(dimension, [])
                valid = [s for s in runs if not (isinstance(s, float) and np.isnan(s))]
                if valid:
                    llm_vals_a.append(float(np.median(valid)))
                else:
                    continue
            gold_vals_a.append(gold)

        # Random selection: route random items to human
        np.random.seed(42)
        random_order = np.random.permutation(len(uuids))
        human_uuids_random = set(uuids[i] for i in random_order[:n_human])
        llm_vals_r, gold_vals_r = [], []
        for uuid in uuids:
            gold = human_consensus.get(uuid)
            if gold is None or np.isnan(gold):
                continue
            if uuid in human_uuids_random:
                llm_vals_r.append(gold)
            else:
                runs = scores[uuid].get(dimension, [])
                valid = [s for s in runs if not (isinstance(s, float) and np.isnan(s))]
                if valid:
                    llm_vals_r.append(float(np.median(valid)))
                else:
                    continue
            gold_vals_r.append(gold)

        alpha_active = np.nan
        alpha_random = np.nan
        if len(llm_vals_a) >= 5 and len(gold_vals_a) >= 5:
            alpha_active = compute_krippendorff_alpha_multi(
                np.array(llm_vals_a), np.array(gold_vals_a)
            )["alpha"]
        if len(llm_vals_r) >= 5 and len(gold_vals_r) >= 5:
            alpha_random = compute_krippendorff_alpha_multi(
                np.array(llm_vals_r), np.array(gold_vals_r)
            )["alpha"]

        gain = (alpha_active - alpha_random) if not (np.isnan(alpha_active) or np.isnan(alpha_random)) else np.nan

        results.append({
            "budget_pct": round(frac * 100, 1),
            "n_human_reviewed": n_human,
            "alpha_active": round(alpha_active, 4) if not np.isnan(alpha_active) else np.nan,
            "alpha_random": round(alpha_random, 4) if not np.isnan(alpha_random) else np.nan,
            "alpha_gain": round(gain, 4) if not np.isnan(gain) else np.nan,
        })

    return pd.DataFrame(results)


def compute_optimal_budget(
    sim_results: pd.DataFrame,
    target_alpha: float = 0.50,
) -> dict:
    """Find minimum human budget to achieve target alpha with active selection."""
    for _, row in sim_results.iterrows():
        if not np.isnan(row["alpha_active"]) and row["alpha_active"] >= target_alpha:
            return {
                "min_budget_pct": row["budget_pct"],
                "n_human_reviewed": row["n_human_reviewed"],
                "alpha_achieved": row["alpha_active"],
                "target_alpha": target_alpha,
            }
    return {
        "min_budget_pct": 100.0,
        "n_human_reviewed": sim_results["n_human_reviewed"].max(),
        "alpha_achieved": sim_results["alpha_active"].max(),
        "target_alpha": target_alpha,
        "note": "Target not achievable with this configuration",
    }
