"""
Few-Shot Example Selector.
Selects stratified examples from human annotations for few-shot prompting.
"""

import random
from typing import Optional

import pandas as pd
import numpy as np

from src.utils import ANNOTATION_DIMENSIONS, setup_logging

logger = setup_logging()


def select_few_shot_examples(
    df_katerina: pd.DataFrame,
    df_linh: pd.DataFrame,
    n_examples: int = 8,
    seed: int = 42,
    agreement_only: bool = True,
) -> list[dict]:
    """
    Select stratified few-shot examples where both annotators agree.

    Strategy:
    - Pick examples spanning all 4 score levels (1-4) across dimensions
    - Prefer items where Katerina and Linh agree (higher confidence ground truth)
    - Include diverse text sources (sociology, history, anthropology, etc.)

    Args:
        df_katerina: Katerina's annotations
        df_linh: Linh's annotations
        n_examples: Number of examples to select (default 8)
        seed: Random seed for reproducibility
        agreement_only: Only use items where both annotators agree on Relevance

    Returns:
        List of example dicts with text, question, answer, option, and scores
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find items where both annotators agree on primary dimension (Relevance)
    primary_dim = "Relevance"
    candidates = []

    for idx in range(min(len(df_katerina), len(df_linh))):
        row_k = df_katerina.iloc[idx]
        row_l = df_linh.iloc[idx]

        k_score = row_k.get(primary_dim)
        l_score = row_l.get(primary_dim)

        # Skip if either is missing
        if pd.isna(k_score) or pd.isna(l_score):
            continue
        if k_score in (999, "NA") or l_score in (999, "NA"):
            continue

        k_score = int(float(k_score))
        l_score = int(float(l_score))

        if agreement_only and k_score != l_score:
            continue

        # Build example with averaged/agreed scores
        example = {
            "index": idx,
            "question_uuid": str(row_k.get("question_uuid", "")),
            "textID": str(row_k.get("textID", "")),
            "text": str(row_k.get("text", "")),
            "question": str(row_k.get("question", "")),
            "answer": str(row_k.get("answer", "")),
            "option_choice": str(row_k.get("option_choice", "")),
            "source": str(row_k.get("Source", "")),
        }

        # Use agreed scores (or Katerina's if agreement_only=False)
        for dim in ANNOTATION_DIMENSIONS:
            k_val = row_k.get(dim)
            l_val = row_l.get(dim)
            if pd.notna(k_val) and k_val not in (999, "NA"):
                example[dim] = int(float(k_val))
            elif pd.notna(l_val) and l_val not in (999, "NA"):
                example[dim] = int(float(l_val))
            else:
                example[dim] = None

        example[f"{primary_dim}_score"] = k_score
        candidates.append(example)

    logger.info(f"Found {len(candidates)} candidate examples (agreement_only={agreement_only})")

    if len(candidates) < n_examples:
        logger.warning(
            f"Only {len(candidates)} candidates found. "
            f"Relaxing agreement requirement."
        )
        return select_few_shot_examples(
            df_katerina, df_linh, n_examples, seed, agreement_only=False
        )

    # Stratified selection: try to get examples from each score level
    selected = _stratified_select(candidates, primary_dim, n_examples)

    logger.info(f"Selected {len(selected)} few-shot examples:")
    for ex in selected:
        scores = {dim: ex.get(dim) for dim in ANNOTATION_DIMENSIONS}
        logger.info(f"  [{ex['textID']}] {scores}")

    return selected


def _stratified_select(
    candidates: list[dict],
    primary_dim: str,
    n_examples: int,
) -> list[dict]:
    """Select examples ensuring coverage of all score levels."""
    # Group by primary dimension score
    by_score = {}
    for c in candidates:
        score = c.get(f"{primary_dim}_score", c.get(primary_dim))
        if score is not None:
            by_score.setdefault(score, []).append(c)

    selected = []
    scores_available = sorted(by_score.keys())

    # First pass: pick at least one from each score level
    for score in scores_available:
        if len(selected) >= n_examples:
            break
        pool = by_score[score]
        # Prefer diverse sources
        chosen = random.choice(pool)
        selected.append(chosen)
        pool.remove(chosen)

    # Second pass: fill remaining slots
    remaining = [c for c in candidates if c not in selected]
    random.shuffle(remaining)

    while len(selected) < n_examples and remaining:
        selected.append(remaining.pop(0))

    return selected


def validate_examples(examples: list[dict]) -> bool:
    """Validate that selected examples cover the required score range."""
    scores_seen = set()
    for ex in examples:
        for dim in ANNOTATION_DIMENSIONS:
            if ex.get(dim) is not None:
                scores_seen.add(ex[dim])

    coverage = len(scores_seen)
    if coverage < 3:
        logger.warning(
            f"Low score coverage: only {coverage}/4 score levels represented. "
            f"Scores seen: {sorted(scores_seen)}"
        )
        return False

    logger.info(f"Example validation passed. Score levels covered: {sorted(scores_seen)}")
    return True
