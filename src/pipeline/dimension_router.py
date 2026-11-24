"""
Dimension Router: Tier A/B/C splitting logic.

Splits each item's dimensions into tiers based on per-dimension viability
evidence (Krippendorff alpha thresholds):

    Tier A (Auto):  alpha >= 0.50: auto-accept LLM annotation
    Tier B (Route): 0.30 <= alpha < 0.50: flag for selective human review
    Tier C (Human): alpha < 0.30: route to human annotation

Default tier assignment based on the experiment results:
    Tier A: Linguistic_Feature (alpha=0.54, Gemini Flash P0),
            Relevance (alpha=0.50, GPT-5.4 P1)
    Tier B: (none currently; Semantic_Uniqueness at 0.26 is closest)
    Tier C: Clarity, Semantic_Plausibility, Semantic_Uniqueness

Routing happens at the *dimension* level, not the item level.
A single item may have LF and Relevance auto-accepted while its
other three dimensions go to blind human review.
"""

from typing import Any

import numpy as np

from src.utils import setup_logging

logger = setup_logging()

# Tier thresholds (configurable for sensitivity analysis)
DEFAULT_TIER_A_THRESHOLD = 0.50
DEFAULT_TIER_B_THRESHOLD = 0.30

# Default tier assignments based on best alpha per dimension
TIER_A_DIMENSIONS = ["Linguistic_Feature", "Relevance"]
TIER_B_DIMENSIONS: list[str] = []
TIER_C_DIMENSIONS = [
    "Clarity",
    "Semantic_Plausibility",
    "Semantic_Uniqueness",
]

# Best model-prompt configuration per Tier A dimension, with its measured alpha
# from Data/results/neurips_experiments/results_with_cis.csv
TIER_A_CONFIGS = {
    "Linguistic_Feature": {"model": "gemini_flash", "prompt": "P0_baseline", "alpha": 0.536},
    "Relevance": {"model": "gpt54", "prompt": "P1_manual", "alpha": 0.505},
}

ALL_PIPELINE_DIMENSIONS = TIER_A_DIMENSIONS + TIER_B_DIMENSIONS + TIER_C_DIMENSIONS


class DimensionRouter:
    """Routes each dimension of an annotated item to the appropriate tier."""

    def __init__(
        self,
        tier_a: list[str] | None = None,
        tier_b: list[str] | None = None,
        tier_c: list[str] | None = None,
        tier_a_threshold: float = DEFAULT_TIER_A_THRESHOLD,
        tier_b_threshold: float = DEFAULT_TIER_B_THRESHOLD,
    ):
        self.tier_a = tier_a if tier_a is not None else list(TIER_A_DIMENSIONS)
        self.tier_b = tier_b if tier_b is not None else list(TIER_B_DIMENSIONS)
        self.tier_c = tier_c if tier_c is not None else list(TIER_C_DIMENSIONS)
        self.tier_a_threshold = tier_a_threshold
        self.tier_b_threshold = tier_b_threshold

    @classmethod
    def from_alphas(
        cls,
        alpha_scores: dict[str, float],
        tier_a_threshold: float = DEFAULT_TIER_A_THRESHOLD,
        tier_b_threshold: float = DEFAULT_TIER_B_THRESHOLD,
    ) -> "DimensionRouter":
        """Create a router with tiers assigned from observed alpha values."""
        tier_a, tier_b, tier_c = [], [], []
        for dim, alpha in alpha_scores.items():
            if np.isnan(alpha):
                tier_c.append(dim)
            elif alpha >= tier_a_threshold:
                tier_a.append(dim)
            elif alpha >= tier_b_threshold:
                tier_b.append(dim)
            else:
                tier_c.append(dim)
        return cls(tier_a=tier_a, tier_b=tier_b, tier_c=tier_c,
                   tier_a_threshold=tier_a_threshold,
                   tier_b_threshold=tier_b_threshold)

    def get_tier_summary(self) -> dict:
        """Return a summary of current tier assignments."""
        return {
            "tier_a": self.tier_a,
            "tier_b": self.tier_b,
            "tier_c": self.tier_c,
            "thresholds": {
                "tier_a": self.tier_a_threshold,
                "tier_b": self.tier_b_threshold,
            },
            "auto_accept_pct": len(self.tier_a) / max(1, len(self.tier_a) + len(self.tier_b) + len(self.tier_c)) * 100,
        }

    def route(self, item_result: dict) -> dict:
        """
        Split one item's annotation runs by dimension tier.

        Args:
            item_result: dict with keys:
                ``question_uuid`` (str)
                ``runs`` (list of per-run dicts from AnnotationEngine,
                          each containing ``scores`` dict keyed by dim)

        Returns:
            dict with:
                question_uuid: str
                tier_a: {dim: [score_run1, score_run2, score_run3]}
                tier_b: {} (empty in v1)
                tier_c: {dim: [score_run1, score_run2, score_run3]}
        """
        uuid = item_result["question_uuid"]
        runs = item_result["runs"]

        tier_a_scores: dict[str, list] = {}
        tier_b_scores: dict[str, list] = {}
        tier_c_scores: dict[str, list] = {}

        for dim in self.tier_a:
            tier_a_scores[dim] = self._extract_dim_scores(runs, dim)

        for dim in self.tier_b:
            tier_b_scores[dim] = self._extract_dim_scores(runs, dim)

        for dim in self.tier_c:
            tier_c_scores[dim] = self._extract_dim_scores(runs, dim)

        return {
            "question_uuid": uuid,
            "tier_a": tier_a_scores,
            "tier_b": tier_b_scores,
            "tier_c": tier_c_scores,
        }

    def route_batch(self, batch_results: list[dict]) -> list[dict]:
        """Route all items in a batch."""
        routed = []
        for item in batch_results:
            routed.append(self.route(item))
        logger.info(
            f"Routed {len(routed)} items: "
            f"Tier A dims={self.tier_a}, Tier C dims={self.tier_c}"
        )
        return routed

    @staticmethod
    def _extract_dim_scores(runs: list[dict], dim: str) -> list:
        """Extract scores for a single dimension across all runs."""
        scores = []
        for run in runs:
            run_scores = run.get("scores", {})
            val = run_scores.get(dim, np.nan)
            scores.append(val)
        return scores
