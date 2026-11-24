"""
Confidence Gate: inter-run agreement routing for ALL dimensions.

Uses 3-run agreement as the uncertainty signal per dimension:

    3 of 3 agree  →  Auto-accept LLM label
    2 of 3 agree  →  Auto-accept majority, flag for spot-check
    All disagree   →  Route to human review

For Tier A dimensions: all three decisions apply.
For Tier B dimensions: only 3/3 agreement → auto-accept; else → human.
For Tier C dimensions: always route to human (gate is bypassed).

Inter-run agreement replaces self-reported confidence, which Phase 1 proved
useless (ECE 0.70, confidence 94% on every item).
"""

from collections import Counter
from typing import Any

import numpy as np

from src.utils import setup_logging

logger = setup_logging()

# Agreement decision constants
AGREE_3_OF_3 = "auto_accept"
AGREE_2_OF_3 = "auto_accept_flagged"
AGREE_SPLIT = "route_to_human"


class ConfidenceGate:
    """Determines routing decision for Tier A items based on inter-run agreement."""

    def evaluate(self, dim_scores: list) -> dict:
        """
        Evaluate inter-run agreement for a single dimension.

        Args:
            dim_scores: List of scores from the 3 independent runs.
                        May contain NaN for failed API calls.

        Returns:
            dict with:
                decision: one of AGREE_3_OF_3, AGREE_2_OF_3, AGREE_SPLIT
                accepted_score: int or None
                agreement_level: str ("3of3", "2of3", "split")
                scores: original scores list
        """
        # Filter out NaN
        valid = [int(s) for s in dim_scores if not _is_nan(s)]

        if len(valid) == 0:
            return {
                "decision": AGREE_SPLIT,
                "accepted_score": None,
                "agreement_level": "no_valid_scores",
                "scores": dim_scores,
            }

        counts = Counter(valid)
        most_common_score, most_common_count = counts.most_common(1)[0]

        if len(valid) == 3 and most_common_count == 3:
            # 3 of 3 agree
            return {
                "decision": AGREE_3_OF_3,
                "accepted_score": most_common_score,
                "agreement_level": "3of3",
                "scores": dim_scores,
            }
        elif most_common_count >= 2:
            # 2 of 3 agree (or 2 of 2 if one run failed)
            return {
                "decision": AGREE_2_OF_3,
                "accepted_score": most_common_score,
                "agreement_level": "2of3",
                "scores": dim_scores,
            }
        else:
            # All different
            return {
                "decision": AGREE_SPLIT,
                "accepted_score": None,
                "agreement_level": "split",
                "scores": dim_scores,
            }

    def evaluate_item(self, tier_a_runs: dict) -> dict:
        """
        Evaluate all Tier A dimensions for a single item.

        Args:
            tier_a_runs: dict mapping dim name -> list of per-run scores.

        Returns:
            dict mapping dim -> gate decision dict.
        """
        decisions = {}
        for dim, scores in tier_a_runs.items():
            decisions[dim] = self.evaluate(scores)
        return decisions

    def evaluate_all_tiers(self, item: dict) -> dict:
        """
        Evaluate ALL dimensions for a single item with tier-aware logic.

        Tier A: 3/3 → auto, 2/3 → auto_flagged, split → human
        Tier B: 3/3 → auto, else → human
        Tier C: always → human (LLM scores recorded but not accepted)
        """
        decisions = {}

        for dim, scores in item.get("tier_a", {}).items():
            decisions[dim] = self.evaluate(scores)

        for dim, scores in item.get("tier_b", {}).items():
            result = self.evaluate(scores)
            if result["decision"] != AGREE_3_OF_3:
                result["decision"] = AGREE_SPLIT
                result["accepted_score"] = None
                result["agreement_level"] = "tier_b_requires_3of3"
            decisions[dim] = result

        for dim, scores in item.get("tier_c", {}).items():
            valid = [int(s) for s in scores if not _is_nan(s)]
            decisions[dim] = {
                "decision": AGREE_SPLIT,
                "accepted_score": None,
                "agreement_level": "tier_c_always_human",
                "scores": scores,
                "llm_median": float(np.median(valid)) if valid else None,
            }

        return decisions

    def evaluate_batch(self, routed_batch: list[dict]) -> list[dict]:
        """
        Apply confidence gating to a batch of routed items.

        Args:
            routed_batch: List of dicts from DimensionRouter.route_batch(),
                         each with keys: question_uuid, tier_a, tier_b, tier_c.

        Returns:
            List of dicts, each augmented with ``gate_decisions`` key.
        """
        gated = []
        auto_count = 0
        flag_count = 0
        human_count = 0

        for item in routed_batch:
            decisions = self.evaluate_all_tiers(item)

            for dim, dec in decisions.items():
                if dec["decision"] == AGREE_3_OF_3:
                    auto_count += 1
                elif dec["decision"] == AGREE_2_OF_3:
                    flag_count += 1
                else:
                    human_count += 1

            gated.append({
                **item,
                "gate_decisions": decisions,
            })

        total = auto_count + flag_count + human_count
        if total > 0:
            logger.info(
                f"Confidence gate: {len(gated)} items | "
                f"3/3 auto: {auto_count} ({auto_count/total*100:.1f}%) | "
                f"2/3 flagged: {flag_count} ({flag_count/total*100:.1f}%) | "
                f"split→human: {human_count} ({human_count/total*100:.1f}%)"
            )

        return gated

    @staticmethod
    def batch_stats(gated_batch: list[dict]) -> dict:
        """Compute aggregate routing statistics for a gated batch."""
        per_dim = {}
        for item in gated_batch:
            for dim, dec in item.get("gate_decisions", {}).items():
                if dim not in per_dim:
                    per_dim[dim] = {"auto": 0, "flagged": 0, "human": 0, "total": 0}
                per_dim[dim]["total"] += 1
                if dec["decision"] == AGREE_3_OF_3:
                    per_dim[dim]["auto"] += 1
                elif dec["decision"] == AGREE_2_OF_3:
                    per_dim[dim]["flagged"] += 1
                else:
                    per_dim[dim]["human"] += 1

        total_decisions = sum(d["total"] for d in per_dim.values())
        total_auto = sum(d["auto"] + d["flagged"] for d in per_dim.values())
        return {
            "per_dimension": per_dim,
            "total_decisions": total_decisions,
            "total_auto_accepted": total_auto,
            "workload_reduction_pct": round(total_auto / max(1, total_decisions) * 100, 1),
        }


def _is_nan(val) -> bool:
    """Check if a value is NaN (works for float, numpy, and non-numeric)."""
    try:
        return np.isnan(float(val))
    except (ValueError, TypeError):
        return False
