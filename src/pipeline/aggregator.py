"""
Pipeline Aggregator: final score assembly + quality metrics.

Responsibilities:
    - Assemble final annotation output with per-item provenance
    - Compute Krippendorff Alpha (ordinal) per dimension
    - Compute three-rater Alpha (rater K + rater L + LLM):
          Rater 3 only has values for Tier A (Linguistic_Feature);
          NaN for all Tier C dimensions.
    - Produce workload metrics (auto-accepted %, routed %, time saved)
    - Per-dimension confusion matrices
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.pipeline.confidence_gate import AGREE_3_OF_3, AGREE_2_OF_3, AGREE_SPLIT
from src.pipeline.dimension_router import (
    TIER_A_DIMENSIONS,
    TIER_C_DIMENSIONS,
    ALL_PIPELINE_DIMENSIONS,
)
from src.pipeline.metrics import compute_krippendorff_alpha_multi
from src.utils import setup_logging

logger = setup_logging()

class PipelineAggregator:
    """Assembles final annotations and computes quality metrics."""

    def assemble(
        self,
        gated_items: list[dict],
        human_reviews: Optional[list[dict]] = None,
    ) -> pd.DataFrame:
        """
        Merge auto-accepted LLM labels with human review labels
        into the final annotated output.

        Output schema:
            question_uuid, dimension, score, source, llm_runs,
            inter_run_agreement, human_annotator_id, routing_reason

        Args:
            gated_items: Output from ConfidenceGate.evaluate_batch().
                         Each dict has: question_uuid, tier_a, tier_b,
                         tier_c, gate_decisions.
            human_reviews: Optional list of human review dicts, each with:
                           question_uuid, dimension, score, annotator_id.

        Returns:
            DataFrame in long format (one row per question × dimension).
        """
        # Index human reviews for fast lookup
        human_lookup: dict[tuple[str, str], dict] = {}
        if human_reviews:
            for hr in human_reviews:
                key = (hr["question_uuid"], hr["dimension"])
                human_lookup[key] = hr

        rows = []
        for item in gated_items:
            uuid = item["question_uuid"]
            gate_decisions = item.get("gate_decisions", {})

            # --- Tier A dimensions ---
            for dim, scores in item["tier_a"].items():
                decision = gate_decisions.get(dim, {})
                dec_type = decision.get("decision", AGREE_SPLIT)
                accepted = decision.get("accepted_score")
                agreement = decision.get("agreement_level", "unknown")

                if dec_type in (AGREE_3_OF_3, AGREE_2_OF_3) and accepted is not None:
                    # Auto-accepted
                    source = "auto" if dec_type == AGREE_3_OF_3 else "auto_flagged"
                    routing = f"tier_a_{agreement}"
                    rows.append({
                        "question_uuid": uuid,
                        "dimension": dim,
                        "score": accepted,
                        "source": source,
                        "llm_runs": scores,
                        "inter_run_agreement": agreement,
                        "human_annotator_id": None,
                        "routing_reason": routing,
                    })
                else:
                    # Routed to human
                    hr = human_lookup.get((uuid, dim))
                    rows.append({
                        "question_uuid": uuid,
                        "dimension": dim,
                        "score": hr["score"] if hr else None,
                        "source": self._review_source(hr),
                        "llm_runs": scores,
                        "inter_run_agreement": agreement,
                        "human_annotator_id": hr.get("annotator_id") if hr else None,
                        "routing_reason": f"tier_a_{agreement}_routed",
                    })

            # --- Tier C dimensions (always human) ---
            for dim, scores in item["tier_c"].items():
                hr = human_lookup.get((uuid, dim))
                rows.append({
                    "question_uuid": uuid,
                    "dimension": dim,
                    "score": hr["score"] if hr else None,
                    "source": self._review_source(hr),
                    "llm_runs": scores,
                    "inter_run_agreement": self._compute_agreement_label(scores),
                    "human_annotator_id": hr.get("annotator_id") if hr else None,
                    "routing_reason": "tier_c_always_human",
                })

        df = pd.DataFrame(rows)
        logger.info(
            f"Assembled {len(df)} annotation rows "
            f"({df['question_uuid'].nunique()} items × "
            f"{df['dimension'].nunique()} dimensions)"
        )
        return df

    def compute_three_rater_alpha(
        self,
        k_scores: np.ndarray,
        l_scores: np.ndarray,
        llm_scores: np.ndarray,
        level: str = "ordinal",
    ) -> float:
        """
        Compute Krippendorff Alpha for three raters.

        Constraints:
            - llm_scores must be np.nan for Tier C dimensions.
            - Alpha computation must correctly handle NaN (skip, not category).

        Args:
            k_scores: 1-D array of rater K's scores.
            l_scores: 1-D array of rater L's scores.
            llm_scores: 1-D array of LLM scores (NaN for Tier C).
            level: Measurement level for krippendorff.alpha().

        Returns:
            Alpha value (float).
        """
        result = compute_krippendorff_alpha_multi(
            np.asarray(k_scores, dtype=float),
            np.asarray(l_scores, dtype=float),
            np.asarray(llm_scores, dtype=float),
            level=level,
        )
        return result["alpha"]

    def compute_all_alphas(
        self,
        df_human: pd.DataFrame,
        assembled: pd.DataFrame,
        human_k_suffix: str = "_KC",
        human_l_suffix: str = "_LH",
    ) -> dict:
        """
        Compute per-dimension three-rater Alpha and two-rater human baseline.

        Rater 3 (LLM) only has values for Tier A dims; it is NaN for all
        Tier C dims.

        Args:
            df_human: Human baseline DataFrame with columns like
                      ``Relevance_KC``, ``Relevance_LH``, etc.
            assembled: Output of ``self.assemble()`` in long format.

        Returns:
            dict mapping dim -> {alpha_human_KL, alpha_three_rater, n_items}
        """
        results = {}

        # Build LLM score lookup from assembled (only auto/auto_flagged sources)
        llm_lookup: dict[tuple[str, str], int] = {}
        for _, row in assembled.iterrows():
            if row["source"] in ("auto", "auto_flagged"):
                llm_lookup[(row["question_uuid"], row["dimension"])] = row["score"]

        for dim in ALL_PIPELINE_DIMENSIONS:
            # Normalize dim name for human columns (Semantic_Unique vs Semantic_Uniqueness)
            human_dim = dim
            if dim == "Semantic_Uniqueness":
                human_dim = "Semantic_Unique"

            k_col = f"{human_dim}{human_k_suffix}"
            l_col = f"{human_dim}{human_l_suffix}"

            if k_col not in df_human.columns or l_col not in df_human.columns:
                logger.warning(f"Skipping {dim}: columns {k_col}/{l_col} not found")
                continue

            k_vals = df_human[k_col].values.astype(float)
            l_vals = df_human[l_col].values.astype(float)

            # Human-only Alpha (K vs L)
            human_alpha = compute_krippendorff_alpha_multi(k_vals, l_vals)

            # Build LLM array: NaN for Tier C, actual scores for Tier A
            llm_vals = np.full(len(k_vals), np.nan)
            if dim in TIER_A_DIMENSIONS:
                uuids = df_human["question_uuid"].values
                for i, uuid in enumerate(uuids):
                    score = llm_lookup.get((str(uuid), dim))
                    if score is not None:
                        llm_vals[i] = float(score)

            three_rater_alpha = compute_krippendorff_alpha_multi(
                k_vals, l_vals, llm_vals
            )

            results[dim] = {
                "alpha_human_KL": human_alpha["alpha"],
                "alpha_three_rater": three_rater_alpha["alpha"],
                "n_items": int(three_rater_alpha["n_items"]),
                "tier": "A" if dim in TIER_A_DIMENSIONS else "C",
            }
            logger.info(
                f"  {dim:25s} | Human α={human_alpha['alpha']:.4f} | "
                f"Three-rater α={three_rater_alpha['alpha']:.4f}"
            )

        return results

    def workload_report(self, assembled: pd.DataFrame) -> dict:
        """
        Compute workload reduction metrics.

        Returns dict with:
            pct_auto_accepted, pct_auto_flagged, pct_routed_human,
            pct_pending, estimated_time_saved_pct (capped at 20% for v1)
        """
        total = len(assembled)
        if total == 0:
            return {
                "pct_auto_accepted": 0.0,
                "pct_auto_flagged": 0.0,
                "pct_auto_lowconf": 0.0,
                "pct_routed_human": 0.0,
                "pct_pending": 0.0,
                "estimated_time_saved_pct": 0.0,
                "total_annotations": 0,
            }

        auto = (assembled["source"] == "auto").sum()
        flagged = (assembled["source"] == "auto_flagged").sum()
        lowconf = (assembled["source"] == "auto_lowconf").sum()
        human = (assembled["source"] == "human").sum()
        pending = (assembled["source"] == "pending_human").sum()

        auto_pct = auto / total * 100
        flagged_pct = flagged / total * 100
        lowconf_pct = lowconf / total * 100
        human_pct = human / total * 100
        pending_pct = pending / total * 100

        # Time saved = fraction of annotations that didn't need a human
        raw_saved = (auto + flagged) / total * 100
        estimated_saved = raw_saved

        report = {
            "pct_auto_accepted": round(auto_pct, 2),
            "pct_auto_flagged": round(flagged_pct, 2),
            "pct_auto_lowconf": round(lowconf_pct, 2),
            "pct_routed_human": round(human_pct, 2),
            "pct_pending": round(pending_pct, 2),
            "raw_auto_pct": round(raw_saved, 2),
            "estimated_time_saved_pct": round(estimated_saved, 2),
            "total_annotations": total,
            "n_items": int(assembled["question_uuid"].nunique()),
            "n_dimensions": int(assembled["dimension"].nunique()),
        }

        logger.info(
            f"Workload report: {report['n_items']} items × "
            f"{report['n_dimensions']} dims | "
            f"Auto: {report['pct_auto_accepted']:.1f}% | "
            f"Flagged: {report['pct_auto_flagged']:.1f}% | "
            f"Low-conf auto: {report['pct_auto_lowconf']:.1f}% | "
            f"Human: {report['pct_routed_human']:.1f}% | "
            f"Time saved: {report['estimated_time_saved_pct']:.1f}%"
        )
        return report

    @staticmethod
    def _review_source(hr: Optional[dict]) -> str:
        """
        Source label for a human-routed decision. Synthesized full-auto
        medians (annotator_id == 'auto_lowconf') must not be counted as
        genuine human reviews in the workload metrics.
        """
        if hr is None:
            return "pending_human"
        if hr.get("annotator_id") == "auto_lowconf":
            return "auto_lowconf"
        return "human"

    @staticmethod
    def _compute_agreement_label(scores: list) -> str:
        """Compute agreement label string from a list of scores."""
        valid = [s for s in scores if not (isinstance(s, float) and np.isnan(s))]
        if len(valid) == 0:
            return "no_valid"
        unique = len(set(valid))
        if unique == 1:
            return "3of3"
        elif len(valid) - unique <= 1:
            return "2of3" if len(valid) >= 2 else "1of1"
        return "split"
