"""
Hybrid Orchestrator: end-to-end Hybrid Human-in-the-Loop Routing Protocol.

Ties together the existing pipeline stages into one auditable flow:

    AnnotationEngine (3 runs @ temp 0.3)
        → DimensionRouter (Tier A / B / C split)
        → ConfidenceGate (3/3 auto | 2/3 flag | split → human)
        → human review queue export (blind for Tier C)
        → PipelineAggregator (final assembly + workload report)

Routing contract:
    Tier A + 3/3 agreement   → auto-accept
    Tier A + 2/3 agreement   → auto-accept, flagged for verification
    Tier A + 0/3 agreement   → human review queue
    Tier C (any agreement)   → human review queue, LLM scores withheld

Usage:
    from src.pipeline.hybrid_orchestrator import HybridOrchestrator

    orch = HybridOrchestrator()
    state = orch.run(items, system_prompt)
    queue = orch.export_human_queue(state)          # blind review CSV
    final = orch.finalize(state, human_reviews)     # after humans annotate
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.pipeline.annotation_engine import AnnotationEngine
from src.pipeline.dimension_router import DimensionRouter, TIER_C_DIMENSIONS
from src.pipeline.confidence_gate import (
    ConfidenceGate,
    AGREE_3_OF_3,
    AGREE_2_OF_3,
)
from src.pipeline.aggregator import PipelineAggregator
from src.utils import setup_logging, RESULTS_DIR

logger = setup_logging()

QUEUE_DIR = RESULTS_DIR / "human_queue"


class HybridOrchestrator:
    """Single entry point for the hybrid routing protocol."""

    def __init__(
        self,
        engine: Optional[AnnotationEngine] = None,
        router: Optional[DimensionRouter] = None,
        gate: Optional[ConfidenceGate] = None,
        aggregator: Optional[PipelineAggregator] = None,
    ):
        self.engine = engine
        self.router = router or DimensionRouter()
        self.gate = gate or ConfidenceGate()
        self.aggregator = aggregator or PipelineAggregator()

    # ------------------------------------------------------------------
    # Stage 1-3: annotate → route → gate
    # ------------------------------------------------------------------

    def run(
        self,
        items: list[dict],
        system_prompt: str,
        condition_name: str = "hybrid_routing",
        precomputed_results: Optional[list[dict]] = None,
    ) -> dict:
        """
        Execute the LLM stages of the protocol.

        Args:
            items: list of dicts (question_uuid, text, question, answer).
            system_prompt: strict prompt payload (see prompt_payloads.py).
            condition_name: label for metadata logging.
            precomputed_results: optional cached AnnotationEngine output
                                 (skips API calls; used for replays/tests).

        Returns:
            state dict with keys: gated_items, gate_stats, condition_name.
        """
        if precomputed_results is not None:
            raw_results = precomputed_results
            logger.info(f"[{condition_name}] Using {len(raw_results)} precomputed results")
        else:
            if self.engine is None:
                self.engine = AnnotationEngine()
            raw_results = self.engine.annotate_batch(
                items, system_prompt=system_prompt, condition_name=condition_name
            )

        routed = self.router.route_batch(raw_results)
        gated = self.gate.evaluate_batch(routed)
        stats = self.gate.batch_stats(gated)

        logger.info(
            f"[{condition_name}] Routing complete | "
            f"workload reduction: {stats['workload_reduction_pct']}%"
        )
        return {
            "gated_items": gated,
            "gate_stats": stats,
            "condition_name": condition_name,
        }

    # ------------------------------------------------------------------
    # Stage 4: blind human review queue
    # ------------------------------------------------------------------

    def export_human_queue(
        self,
        state: dict,
        items_lookup: Optional[dict[str, dict]] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Build the human review queue.

        CRITICAL: For Tier C dimensions the LLM scores are withheld
        (blind review). For Tier A escalations the disagreeing runs ARE
        shown, since the human is adjudicating the disagreement.

        Args:
            state: output of ``run()``.
            items_lookup: optional {question_uuid: item dict} to attach
                          question text for the reviewer UI.
            save: write CSV to Data/results/human_queue/.

        Returns:
            DataFrame: one row per (question_uuid, dimension) needing a human.
        """
        rows = []
        for item in state["gated_items"]:
            uuid = item["question_uuid"]
            src = items_lookup.get(uuid, {}) if items_lookup else {}
            for dim, dec in item.get("gate_decisions", {}).items():
                if dec["decision"] in (AGREE_3_OF_3,):
                    continue  # fully auto-accepted
                is_tier_c = dim in TIER_C_DIMENSIONS
                needs_human = dec["decision"] not in (AGREE_3_OF_3, AGREE_2_OF_3)
                if not (is_tier_c or needs_human or dec["decision"] == AGREE_2_OF_3):
                    continue
                rows.append({
                    "question_uuid": uuid,
                    "dimension": dim,
                    "tier": "C" if is_tier_c else "A",
                    "queue_reason": (
                        "tier_c_always_human" if is_tier_c
                        else dec["agreement_level"]
                    ),
                    # BLIND: never leak LLM scores for Tier C
                    "llm_runs_visible": (
                        None if is_tier_c else json.dumps(dec.get("scores"))
                    ),
                    "priority": (
                        1 if dec["agreement_level"] == "split"
                        else (2 if is_tier_c else 3)
                    ),
                    "question": src.get("question"),
                    "answer": src.get("answer"),
                })

        queue = pd.DataFrame(rows).sort_values(
            ["priority", "question_uuid"], ignore_index=True
        ) if rows else pd.DataFrame(columns=[
            "question_uuid", "dimension", "tier", "queue_reason",
            "llm_runs_visible", "priority", "question", "answer",
        ])

        if save and len(queue):
            QUEUE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = QUEUE_DIR / f"human_queue_{state['condition_name']}_{stamp}.csv"
            queue.to_csv(path, index=False)
            logger.info(f"Human queue exported: {path} ({len(queue)} rows)")

        return queue

    # ------------------------------------------------------------------
    # Stage 5: final assembly
    # ------------------------------------------------------------------

    def finalize(
        self,
        state: dict,
        human_reviews: Optional[list[dict]] = None,
        full_auto: bool = False,
    ) -> dict:
        """
        Merge human reviews with auto-accepted labels and report workload.

        Args:
            state: output of ``run()``.
            human_reviews: list of dicts with question_uuid, dimension,
                           score, annotator_id.
            full_auto: if True, force median fallback for Tier C when no
                       human review is present (LLM-only mode).

        Returns:
            dict with assembled DataFrame and workload report.
        """
        # If full_auto, synthesize an LLM median for every decision that
        # would otherwise wait on a human: Tier A splits AND all Tier C dims
        # (the gate forces AGREE_SPLIT for Tier C, so both are covered by
        # the not-auto-accepted check).
        if full_auto:
            provided = {
                (hr["question_uuid"], hr["dimension"])
                for hr in (human_reviews or [])
            }
            synthesized = []
            for item in state["gated_items"]:
                uuid = item["question_uuid"]
                for dim, dec in item.get("gate_decisions", {}).items():
                    if dec["decision"] in (AGREE_3_OF_3, AGREE_2_OF_3):
                        continue  # already auto-accepted by the gate
                    if (uuid, dim) in provided:
                        continue  # a real human review always wins
                    # median of the 3 runs (NaN-safe)
                    valid = sorted(
                        s for s in dec["scores"]
                        if not (isinstance(s, float) and s != s)
                    )
                    median = int(valid[len(valid) // 2]) if valid else 3
                    synthesized.append({
                        "question_uuid": uuid,
                        "dimension": dim,
                        "score": median,
                        "annotator_id": "auto_lowconf",
                    })
            human_reviews = (human_reviews or []) + synthesized
            logger.info(
                f"[full_auto] Synthesized {len(synthesized)} low-confidence "
                f"LLM medians (annotator_id=auto_lowconf, flagged for audit)"
            )

        assembled = self.aggregator.assemble(
            state["gated_items"], human_reviews=human_reviews
        )
        report = self.aggregator.workload_report(assembled)
        return {"assembled": assembled, "workload_report": report}
