"""
Tests for the dimension router, confidence gate and aggregator.

Run:  python -m pytest tests/test_router_gate.py -v
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _make_item(uuid: str, lf_scores: list, rel_scores: list = None) -> dict:
    """Helper to create a mock engine output item."""
    if rel_scores is None:
        rel_scores = [3, 3, 3]
    runs = []
    for i in range(3):
        runs.append({
            "scores": {
                "Relevance": rel_scores[i] if i < len(rel_scores) else np.nan,
                "Clarity": 4,
                "Linguistic_Feature": lf_scores[i] if i < len(lf_scores) else np.nan,
                "Semantic_Plausibility": 3,
                "Semantic_Uniqueness": 3,
            }
        })
    return {"question_uuid": uuid, "runs": runs}


# -----------------------------------------------------------------------
# DimensionRouter
# -----------------------------------------------------------------------

class TestDimensionRouter:

    def test_route_splits_by_tier(self):
        from src.pipeline.dimension_router import DimensionRouter
        router = DimensionRouter()
        item = _make_item("uuid1", [4, 4, 4])
        result = router.route(item)

        assert result["question_uuid"] == "uuid1"
        assert "Linguistic_Feature" in result["tier_a"]
        assert "Relevance" in result["tier_a"]
        assert "Clarity" in result["tier_c"]
        assert result["tier_b"] == {}

    def test_tier_a_extracts_all_run_scores(self):
        from src.pipeline.dimension_router import DimensionRouter
        router = DimensionRouter()
        item = _make_item("uuid1", [2, 3, 4])
        result = router.route(item)
        assert result["tier_a"]["Linguistic_Feature"] == [2, 3, 4]

    def test_route_batch(self):
        from src.pipeline.dimension_router import DimensionRouter
        router = DimensionRouter()
        items = [_make_item(f"uuid{i}", [4, 4, 4]) for i in range(5)]
        routed = router.route_batch(items)
        assert len(routed) == 5


# -----------------------------------------------------------------------
# ConfidenceGate
# -----------------------------------------------------------------------

class TestConfidenceGate:

    def test_3_of_3_agree(self):
        from src.pipeline.confidence_gate import ConfidenceGate, AGREE_3_OF_3
        gate = ConfidenceGate()
        result = gate.evaluate([4, 4, 4])
        assert result["decision"] == AGREE_3_OF_3
        assert result["accepted_score"] == 4
        assert result["agreement_level"] == "3of3"

    def test_2_of_3_agree(self):
        from src.pipeline.confidence_gate import ConfidenceGate, AGREE_2_OF_3
        gate = ConfidenceGate()
        result = gate.evaluate([4, 4, 3])
        assert result["decision"] == AGREE_2_OF_3
        assert result["accepted_score"] == 4
        assert result["agreement_level"] == "2of3"

    def test_all_different(self):
        from src.pipeline.confidence_gate import ConfidenceGate, AGREE_SPLIT
        gate = ConfidenceGate()
        result = gate.evaluate([1, 2, 3])
        assert result["decision"] == AGREE_SPLIT
        assert result["accepted_score"] is None
        assert result["agreement_level"] == "split"

    def test_nan_handling(self):
        from src.pipeline.confidence_gate import ConfidenceGate, AGREE_2_OF_3
        gate = ConfidenceGate()
        result = gate.evaluate([4, np.nan, 4])
        assert result["decision"] == AGREE_2_OF_3
        assert result["accepted_score"] == 4

    def test_all_nan(self):
        from src.pipeline.confidence_gate import ConfidenceGate, AGREE_SPLIT
        gate = ConfidenceGate()
        result = gate.evaluate([np.nan, np.nan, np.nan])
        assert result["decision"] == AGREE_SPLIT
        assert result["agreement_level"] == "no_valid_scores"

    def test_evaluate_batch(self):
        from src.pipeline.dimension_router import DimensionRouter
        from src.pipeline.confidence_gate import ConfidenceGate
        router = DimensionRouter()
        gate = ConfidenceGate()
        items = [
            _make_item("a", [4, 4, 4]),
            _make_item("b", [2, 3, 4]),
            _make_item("c", [1, 1, 2]),
        ]
        routed = router.route_batch(items)
        gated = gate.evaluate_batch(routed)
        assert len(gated) == 3
        assert "gate_decisions" in gated[0]


# -----------------------------------------------------------------------
# PipelineAggregator
# -----------------------------------------------------------------------

class TestPipelineAggregator:

    def test_assemble_output_schema(self):
        from src.pipeline.dimension_router import DimensionRouter
        from src.pipeline.confidence_gate import ConfidenceGate
        from src.pipeline.aggregator import PipelineAggregator

        router = DimensionRouter()
        gate = ConfidenceGate()
        agg = PipelineAggregator()

        items = [_make_item("uuid1", [4, 4, 4])]
        routed = router.route_batch(items)
        gated = gate.evaluate_batch(routed)
        assembled = agg.assemble(gated)

        expected_cols = {
            "question_uuid", "dimension", "score", "source",
            "llm_runs", "inter_run_agreement", "human_annotator_id",
            "routing_reason",
        }
        assert expected_cols.issubset(set(assembled.columns))
        assert len(assembled) == 5  # 5 dimensions per item

    def test_auto_accepted_source(self):
        from src.pipeline.dimension_router import DimensionRouter
        from src.pipeline.confidence_gate import ConfidenceGate
        from src.pipeline.aggregator import PipelineAggregator

        router = DimensionRouter()
        gate = ConfidenceGate()
        agg = PipelineAggregator()

        items = [_make_item("uuid1", [4, 4, 4])]
        routed = router.route_batch(items)
        gated = gate.evaluate_batch(routed)
        assembled = agg.assemble(gated)

        lf_row = assembled[assembled["dimension"] == "Linguistic_Feature"].iloc[0]
        assert lf_row["source"] == "auto"
        assert lf_row["score"] == 4

        rel_row = assembled[assembled["dimension"] == "Relevance"].iloc[0]
        assert rel_row["source"] == "auto"  # Relevance is now Tier A

    def test_workload_report(self):
        from src.pipeline.dimension_router import DimensionRouter
        from src.pipeline.confidence_gate import ConfidenceGate
        from src.pipeline.aggregator import PipelineAggregator

        router = DimensionRouter()
        gate = ConfidenceGate()
        agg = PipelineAggregator()

        items = [_make_item(f"uuid{i}", [4, 4, 4]) for i in range(10)]
        routed = router.route_batch(items)
        gated = gate.evaluate_batch(routed)
        assembled = agg.assemble(gated)
        report = agg.workload_report(assembled)

        # 10 items × 2 auto dims (LF + Relevance) / 5 total = 40% auto
        assert report["raw_auto_pct"] == 40.0
        assert report["estimated_time_saved_pct"] == report["raw_auto_pct"]
        assert report["total_annotations"] == 50

    def test_three_rater_alpha_basic(self):
        from src.pipeline.aggregator import PipelineAggregator
        agg = PipelineAggregator()
        k = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        l = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        llm = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        alpha = agg.compute_three_rater_alpha(k, l, llm)
        assert alpha == 1.0

    def test_three_rater_alpha_nan_tier_c(self):
        """LLM all NaN → Alpha should equal human-only Alpha."""
        from src.pipeline.aggregator import PipelineAggregator
        agg = PipelineAggregator()
        k = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        l = np.array([1, 2, 3, 4, 2, 2, 3, 4])
        llm = np.array([np.nan] * 8)
        alpha_three = agg.compute_three_rater_alpha(k, l, llm)
        from src.pipeline.metrics import compute_krippendorff_alpha_multi
        alpha_two = compute_krippendorff_alpha_multi(k, l)["alpha"]
        assert abs(alpha_three - alpha_two) < 0.001


# -----------------------------------------------------------------------
# Phase 1 data validation (integration)
# -----------------------------------------------------------------------

class TestPhase1Validation:
    """Validate against actual Phase 1 200-item data."""

    def test_auto_accept_rate_above_70_pct(self):
        """Auto-accept rate over Tier A decisions must be at least 70%."""
        results_path = ROOT / "Data" / "results" / "routing_validation" / "routing_validation.json"
        if not results_path.exists():
            pytest.skip("Run scripts/validate_routing.py first")
        with open(results_path) as f:
            results = json.load(f)
        assert results["auto_accept_rate_pct"] >= 70.0

    def test_tier_c_nan_handled_correctly(self):
        """Tier C three-rater Alpha must equal human-only Alpha."""
        results_path = ROOT / "Data" / "results" / "routing_validation" / "routing_validation.json"
        if not results_path.exists():
            pytest.skip("Run scripts/validate_routing.py first")
        with open(results_path) as f:
            results = json.load(f)
        from src.pipeline.dimension_router import TIER_C_DIMENSIONS
        for dim in TIER_C_DIMENSIONS:
            info = results["alphas"].get(dim, {})
            if info:
                assert abs(info["alpha_three_rater"] - info["alpha_human_KL"]) < 0.001, (
                    f"{dim}: three-rater ({info['alpha_three_rater']}) != "
                    f"human ({info['alpha_human_KL']})"
                )
