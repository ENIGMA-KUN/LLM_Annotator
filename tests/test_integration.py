"""
Integration tests for the pipeline modules.

Verify that:
1. All pipeline modules import correctly from both old and new paths
2. Metric functions via pipeline.metrics produce identical results to
   the Phase 1 evaluation JSON (read-only historical evidence)
3. InputHandler validates and loads the existing data correctly
4. compute_krippendorff_alpha_multi handles NaN (Tier C) correctly
5. The prompt registry is well-formed

Run:  python -m pytest tests/test_integration.py -v
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent

# The research data and prompts are not distributed with the repository, so tests
# that read them skip on a fresh clone instead of failing.
needs_data = pytest.mark.skipif(
    not (ROOT / "Data" / "processed" / "full_scores_both_annotators.csv").exists(),
    reason="needs the unpublished Data/ folder",
)
needs_prompts = pytest.mark.skipif(
    not (ROOT / "prompts" / "prompt_registry.json").exists(),
    reason="needs the unpublished prompts/ folder",
)


# -----------------------------------------------------------------------
# 1. Import compatibility tests
# -----------------------------------------------------------------------

class TestImports:
    """Verify all import paths work."""

    def test_legacy_evaluator_imports(self):
        from src.evaluator import (
            compute_cohens_kappa,
            compute_krippendorff_alpha,
            compute_exact_match,
            compute_within_one,
            compute_per_dimension_f1,
            compute_confusion_matrix,
            compute_expected_calibration_error,
            evaluate_full,
            evaluate_human_agreement,
            print_comparison_table,
        )
        assert callable(evaluate_full)

    def test_new_pipeline_metrics_imports(self):
        from src.pipeline.metrics import (
            compute_cohens_kappa,
            compute_krippendorff_alpha,
            compute_krippendorff_alpha_multi,
            compute_exact_match,
            compute_within_one,
            compute_per_dimension_f1,
            compute_confusion_matrix,
            compute_expected_calibration_error,
            evaluate_full,
            evaluate_human_agreement,
            print_comparison_table,
        )
        assert callable(compute_krippendorff_alpha_multi)

    def test_identity_between_paths(self):
        """Same function object at both import paths."""
        from src.evaluator import evaluate_full as ef_old
        from src.pipeline.metrics import evaluate_full as ef_new
        assert ef_old is ef_new

    def test_pipeline_package_barrel(self):
        from src.pipeline import (
            InputHandler,
            AnnotationEngine,
            DimensionRouter,
            ConfidenceGate,
            PipelineAggregator,
            QualityMonitor,
        )
        assert InputHandler is not None

    def test_legacy_annotator_still_importable(self):
        from src.annotator import LLMAnnotator
        assert callable(LLMAnnotator)


# -----------------------------------------------------------------------
# 2. Metric identity tests (compare pipeline.metrics to Phase 1 JSON)
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def phase1_eval():
    """Load the Phase 1 evaluation JSON (read-only evidence)."""
    path = ROOT / "Data" / "results" / "phase1_question_level" / "qlevel_evaluation.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def human_data():
    """Load the 200-item human baseline CSV."""
    path = ROOT / "Data" / "processed" / "full_scores_both_annotators.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def llm_zero_shot():
    """Load the zero-shot LLM results CSV."""
    path = (
        ROOT / "Data" / "results" / "phase1_question_level"
        / "zero_shot_qlevel" / "zero_shot_qlevel_results.csv"
    )
    return pd.read_csv(path)


DIMS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility", "Semantic_Uniqueness"]
DIM_HUMAN_SUFFIX = {"Relevance": "_KC", "Clarity": "_KC", "Linguistic_Feature": "_KC",
                     "Semantic_Plausibility": "_KC", "Semantic_Unique": "_KC"}


@needs_data
class TestMetricIdentity:
    """Verify that pipeline.metrics produces the same Alpha values as Phase 1."""

    def test_krippendorff_alpha_linguistic_feature_vs_K(self, human_data, llm_zero_shot, phase1_eval):
        """Alpha for Linguistic Feature (LLM vs rater K) must match Phase 1."""
        from src.pipeline.metrics import compute_krippendorff_alpha

        dim = "Linguistic_Feature"
        human_col = f"{dim}_KC"
        h = human_data[human_col].dropna().values[:200]
        l = llm_zero_shot[dim].dropna().values[:200]
        min_len = min(len(h), len(l))
        h_s = pd.Series(h[:min_len])
        l_s = pd.Series(l[:min_len])

        result = compute_krippendorff_alpha(h_s, l_s)
        expected = phase1_eval["zero_shot_qlevel"]["dimensions"][dim]["alpha_vs_K"]

        assert abs(result["alpha"] - expected) < 0.01, (
            f"Alpha mismatch for {dim} vs K: got {result['alpha']}, expected {expected}"
        )

    def test_krippendorff_alpha_relevance_vs_K(self, human_data, llm_zero_shot, phase1_eval):
        """Alpha for Relevance (LLM vs rater K): even negative values must match."""
        from src.pipeline.metrics import compute_krippendorff_alpha

        dim = "Relevance"
        h = human_data[f"{dim}_KC"].dropna().values[:200]
        l = llm_zero_shot[dim].dropna().values[:200]
        min_len = min(len(h), len(l))
        h_s = pd.Series(h[:min_len])
        l_s = pd.Series(l[:min_len])

        result = compute_krippendorff_alpha(h_s, l_s)
        expected = phase1_eval["zero_shot_qlevel"]["dimensions"][dim]["alpha_vs_K"]

        assert abs(result["alpha"] - expected) < 0.01, (
            f"Alpha mismatch for {dim} vs K: got {result['alpha']}, expected {expected}"
        )


# -----------------------------------------------------------------------
# 3. InputHandler tests
# -----------------------------------------------------------------------

class TestInputHandler:
    """Verify InputHandler loads and validates data correctly."""

    @needs_data
    def test_load_existing_csv(self):
        from src.pipeline.input_handler import InputHandler
        handler = InputHandler()
        path = ROOT / "Data" / "processed" / "full_scores_both_annotators.csv"
        df = handler.load_csv(path)
        assert len(df) == 200
        assert "question_uuid" in df.columns

    def test_schema_validation_missing_column(self, tmp_path):
        """Must raise ValueError if required column is missing."""
        from src.pipeline.input_handler import InputHandler
        handler = InputHandler()
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"col_a": [1, 2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            handler.load_csv(bad_csv)

    def test_file_not_found(self):
        from src.pipeline.input_handler import InputHandler
        handler = InputHandler()
        with pytest.raises(FileNotFoundError):
            handler.load_csv(Path("nonexistent.csv"))

    @needs_data
    def test_batch_splitting(self):
        from src.pipeline.input_handler import InputHandler
        handler = InputHandler(batch_size=50)
        path = ROOT / "Data" / "processed" / "full_scores_both_annotators.csv"
        df = handler.load_csv(path)
        batches = handler.split_batches(df)
        assert len(batches) == 4  # 200 / 50
        assert len(batches[0]) == 50
        assert len(batches[-1]) == 50


# -----------------------------------------------------------------------
# 4. Three-rater Alpha (NaN handling for Tier C)
# -----------------------------------------------------------------------

class TestThreeRaterAlpha:
    """Verify compute_krippendorff_alpha_multi handles NaN correctly."""

    def test_two_raters_basic(self):
        from src.pipeline.metrics import compute_krippendorff_alpha_multi
        r1 = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        r2 = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        result = compute_krippendorff_alpha_multi(r1, r2)
        assert result["alpha"] == 1.0  # perfect agreement

    def test_three_raters_partial_nan(self):
        """Rater 3 has NaN on some items; must not crash or treat NaN as category."""
        from src.pipeline.metrics import compute_krippendorff_alpha_multi
        r1 = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        r2 = np.array([1, 2, 3, 4, 2, 2, 3, 4])
        r3 = np.array([1, 2, np.nan, np.nan, np.nan, 2, np.nan, 4])
        result = compute_krippendorff_alpha_multi(r1, r2, r3)
        assert not math.isnan(result["alpha"])
        assert result["n_raters"] == 3

    def test_all_nan_rater_returns_valid(self):
        """If rater 3 is all NaN (Tier C scenario), Alpha should still compute for raters 1 & 2."""
        from src.pipeline.metrics import compute_krippendorff_alpha_multi
        r1 = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        r2 = np.array([1, 2, 3, 4, 2, 2, 3, 4])
        r3 = np.array([np.nan] * 8)
        result = compute_krippendorff_alpha_multi(r1, r2, r3)
        # With all NaN for rater 3, krippendorff library treats it as
        # missing data; result should still be valid (just raters 1 & 2)
        assert not math.isnan(result["alpha"])

    def test_insufficient_data(self):
        from src.pipeline.metrics import compute_krippendorff_alpha_multi
        r1 = np.array([1])
        r2 = np.array([1])
        result = compute_krippendorff_alpha_multi(r1, r2)
        assert math.isnan(result["alpha"])


# -----------------------------------------------------------------------
# 5. Prompt Registry
# -----------------------------------------------------------------------

@needs_prompts
class TestPromptRegistry:
    """Verify prompt_registry.json is well-formed."""

    def test_registry_loads(self):
        path = ROOT / "prompts" / "prompt_registry.json"
        with open(path) as f:
            reg = json.load(f)
        assert "prompts" in reg
        assert "canonical" in reg

    def test_all_prompt_files_exist(self):
        path = ROOT / "prompts" / "prompt_registry.json"
        with open(path) as f:
            reg = json.load(f)
        prompts_dir = ROOT / "prompts"
        for name, info in reg["prompts"].items():
            prompt_file = prompts_dir / info["file"]
            assert prompt_file.exists(), f"Prompt file missing: {prompt_file}"

    def test_canonical_prompt_is_active(self):
        path = ROOT / "prompts" / "prompt_registry.json"
        with open(path) as f:
            reg = json.load(f)
        canonical = reg["canonical"]
        assert canonical in reg["prompts"]
        assert reg["prompts"][canonical]["status"] == "active"
