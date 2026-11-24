"""Tests for QualityMonitor."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.quality_monitor import (
    QualityMonitor,
    ALPHA_DROP_THRESHOLD,
    GOLD_STANDARD_SIZE,
)
from src.pipeline.dimension_router import ALL_PIPELINE_DIMENSIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def human_df():
    """Minimal human DataFrame that mimics full_scores_both_annotators.csv."""
    np.random.seed(42)
    n = 60
    data = {"question_uuid": [f"uuid_{i}" for i in range(n)]}
    # K and L agree on first 35 items, disagree on rest
    for dim, human_dim in [
        ("Relevance", "Relevance"), ("Clarity", "Clarity"),
        ("Linguistic_Feature", "Linguistic_Feature"),
        ("Semantic_Plausibility", "Semantic_Plausibility"),
        ("Semantic_Uniqueness", "Semantic_Unique"),
    ]:
        k_scores = np.random.randint(1, 5, size=n)
        l_scores = k_scores.copy()
        l_scores[35:] = np.random.randint(1, 5, size=n - 35)
        data[f"{human_dim}_KC"] = k_scores
        data[f"{human_dim}_LH"] = l_scores
    return pd.DataFrame(data)


@pytest.fixture
def gold_df(human_df):
    """Create synthetic gold from the fixture human data."""
    return QualityMonitor.create_synthetic_gold(human_df, n_items=10, seed=42)


@pytest.fixture
def sample_gated_items(gold_df):
    """Create minimal gated items matching the gold standard UUIDs."""
    items = []
    for _, row in gold_df.iterrows():
        uuid = row["question_uuid"]
        tier_a = {"Linguistic_Feature": [int(row["Linguistic_Feature"])] * 3}
        tier_c = {}
        for dim in ["Relevance", "Clarity", "Semantic_Plausibility", "Semantic_Uniqueness"]:
            tier_c[dim] = [int(row[dim])] * 3
        gate_decisions = {
            "Linguistic_Feature": {
                "decision": "auto_accept",
                "accepted_score": int(row["Linguistic_Feature"]),
                "agreement_level": "3of3",
            }
        }
        items.append({
            "question_uuid": uuid,
            "tier_a": tier_a,
            "tier_b": {},
            "tier_c": tier_c,
            "gate_decisions": gate_decisions,
        })
    return items


# ---------------------------------------------------------------------------
# Tests: Gold standard creation
# ---------------------------------------------------------------------------

class TestGoldStandardCreation:

    def test_synthetic_gold_shape(self, human_df):
        gold = QualityMonitor.create_synthetic_gold(human_df, n_items=10, seed=42)
        assert len(gold) == 10
        assert "question_uuid" in gold.columns
        for dim in ALL_PIPELINE_DIMENSIONS:
            assert dim in gold.columns

    def test_synthetic_gold_all_dims_present(self, gold_df):
        for dim in ALL_PIPELINE_DIMENSIONS:
            assert dim in gold_df.columns
            assert gold_df[dim].notna().all()

    def test_synthetic_gold_reproducible(self, human_df):
        g1 = QualityMonitor.create_synthetic_gold(human_df, n_items=10, seed=42)
        g2 = QualityMonitor.create_synthetic_gold(human_df, n_items=10, seed=42)
        pd.testing.assert_frame_equal(g1, g2)

    def test_synthetic_gold_different_seeds(self, human_df):
        g1 = QualityMonitor.create_synthetic_gold(human_df, n_items=10, seed=42)
        g2 = QualityMonitor.create_synthetic_gold(human_df, n_items=10, seed=99)
        assert not g1["question_uuid"].equals(g2["question_uuid"])

    def test_synthetic_gold_saves_to_disk(self, human_df, tmp_path):
        path = tmp_path / "gold.csv"
        QualityMonitor.create_synthetic_gold(human_df, n_items=5, output_path=path)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 5

    def test_caps_at_available_items(self, human_df):
        gold = QualityMonitor.create_synthetic_gold(human_df, n_items=999, seed=42)
        assert len(gold) <= len(human_df)


# ---------------------------------------------------------------------------
# Tests: Evaluation on gold
# ---------------------------------------------------------------------------

class TestEvaluateOnGold:

    def test_perfect_scores(self, gold_df, sample_gated_items):
        """When LLM matches gold perfectly, Alpha should be 1.0 and exact match 100%."""
        monitor = QualityMonitor()
        monitor.set_gold_df(gold_df)
        result = monitor.evaluate_on_gold(sample_gated_items, label="perfect")

        assert result["n_matched"] == len(gold_df)
        lf = result["alphas"]["Linguistic_Feature"]
        assert lf["exact_match_pct"] == 100.0
        assert lf["within_1_pct"] == 100.0
        assert lf["bias"] == 0.0

    def test_degraded_scores_lower_alpha(self, gold_df, sample_gated_items):
        """When we add noise to LLM scores, Alpha should drop."""
        # Corrupt some scores
        for item in sample_gated_items[:5]:
            item["gate_decisions"]["Linguistic_Feature"]["accepted_score"] = 4

        monitor = QualityMonitor()
        monitor.set_gold_df(gold_df)
        result = monitor.evaluate_on_gold(sample_gated_items, label="degraded")

        lf = result["alphas"]["Linguistic_Feature"]
        assert lf["exact_match_pct"] < 100.0

    def test_no_overlap_returns_error(self, gold_df):
        """Items with no UUID overlap should return error dict."""
        monitor = QualityMonitor()
        monitor.set_gold_df(gold_df)
        result = monitor.evaluate_on_gold(
            [{"question_uuid": "nonexistent", "tier_a": {}, "tier_b": {},
              "tier_c": {}, "gate_decisions": {}}],
            label="no_overlap",
        )
        assert result.get("error") == "no_overlap"

    def test_raises_without_gold(self, sample_gated_items):
        monitor = QualityMonitor()
        with pytest.raises(ValueError, match="No gold standard"):
            monitor.evaluate_on_gold(sample_gated_items)


# ---------------------------------------------------------------------------
# Tests: Drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection:

    def test_no_drift_no_alerts(self):
        monitor = QualityMonitor()
        baseline = {"Linguistic_Feature": {"alpha": 0.7}}
        current = {"Linguistic_Feature": {"alpha": 0.65}}
        alerts = monitor.check_drift(current, baseline_override=baseline)
        assert len(alerts) == 0

    def test_drift_triggers_alert(self):
        monitor = QualityMonitor()
        baseline = {"Linguistic_Feature": {"alpha": 0.7}}
        current = {"Linguistic_Feature": {"alpha": 0.5}}
        alerts = monitor.check_drift(current, baseline_override=baseline)
        assert len(alerts) == 1
        assert alerts[0]["dimension"] == "Linguistic_Feature"
        assert alerts[0]["drop"] == pytest.approx(0.2, abs=0.01)

    def test_critical_severity(self):
        monitor = QualityMonitor()
        baseline = {"Linguistic_Feature": {"alpha": 0.8}}
        current = {"Linguistic_Feature": {"alpha": 0.4}}
        alerts = monitor.check_drift(current, baseline_override=baseline)
        assert alerts[0]["severity"] == "critical"

    def test_no_baseline_no_alerts(self):
        monitor = QualityMonitor()
        current = {"Linguistic_Feature": {"alpha": 0.5}}
        assert monitor.check_drift(current) == []


# ---------------------------------------------------------------------------
# Tests: History
# ---------------------------------------------------------------------------

class TestHistory:

    def test_history_accumulates(self, gold_df, sample_gated_items):
        monitor = QualityMonitor()
        monitor.set_gold_df(gold_df)
        monitor.evaluate_on_gold(sample_gated_items, label="batch_1")
        monitor.evaluate_on_gold(sample_gated_items, label="batch_2")

        history = monitor.get_history()
        assert len(history) > 0
        assert set(history["label"].unique()) == {"batch_1", "batch_2"}

    def test_history_persists(self, gold_df, sample_gated_items, tmp_path):
        hist_path = tmp_path / "history.json"
        monitor = QualityMonitor(history_path=hist_path)
        monitor.set_gold_df(gold_df)
        monitor.evaluate_on_gold(sample_gated_items, label="test")

        assert hist_path.exists()

        # Reload and verify
        monitor2 = QualityMonitor(history_path=hist_path)
        assert len(monitor2._history) == 1

    def test_empty_history(self):
        monitor = QualityMonitor()
        assert monitor.get_history().empty


# ---------------------------------------------------------------------------
# Tests: Confusion matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:

    def test_perfect_confusion_matrix(self, gold_df, sample_gated_items):
        monitor = QualityMonitor()
        monitor.set_gold_df(gold_df)
        cm = monitor.confusion_matrix(sample_gated_items, "Linguistic_Feature")
        assert cm is not None
        # Perfect match means all values on diagonal
        assert cm.values.sum() == len(gold_df)

    def test_no_gold_returns_none(self, sample_gated_items):
        monitor = QualityMonitor()
        assert monitor.confusion_matrix(sample_gated_items, "Linguistic_Feature") is None


# ---------------------------------------------------------------------------
# Tests: Real data integration
# ---------------------------------------------------------------------------

class TestRealDataIntegration:

    @pytest.fixture
    def real_gold_path(self):
        p = ROOT / "Data" / "gold_standard" / "synthetic_gold_30.csv"
        if not p.exists():
            pytest.skip("Real gold standard not found")
        return p

    def test_real_gold_loads(self, real_gold_path):
        monitor = QualityMonitor(gold_standard_path=real_gold_path)
        gold = monitor.get_gold_df()
        assert gold is not None
        assert len(gold) == 30
        for dim in ALL_PIPELINE_DIMENSIONS:
            assert dim in gold.columns
