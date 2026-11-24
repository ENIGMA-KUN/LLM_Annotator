"""
Quality Monitor: drift detection on a fixed gold standard.

Responsibilities:
    - Load / create a gold standard from adjudicated human scores
    - Evaluate pipeline output against the gold standard per dimension
    - Track rolling Alpha history across batches
    - Alert if any dimension's Alpha drops > 0.1 from baseline
    - Provide time-series data for dashboard drift plot
    - Per-dimension confusion matrices on the gold set

Gold standard can come from:
    (a) an expert-adjudicated 30-item set (preferred),
    (b) Synthetic: items where K and L agree on all 5 dims (interim).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.pipeline.dimension_router import ALL_PIPELINE_DIMENSIONS, TIER_A_DIMENSIONS
from src.pipeline.metrics import compute_krippendorff_alpha_multi
from src.utils import setup_logging

logger = setup_logging()

# Drift detection threshold: alert when alpha drops by more than this
ALPHA_DROP_THRESHOLD = 0.1

# Default gold-standard size
GOLD_STANDARD_SIZE = 30

# Column name mapping for human data
_HUMAN_DIM_MAP = {
    "Relevance": "Relevance",
    "Clarity": "Clarity",
    "Linguistic_Feature": "Linguistic_Feature",
    "Semantic_Plausibility": "Semantic_Plausibility",
    "Semantic_Uniqueness": "Semantic_Unique",
}
_K_SUFFIX = "_KC"
_L_SUFFIX = "_LH"


class QualityMonitor:
    """Monitors annotation quality over time using a fixed gold standard."""

    def __init__(
        self,
        gold_standard_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
    ):
        """
        Args:
            gold_standard_path: Path to gold standard CSV. If None, must call
                                create_synthetic_gold() or load later.
            history_path: Path to JSON file persisting evaluation history.
        """
        self.gold_standard_path = gold_standard_path
        self.history_path = history_path
        self._gold_df: Optional[pd.DataFrame] = None
        self._history: list[dict] = []
        self._baseline: Optional[dict] = None

        if gold_standard_path and gold_standard_path.exists():
            self._gold_df = pd.read_csv(gold_standard_path)
            logger.info(f"Loaded gold standard: {len(self._gold_df)} items from {gold_standard_path}")

        if history_path and history_path.exists():
            with open(history_path) as f:
                self._history = json.load(f)
            if self._history:
                self._baseline = self._history[0].get("alphas")
            logger.info(f"Loaded {len(self._history)} history entries from {history_path}")

    # ------------------------------------------------------------------
    # Gold standard creation
    # ------------------------------------------------------------------

    @staticmethod
    def create_synthetic_gold(
        human_df: pd.DataFrame,
        n_items: int = GOLD_STANDARD_SIZE,
        output_path: Optional[Path] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Create a synthetic gold standard from items where K and L agree
        on ALL ordinal dimensions (exact match). These are high-confidence
        consensus items.

        Args:
            human_df: Full human scores (full_scores_both_annotators.csv).
            n_items: Number of gold items to select.
            output_path: If provided, save to CSV.
            seed: Random seed for reproducible selection.

        Returns:
            DataFrame with question_uuid + gold scores per dimension.
        """
        agree_mask = pd.Series(True, index=human_df.index)
        for dim, human_dim in _HUMAN_DIM_MAP.items():
            k_col = f"{human_dim}{_K_SUFFIX}"
            l_col = f"{human_dim}{_L_SUFFIX}"
            if k_col in human_df.columns and l_col in human_df.columns:
                agree_mask &= (human_df[k_col] == human_df[l_col])

        agreed = human_df[agree_mask].copy()
        logger.info(
            f"Gold standard candidates: {len(agreed)}/{len(human_df)} items "
            f"where K and L agree on all dims"
        )

        if len(agreed) < n_items:
            logger.warning(
                f"Only {len(agreed)} unanimous items available, "
                f"requested {n_items}. Using all available."
            )
            n_items = len(agreed)

        sample = agreed.sample(n=n_items, random_state=seed)

        # Build gold DF with canonical column names
        gold_rows = []
        for _, row in sample.iterrows():
            gold_row = {"question_uuid": row["question_uuid"]}
            for dim, human_dim in _HUMAN_DIM_MAP.items():
                k_col = f"{human_dim}{_K_SUFFIX}"
                gold_row[dim] = int(row[k_col])  # K == L, either works
            gold_rows.append(gold_row)

        gold_df = pd.DataFrame(gold_rows)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gold_df.to_csv(output_path, index=False)
            logger.info(f"Saved synthetic gold standard ({n_items} items) to {output_path}")

        return gold_df

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_on_gold(
        self,
        gated_items: list[dict],
        label: str = "",
    ) -> dict:
        """
        Evaluate pipeline output against the gold standard.

        Computes per-dimension:
            - Alpha (LLM auto-accepted scores vs gold)
            - Exact match percentage
            - Mean absolute error
            - Within-1 percentage

        Args:
            gated_items: Output from ConfidenceGate.evaluate_batch().
            label: Human-readable label for this evaluation (e.g. batch name).

        Returns:
            dict with per-dimension metrics, alerts, and timestamp.
        """
        if self._gold_df is None:
            raise ValueError("No gold standard loaded. Call create_synthetic_gold() first.")

        gold_uuids = set(self._gold_df["question_uuid"].astype(str))

        # Filter gated items to gold set only
        gold_items = [
            item for item in gated_items
            if str(item["question_uuid"]) in gold_uuids
        ]

        if not gold_items:
            logger.warning("No gated items overlap with gold standard")
            return {"error": "no_overlap", "n_matched": 0}

        # Build lookup: uuid -> {dim -> gold_score}
        gold_lookup = {}
        for _, row in self._gold_df.iterrows():
            uuid = str(row["question_uuid"])
            gold_lookup[uuid] = {
                dim: row[dim] for dim in ALL_PIPELINE_DIMENSIONS if dim in row
            }

        # Evaluate per dimension
        alphas = {}
        for dim in ALL_PIPELINE_DIMENSIONS:
            llm_scores = []
            gold_scores = []

            for item in gold_items:
                uuid = str(item["question_uuid"])
                gold_score = gold_lookup.get(uuid, {}).get(dim)
                if gold_score is None or np.isnan(gold_score):
                    continue

                # Get LLM score for this dimension
                llm_score = self._extract_llm_score(item, dim)
                if llm_score is None or np.isnan(llm_score):
                    continue

                llm_scores.append(llm_score)
                gold_scores.append(gold_score)

            n = len(llm_scores)
            if n < 3:
                alphas[dim] = {
                    "alpha": float("nan"),
                    "exact_match_pct": 0.0,
                    "within_1_pct": 0.0,
                    "mae": float("nan"),
                    "bias": float("nan"),
                    "n_items": n,
                }
                continue

            llm_arr = np.array(llm_scores, dtype=float)
            gold_arr = np.array(gold_scores, dtype=float)

            alpha_result = compute_krippendorff_alpha_multi(llm_arr, gold_arr)
            exact = (llm_arr == gold_arr).mean() * 100
            within1 = (np.abs(llm_arr - gold_arr) <= 1).mean() * 100
            mae = np.abs(llm_arr - gold_arr).mean()
            bias = (llm_arr - gold_arr).mean()

            alphas[dim] = {
                "alpha": round(alpha_result["alpha"], 4),
                "exact_match_pct": round(exact, 1),
                "within_1_pct": round(within1, 1),
                "mae": round(float(mae), 3),
                "bias": round(float(bias), 3),
                "n_items": n,
            }

        # Check for drift
        alerts = self.check_drift(alphas)

        evaluation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "n_gold_items": len(self._gold_df),
            "n_matched": len(gold_items),
            "alphas": alphas,
            "alerts": alerts,
        }

        self._history.append(evaluation)

        # Set baseline from first evaluation
        if self._baseline is None:
            self._baseline = alphas
            logger.info("Set baseline alphas from first gold evaluation")

        # Persist history
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump(self._history, f, indent=2, default=str)

        logger.info(
            f"Gold evaluation ({label}): {len(gold_items)} items matched | "
            f"Alerts: {len(alerts)}"
        )
        for dim, metrics in alphas.items():
            logger.info(
                f"  {dim:25s} | α={metrics['alpha']:.4f} | "
                f"exact={metrics['exact_match_pct']:.1f}% | "
                f"w/in1={metrics['within_1_pct']:.1f}% | "
                f"bias={metrics['bias']:+.3f}"
            )

        return evaluation

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(
        self,
        current_alphas: dict,
        baseline_override: Optional[dict] = None,
    ) -> list[dict]:
        """
        Compare current gold-standard Alpha to baseline.

        Returns a list of alert dicts for dimensions where Alpha
        dropped more than ALPHA_DROP_THRESHOLD.
        """
        baseline = baseline_override or self._baseline
        if not baseline:
            return []  # No baseline yet

        alerts = []
        for dim, current in current_alphas.items():
            base = baseline.get(dim)
            if not base:
                continue

            current_alpha = current["alpha"] if isinstance(current, dict) else current
            base_alpha = base["alpha"] if isinstance(base, dict) else base

            if np.isnan(current_alpha) or np.isnan(base_alpha):
                continue

            drop = base_alpha - current_alpha
            if drop > ALPHA_DROP_THRESHOLD:
                alert = {
                    "dimension": dim,
                    "baseline_alpha": round(base_alpha, 4),
                    "current_alpha": round(current_alpha, 4),
                    "drop": round(drop, 4),
                    "severity": "critical" if drop > 2 * ALPHA_DROP_THRESHOLD else "warning",
                }
                alerts.append(alert)
                logger.warning(
                    f"DRIFT ALERT [{alert['severity']}]: {dim} Alpha dropped "
                    f"{drop:.4f} (from {base_alpha:.4f} to {current_alpha:.4f})"
                )

        return alerts

    # ------------------------------------------------------------------
    # History / time-series
    # ------------------------------------------------------------------

    def get_history(self) -> pd.DataFrame:
        """
        Return time-series of gold-standard evaluations.

        Returns:
            DataFrame with columns: timestamp, label, dimension,
            alpha, exact_match_pct, within_1_pct, mae, bias
        """
        if not self._history:
            return pd.DataFrame()

        rows = []
        for entry in self._history:
            for dim, metrics in entry.get("alphas", {}).items():
                rows.append({
                    "timestamp": entry["timestamp"],
                    "label": entry.get("label", ""),
                    "dimension": dim,
                    "alpha": metrics.get("alpha", float("nan")),
                    "exact_match_pct": metrics.get("exact_match_pct", 0),
                    "within_1_pct": metrics.get("within_1_pct", 0),
                    "mae": metrics.get("mae", float("nan")),
                    "bias": metrics.get("bias", float("nan")),
                    "n_items": metrics.get("n_items", 0),
                })

        return pd.DataFrame(rows)

    def get_gold_df(self) -> Optional[pd.DataFrame]:
        """Return the gold standard DataFrame."""
        return self._gold_df

    def set_gold_df(self, df: pd.DataFrame) -> None:
        """Set the gold standard DataFrame directly."""
        self._gold_df = df
        logger.info(f"Gold standard set: {len(df)} items")

    # ------------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------------

    def confusion_matrix(
        self,
        gated_items: list[dict],
        dimension: str,
    ) -> Optional[pd.DataFrame]:
        """
        Build a confusion matrix (gold vs LLM) for one dimension.

        Returns:
            DataFrame crosstab, or None if insufficient data.
        """
        if self._gold_df is None:
            return None

        gold_uuids = set(self._gold_df["question_uuid"].astype(str))
        gold_lookup = {}
        for _, row in self._gold_df.iterrows():
            if dimension in row:
                gold_lookup[str(row["question_uuid"])] = row[dimension]

        gold_vals, llm_vals = [], []
        for item in gated_items:
            uuid = str(item["question_uuid"])
            if uuid not in gold_uuids:
                continue
            gold_score = gold_lookup.get(uuid)
            llm_score = self._extract_llm_score(item, dimension)
            if gold_score is not None and llm_score is not None:
                if not (np.isnan(gold_score) or np.isnan(llm_score)):
                    gold_vals.append(int(gold_score))
                    llm_vals.append(int(llm_score))

        if len(gold_vals) < 3:
            return None

        return pd.crosstab(
            pd.Series(gold_vals, name="Gold"),
            pd.Series(llm_vals, name="LLM"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_llm_score(item: dict, dimension: str) -> Optional[float]:
        """
        Extract the LLM score for a dimension from a gated item.

        For Tier A: uses the gate decision's accepted_score (majority/unanimous).
        For Tier C: uses the median of the 3 LLM runs.
        """
        # Try Tier A gate decision first
        gate_dec = item.get("gate_decisions", {}).get(dimension, {})
        accepted = gate_dec.get("accepted_score")
        if accepted is not None and not (isinstance(accepted, float) and np.isnan(accepted)):
            return float(accepted)

        # Fall back to median of runs from tier_a or tier_c
        for tier_key in ("tier_a", "tier_c"):
            scores = item.get(tier_key, {}).get(dimension, [])
            valid = [s for s in scores if not (isinstance(s, float) and np.isnan(s))]
            if valid:
                return float(np.median(valid))

        return None
