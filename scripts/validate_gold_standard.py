"""
CI-style gold standard validation script.

Runs Phase 1 zero-shot results through the pipeline and evaluates
against the gold standard. Exits with code 1 if any Tier A dimension's
Alpha drops below threshold.

Usage:
    python scripts/validate_gold_standard.py
"""
import sys
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import DimensionRouter, ConfidenceGate, QualityMonitor

GOLD_PATH = ROOT / "Data" / "gold_standard" / "synthetic_gold_30.csv"
HISTORY_PATH = ROOT / "Data" / "gold_standard" / "evaluation_history.json"
PHASE1_PATH = ROOT / "Data" / "results" / "phase1_question_level" / "zero_shot_qlevel" / "zero_shot_qlevel_results.csv"

DIMS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility", "Semantic_Uniqueness"]


def load_and_gate(csv_path: Path) -> list[dict]:
    """Load Phase 1 CSV, reshape, route + gate."""
    df = pd.read_csv(csv_path)
    engine_items = []
    for _, row in df.iterrows():
        uuid = str(row["question_uuid"])
        runs = []
        for ri in range(3):
            scores = {}
            for dim in DIMS:
                col = f"{dim}_all_runs"
                if col in df.columns:
                    try:
                        rl = ast.literal_eval(str(row[col]))
                        scores[dim] = rl[ri] if ri < len(rl) else np.nan
                    except:
                        scores[dim] = np.nan
                else:
                    scores[dim] = row.get(dim, np.nan)
            runs.append({"scores": scores})
        engine_items.append({"question_uuid": uuid, "runs": runs})

    router = DimensionRouter()
    gate = ConfidenceGate()
    routed = router.route_batch(engine_items)
    return gate.evaluate_batch(routed)


def main():
    print("=" * 70)
    print("GOLD STANDARD VALIDATION")
    print("=" * 70)

    if not GOLD_PATH.exists():
        print(f"ERROR: Gold standard not found at {GOLD_PATH}")
        sys.exit(1)

    if not PHASE1_PATH.exists():
        print(f"ERROR: Phase 1 results not found at {PHASE1_PATH}")
        sys.exit(1)

    # Run pipeline
    gated = load_and_gate(PHASE1_PATH)
    print(f"Loaded {len(gated)} items from Phase 1 zero-shot")

    # Evaluate on gold standard
    monitor = QualityMonitor(
        gold_standard_path=GOLD_PATH,
        history_path=HISTORY_PATH,
    )
    result = monitor.evaluate_on_gold(gated, label="ci_validation")

    # Report
    print(f"\nMatched {result['n_matched']}/{result['n_gold_items']} gold items\n")
    print(f"{'Dimension':<25} {'Alpha':>7} {'Exact%':>7} {'W/in1%':>7} {'MAE':>6} {'Bias':>6} {'N':>4}")
    print("-" * 70)

    failed = False
    for dim, m in result["alphas"].items():
        alpha = m["alpha"]
        flag = ""
        if dim == "Linguistic_Feature" and (np.isnan(alpha) or alpha < 0.3):
            flag = " *** FAIL ***"
            failed = True
        print(
            f"{dim:<25} {alpha:>7.4f} {m['exact_match_pct']:>6.1f}% "
            f"{m['within_1_pct']:>6.1f}% {m['mae']:>6.3f} {m['bias']:>+5.3f} {m['n_items']:>4}{flag}"
        )

    # Alerts
    if result.get("alerts"):
        print(f"\n{'!' * 50}")
        for alert in result["alerts"]:
            print(
                f"  DRIFT ALERT [{alert['severity']}]: {alert['dimension']} "
                f"dropped {alert['drop']:.4f} "
                f"(from {alert['baseline_alpha']:.4f} to {alert['current_alpha']:.4f})"
            )
        print(f"{'!' * 50}")

    # Confusion matrix for Linguistic Feature
    cm = monitor.confusion_matrix(gated, "Linguistic_Feature")
    if cm is not None:
        print("\nConfusion Matrix: Linguistic Feature (Gold vs LLM):")
        print(cm.to_string())

    if failed:
        print("\n*** VALIDATION FAILED: Linguistic Feature Alpha below threshold ***")
        sys.exit(1)
    else:
        print("\n[OK] Validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
