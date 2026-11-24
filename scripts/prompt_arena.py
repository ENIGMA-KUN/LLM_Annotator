"""
Prompt Arena: promptfoo-style side-by-side prompt comparison.

Runs N prompt variations (3-5 recommended) against the 200-item
human-annotated golden baseline and ranks them by Krippendorff's alpha
(ordinal) per dimension, with special focus on Linguistic_Feature
(clause boundary violations / run-on detection).

Each prompt variation runs the full 3-run protocol so the arena also
reports inter-run stability (3/3 consensus rate) per prompt; a prompt
that scores well but is unstable is flagged.

Usage:
    # Compare specific prompt files
    python scripts/prompt_arena.py --prompts prompts/best_prompt_v1.txt \
        prompts/strict_boundary_v1.txt prompts/zero_shot_qlevel.txt \
        --n-items 100

    # Dry run: estimate cost only, no API calls
    python scripts/prompt_arena.py --prompts prompts/*.txt --dry-run

Output:
    Data/results/prompt_arena/arena_leaderboard_<timestamp>.csv
    Data/results/prompt_arena/arena_raw_<prompt>_<timestamp>.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.pipeline.annotation_engine import AnnotationEngine, Q_LEVEL_DIMENSIONS
from src.pipeline.confidence_gate import ConfidenceGate, AGREE_3_OF_3
from src.pipeline.metrics import compute_krippendorff_alpha_multi

OUTPUT_DIR = ROOT / "Data" / "results" / "prompt_arena"

HUMAN_DIM_MAP = {
    "Relevance": "Relevance",
    "Clarity": "Clarity",
    "Linguistic_Feature": "Linguistic_Feature",
    "Semantic_Plausibility": "Semantic_Plausibility",
    "Semantic_Uniqueness": "Semantic_Unique",
}
K_SUFFIX = "_KC"
L_SUFFIX = "_LH"

# Cost model (Gemini 2.5 Flash); rough per-call estimate for --dry-run
EST_TOKENS_PER_CALL = 4000
EST_COST_PER_CALL = 4000 * 0.15 / 1e6 + 1500 * 0.60 / 1e6


def load_golden_baseline(n_items: int) -> pd.DataFrame:
    """Load the human-annotated golden dataset with per-item consensus."""
    df = pd.read_csv(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")
    for dim, hd in HUMAN_DIM_MAP.items():
        k_col, l_col = f"{hd}{K_SUFFIX}", f"{hd}{L_SUFFIX}"
        df[f"{dim}_consensus"] = ((df[k_col] + df[l_col]) / 2).round().astype(int)
    return df.head(n_items)


def build_items(df: pd.DataFrame) -> list[dict]:
    """Convert golden rows into AnnotationEngine items."""
    return [
        {
            "question_uuid": str(row["question_uuid"]),
            "text": str(row.get("text", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        }
        for _, row in df.iterrows()
    ]


def median_scores(results: list[dict]) -> dict[str, dict[str, float]]:
    """{uuid: {dim: median score across 3 runs}}."""
    out = {}
    for item in results:
        per_dim = {}
        for dim in Q_LEVEL_DIMENSIONS:
            vals = [
                r["scores"].get(dim, np.nan)
                for r in item["runs"]
                if not (isinstance(r["scores"].get(dim), float)
                        and np.isnan(r["scores"].get(dim)))
            ]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            per_dim[dim] = float(np.median(vals)) if vals else np.nan
        out[item["question_uuid"]] = per_dim
    return out


def consensus_rate(results: list[dict], dim: str) -> float:
    """Fraction of items with 3/3 inter-run consensus on a dimension."""
    gate = ConfidenceGate()
    n_auto = 0
    for item in results:
        scores = [r["scores"].get(dim, np.nan) for r in item["runs"]]
        if gate.evaluate(scores)["decision"] == AGREE_3_OF_3:
            n_auto += 1
    return n_auto / max(1, len(results))


def evaluate_prompt(
    prompt_path: Path,
    items: list[dict],
    golden: pd.DataFrame,
    stamp: str,
) -> dict:
    """Run one prompt through the 3-run protocol and score it."""
    prompt_text = prompt_path.read_text(encoding="utf-8")
    name = prompt_path.stem

    engine = AnnotationEngine(prompt_version=f"arena_{name}")
    results = engine.annotate_batch(
        items, system_prompt=prompt_text, condition_name=f"arena_{name}"
    )
    cost = engine.get_cost_summary()

    # Persist raw results for replay/audit
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / f"arena_raw_{name}_{stamp}.json"
    serializable = json.loads(json.dumps(results, default=str))
    raw_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    # Score: alpha (LLM median vs each human + consensus) per dimension
    med = median_scores(results)
    row = {"prompt": name, "cost_usd": cost["total_cost_usd"], "n_items": len(items)}
    for dim in Q_LEVEL_DIMENSIONS:
        llm = np.array([
            med.get(str(u), {}).get(dim, np.nan)
            for u in golden["question_uuid"].astype(str)
        ])
        human = golden[f"{dim}_consensus"].values.astype(float)
        alpha = compute_krippendorff_alpha_multi(human, llm, level="ordinal")
        row[f"alpha_{dim}"] = alpha["alpha"]
        row[f"consensus3of3_{dim}"] = round(consensus_rate(results, dim), 3)
    return row


def main():
    parser = argparse.ArgumentParser(description="Promptfoo-style prompt arena")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="3-5 prompt .txt files to compare")
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost only, no API calls")
    args = parser.parse_args()

    prompt_paths = [Path(p) for p in args.prompts]
    missing = [p for p in prompt_paths if not p.exists()]
    if missing:
        sys.exit(f"Prompt files not found: {missing}")

    n_calls = len(prompt_paths) * args.n_items * 3
    est_cost = n_calls * EST_COST_PER_CALL
    print(f"Arena: {len(prompt_paths)} prompts x {args.n_items} items x 3 runs "
          f"= {n_calls} calls (~${est_cost:.2f} estimated)")

    if args.dry_run:
        print("Dry run: no API calls made.")
        return

    golden = load_golden_baseline(args.n_items)
    items = build_items(golden)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = []
    for p in prompt_paths:
        print(f"\n=== Evaluating prompt: {p.stem} ===")
        rows.append(evaluate_prompt(p, items, golden, stamp))

    board = pd.DataFrame(rows).sort_values(
        "alpha_Linguistic_Feature", ascending=False, ignore_index=True
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"arena_leaderboard_{stamp}.csv"
    board.to_csv(out_path, index=False)

    print("\n=== LEADERBOARD (ranked by Linguistic_Feature alpha) ===")
    cols = ["prompt", "alpha_Linguistic_Feature",
            "consensus3of3_Linguistic_Feature", "cost_usd"]
    print(board[cols].to_string(index=False))
    print(f"\nFull leaderboard saved: {out_path}")


if __name__ == "__main__":
    main()
