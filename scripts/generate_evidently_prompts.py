"""
Generate optimized annotation prompts using Evidently AI's prompt optimizer.

For each of the 5 ordinal dimensions, feeds K+L human consensus labels + question
text into Evidently's optimizer and saves the resulting prompts.

Usage:
    python scripts/generate_evidently_prompts.py [--n-optimize 100] [--n-iter 5]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

DIMS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility", "Semantic_Uniqueness"]
HUMAN_DIM_MAP = {
    "Relevance": "Relevance",
    "Clarity": "Clarity",
    "Linguistic_Feature": "Linguistic_Feature",
    "Semantic_Plausibility": "Semantic_Plausibility",
    "Semantic_Uniqueness": "Semantic_Unique",
}
K_SUFFIX = "_KC"
L_SUFFIX = "_LH"

OUTPUT_DIR = ROOT / "prompts" / "evidently_optimized"


def load_data(n_optimize: int):
    """Load human data + item text for the optimization split."""
    human_df = pd.read_csv(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")
    items_df = pd.read_csv(ROOT / "Data" / "processed" / "pipeline_upload_200items.csv")

    # Merge text info
    merged = human_df.merge(items_df[["question_uuid", "text", "question", "answer"]],
                            on="question_uuid", how="left")

    # Use first n_optimize items for prompt optimization (train split)
    train = merged.head(n_optimize).copy()

    # Compute consensus per dimension
    for dim, hd in HUMAN_DIM_MAP.items():
        k_col = f"{hd}{K_SUFFIX}"
        l_col = f"{hd}{L_SUFFIX}"
        train[f"{dim}_consensus"] = ((train[k_col] + train[l_col]) / 2).round().astype(int)

    return train


def build_item_text(row) -> str:
    """Build the evaluation text for one item."""
    text = str(row.get("text", ""))[:2000]
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    return (
        f"Text passage: {text}\n\n"
        f"Question: {question}\n\n"
        f"Correct answer: {answer}"
    )


def optimize_with_evidently(train_df: pd.DataFrame, dim: str, n_iter: int = 5):
    """
    Use Evidently's prompt optimizer for a single dimension.

    Returns the optimized prompt string.
    """
    try:
        from evidently.llm.prompts import PromptOptimizer
    except ImportError:
        # Try alternative import paths (Evidently API changes across versions)
        try:
            from evidently.features.llm_judge import PromptOptimizer
        except ImportError:
            return _optimize_manual_fallback(train_df, dim)

    # Build labeled dataset for this dimension
    items = []
    labels = []
    for _, row in train_df.iterrows():
        item_text = build_item_text(row)
        consensus = row[f"{dim}_consensus"]
        items.append(item_text)
        labels.append(str(consensus))

    # Initial criteria
    dim_descriptions = {
        "Relevance": "Rate how well the question addresses concepts central to the passage (1=tangential, 4=directly covers main point)",
        "Clarity": "Rate how clearly the question stem and options are worded (1=confusing, 4=perfectly clear)",
        "Linguistic_Feature": "Rate how similar the distractors are to the correct answer in length, structure, and style (1=visibly different, 4=uniform)",
        "Semantic_Plausibility": "Rate whether distractors would seem plausible to someone who hasn't read the passage (1=obviously wrong, 4=all plausible)",
        "Semantic_Uniqueness": "Rate whether each distractor is wrong for a different reason (1=duplicates, 4=all unique errors)",
    }

    criteria = dim_descriptions.get(dim, f"Rate {dim} on a 1-4 scale")

    try:
        optimizer = PromptOptimizer(
            criteria=criteria,
            target_labels=["1", "2", "3", "4"],
            n_iterations=n_iter,
        )
        result = optimizer.optimize(
            texts=items,
            labels=labels,
        )
        return result.best_prompt
    except Exception as e:
        print(f"  Evidently optimizer failed for {dim}: {e}")
        print(f"  Falling back to manual optimization...")
        return _optimize_manual_fallback(train_df, dim)


def _optimize_manual_fallback(train_df: pd.DataFrame, dim: str) -> str:
    """
    Fallback: Use LLM to generate an optimized prompt by analyzing label patterns.
    This mimics what Evidently does but uses direct Gemini API calls.
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Build examples showing score distribution
    examples = []
    for score in [1, 2, 3, 4]:
        subset = train_df[train_df[f"{dim}_consensus"] == score]
        if len(subset) > 0:
            sample = subset.sample(min(3, len(subset)), random_state=42)
            for _, row in sample.iterrows():
                examples.append(f"Score {score}: {build_item_text(row)[:500]}")

    examples_text = "\n\n---\n\n".join(examples[:12])

    # Score distribution
    dist = train_df[f"{dim}_consensus"].value_counts().sort_index().to_dict()
    dist_text = ", ".join(f"Score {k}: {v} items" for k, v in dist.items())

    meta_prompt = f"""You are a prompt engineering expert. Your task is to write an optimized evaluation prompt for an LLM acting as an annotation judge.

DIMENSION: {dim}
SCALE: 1-4 ordinal (1=Strongly Disagree, 4=Strongly Agree)
SCORE DISTRIBUTION IN EXPERT DATA: {dist_text}

Here are examples of items and their expert-assigned scores:

{examples_text}

Based on these examples, write a precise evaluation prompt that:
1. Defines exactly what each score level (1-4) means for this dimension
2. Includes concrete decision rules that distinguish adjacent score levels
3. Addresses common scoring errors (e.g., positive bias: experts give lower scores than naive raters)
4. Matches the expert score distribution (note: scores are NOT uniformly distributed)

Output ONLY the evaluation prompt text. No preamble. The prompt should instruct an LLM to score a given MCQ on this dimension using a 1-4 scale and return JSON with the key "{dim}" containing an integer 1-4.
"""

    response = model.generate_content(meta_prompt)
    return response.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-optimize", type=int, default=100, help="Items for optimization")
    parser.add_argument("--n-iter", type=int, default=5, help="Evidently optimization iterations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = load_data(args.n_optimize)
    print(f"  Train split: {len(train_df)} items")

    prompts = {}
    for dim in DIMS:
        print(f"\nOptimizing prompt for: {dim}")
        dist = train_df[f"{dim}_consensus"].value_counts().sort_index().to_dict()
        print(f"  Score distribution: {dist}")

        prompt = optimize_with_evidently(train_df, dim, n_iter=args.n_iter)

        # Save individual prompt
        prompt_path = OUTPUT_DIR / f"{dim.lower()}_optimized.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"  Saved to {prompt_path}")
        print(f"  Prompt length: {len(prompt)} chars")

        prompts[dim] = prompt

    # Save combined prompt (all 5 dimensions in one)
    combined = _build_combined_prompt(prompts)
    combined_path = OUTPUT_DIR / "combined_all_dims.txt"
    combined_path.write_text(combined, encoding="utf-8")
    print(f"\nSaved combined prompt to {combined_path}")

    # Save metadata
    meta = {
        "n_optimize_items": args.n_optimize,
        "n_iterations": args.n_iter,
        "dimensions": DIMS,
        "prompt_lengths": {dim: len(p) for dim, p in prompts.items()},
    }
    with open(OUTPUT_DIR / "optimization_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone! All prompts saved to prompts/evidently_optimized/")


def _build_combined_prompt(prompts: dict) -> str:
    """Combine per-dimension optimized prompts into a single evaluation prompt."""
    sections = []
    for dim, prompt in prompts.items():
        sections.append(f"## {dim}\n\n{prompt}")

    header = """You are a strict educational assessment evaluator. Score the following MCQ on each dimension using the criteria below.

You MUST respond with a valid JSON object containing integer scores (1-4) for each dimension:
{
  "Relevance": <1-4>,
  "Clarity": <1-4>,
  "Linguistic_Feature": <1-4>,
  "Semantic_Plausibility": <1-4>,
  "Semantic_Uniqueness": <1-4>,
  "confidence": <0-100>,
  "reasoning": "<brief reasoning>"
}

"""
    return header + "\n\n".join(sections)


if __name__ == "__main__":
    main()
