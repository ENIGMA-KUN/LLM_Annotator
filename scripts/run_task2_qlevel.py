"""
Task 2: Question-Level Re-run with V3-4level Rubric.
New API calls: runs 3 conditions (zero-shot, rubric-guided, few-shot) on
200 question-level items, 3 runs each, using Gemini 2.5 Flash.

Evaluates LLM against BOTH annotators (K and L) on the same 200 items.
"""

import json
import ast
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import krippendorff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
HUMAN_CSV = DATA / "processed" / "full_scores_both_annotators.csv"
RESULTS = DATA / "results"
OUT_DIR = RESULTS / "phase1_question_level"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR = ROOT / "prompts"

# Question-level dimensions
QLEVEL_ORDINAL_DIMS = [
    "Relevance", "Clarity", "Linguistic_Feature",
    "Semantic_Plausibility", "Semantic_Uniqueness",
]
QLEVEL_BINARY_DIMS = ["Correctness", "Incorrectness"]
QLEVEL_ALL_DIMS = QLEVEL_BINARY_DIMS + QLEVEL_ORDINAL_DIMS

CONDITIONS = ["zero_shot_qlevel", "rubric_guided_qlevel", "few_shot_qlevel"]
COND_LABELS = {
    "zero_shot_qlevel": "C1: Zero-shot",
    "rubric_guided_qlevel": "C2: Rubric-guided",
    "few_shot_qlevel": "C3: Few-shot",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def alpha_ordinal(rater1, rater2):
    r1, r2 = np.array(rater1, dtype=float), np.array(rater2, dtype=float)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    if len(r1) < 2:
        return np.nan
    try:
        return krippendorff.alpha(
            reliability_data=np.array([r1, r2]),
            level_of_measurement="ordinal",
        )
    except ValueError:
        return np.nan


def alpha_multi_rater(ratings_list):
    data = np.array(ratings_list, dtype=float)
    if data.shape[1] < 2:
        return np.nan
    try:
        return krippendorff.alpha(
            reliability_data=data, level_of_measurement="ordinal"
        )
    except ValueError:
        return np.nan


def kappa_weighted(h, l):
    h, l = np.array(h, dtype=float), np.array(l, dtype=float)
    mask = ~(np.isnan(h) | np.isnan(l))
    h, l = h[mask].astype(int), l[mask].astype(int)
    if len(h) < 2:
        return np.nan
    return cohen_kappa_score(h, l, weights="quadratic", labels=[1, 2, 3, 4])


def parse_all_runs(series):
    def _parse(val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []
    return series.apply(_parse)


# ── 1. Load human data ──────────────────────────────────────────────────────

print("=" * 70)
print("TASK 2: Question-Level Re-run with V3-4level Rubric")
print("=" * 70)

df_human = pd.read_csv(HUMAN_CSV)
print(f"Human data: {len(df_human)} questions, columns: {df_human.columns.tolist()}")

# Map human columns: Relevance_KC -> Relevance for annotator K, etc.
HUMAN_DIM_MAP = {
    "Relevance": ("Relevance_KC", "Relevance_LH"),
    "Clarity": ("Clarity_KC", "Clarity_LH"),
    "Linguistic_Feature": ("Linguistic_Feature_KC", "Linguistic_Feature_LH"),
    "Semantic_Plausibility": ("Semantic_Plausibility_KC", "Semantic_Plausibility_LH"),
    "Semantic_Uniqueness": ("Semantic_Unique_KC", "Semantic_Unique_LH"),
    "Correctness": ("Correctness_KC", "Correctness_LH"),
    "Incorrectness": ("Incorrectness_KC", "Incorrectness_LH"),
}


# ── 2. Few-shot example selection ───────────────────────────────────────────

def select_few_shot_examples(df, n_examples=8):
    """
    Select stratified few-shot examples where both annotators agreed.
    Returns list of example dicts and their indices (to exclude from eval).
    """
    # Find items where K and L agree on all ordinal dimensions (within 1)
    agree_mask = pd.Series(True, index=df.index)
    for dim in QLEVEL_ORDINAL_DIMS:
        kc_col, lh_col = HUMAN_DIM_MAP[dim]
        if kc_col in df.columns and lh_col in df.columns:
            agree_mask &= (abs(df[kc_col] - df[lh_col]) <= 1)

    agreed = df[agree_mask].copy()
    print(f"  Items where K and L agree (within 1): {len(agreed)} / {len(df)}")

    # Stratify by Relevance score (use KC as reference)
    selected_indices = []
    per_stratum = max(1, n_examples // 4)

    for score in [1, 2, 3, 4]:
        pool = agreed[agreed["Relevance_KC"] == score]
        if len(pool) == 0:
            pool = agreed  # fallback
        # Try to pick from different textIDs
        sample = pool.groupby("textID").first().reset_index()
        if len(sample) >= per_stratum:
            sample = sample.sample(n=per_stratum, random_state=42)
        else:
            sample = pool.sample(n=min(per_stratum, len(pool)), random_state=42)
        selected_indices.extend(sample.index.tolist() if "question_uuid" not in sample.columns
                                else df[df["question_uuid"].isin(sample["question_uuid"])].index.tolist())

    # Deduplicate and limit
    selected_indices = list(dict.fromkeys(selected_indices))[:n_examples]
    selected = df.loc[selected_indices]

    examples = []
    for _, row in selected.iterrows():
        # Use consensus (average of K and L, rounded) as the example scores
        ex = {
            "question_uuid": row["question_uuid"],
            "textID": row["textID"],
            "text": str(row["text"])[:800],
            "question": row["question"],
            "answer": row["answer"],
        }
        for dim in QLEVEL_ORDINAL_DIMS:
            kc_col, lh_col = HUMAN_DIM_MAP[dim]
            ex[dim] = int(round((row[kc_col] + row[lh_col]) / 2))
        for dim in QLEVEL_BINARY_DIMS:
            kc_col, lh_col = HUMAN_DIM_MAP[dim]
            ex[dim] = int(row[kc_col])  # binary, so use K
        examples.append(ex)

    return examples, selected_indices


def format_examples_qlevel(examples):
    """Format few-shot examples for injection into the prompt."""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"""--- Example {i} ---
Text (excerpt): {ex['text']}...
Question: {ex['question']}
Correct Answer: {ex['answer']}
Expert Annotations:
  Correctness: {ex['Correctness']}
  Incorrectness: {ex['Incorrectness']}
  Relevance: {ex['Relevance']}
  Clarity: {ex['Clarity']}
  Linguistic_Feature: {ex['Linguistic_Feature']}
  Semantic_Plausibility: {ex['Semantic_Plausibility']}
  Semantic_Uniqueness: {ex['Semantic_Uniqueness']}
""")
    return "\n".join(parts)


# ── 3. LLM Annotation Engine ────────────────────────────────────────────────

def run_annotation_qlevel(
    df_items, system_prompt, condition_name, num_runs=3,
    exclude_indices=None,
):
    """
    Run LLM annotation at question level.
    Returns DataFrame of aggregated results and list of raw responses.
    """
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from src.utils import load_env
    from src.metadata import get_metadata_logger
    import re

    api_key = load_env()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )

    # Log metadata
    meta = get_metadata_logger()
    meta.log_model(
        model_name="gemini-2.5-flash",
        temperature=0.3,
        num_runs=num_runs,
        response_format="application/json",
        extra={"condition": condition_name, "level": "question"},
    )
    meta.log_experiment(
        condition_name=condition_name,
        prompt_name=condition_name,
        prompt_text=system_prompt[:500],
        n_items=len(df_items),
        num_runs=num_runs,
    )
    meta.start_timer(f"experiment_{condition_name}")

    # Filter out few-shot examples for C3
    if exclude_indices is not None:
        eval_df = df_items.drop(index=exclude_indices, errors="ignore").reset_index(drop=True)
        print(f"  Excluding {len(exclude_indices)} few-shot examples. Evaluating {len(eval_df)} items.")
    else:
        eval_df = df_items.copy()

    all_results = []
    raw_responses = []
    total_tokens = 0
    total_cost = 0.0

    for idx, row in eval_df.iterrows():
        user_prompt = f"""Please evaluate the following multiple-choice question at the QUESTION level.

**Text Passage:**
{str(row['text'])[:3000]}

**Question:**
{row['question']}

**Correct Answer:**
{row['answer']}

Evaluate the overall quality of this question (stem + correct answer + all distractors together) using the rubric. Respond with a JSON object."""

        run_annotations = []
        for run_id in range(num_runs):
            parsed = None
            for attempt in range(5):
                try:
                    response = model.generate_content(
                        contents=[{"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}],
                    )
                    usage = response.usage_metadata
                    prompt_tok = getattr(usage, "prompt_token_count", 0) or 0
                    comp_tok = getattr(usage, "candidates_token_count", 0) or 0
                    tokens = prompt_tok + comp_tok
                    total_tokens += tokens
                    cost = (prompt_tok * 0.10 / 1_000_000) + (comp_tok * 0.40 / 1_000_000)
                    total_cost += cost

                    content = response.text
                    parsed = json.loads(content)
                    break
                except (google_exceptions.ResourceExhausted, google_exceptions.DeadlineExceeded,
                        google_exceptions.ServiceUnavailable) as e:
                    delay = 2.0 * (2 ** attempt) + np.random.uniform(0, 1)
                    print(f"    Rate limit (attempt {attempt+1}/5). Waiting {delay:.0f}s...")
                    time.sleep(delay)
                except json.JSONDecodeError:
                    # Try extracting from markdown
                    import re as re_mod
                    m = re_mod.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re_mod.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(1))
                            break
                        except json.JSONDecodeError:
                            pass
                    if attempt == 4:
                        parsed = {}
                    time.sleep(1)
                except Exception as e:
                    print(f"    Error: {e}")
                    if attempt == 4:
                        parsed = {}
                    time.sleep(2)

            if parsed is None:
                parsed = {}

            # Parse annotation
            annotation = {}
            for dim in QLEVEL_ORDINAL_DIMS:
                val = parsed.get(dim) or parsed.get(dim.lower()) or parsed.get(dim.replace("_", " "))
                if val is not None:
                    try:
                        val = int(float(val))
                        val = max(1, min(4, val))
                    except (ValueError, TypeError):
                        val = np.nan
                else:
                    val = np.nan
                annotation[dim] = val

            for dim in QLEVEL_BINARY_DIMS:
                val = parsed.get(dim) or parsed.get(dim.lower())
                if val is not None:
                    try:
                        val = int(float(val))
                        val = max(0, min(1, val))
                    except (ValueError, TypeError):
                        val = np.nan
                else:
                    val = np.nan
                annotation[dim] = val

            annotation["confidence"] = parsed.get("confidence", parsed.get("Confidence", 50))
            annotation["reasoning"] = parsed.get("reasoning", parsed.get("Reasoning", ""))
            try:
                annotation["confidence"] = int(float(annotation["confidence"]))
                annotation["confidence"] = max(0, min(100, annotation["confidence"]))
            except (ValueError, TypeError):
                annotation["confidence"] = 50

            annotation["run_id"] = run_id
            run_annotations.append(annotation)

            raw_responses.append({
                "question_uuid": row["question_uuid"],
                "run_id": run_id,
                "raw_response": parsed,
            })

        # Aggregate runs (majority vote for ordinal, mode for binary)
        aggregated = {"question_uuid": row["question_uuid"], "textID": row["textID"]}
        for dim in QLEVEL_ORDINAL_DIMS:
            scores = [r[dim] for r in run_annotations if not pd.isna(r.get(dim))]
            if scores:
                from statistics import mode, StatisticsError
                try:
                    aggregated[dim] = mode(scores)
                except StatisticsError:
                    aggregated[dim] = round(np.mean(scores))
                aggregated[f"{dim}_consistency"] = scores.count(aggregated[dim]) / len(scores)
                aggregated[f"{dim}_all_runs"] = scores
            else:
                aggregated[dim] = np.nan
                aggregated[f"{dim}_consistency"] = 0.0
                aggregated[f"{dim}_all_runs"] = []

        for dim in QLEVEL_BINARY_DIMS:
            scores = [r[dim] for r in run_annotations if not pd.isna(r.get(dim))]
            if scores:
                aggregated[dim] = int(round(np.mean(scores)))
                aggregated[f"{dim}_all_runs"] = scores
            else:
                aggregated[dim] = np.nan
                aggregated[f"{dim}_all_runs"] = []

        confidences = [r.get("confidence", 50) for r in run_annotations]
        aggregated["confidence"] = round(np.mean(confidences))
        aggregated["reasoning"] = run_annotations[0].get("reasoning", "")

        all_results.append(aggregated)

        # Progress
        if (len(all_results)) % 20 == 0:
            print(f"  [{condition_name}] {len(all_results)}/{len(eval_df)} done | Tokens: {total_tokens:,} | Cost: ${total_cost:.4f}")

    elapsed = meta.stop_timer(f"experiment_{condition_name}")
    meta.log_experiment_cost(
        condition_name=condition_name,
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 4),
        num_calls=len(raw_responses),
        successful_calls=len([r for r in raw_responses if r["raw_response"]]),
        duration_seconds=elapsed,
    )

    print(f"  [{condition_name}] COMPLETE | Items: {len(all_results)} | Tokens: {total_tokens:,} | Cost: ${total_cost:.4f}")
    return pd.DataFrame(all_results), raw_responses, {"total_tokens": total_tokens, "total_cost": round(total_cost, 4)}


# ── 4. Run conditions ───────────────────────────────────────────────────────

def run_all_conditions(df_human, skip_if_exists=True):
    """Run all 3 conditions and save results."""

    # Select few-shot examples first
    print("\n--- Selecting few-shot examples ---")
    examples, example_indices = select_few_shot_examples(df_human)
    examples_path = OUT_DIR / "few_shot_qlevel" / "selected_examples_qlevel.json"
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, default=str)
    print(f"  Saved {len(examples)} examples to {examples_path}")

    for cond in CONDITIONS:
        cond_dir = OUT_DIR / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        results_csv = cond_dir / f"{cond}_results.csv"

        if skip_if_exists and results_csv.exists():
            print(f"\n--- {cond}: Results exist, skipping ---")
            continue

        print(f"\n--- Running {cond} ---")
        prompt_path = PROMPTS_DIR / f"{cond}.txt"
        system_prompt = prompt_path.read_text(encoding="utf-8")

        exclude = example_indices if cond == "few_shot_qlevel" else None
        if cond == "few_shot_qlevel":
            examples_text = format_examples_qlevel(examples)
            system_prompt = system_prompt.replace("{{EXAMPLES}}", examples_text)

        df_results, raw_responses, cost_info = run_annotation_qlevel(
            df_human, system_prompt, cond, num_runs=3, exclude_indices=exclude,
        )

        # Save
        df_results.to_csv(results_csv, index=False)
        with open(cond_dir / f"{cond}_raw_responses.json", "w", encoding="utf-8") as f:
            json.dump(raw_responses, f, indent=2, default=str)
        with open(cond_dir / f"{cond}_cost.json", "w", encoding="utf-8") as f:
            json.dump(cost_info, f, indent=2)

        print(f"  Saved to {cond_dir}")


# ── 5. Evaluation ───────────────────────────────────────────────────────────

def evaluate_all_conditions(df_human):
    """Evaluate all conditions against both annotators."""

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    all_eval = {}
    comparison_rows = []

    for cond in CONDITIONS:
        cond_dir = OUT_DIR / cond
        results_csv = cond_dir / f"{cond}_results.csv"
        if not results_csv.exists():
            print(f"  {cond}: No results found, skipping evaluation")
            continue

        df_llm = pd.read_csv(results_csv)
        print(f"\n--- {cond} ({len(df_llm)} items) ---")

        # Merge with human data on question_uuid
        merged = df_human.merge(df_llm, on="question_uuid", suffixes=("_human", "_llm"))
        print(f"  Merged: {len(merged)} items")

        cond_eval = {"dimensions": {}, "aggregate": {}, "three_rater": {}}
        alphas_k, alphas_l, alphas_cons = [], [], []

        for dim in QLEVEL_ORDINAL_DIMS:
            kc_col, lh_col = HUMAN_DIM_MAP[dim]

            # Handle column name conflicts from merge
            kc = merged[kc_col].values.astype(float) if kc_col in merged.columns else None
            lh = merged[lh_col].values.astype(float) if lh_col in merged.columns else None

            # LLM scores - might have _llm suffix or not depending on merge
            llm_col = dim if dim in merged.columns else f"{dim}_llm"
            llm = merged[llm_col].values.astype(float) if llm_col in merged.columns else None

            if kc is None or lh is None or llm is None:
                print(f"  WARNING: Missing columns for {dim}")
                continue

            consensus = (kc + lh) / 2.0

            # Alpha: LLM vs K, LLM vs L, LLM vs consensus
            alpha_vs_k = alpha_ordinal(kc, llm)
            alpha_vs_l = alpha_ordinal(lh, llm)
            alpha_vs_cons = alpha_ordinal(consensus, llm)

            # Kappa (for comparison)
            kappa_vs_k = kappa_weighted(kc, llm)
            kappa_vs_l = kappa_weighted(lh, llm)

            # Bias
            bias_vs_k = float(np.nanmean(llm - kc))
            bias_vs_l = float(np.nanmean(llm - lh))
            bias_vs_cons = float(np.nanmean(llm - consensus))
            abs_dist_cons = float(np.nanmean(np.abs(llm - consensus)))

            # Human baseline Alpha for this dimension
            alpha_human = alpha_ordinal(kc, lh)

            # Three-rater Alpha: {K, L, LLM}
            alpha_3rater = alpha_multi_rater([kc, lh, llm])

            # Self-consistency
            runs_col = f"{dim}_all_runs"
            if runs_col in merged.columns:
                runs = parse_all_runs(merged[runs_col])
                sc = runs.apply(lambda r: len(set(r)) == 1 if len(r) > 0 else np.nan).mean()
            else:
                sc = np.nan

            dim_eval = {
                "alpha_vs_K": round(alpha_vs_k, 4) if not np.isnan(alpha_vs_k) else None,
                "alpha_vs_L": round(alpha_vs_l, 4) if not np.isnan(alpha_vs_l) else None,
                "alpha_vs_consensus": round(alpha_vs_cons, 4) if not np.isnan(alpha_vs_cons) else None,
                "alpha_human_KL": round(alpha_human, 4) if not np.isnan(alpha_human) else None,
                "alpha_three_rater_KL_LLM": round(alpha_3rater, 4) if not np.isnan(alpha_3rater) else None,
                "kappa_vs_K": round(kappa_vs_k, 4) if not np.isnan(kappa_vs_k) else None,
                "kappa_vs_L": round(kappa_vs_l, 4) if not np.isnan(kappa_vs_l) else None,
                "bias_vs_K": round(bias_vs_k, 4),
                "bias_vs_L": round(bias_vs_l, 4),
                "bias_vs_consensus": round(bias_vs_cons, 4),
                "abs_distance_consensus": round(abs_dist_cons, 4),
                "self_consistency": round(float(sc), 4) if not np.isnan(sc) else None,
                "n": len(merged),
            }
            cond_eval["dimensions"][dim] = dim_eval

            print(f"  {dim:25s} | alpha_K={dim_eval['alpha_vs_K']} | alpha_L={dim_eval['alpha_vs_L']} | "
                  f"α_cons={dim_eval['alpha_vs_consensus']} | α_human={dim_eval['alpha_human_KL']} | "
                  f"α_3rater={dim_eval['alpha_three_rater_KL_LLM']} | bias={bias_vs_cons:+.3f}")

            if dim_eval["alpha_vs_K"] is not None:
                alphas_k.append(dim_eval["alpha_vs_K"])
            if dim_eval["alpha_vs_L"] is not None:
                alphas_l.append(dim_eval["alpha_vs_L"])
            if dim_eval["alpha_vs_consensus"] is not None:
                alphas_cons.append(dim_eval["alpha_vs_consensus"])

            comparison_rows.append({
                "condition": cond,
                "dimension": dim,
                "alpha_vs_K": dim_eval["alpha_vs_K"],
                "alpha_vs_L": dim_eval["alpha_vs_L"],
                "alpha_vs_consensus": dim_eval["alpha_vs_consensus"],
                "alpha_human_KL": dim_eval["alpha_human_KL"],
                "alpha_3rater": dim_eval["alpha_three_rater_KL_LLM"],
                "bias_vs_consensus": dim_eval["bias_vs_consensus"],
                "abs_distance": dim_eval["abs_distance_consensus"],
                "self_consistency": dim_eval["self_consistency"],
            })

        cond_eval["aggregate"] = {
            "mean_alpha_vs_K": round(np.mean(alphas_k), 4) if alphas_k else None,
            "mean_alpha_vs_L": round(np.mean(alphas_l), 4) if alphas_l else None,
            "mean_alpha_vs_consensus": round(np.mean(alphas_cons), 4) if alphas_cons else None,
        }
        all_eval[cond] = cond_eval

    # Save evaluation
    with open(OUT_DIR / "qlevel_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(all_eval, f, indent=2, default=str)

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(OUT_DIR / "qlevel_comparison_table.csv", index=False)
    with open(OUT_DIR / "qlevel_comparison_table.txt", "w", encoding="utf-8") as f:
        f.write(df_comp.to_string(index=False))
    print(f"\nComparison table:\n{df_comp.to_string(index=False)}")

    return all_eval, df_comp


# ── 6. Figures ───────────────────────────────────────────────────────────────

def generate_figures(df_human, all_eval, df_comp):
    """Generate all Task 2 figures."""

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Collect available conditions
    avail_conds = [c for c in CONDITIONS if c in all_eval]
    if not avail_conds:
        print("No conditions to plot.")
        return

    # Fig 1: Alpha vs consensus by dimension
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(QLEVEL_ORDINAL_DIMS))
    width = 0.2
    for i, cond in enumerate(avail_conds):
        vals = [all_eval[cond]["dimensions"].get(d, {}).get("alpha_vs_consensus") or 0
                for d in QLEVEL_ORDINAL_DIMS]
        ax.bar(x + i * width, vals, width, label=COND_LABELS.get(cond, cond), zorder=3)

    # Human baseline bar
    human_alphas = []
    for d in QLEVEL_ORDINAL_DIMS:
        # Get from first available condition
        for c in avail_conds:
            val = all_eval[c]["dimensions"].get(d, {}).get("alpha_human_KL")
            if val is not None:
                human_alphas.append(val)
                break
        else:
            human_alphas.append(0)
    ax.bar(x + len(avail_conds) * width, human_alphas, width,
           label="Human (K vs L)", color="gold", edgecolor="black", zorder=3)

    ax.set_xticks(x + width * len(avail_conds) / 2)
    ax.set_xticklabels([d.replace("_", "\n") for d in QLEVEL_ORDINAL_DIMS])
    ax.set_ylabel("Krippendorff's Alpha (ordinal)")
    ax.set_title("Alpha vs Consensus: Question Level (n=200)")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_alpha_by_dimension_qlevel.png", dpi=150)
    plt.close()
    print("Saved fig1_alpha_by_dimension_qlevel.png")

    # Fig 2: Three-rater Alpha comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    dims_for_plot = QLEVEL_ORDINAL_DIMS
    x = np.arange(len(dims_for_plot))
    width = 0.25

    for i, cond in enumerate(avail_conds):
        vals_3r = [all_eval[cond]["dimensions"].get(d, {}).get("alpha_three_rater_KL_LLM") or 0
                   for d in dims_for_plot]
        ax.bar(x + i * width, vals_3r, width, label=f"{COND_LABELS.get(cond, cond)} (K+L+LLM)", zorder=3)

    ax.bar(x + len(avail_conds) * width, human_alphas[:len(dims_for_plot)], width,
           label="Human only (K vs L)", color="gold", edgecolor="black", zorder=3)

    ax.set_xticks(x + width * len(avail_conds) / 2)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims_for_plot])
    ax.set_ylabel("Krippendorff's Alpha")
    ax.set_title("Three-Rater Alpha: Does Adding LLM Help or Hurt?")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_three_rater_alpha_qlevel.png", dpi=150)
    plt.close()
    print("Saved fig2_three_rater_alpha_qlevel.png")

    # Fig 3: Bias histogram
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    for idx, dim in enumerate(QLEVEL_ORDINAL_DIMS):
        ax = axes_flat[idx]
        kc_col, lh_col = HUMAN_DIM_MAP[dim]
        consensus = (df_human[kc_col] + df_human[lh_col]) / 2.0

        for cond in avail_conds:
            cond_dir = OUT_DIR / cond
            results_csv = cond_dir / f"{cond}_results.csv"
            if not results_csv.exists():
                continue
            df_llm = pd.read_csv(results_csv)
            merged = df_human.merge(df_llm, on="question_uuid")
            llm_col = dim if dim in merged.columns else f"{dim}_y"
            if llm_col not in merged.columns:
                continue
            diff = merged[llm_col].values - consensus.loc[merged.index.intersection(consensus.index)].values[:len(merged)]
            ax.hist(diff, bins=np.arange(-3.5, 4.5, 1), alpha=0.4,
                    label=COND_LABELS.get(cond, cond), edgecolor="black")

        ax.set_title(dim.replace("_", " "))
        ax.set_xlabel("LLM − Consensus")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax.legend(fontsize=7)

    # Hide unused subplot
    if len(QLEVEL_ORDINAL_DIMS) < len(axes_flat):
        axes_flat[-1].set_visible(False)

    fig.suptitle("Score Difference Distribution: Question Level", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_bias_histogram_qlevel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_bias_histogram_qlevel.png")

    # Fig 4: Heatmap
    heatmap_data = []
    for cond in avail_conds:
        row = [all_eval[cond]["dimensions"].get(d, {}).get("alpha_vs_consensus") or 0
               for d in QLEVEL_ORDINAL_DIMS]
        heatmap_data.append(row)
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[COND_LABELS.get(c, c) for c in avail_conds],
        columns=[d.replace("_", " ") for d in QLEVEL_ORDINAL_DIMS],
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                vmin=-0.3, vmax=0.8, ax=ax, linewidths=0.5)
    ax.set_title("Alpha vs Consensus Heatmap: Question Level")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_alpha_heatmap_qlevel.png", dpi=150)
    plt.close()
    print("Saved fig4_alpha_heatmap_qlevel.png")


# ── 7. Summary ──────────────────────────────────────────────────────────────

def write_summary(all_eval, df_comp):
    """Write Task 2 summary markdown."""
    lines = [
        "# Task 2: Question-Level Re-run: Summary",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Items:** 200 questions (V3-4level rubric)",
        f"**Model:** Gemini 2.5 Flash, temp=0.3, 3 runs per item",
        f"**Primary metric:** Krippendorff's Alpha (ordinal)",
        "",
        "## Results Table",
        "",
        df_comp.to_string(index=False),
        "",
        "## Aggregate",
        "",
        "| Condition | Mean α vs K | Mean α vs L | Mean α vs Consensus |",
        "|-----------|------------|------------|-------------------|",
    ]
    for cond in CONDITIONS:
        if cond in all_eval:
            agg = all_eval[cond]["aggregate"]
            lines.append(f"| {cond} | {agg['mean_alpha_vs_K']} | {agg['mean_alpha_vs_L']} | {agg['mean_alpha_vs_consensus']} |")

    lines.extend([
        "",
        "## Three-Rater Alpha (K + L + LLM)",
        "",
        "| Condition | Dimension | Human α(K,L) | Three-Rater α(K,L,LLM) | Δ |",
        "|-----------|-----------|-------------|----------------------|---|",
    ])
    for cond in CONDITIONS:
        if cond not in all_eval:
            continue
        for dim in QLEVEL_ORDINAL_DIMS:
            d = all_eval[cond]["dimensions"].get(dim, {})
            h_alpha = d.get("alpha_human_KL")
            t_alpha = d.get("alpha_three_rater_KL_LLM")
            delta = round(t_alpha - h_alpha, 4) if h_alpha is not None and t_alpha is not None else None
            lines.append(f"| {cond} | {dim} | {h_alpha} | {t_alpha} | {delta} |")

    lines.extend([
        "",
        "## Figures",
        "- `figures/fig1_alpha_by_dimension_qlevel.png`",
        "- `figures/fig2_three_rater_alpha_qlevel.png`",
        "- `figures/fig3_bias_histogram_qlevel.png`",
        "- `figures/fig4_alpha_heatmap_qlevel.png`",
    ])

    with open(OUT_DIR / "qlevel_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to {OUT_DIR / 'qlevel_summary.md'}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Question-level LLM annotation")
    parser.add_argument("--skip-run", action="store_true", help="Skip annotation, only evaluate existing results")
    parser.add_argument("--force-rerun", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    if not args.skip_run:
        run_all_conditions(df_human, skip_if_exists=not args.force_rerun)

    all_eval, df_comp = evaluate_all_conditions(df_human)
    generate_figures(df_human, all_eval, df_comp)
    write_summary(all_eval, df_comp)

    print("\n" + "=" * 70)
    print("TASK 2 COMPLETE")
    print("=" * 70)
