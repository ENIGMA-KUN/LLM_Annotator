"""
Experiment runner: 3 prompts × 3 models × 100 items × 3 runs.

Uses litellm for unified multi-model API access.

Usage:
    python scripts/run_experiments.py [--n-items 100] [--models gemini,gpt,claude] [--dry-run]
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.pipeline.metrics import compute_krippendorff_alpha_multi
from src.pipeline.dimension_router import ALL_PIPELINE_DIMENSIONS

DIMS = list(ALL_PIPELINE_DIMENSIONS)
HUMAN_DIM_MAP = {
    "Relevance": "Relevance",
    "Clarity": "Clarity",
    "Linguistic_Feature": "Linguistic_Feature",
    "Semantic_Plausibility": "Semantic_Plausibility",
    "Semantic_Uniqueness": "Semantic_Unique",
}
K_SUFFIX = "_KC"
L_SUFFIX = "_LH"

OUTPUT_DIR = ROOT / "Data" / "results" / "neurips_experiments"

# Model configs
MODEL_MAP = {
    "gemini_lite": "gemini/gemini-2.5-flash-lite",
    "gemini_flash": "gemini/gemini-2.5-flash",
    "gpt54": "gpt-5.4",
    "llama70b": "groq/llama-3.3-70b-versatile",
    "llama4scout": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    # NVIDIA NIM (build.nvidia.com). Free tier: 1000 credits on signup, 40 req/min,
    # and models marked free on the catalogue do not draw down credits at all.
    # Needs NVIDIA_NIM_API_KEY in .env. OpenAI-compatible, so LiteLLM routes it directly.
    "nim_llama70b": "nvidia_nim/meta/llama-3.3-70b-instruct",
    "nim_qwen": "nvidia_nim/qwen/qwen2.5-72b-instruct",
    "nim_deepseek": "nvidia_nim/deepseek-ai/deepseek-r1",
    "nim_nemotron": "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct",
}

PROMPT_MAP = {
    "P0_baseline": ROOT / "prompts" / "zero_shot_qlevel.txt",
    "P1_manual": ROOT / "prompts" / "best_prompt_v1.txt",
    "P2_evidently": ROOT / "prompts" / "evidently_optimized" / "combined_all_dims.txt",
}

NUM_RUNS = 3
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 4096


def load_human_data():
    """Load human scores and compute consensus."""
    df = pd.read_csv(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")
    for dim, hd in HUMAN_DIM_MAP.items():
        df[f"{dim}_consensus"] = ((df[f"{hd}{K_SUFFIX}"] + df[f"{hd}{L_SUFFIX}"]) / 2).round().astype(int)
    return df


def load_items():
    """Load item data for API calls."""
    return pd.read_csv(ROOT / "Data" / "processed" / "pipeline_upload_200items.csv")


def load_phase1_baseline(n_items: int) -> dict:
    """Load Phase 1 results as pre-computed scores (no API cost)."""
    csv_path = ROOT / "Data" / "results" / "phase1_question_level" / "zero_shot_qlevel" / "zero_shot_qlevel_results.csv"
    df = pd.read_csv(csv_path).head(n_items)

    scores = {}  # uuid -> {dim: [run1, run2, run3]}
    for _, row in df.iterrows():
        uuid = str(row["question_uuid"])
        item_scores = {}
        for dim in DIMS:
            col = f"{dim}_all_runs"
            if col in df.columns:
                try:
                    rl = ast.literal_eval(str(row[col]))
                    item_scores[dim] = [float(x) for x in rl[:3]]
                except:
                    item_scores[dim] = [np.nan] * 3
            else:
                item_scores[dim] = [float(row.get(dim, np.nan))] * 3
        scores[uuid] = item_scores
    return scores


# --- Pricing per 1M tokens (USD) ---
MODEL_PRICING = {
    "gemini/gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini/gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gpt-5.4": {"input": 2.50, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "groq/llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},  # free tier
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.00, "output": 0.00},
    # NVIDIA NIM free tier. Zero cost, but the run still consumes rate-limit budget
    # (40 req/min) and credits on non-free models, so the ledger records the calls.
    "nvidia_nim/meta/llama-3.3-70b-instruct": {"input": 0.00, "output": 0.00},
    "nvidia_nim/qwen/qwen2.5-72b-instruct": {"input": 0.00, "output": 0.00},
    "nvidia_nim/deepseek-ai/deepseek-r1": {"input": 0.00, "output": 0.00},
    "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct": {"input": 0.00, "output": 0.00},
}


def compute_call_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """Compute USD cost for a single API call."""
    pricing = MODEL_PRICING.get(model_id, {"input": 0, "output": 0})
    return (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_environment_info() -> dict:
    """Capture environment for reproducibility."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Git hash
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        env["git_commit"] = "unavailable"
    # Key package versions
    for pkg in ["litellm", "numpy", "pandas", "krippendorff"]:
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[f"{pkg}_version"] = "not installed"
    return env


def call_llm(model_id: str, system_prompt: str, user_prompt: str,
             attempt: int = 0, log_file=None, item_uuid: str = "", run_i: int = 0) -> dict:
    """Call an LLM via litellm and parse scores. Logs metadata per call."""
    import litellm

    t0 = time.time()
    try:
        response = litellm.completion(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"} if ("gpt" in model_id or "claude" in model_id or "groq" in model_id) else None,
        )
        latency_ms = round((time.time() - t0) * 1000)

        raw = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        cost = compute_call_cost(model_id, tokens_in, tokens_out)

        tokens = {"input": tokens_in, "output": tokens_out}
        scores = parse_scores(raw)

        # Log metadata
        if log_file:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_id,
                "item_uuid": item_uuid,
                "run": run_i,
                "attempt": attempt,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "cost_usd": round(cost, 6),
                "scores": {d: (v if not (isinstance(v, float) and np.isnan(v)) else None) for d, v in scores.items()},
                "success": True,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        return {"scores": scores, "raw": raw, "tokens": tokens, "cost": cost, "success": True}

    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000)
        is_rate_limit = "RateLimit" in str(type(e).__name__) or "429" in str(e) or "rate limit" in str(e).lower()
        max_retries = 8 if is_rate_limit else 3
        if attempt < max_retries:
            wait = min(60, 2 ** (attempt + 1)) if is_rate_limit else 2 ** (attempt + 1)
            if is_rate_limit and attempt == 0:
                print(f"    Rate limited, backing off...")
            elif not is_rate_limit:
                print(f"    Retry {attempt+1}/{max_retries} for {model_id}: {str(e)[:100]}")
            time.sleep(wait)
            return call_llm(model_id, system_prompt, user_prompt, attempt + 1, log_file, item_uuid, run_i)

        # Log failed call
        if log_file:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_id,
                "item_uuid": item_uuid,
                "run": run_i,
                "attempt": attempt,
                "latency_ms": latency_ms,
                "error": str(e)[:300],
                "success": False,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        return {"scores": {d: np.nan for d in DIMS}, "raw": str(e), "tokens": {"input": 0, "output": 0}, "cost": 0, "success": False}


def parse_scores(raw: str) -> dict:
    """Parse dimension scores from LLM response."""
    scores = {}

    # Strip markdown code fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(lines).strip()

    # Try JSON parse
    try:
        parsed = json.loads(clean)
        for dim in DIMS:
            val = parsed.get(dim) or parsed.get(dim.lower()) or parsed.get(dim.replace("_", " "))
            if val is not None:
                # Handle nested dict like {"score": 3, "evidence": "..."}
                if isinstance(val, dict):
                    score_val = val.get("score") or val.get("Score") or val.get("rating")
                    if score_val is not None:
                        try:
                            scores[dim] = max(1, min(4, int(float(score_val))))
                        except (ValueError, TypeError):
                            scores[dim] = np.nan
                    else:
                        scores[dim] = np.nan
                else:
                    try:
                        scores[dim] = max(1, min(4, int(float(val))))
                    except (ValueError, TypeError):
                        scores[dim] = np.nan
            else:
                scores[dim] = np.nan

        # If some missing, search nested
        for dim in DIMS:
            if dim not in scores or (isinstance(scores[dim], float) and np.isnan(scores[dim])):
                scores[dim] = _find_in_nested(parsed, dim)

        return scores
    except json.JSONDecodeError:
        pass

    # Regex fallback
    for dim in DIMS:
        patterns = [rf'"{dim}"\s*:\s*(\d+)', rf'"{dim.lower()}"\s*:\s*(\d+)']
        val = np.nan
        for pat in patterns:
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                try:
                    val = max(1, min(4, int(m.group(1))))
                except:
                    pass
                break
        scores[dim] = val

    return scores


def _find_in_nested(d: dict, dim: str):
    """Recursively search for a dimension key in nested dicts."""
    for key, val in d.items():
        if isinstance(val, dict):
            result = _find_in_nested(val, dim)
            if not (isinstance(result, float) and np.isnan(result)):
                return result
        if key.lower().replace(" ", "_") == dim.lower() or key == dim:
            try:
                return max(1, min(4, int(float(val))))
            except:
                pass
    return np.nan


def build_user_prompt(row) -> str:
    """Build user prompt for one item."""
    return (
        f"Please evaluate the following MCQ.\n\n"
        f"**Text Passage:**\n{str(row.get('text', ''))[:3000]}\n\n"
        f"**Question:**\n{row.get('question', '')}\n\n"
        f"**Correct Answer:**\n{row.get('answer', '')}\n\n"
        f"Respond with a JSON object containing scores for each dimension."
    )


def run_experiment(
    model_key: str,
    prompt_key: str,
    items_df: pd.DataFrame,
    n_items: int,
    phase1_scores: dict = None,
) -> dict:
    """Run one cell of the experiment matrix."""
    model_id = MODEL_MAP[model_key]
    prompt_path = PROMPT_MAP[prompt_key]

    # Special case: Phase 1 baseline ONLY for gemini_flash P0 (original Phase 1 model)
    if prompt_key == "P0_baseline" and model_key == "gemini_flash" and phase1_scores:
        print(f"  Using Phase 1 cached results (Gemini Flash was the Phase 1 model)")
        return phase1_scores

    if not prompt_path.exists():
        print(f"  ERROR: Prompt not found: {prompt_path}")
        return {}

    system_prompt = prompt_path.read_text(encoding="utf-8")

    all_scores = {}  # uuid -> {dim: [run1, run2, run3]}
    total_tokens = {"input": 0, "output": 0}
    total_cost = 0.0
    success_count = 0
    fail_count = 0

    # Create per-cell metadata log (JSONL)
    cell_dir = OUTPUT_DIR / f"{model_key}_{prompt_key}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell_dir / "call_log.jsonl"
    log_file = open(log_path, "w", encoding="utf-8")

    # Write cell-level metadata header
    cell_meta = {
        "_type": "cell_metadata",
        "model_key": model_key,
        "model_id": model_id,
        "prompt_key": prompt_key,
        "prompt_hash": hashlib.sha256(system_prompt.encode()).hexdigest()[:16],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "num_runs": NUM_RUNS,
        "n_items": n_items,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    log_file.write(json.dumps(cell_meta) + "\n")
    log_file.flush()

    subset = items_df.head(n_items)
    for idx, (_, row) in enumerate(subset.iterrows()):
        uuid = str(row["question_uuid"])
        user_prompt = build_user_prompt(row)
        item_runs = {d: [] for d in DIMS}

        for run_i in range(NUM_RUNS):
            result = call_llm(model_id, system_prompt, user_prompt,
                              log_file=log_file, item_uuid=uuid, run_i=run_i)
            if result["success"]:
                success_count += 1
            else:
                fail_count += 1
            total_tokens["input"] += result["tokens"]["input"]
            total_tokens["output"] += result["tokens"]["output"]
            total_cost += result.get("cost", 0)

            for d in DIMS:
                item_runs[d].append(result["scores"].get(d, np.nan))

            # Rate limit buffer: Groq free tier ~6K TPM → ~3 calls/min for long prompts
            delay = 25.0 if "groq" in model_id else 1.5
            time.sleep(delay)

        all_scores[uuid] = item_runs

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{n_items} done | Success: {success_count} | Fail: {fail_count} | Tokens: {total_tokens['input']+total_tokens['output']:,} | Cost: ${total_cost:.4f}")

    # Write cell summary
    cell_summary = {
        "_type": "cell_summary",
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_calls": success_count + fail_count,
        "success_calls": success_count,
        "failed_calls": fail_count,
        "total_tokens_in": total_tokens["input"],
        "total_tokens_out": total_tokens["output"],
        "total_cost_usd": round(total_cost, 6),
    }
    log_file.write(json.dumps(cell_summary) + "\n")
    log_file.close()

    print(f"  Completed: {success_count}/{success_count+fail_count} calls | Tokens: {total_tokens['input']+total_tokens['output']:,} | Cost: ${total_cost:.4f}")
    print(f"  Metadata log: {log_path}")
    return all_scores


def compute_metrics(
    scores: dict,
    df_human: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-dimension metrics for one experiment cell."""
    rows = []
    for dim in DIMS:
        hd = HUMAN_DIM_MAP[dim]

        llm_vals, cons_vals, k_vals, l_vals = [], [], [], []
        for _, row in df_human.iterrows():
            uuid = str(row["question_uuid"])
            if uuid in scores and dim in scores[uuid]:
                run_scores = [s for s in scores[uuid][dim] if not (isinstance(s, float) and np.isnan(s))]
                if run_scores:
                    median_score = float(np.median(run_scores))
                    llm_vals.append(median_score)
                    cons_vals.append(float(row[f"{dim}_consensus"]))
                    k_vals.append(float(row[f"{hd}{K_SUFFIX}"]))
                    l_vals.append(float(row[f"{hd}{L_SUFFIX}"]))

        if len(llm_vals) < 5:
            rows.append({"dimension": dim, "alpha": np.nan, "alpha_kl": np.nan,
                         "bias": np.nan, "exact_match": np.nan, "within_1": np.nan,
                         "mae": np.nan, "consistency": np.nan, "n": len(llm_vals)})
            continue

        llm_arr = np.array(llm_vals)
        cons_arr = np.array(cons_vals)
        k_arr = np.array(k_vals)
        l_arr = np.array(l_vals)

        alpha = compute_krippendorff_alpha_multi(llm_arr, cons_arr)["alpha"]
        alpha_kl = compute_krippendorff_alpha_multi(k_arr, l_arr)["alpha"]
        bias = float((llm_arr - cons_arr).mean())
        exact = float((llm_arr == cons_arr).mean() * 100)
        within1 = float((np.abs(llm_arr - cons_arr) <= 1).mean() * 100)
        mae = float(np.abs(llm_arr - cons_arr).mean())

        # Inter-run consistency (proportion of items where all 3 runs agree)
        n_consistent = 0
        n_total = 0
        for uuid in scores:
            if dim in scores[uuid]:
                rs = [s for s in scores[uuid][dim] if not (isinstance(s, float) and np.isnan(s))]
                if len(rs) == 3:
                    n_total += 1
                    if rs[0] == rs[1] == rs[2]:
                        n_consistent += 1
        consistency = n_consistent / n_total * 100 if n_total > 0 else np.nan

        rows.append({
            "dimension": dim,
            "alpha": round(alpha, 4),
            "alpha_kl": round(alpha_kl, 4),
            "bias": round(bias, 3),
            "exact_match": round(exact, 1),
            "within_1": round(within1, 1),
            "mae": round(mae, 3),
            "consistency": round(consistency, 1),
            "n": len(llm_vals),
        })

    return pd.DataFrame(rows)


def report_and_save(all_results: dict, output_dir: Path):
    """Generate comparison tables and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build master table
    master_rows = []
    for (model, prompt), metrics_df in all_results.items():
        for _, row in metrics_df.iterrows():
            master_rows.append({
                "model": model,
                "prompt": prompt,
                **row.to_dict(),
            })

    master = pd.DataFrame(master_rows)
    master.to_csv(output_dir / "master_results.csv", index=False)

    # Print headline table
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS: per-dimension alpha")
    print("=" * 100)

    for dim in DIMS:
        dim_data = master[master["dimension"] == dim]
        print(f"\n--- {dim} ---")
        print(f"  {'Model':<12} {'P0(base)':>10} {'P1(manual)':>10} {'P2(evid)':>10} | {'Bias P0':>8} {'Bias P1':>8} {'Bias P2':>8} | alpha(K,L)")
        print(f"  {'-'*90}")

        for model in MODEL_MAP:
            vals = {}
            for prompt in PROMPT_MAP:
                cell = dim_data[(dim_data["model"] == model) & (dim_data["prompt"] == prompt)]
                if len(cell) > 0:
                    vals[prompt] = (cell.iloc[0]["alpha"], cell.iloc[0]["bias"])
                else:
                    vals[prompt] = (np.nan, np.nan)

            model_data = dim_data[dim_data["model"] == model]
            alpha_kl = model_data.iloc[0].get("alpha_kl", np.nan) if len(model_data) > 0 else np.nan

            a0, b0 = vals.get("P0_baseline", (np.nan, np.nan))
            a1, b1 = vals.get("P1_manual", (np.nan, np.nan))
            a2, b2 = vals.get("P2_evidently", (np.nan, np.nan))

            tier_flag = ""
            best_a = max(a for a in [a0, a1, a2] if not np.isnan(a)) if any(not np.isnan(a) for a in [a0, a1, a2]) else np.nan
            if not np.isnan(best_a) and best_a >= 0.5:
                tier_flag = " ** TIER A **"

            print(f"  {model:<12} {a0:>10.4f} {a1:>10.4f} {a2:>10.4f} | {b0:>+7.3f} {b1:>+7.3f} {b2:>+7.3f} | {alpha_kl:.4f}{tier_flag}")

    # Tier summary
    print("\n" + "=" * 100)
    print("TIER ASSIGNMENT SUMMARY (best Alpha across all models x prompts)")
    print("=" * 100)

    for dim in DIMS:
        dim_data = master[master["dimension"] == dim]
        best = dim_data["alpha"].max()
        best_row = dim_data.loc[dim_data["alpha"].idxmax()] if not np.isnan(best) else None
        tier = "A" if best >= 0.5 else "B" if best >= 0.3 else "C"
        config = f"{best_row['model']}/{best_row['prompt']}" if best_row is not None else "N/A"
        print(f"  {dim:<25} alpha={best:.4f}  Tier {tier}  (best: {config})")

    # Save enriched summary with full metadata
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": get_environment_info(),
        "models": {k: v for k, v in MODEL_MAP.items()},
        "prompts": {k: str(v) for k, v in PROMPT_MAP.items()},
        "prompt_hashes": {k: sha256_file(v)[:16] if v.exists() else "missing"
                         for k, v in PROMPT_MAP.items()},
        "data_checksums": {
            "human_labels": sha256_file(ROOT / "Data" / "processed" / "full_scores_both_annotators.csv")[:16]
            if (ROOT / "Data" / "processed" / "full_scores_both_annotators.csv").exists() else "missing",
            "items": sha256_file(ROOT / "Data" / "processed" / "pipeline_upload_200items.csv")[:16]
            if (ROOT / "Data" / "processed" / "pipeline_upload_200items.csv").exists() else "missing",
        },
        "config": {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "num_runs": NUM_RUNS,
        },
        "results_file": "master_results.csv",
    }
    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--models", type=str, default="gemini_lite",
                        help="Comma-separated: gemini_lite,gemini_flash,llama70b,qwen32b")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run Phase 1 baseline (no API calls)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    n_items = args.n_items

    # Load data
    df_human = load_human_data()
    items_df = load_items()
    phase1_scores = load_phase1_baseline(n_items)

    print(f"Experiment: {len(models)} models x 3 prompts x {n_items} items x 3 runs")
    print(f"Models: {models}")
    print(f"Human data: {len(df_human)} items")

    all_results = {}

    for model in models:
        if model not in MODEL_MAP:
            print(f"Unknown model: {model}. Skipping.")
            continue

        for prompt_key in PROMPT_MAP:
            label = f"{model}/{prompt_key}"
            cell_dir = OUTPUT_DIR / f"{model}_{prompt_key}"
            cached_path = cell_dir / "raw_scores.json"

            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            # Resume support: skip cells that already have results
            if cached_path.exists() and cached_path.stat().st_size > 100:
                print(f"  CACHED: loading from {cached_path}")
                with open(cached_path) as f:
                    cached = json.load(f)
                # Reconstruct scores dict with proper types
                scores = {}
                for uuid, dims in cached.items():
                    scores[uuid] = {d: [float(v) if v is not None else np.nan for v in vs] for d, vs in dims.items()}
                metrics = compute_metrics(scores, df_human.head(n_items))
                all_results[(model, prompt_key)] = metrics
                continue

            if args.dry_run and not (prompt_key == "P0_baseline" and model.startswith("gemini")):
                print(f"  [DRY RUN] Skipping")
                continue

            scores = run_experiment(model, prompt_key, items_df, n_items, phase1_scores)

            if scores:
                metrics = compute_metrics(scores, df_human.head(n_items))
                all_results[(model, prompt_key)] = metrics

                # Save per-cell raw scores
                cell_dir.mkdir(parents=True, exist_ok=True)
                with open(cached_path, "w") as f:
                    # Convert numpy to native types
                    clean = {}
                    for uuid, dims in scores.items():
                        clean[uuid] = {d: [float(x) if not np.isnan(x) else None for x in vs] for d, vs in dims.items()}
                    json.dump(clean, f, indent=2)

    if all_results:
        report_and_save(all_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
