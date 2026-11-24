"""
LLM as Judge (Annotating): core annotation engine.
Handles zero-shot and few-shot annotation of MCQ distractors using Google Gemini.
"""

import json
import time
import random
import re
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tqdm import tqdm

from src.utils import (
    load_env,
    setup_logging,
    load_prompt,
    build_annotation_item,
    save_results,
    ANNOTATION_DIMENSIONS,
    RESULTS_DIR,
)
from src.metadata import get_metadata_logger

logger = setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds
NUM_RUNS = 3  # self-consistency runs


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMAnnotator:
    """
    Annotates MCQ distractors using Google Gemini.
    Supports zero-shot and few-shot conditions.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or load_env()
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        self.temperature = temperature
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self._call_log: list[dict] = []

        # Log model metadata
        meta = get_metadata_logger()
        meta.log_model(
            model_name=model,
            temperature=temperature,
            num_runs=NUM_RUNS,
            response_format="application/json",
            extra={"max_retries": MAX_RETRIES, "retry_base_delay": RETRY_BASE_DELAY},
        )

    # -----------------------------------------------------------------------
    # Core API Call
    # -----------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Call Gemini API with retry logic and cost tracking.
        Returns parsed JSON response.
        """
        content = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(
                    contents=[
                        {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]},
                    ],
                )

                # Track usage
                usage_meta = response.usage_metadata
                prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
                completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
                tokens = prompt_tokens + completion_tokens
                self.total_tokens_used += tokens

                # Gemini 2.0 Flash pricing: $0.10/1M input, $0.40/1M output
                cost = (prompt_tokens * 0.10 / 1_000_000) + (
                    completion_tokens * 0.40 / 1_000_000
                )
                self.total_cost += cost

                # Parse response
                content = response.text
                parsed = json.loads(content)

                # Log the call
                self._call_log.append({
                    "model": self.model_name,
                    "tokens": tokens,
                    "cost": cost,
                    "success": True,
                })

                return parsed

            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ServiceUnavailable,
            ) as e:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Rate limit/timeout (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

            except google_exceptions.GoogleAPIError as e:
                logger.error(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_BASE_DELAY)

            except json.JSONDecodeError as e:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                logger.error(f"JSON parse error: {e}. Raw: {content[:200]}")
                if attempt == MAX_RETRIES - 1:
                    return self._empty_annotation("JSON parse failure")
                time.sleep(1)

        return self._empty_annotation("Max retries exceeded")

    # -----------------------------------------------------------------------
    # Annotation Methods
    # -----------------------------------------------------------------------

    def annotate_zero_shot(
        self,
        df: pd.DataFrame,
        num_runs: int = NUM_RUNS,
        save_tag: str = "zero_shot",
    ) -> pd.DataFrame:
        """
        Run zero-shot annotation on all items in the DataFrame.
        Runs each item `num_runs` times for self-consistency.
        """
        system_prompt = load_prompt("zero_shot")
        return self._run_experiment(df, system_prompt, num_runs, save_tag)

    def annotate_rubric_guided(
        self,
        df: pd.DataFrame,
        num_runs: int = NUM_RUNS,
        save_tag: str = "rubric_guided",
    ) -> pd.DataFrame:
        """
        Run rubric-guided annotation (detailed rubric, no examples).
        Condition 2 in the AIME three-condition framework.
        """
        system_prompt = load_prompt("rubric_guided")
        return self._run_experiment(df, system_prompt, num_runs, save_tag)

    def annotate_few_shot(
        self,
        df: pd.DataFrame,
        examples: list[dict],
        num_runs: int = NUM_RUNS,
        save_tag: str = "few_shot",
    ) -> pd.DataFrame:
        """
        Run few-shot annotation with provided examples.
        """
        system_prompt = load_prompt("few_shot")
        # Inject examples into system prompt
        examples_text = self._format_examples(examples)
        system_prompt = system_prompt.replace("{{EXAMPLES}}", examples_text)
        return self._run_experiment(df, system_prompt, num_runs, save_tag)

    def _run_experiment(
        self,
        df: pd.DataFrame,
        system_prompt: str,
        num_runs: int,
        save_tag: str,
    ) -> pd.DataFrame:
        """
        Core experiment runner. Annotates each row `num_runs` times.
        Returns DataFrame with LLM annotations and metadata.
        """
        all_results = []
        raw_responses = []  # Save raw responses for reproducibility
        logger.info(
            f"Starting {save_tag} experiment: {len(df)} items x {num_runs} runs"
        )

        # Log experiment metadata
        meta = get_metadata_logger()
        meta.log_experiment(
            condition_name=save_tag,
            prompt_name=save_tag,
            prompt_text=system_prompt,
            n_items=len(df),
            num_runs=num_runs,
        )
        meta.start_timer(f"experiment_{save_tag}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{save_tag}"):
            item = build_annotation_item(row)
            user_prompt = self._build_user_prompt(item)

            run_annotations = []
            for run_id in range(num_runs):
                response = self._call_llm(system_prompt, user_prompt)

                # Save raw response
                raw_responses.append({
                    "item_index": idx,
                    "run_id": run_id,
                    "question_uuid": item["question_uuid"],
                    "raw_response": response,
                })

                annotation = self._parse_annotation(response)
                annotation["run_id"] = run_id
                annotation["item_index"] = idx
                annotation["question_uuid"] = item["question_uuid"]
                annotation["textID"] = item["textID"]
                annotation["option_choice"] = item["option_choice"]
                run_annotations.append(annotation)

            # Compute self-consistency (majority vote + agreement rate)
            aggregated = self._aggregate_runs(run_annotations)
            aggregated["question_uuid"] = item["question_uuid"]
            aggregated["textID"] = item["textID"]
            aggregated["option_choice"] = item["option_choice"]
            aggregated["item_index"] = idx

            all_results.append(aggregated)

            # Log progress periodically
            if (idx + 1) % 50 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{len(df)} | "
                    f"Tokens: {self.total_tokens_used:,} | "
                    f"Cost: ${self.total_cost:.4f}"
                )
                # Save intermediate results
                save_results(all_results, f"{save_tag}_results_partial", subdir=save_tag)

        # Save all results
        save_results(all_results, f"{save_tag}_results", subdir=save_tag)
        save_results(raw_responses, f"{save_tag}_raw_responses", subdir=save_tag)
        logger.info(
            f"Experiment complete. Total tokens: {self.total_tokens_used:,}, "
            f"Total cost: ${self.total_cost:.4f}"
        )

        # Log experiment cost metadata
        elapsed = meta.stop_timer(f"experiment_{save_tag}")
        cost_summary = self.get_cost_summary()
        meta.log_experiment_cost(
            condition_name=save_tag,
            total_tokens=cost_summary["total_tokens"],
            total_cost_usd=cost_summary["total_cost_usd"],
            num_calls=cost_summary["num_calls"],
            successful_calls=cost_summary["successful_calls"],
            duration_seconds=elapsed,
        )

        return pd.DataFrame(all_results)

    # -----------------------------------------------------------------------
    # Prompt Building
    # -----------------------------------------------------------------------

    def _build_user_prompt(self, item: dict) -> str:
        """Build the user message for a single annotation item."""
        return f"""Please annotate the following MCQ distractor option.

**Text Passage:**
{item['text'][:3000]}

**Question:**
{item['question']}

**Correct Answer:**
{item['answer']}

**Option to Evaluate (Distractor):**
{item['option_choice']}

Provide your annotation as a JSON object with scores for each dimension (1-4 scale), confidence (0-100), and reasoning."""

    def _format_examples(self, examples: list[dict]) -> str:
        """Format few-shot examples for injection into the prompt."""
        parts = []
        for i, ex in enumerate(examples, 1):
            parts.append(f"""--- Example {i} ---
Text: {str(ex.get('text', ''))[:500]}...
Question: {ex.get('question', '')}
Correct Answer: {ex.get('answer', '')}
Option: {ex.get('option_choice', '')}
Annotations:
  Relevance: {ex.get('Relevance', 'N/A')}
  Clarity: {ex.get('Clarity', 'N/A')}
  Linguistic_Feature: {ex.get('Linguistic_Feature', 'N/A')}
  Semantic_Plausibility: {ex.get('Semantic_Plausibility', 'N/A')}
""")
        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # Response Parsing
    # -----------------------------------------------------------------------

    def _parse_annotation(self, response: dict) -> dict:
        """Parse LLM JSON response into standardized annotation dict."""
        result = {}
        for dim in ANNOTATION_DIMENSIONS:
            # Try multiple key formats
            val = (
                response.get(dim)
                or response.get(dim.lower())
                or response.get(dim.replace("_", " "))
                or response.get(dim.replace("_", ""))
            )
            if val is not None:
                try:
                    val = int(float(val))
                    val = max(1, min(4, val))  # Clamp to 1-4
                except (ValueError, TypeError):
                    val = np.nan
            else:
                val = np.nan
            result[dim] = val

        # Extract confidence and reasoning
        result["confidence"] = response.get("confidence", response.get("Confidence", 50))
        result["reasoning"] = response.get("reasoning", response.get("Reasoning", ""))

        try:
            result["confidence"] = int(float(result["confidence"]))
            result["confidence"] = max(0, min(100, result["confidence"]))
        except (ValueError, TypeError):
            result["confidence"] = 50

        return result

    def _aggregate_runs(self, runs: list[dict]) -> dict:
        """
        Aggregate multiple annotation runs using majority vote.
        Also compute self-consistency rate per dimension.
        """
        result = {}

        for dim in ANNOTATION_DIMENSIONS:
            scores = [r[dim] for r in runs if not pd.isna(r.get(dim))]
            if scores:
                # Majority vote (mode)
                from statistics import mode, StatisticsError
                try:
                    result[dim] = mode(scores)
                except StatisticsError:
                    result[dim] = round(np.mean(scores))

                # Self-consistency: fraction of runs that agree with majority
                result[f"{dim}_consistency"] = scores.count(result[dim]) / len(scores)
                result[f"{dim}_all_runs"] = scores
            else:
                result[dim] = np.nan
                result[f"{dim}_consistency"] = 0.0
                result[f"{dim}_all_runs"] = []

        # Aggregate confidence and reasoning from first run
        confidences = [r.get("confidence", 50) for r in runs]
        result["confidence"] = round(np.mean(confidences))
        result["reasoning"] = runs[0].get("reasoning", "") if runs else ""

        # Overall self-consistency
        dim_consistencies = [
            result.get(f"{dim}_consistency", 0) for dim in ANNOTATION_DIMENSIONS
        ]
        result["overall_consistency"] = (
            np.mean(dim_consistencies) if dim_consistencies else 0.0
        )

        return result

    def _empty_annotation(self, reason: str) -> dict:
        """Return empty annotation dict for failed API calls."""
        result = {dim: np.nan for dim in ANNOTATION_DIMENSIONS}
        result["confidence"] = 0
        result["reasoning"] = f"FAILED: {reason}"
        return result

    def get_cost_summary(self) -> dict:
        """Return summary of API usage and costs."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost, 4),
            "num_calls": len(self._call_log),
            "successful_calls": sum(1 for c in self._call_log if c["success"]),
        }
