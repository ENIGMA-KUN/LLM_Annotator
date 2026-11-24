"""
Annotation Engine: Gemini wrapper + 3-run protocol for the hybrid pipeline.

Responsibilities:
    - Wrap the Google Gemini API with retry logic
    - Run 3 independent annotation passes per item (for inter-run agreement)
    - Question-level evaluation only (V3 4-level rubric)
    - Output per-dimension score + reasoning per run as JSON
    - Full metadata logging: tokens, cost, latency, timestamp, model version, prompt version

Migrated and adapted from the Phase 1 ``src/annotator.py``.
The original module remains importable for backward compatibility.
"""

import json
import hashlib
import re
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from src.utils import load_env, setup_logging, RESULTS_DIR
from src.metadata import get_metadata_logger

logger = setup_logging()

# ---------------------------------------------------------------------------
# Constants (aligned with the Phase 1 baseline; keep fixed for comparability)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.3
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff: 2, 4, 8, 16, 32
NUM_RUNS = 3  # inter-run agreement signal requires exactly 3
MAX_OUTPUT_TOKENS = 16384
COST_ALARM_USD = 5.0  # halt if cumulative cost exceeds this

# Question-level dimensions (V3-4level rubric)
Q_LEVEL_DIMENSIONS = [
    "Relevance",
    "Clarity",
    "Linguistic_Feature",
    "Semantic_Plausibility",
    "Semantic_Uniqueness",
]

# Gemini 2.5 Flash pricing (per 1M tokens)
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60


class AnnotationEngine:
    """
    Annotates MCQ questions using Google Gemini with a 3-run protocol.

    Each item is sent to the LLM ``NUM_RUNS`` times.  The engine returns
    *all* per-run results so downstream components (dimension router,
    confidence gate) can inspect inter-run agreement.

    Every API call is logged with model, prompt hash, tokens, latency and cost.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: Optional[str] = None,
        prompt_version: str = "qlevel_v3_4level_v1",
    ):
        self.api_key = api_key or load_env()
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )

        # Cumulative cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_calls = 0
        self.successful_calls = 0
        self._call_log: list[dict] = []

        # Log model metadata
        meta = get_metadata_logger()
        meta.log_model(
            model_name=model,
            temperature=temperature,
            num_runs=NUM_RUNS,
            response_format="application/json",
            extra={
                "max_retries": MAX_RETRIES,
                "retry_base_delay": RETRY_BASE_DELAY,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "prompt_version": prompt_version,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_batch(
        self,
        items: list[dict],
        system_prompt: str,
        num_runs: int = NUM_RUNS,
        condition_name: str = "annotation",
    ) -> list[dict]:
        """
        Annotate a list of items, each ``num_runs`` times.

        Args:
            items: List of dicts, each with keys:
                   question_uuid, text, question, answer
            system_prompt: The full system prompt (loaded from prompt file).
            num_runs: Number of independent runs per item (default 3).
            condition_name: Label for logging/metadata.

        Returns:
            List of dicts, one per item.  Each dict contains:
                - question_uuid
                - runs: list of per-run result dicts
                - Each run dict has: scores (per dimension), reasoning,
                  confidence, metadata (tokens, cost, latency, etc.)
        """
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]
        meta = get_metadata_logger()
        meta.log_experiment(
            condition_name=condition_name,
            prompt_name=self.prompt_version,
            prompt_text=system_prompt,
            n_items=len(items),
            num_runs=num_runs,
        )
        meta.start_timer(f"engine_{condition_name}")

        results = []
        for i, item in enumerate(items):
            user_prompt = self._build_user_prompt(item)
            runs = []
            for run_number in range(1, num_runs + 1):
                run_result = self._call_with_metadata(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    item=item,
                    run_number=run_number,
                    prompt_hash=prompt_hash,
                )
                runs.append(run_result)

            results.append({
                "question_uuid": item["question_uuid"],
                "runs": runs,
            })

            if (i + 1) % 20 == 0:
                logger.info(
                    f"  [{condition_name}] {i + 1}/{len(items)} done | "
                    f"Tokens: {self.total_input_tokens + self.total_output_tokens:,} | "
                    f"Cost: ${self.total_cost_usd:.4f}"
                )

            # Cost alarm
            if self.total_cost_usd > COST_ALARM_USD:
                logger.error(
                    f"COST ALARM: ${self.total_cost_usd:.2f} exceeds "
                    f"${COST_ALARM_USD} ceiling. Halting."
                )
                break

        elapsed = meta.stop_timer(f"engine_{condition_name}")
        meta.log_experiment_cost(
            condition_name=condition_name,
            total_tokens=self.total_input_tokens + self.total_output_tokens,
            total_cost_usd=self.total_cost_usd,
            num_calls=self.total_calls,
            successful_calls=self.successful_calls,
            duration_seconds=elapsed,
        )
        logger.info(
            f"  [{condition_name}] COMPLETE | Items: {len(results)} | "
            f"Cost: ${self.total_cost_usd:.4f}"
        )
        return results

    # ------------------------------------------------------------------
    # LLM call, with per-call metadata
    # ------------------------------------------------------------------

    def _call_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        item: dict,
        run_number: int,
        prompt_hash: str,
    ) -> dict:
        """
        Single LLM call with full metadata capture.

        Returns a dict with:
            scores, reasoning, confidence, metadata, parsed_successfully
        """
        self.total_calls += 1
        call_start = time.time()
        call_metadata = {
            "model": self.model_name,
            "provider": "google",
            "temperature": self.temperature,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "prompt_template_version": self.prompt_version,
            "prompt_hash": prompt_hash,
            "question_uuid": item["question_uuid"],
            "run_number": run_number,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        raw_text = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"text": system_prompt + "\n\n" + user_prompt}
                            ],
                        }
                    ]
                )

                # Token accounting
                usage = response.usage_metadata
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                cost = (
                    input_tokens * PRICE_INPUT_PER_M / 1_000_000
                    + output_tokens * PRICE_OUTPUT_PER_M / 1_000_000
                )
                self.total_cost_usd += cost

                latency = time.time() - call_start
                call_metadata.update({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "call_cost_usd": round(cost, 6),
                    "latency_seconds": round(latency, 2),
                })

                raw_text = response.text
                parsed = json.loads(raw_text)
                scores = self._parse_scores_deep(parsed)
                # If deep parse still has NaN, try regex on raw text
                if any(np.isnan(v) for v in scores.values()):
                    regex_scores = self._regex_extract_scores(raw_text)
                    for dim in Q_LEVEL_DIMENSIONS:
                        if np.isnan(scores.get(dim, np.nan)) and not np.isnan(regex_scores.get(dim, np.nan)):
                            scores[dim] = regex_scores[dim]
                self.successful_calls += 1

                call_metadata["parsed_successfully"] = True
                call_metadata["parse_errors"] = []
                self._call_log.append(call_metadata)

                return {
                    "scores": scores,
                    "reasoning": self._extract_reasoning(parsed),
                    "confidence": self._extract_confidence(parsed),
                    "metadata": call_metadata,
                    "parsed_successfully": True,
                    "raw_response": raw_text,
                }

            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ServiceUnavailable,
            ) as e:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"API transient error (attempt {attempt + 1}/{MAX_RETRIES}): "
                    f"{e}. Retrying in {delay:.1f}s"
                )
                time.sleep(delay)

            except google_exceptions.GoogleAPIError as e:
                logger.error(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(RETRY_BASE_DELAY)

            except json.JSONDecodeError:
                # Try extracting JSON from markdown fences
                json_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL
                )
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        scores = self._parse_scores(parsed)
                        self.successful_calls += 1
                        call_metadata["parsed_successfully"] = True
                        call_metadata["parse_errors"] = ["extracted_from_markdown"]
                        self._call_log.append(call_metadata)
                        return {
                            "scores": scores,
                            "reasoning": self._extract_reasoning(parsed),
                            "confidence": self._extract_confidence(parsed),
                            "metadata": call_metadata,
                            "parsed_successfully": True,
                            "raw_response": raw_text,
                        }
                    except (json.JSONDecodeError, Exception):
                        pass

                # Regex fallback: extract scores from truncated JSON
                scores = self._regex_extract_scores(raw_text)
                if any(not np.isnan(v) for v in scores.values()):
                    self.successful_calls += 1
                    call_metadata["parsed_successfully"] = True
                    call_metadata["parse_errors"] = ["regex_fallback"]
                    self._call_log.append(call_metadata)
                    return {
                        "scores": scores,
                        "reasoning": {},
                        "confidence": 50,
                        "metadata": call_metadata,
                        "parsed_successfully": True,
                        "raw_response": raw_text,
                    }

                logger.error(f"JSON parse failure. Raw[:200]: {raw_text[:200]}")
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(1)

        # All retries exhausted
        call_metadata["parsed_successfully"] = False
        call_metadata["parse_errors"] = ["max_retries_exhausted"]
        self._call_log.append(call_metadata)
        return {
            "scores": {dim: np.nan for dim in Q_LEVEL_DIMENSIONS},
            "reasoning": {},
            "confidence": 0,
            "metadata": call_metadata,
            "parsed_successfully": False,
            "raw_response": raw_text,
        }

    # ------------------------------------------------------------------
    # Prompt Building (question-level, V3 rubric)
    # ------------------------------------------------------------------

    def _build_user_prompt(self, item: dict) -> str:
        """Build the user-facing prompt for a single question-level item."""
        return (
            f"Please evaluate the following MCQ.\n\n"
            f"**Text Passage:**\n{str(item.get('text', ''))[:3000]}\n\n"
            f"**Question:**\n{item.get('question', '')}\n\n"
            f"**Correct Answer:**\n{item.get('answer', '')}\n\n"
            f"Provide your annotation as a JSON object with scores for each "
            f"dimension (1-4 scale), confidence (0-100), and reasoning per dimension."
        )

    # ------------------------------------------------------------------
    # Response Parsing
    # ------------------------------------------------------------------

    def _parse_scores(self, response: dict) -> dict:
        """Extract per-dimension scores from LLM JSON, clamping to 1-4."""
        scores = {}
        for dim in Q_LEVEL_DIMENSIONS:
            val = (
                response.get(dim)
                or response.get(dim.lower())
                or response.get(dim.replace("_", " "))
                or response.get(dim.replace("_", ""))
            )
            if val is not None:
                try:
                    val = int(float(val))
                    val = max(1, min(4, val))
                except (ValueError, TypeError):
                    val = np.nan
            else:
                val = np.nan
            scores[dim] = val
        return scores

    def _regex_extract_scores(self, raw_text: str) -> dict:
        """Fallback: extract dimension scores from raw text using regex."""
        scores = {}
        for dim in Q_LEVEL_DIMENSIONS:
            # Match patterns like "Relevance": 3 or "Relevance":3 or "Relevance" : 3
            patterns = [
                rf'"{dim}"\s*:\s*(\d+)',
                rf'"{dim.lower()}"\s*:\s*(\d+)',
                rf'"{dim.replace("_", " ")}"\s*:\s*(\d+)',
            ]
            val = np.nan
            for pat in patterns:
                m = re.search(pat, raw_text, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        val = max(1, min(4, val))
                    except (ValueError, TypeError):
                        val = np.nan
                    break
            scores[dim] = val
        return scores

    def _parse_scores_deep(self, response: dict) -> dict:
        """Extract scores from possibly nested JSON (Gemini thinking output)."""
        # Try top-level first
        scores = self._parse_scores(response)
        if all(not np.isnan(v) for v in scores.values()):
            return scores

        # Search nested dicts recursively for score keys
        def _find_in_dict(d: dict) -> dict:
            found = {}
            for key, val in d.items():
                if isinstance(val, dict):
                    found.update(_find_in_dict(val))
                for dim in Q_LEVEL_DIMENSIONS:
                    if key.lower().replace(" ", "_") == dim.lower() or key == dim:
                        try:
                            found[dim] = max(1, min(4, int(float(val))))
                        except (ValueError, TypeError):
                            pass
            return found

        deep = _find_in_dict(response)
        for dim in Q_LEVEL_DIMENSIONS:
            if np.isnan(scores.get(dim, np.nan)) and dim in deep:
                scores[dim] = deep[dim]

        return scores

    @staticmethod
    def _extract_reasoning(response: dict) -> dict:
        """Extract per-dimension reasoning strings."""
        reasoning = response.get("reasoning", response.get("Reasoning", {}))
        if isinstance(reasoning, str):
            return {"general": reasoning}
        if isinstance(reasoning, dict):
            return reasoning
        return {}

    @staticmethod
    def _extract_confidence(response: dict) -> int:
        """Extract confidence value, default 50."""
        conf = response.get("confidence", response.get("Confidence", 50))
        try:
            conf = int(float(conf))
            return max(0, min(100, conf))
        except (ValueError, TypeError):
            return 50

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_cost_summary(self) -> dict:
        """Return cumulative cost and token usage."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
        }
