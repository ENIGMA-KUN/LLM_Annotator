"""
Metadata Logger - Centralized experiment metadata collection for reviewer audits.

Captures and persists:
- Environment: Python version, OS, library versions, hardware
- Data: file paths, shapes, checksums, null counts
- Model: name, temperature, API config, pricing
- Experiment: condition, prompt hash, timing, costs, items processed
- Evaluation: metrics computed, sample sizes, statistical parameters
- Provenance: git commit (if available), timestamps, user

Usage:
    from src.metadata import get_metadata_logger
    meta = get_metadata_logger()
    meta.log_environment()
    meta.log_data("katerina", df_k)
    meta.log_model(annotator)
    meta.save()
"""

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import RESULTS_DIR, ANNOTATION_DIMENSIONS, setup_logging

logger = setup_logging()

# Singleton instance
_instance: Optional["MetadataLogger"] = None


def get_metadata_logger(run_id: Optional[str] = None) -> "MetadataLogger":
    """Get or create the singleton MetadataLogger."""
    global _instance
    if _instance is None or run_id is not None:
        _instance = MetadataLogger(run_id=run_id)
    return _instance


class MetadataLogger:
    """Collects and persists experiment metadata for reviewer audits."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._store: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": {},
            "data": {},
            "models": {},
            "experiments": {},
            "evaluations": {},
            "processing": {},
            "timing": {},
            "provenance": {},
        }
        self._timers: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def log_environment(self) -> None:
        """Capture Python, OS, and library versions."""
        env = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cwd": os.getcwd(),
        }

        # Library versions
        libs = {}
        for lib_name in [
            "pandas", "numpy", "scipy", "sklearn", "krippendorff",
            "google.generativeai", "tqdm", "matplotlib", "seaborn",
            "dotenv",
        ]:
            try:
                mod = __import__(lib_name.split(".")[0])
                libs[lib_name] = getattr(mod, "__version__", "installed (no version)")
            except ImportError:
                libs[lib_name] = "NOT INSTALLED"
        env["library_versions"] = libs

        # Git info
        env["git"] = self._get_git_info()

        self._store["environment"] = env
        logger.info(f"[METADATA] Environment: Python {sys.version.split()[0]}, {platform.platform()}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def log_data(
        self,
        name: str,
        df: pd.DataFrame,
        source_path: Optional[str] = None,
    ) -> None:
        """Log metadata about a DataFrame (shape, dtypes, nulls, checksum)."""
        info: dict[str, Any] = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "null_counts": {
                col: int(df[col].isnull().sum())
                for col in df.columns
                if df[col].isnull().sum() > 0
            },
            "total_nulls": int(df.isnull().sum().sum()),
        }

        if source_path:
            info["source_path"] = str(source_path)
            if Path(source_path).exists():
                info["file_size_bytes"] = Path(source_path).stat().st_size

        # Content hash for reproducibility (hash first 10k values)
        sample = df.head(min(100, len(df))).to_csv(index=False)
        info["content_hash_sha256"] = hashlib.sha256(sample.encode()).hexdigest()[:16]

        # Dimension-specific stats
        dim_stats = {}
        for dim in ANNOTATION_DIMENSIONS:
            if dim in df.columns:
                s = df[dim].dropna()
                if len(s) > 0:
                    dim_stats[dim] = {
                        "n_valid": int(len(s)),
                        "n_missing": int(df[dim].isnull().sum()),
                        "mean": round(float(s.mean()), 4),
                        "std": round(float(s.std()), 4),
                        "min": int(s.min()),
                        "max": int(s.max()),
                        "distribution": {int(k): int(v) for k, v in s.value_counts().sort_index().items()},
                    }
        if dim_stats:
            info["dimension_stats"] = dim_stats

        # Unique identifiers
        if "question_uuid" in df.columns:
            info["n_unique_questions"] = int(df["question_uuid"].nunique())
        if "textID" in df.columns:
            info["n_unique_texts"] = int(df["textID"].nunique())
        if "Source" in df.columns:
            info["sources"] = df["Source"].value_counts().to_dict()

        self._store["data"][name] = info
        logger.info(
            f"[METADATA] Data '{name}': {len(df)} rows x {len(df.columns)} cols, "
            f"{info['total_nulls']} total nulls"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def log_model(
        self,
        model_name: str,
        temperature: float,
        num_runs: int,
        response_format: str = "application/json",
        extra: Optional[dict] = None,
    ) -> None:
        """Log LLM model configuration."""
        model_info = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "temperature": temperature,
            "num_runs_per_item": num_runs,
            "response_format": response_format,
            "provider": "Google Gemini",
        }
        if extra:
            model_info.update(extra)
        self._store["models"][model_name] = model_info
        logger.info(f"[METADATA] Model: {model_name}, temp={temperature}, runs={num_runs}")

    # ------------------------------------------------------------------
    # Experiment / Condition
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        condition_name: str,
        prompt_name: str,
        prompt_text: str,
        n_items: int,
        num_runs: int,
        extra: Optional[dict] = None,
    ) -> None:
        """Log experiment condition metadata, including prompt hash."""
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

        exp_info = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "condition": condition_name,
            "prompt_file": prompt_name,
            "prompt_hash_sha256": prompt_hash,
            "prompt_length_chars": len(prompt_text),
            "n_items": n_items,
            "num_runs_per_item": num_runs,
            "total_api_calls_expected": n_items * num_runs,
        }
        if extra:
            exp_info.update(extra)
        self._store["experiments"][condition_name] = exp_info
        logger.info(
            f"[METADATA] Experiment '{condition_name}': {n_items} items x {num_runs} runs, "
            f"prompt_hash={prompt_hash}"
        )

    def log_experiment_cost(
        self,
        condition_name: str,
        total_tokens: int,
        total_cost_usd: float,
        num_calls: int,
        successful_calls: int,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Log API cost and token usage for a condition."""
        cost_info = {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "num_api_calls": num_calls,
            "successful_calls": successful_calls,
            "failed_calls": num_calls - successful_calls,
            "success_rate": round(successful_calls / max(num_calls, 1), 4),
        }
        if duration_seconds is not None:
            cost_info["duration_seconds"] = round(duration_seconds, 2)
            cost_info["items_per_second"] = round(
                successful_calls / max(duration_seconds, 0.01), 4
            )

        if condition_name in self._store["experiments"]:
            self._store["experiments"][condition_name]["cost"] = cost_info
        else:
            self._store["experiments"][condition_name] = {"cost": cost_info}

        logger.info(
            f"[METADATA] Cost '{condition_name}': {total_tokens:,} tokens, "
            f"${total_cost_usd:.4f}, {successful_calls}/{num_calls} calls"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def log_evaluation(
        self,
        condition_name: str,
        human_annotator: str,
        n_items: int,
        metrics_summary: dict,
        extra: Optional[dict] = None,
    ) -> None:
        """Log evaluation metadata for a condition vs human annotator comparison."""
        eval_info = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "condition": condition_name,
            "human_annotator": human_annotator,
            "n_items_evaluated": n_items,
            "kappa_weighting": "quadratic",
            "alpha_level": "ordinal",
            "score_range": [1, 2, 3, 4],
            "metrics_summary": metrics_summary,
        }
        if extra:
            eval_info.update(extra)

        key = f"{condition_name}_vs_{human_annotator}"
        self._store["evaluations"][key] = eval_info
        logger.info(
            f"[METADATA] Evaluation '{key}': n={n_items}, "
            f"kappa={metrics_summary.get('mean_kappa', 'N/A')}"
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def log_processing(
        self,
        step_name: str,
        input_files: list[str],
        output_files: list[str],
        extra: Optional[dict] = None,
    ) -> None:
        """Log data processing step metadata."""
        proc_info = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "step": step_name,
            "input_files": input_files,
            "output_files": output_files,
            "n_inputs": len(input_files),
            "n_outputs": len(output_files),
        }
        if extra:
            proc_info.update(extra)
        self._store["processing"][step_name] = proc_info
        logger.info(f"[METADATA] Processing '{step_name}': {len(input_files)} in -> {len(output_files)} out")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer, return elapsed seconds."""
        if name not in self._timers:
            return 0.0
        elapsed = time.time() - self._timers[name]
        self._store["timing"][name] = {
            "started_at": datetime.fromtimestamp(
                self._timers[name], tz=timezone.utc
            ).isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(f"[METADATA] Timer '{name}': {elapsed:.2f}s")
        return elapsed

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def log_provenance(self, key: str, value: Any) -> None:
        """Log arbitrary provenance information."""
        self._store["provenance"][key] = value

    # ------------------------------------------------------------------
    # Save / Export
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Save all metadata to a JSON file."""
        self._store["saved_at"] = datetime.now(timezone.utc).isoformat()

        if path is None:
            meta_dir = RESULTS_DIR / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            path = meta_dir / f"run_metadata_{self.run_id}.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, default=str)

        logger.info(f"[METADATA] Saved metadata: {path}")
        return path

    def get_store(self) -> dict:
        """Return the full metadata store (read-only copy)."""
        return dict(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _get_git_info() -> dict:
        """Try to capture git commit info for provenance."""
        import subprocess
        info = {}
        try:
            info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            info["dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            info["error"] = "git not available or not a git repository"
        return info
