"""
Input Handler: CSV ingest + schema validation for the hybrid pipeline.

Responsibilities:
    - Accept CSV with text + rubric reference
    - Validate schema against expected columns (question_uuid, text, question, answer, options)
    - Forward-fill sparse context columns
    - Split into batches for processing
    - Compute SHA-256 of input file for reproducibility
"""

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import setup_logging

logger = setup_logging()

REQUIRED_COLUMNS = [
    "question_uuid",
    "text",
    "question",
    "answer",
]

# Optional columns that enhance analysis
OPTIONAL_COLUMNS = [
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "subject",
    "source",
]

# Context columns to forward-fill (sparse in raw data)
CONTEXT_COLUMNS = [
    "question_uuid",
    "textID",
    "Source",
    "text",
    "answer",
    "question",
]


class InputHandler:
    """Validates, cleans, and batches incoming CSV data for the pipeline."""

    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size

    def load_csv(self, path: Path) -> pd.DataFrame:
        """
        Load and validate a CSV file for pipeline ingestion.

        Args:
            path: Path to the input CSV.

        Returns:
            Validated and cleaned DataFrame.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If required columns are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        df = pd.read_csv(path)
        file_hash = self._compute_file_hash(path)
        logger.info(
            f"Loaded {len(df)} rows from {path.name} "
            f"(SHA-256: {file_hash[:16]})"
        )

        self._validate_schema(df, path)
        df = self._clean(df)
        return df

    def _validate_schema(self, df: pd.DataFrame, path: Path) -> None:
        """Check that all required columns are present."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input CSV {path.name} is missing required columns: {missing}. "
                f"Expected: {REQUIRED_COLUMNS}"
            )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill sparse context columns and normalize types."""
        df = df.copy()
        for col in CONTEXT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].ffill()
        return df

    def split_batches(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Split DataFrame into batches of self.batch_size."""
        return [
            df.iloc[i : i + self.batch_size].copy()
            for i in range(0, len(df), self.batch_size)
        ]

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file for reproducibility logging."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
