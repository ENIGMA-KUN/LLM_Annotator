"""
Utility functions for data loading, cleaning, and common operations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = DATA_DIR / "results"
PROCESSED_DIR = DATA_DIR / "processed"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

ANNOTATION_DIMENSIONS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility"]
Q_DIMENSIONS = ["Q-Relevance", "Q-Clarity", "Q-Ling_Feature", "Q-Sem_Plausibility"]
ALL_DIMENSIONS = ANNOTATION_DIMENSIONS + Q_DIMENSIONS

VALID_SCORES = [1, 2, 3, 4]
MISSING_VALUES = {"NA", "na", "N/A", "n/a", "", "999", 999, np.nan}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return project logger."""
    logger = logging.getLogger("llm_annotator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_env() -> str:
    """Load .env and return Gemini API key."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Copy .env.example to .env and add your key."
        )
    return api_key


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in [RESULTS_DIR, PROCESSED_DIR, DATA_DIR / "gold_standard"]:
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Loading & Cleaning
# ---------------------------------------------------------------------------

def load_annotator_csv(name: str) -> pd.DataFrame:
    """
    Load an annotator CSV file by name ('katerina' or 'linh').
    Prefers pre-processed files from Data/processed/ (already cleaned).
    Falls back to raw files in Data/raw/ if processed files don't exist.
    """
    if name.lower() not in ("katerina", "linh"):
        raise ValueError(f"Unknown annotator: {name}. Use 'katerina' or 'linh'.")

    # Prefer processed (already cleaned, forward-filled, validated)
    processed_map = {
        "katerina": PROCESSED_DIR / "katerina_annotations.csv",
        "linh": PROCESSED_DIR / "linh_annotations.csv",
    }
    processed_path = processed_map[name.lower()]
    if processed_path.exists():
        logger = logging.getLogger("llm_annotator")
        logger.info(f"Loading processed annotations: {processed_path}")
        return pd.read_csv(processed_path)

    # Fallback to raw files
    raw_map = {
        "katerina": DATA_DIR / "raw" / "Katerina_Annoted - Katerina-ACL-annotation.csv",
        "linh": DATA_DIR / "raw" / "Linh_Annoted - Linh-ACL-annotation.csv",
    }
    path = raw_map[name.lower()]
    if not path.exists():
        raise FileNotFoundError(
            f"No data found for '{name}'. Run scripts/process_raw_data.py first, "
            f"or place raw CSVs in {DATA_DIR / 'raw'}"
        )
    df = pd.read_csv(path)
    df = clean_annotations(df)
    return df


def clean_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean annotation DataFrame:
    - Replace 999 and 'NA' with np.nan
    - Convert score columns to numeric (float, to allow NaN)
    - Forward-fill question-level columns (text, question, answer, etc.)
    """
    df = df.copy()

    # Forward-fill contextual columns (they are only on first row per question)
    context_cols = ["question_uuid", "textID", "Source", "text", "answer", "question"]
    for col in context_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Clean score columns
    score_cols = [c for c in ALL_DIMENSIONS if c in df.columns]
    for col in score_cols:
        df[col] = df[col].replace({"NA": np.nan, "na": np.nan, "N/A": np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Treat 999 as missing
        df.loc[df[col] == 999, col] = np.nan

    return df


def get_valid_pairs(
    df_human: pd.DataFrame,
    df_llm: pd.DataFrame,
    dimension: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Return aligned (human, llm) score series for a dimension,
    dropping rows where either is NaN.
    """
    mask = df_human[dimension].notna() & df_llm[dimension].notna()
    return df_human.loc[mask, dimension], df_llm.loc[mask, dimension]


def build_annotation_item(row: pd.Series) -> dict:
    """
    Build an annotation item dict from a DataFrame row
    for sending to the LLM.
    """
    return {
        "question_uuid": str(row.get("question_uuid", "")),
        "textID": str(row.get("textID", "")),
        "text": str(row.get("text", "")),
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "option_choice": str(row.get("option_choice", "")),
    }


def save_results(
    results: list[dict],
    filename: str,
    subdir: Optional[str] = None,
) -> Path:
    """Save results as both CSV and JSON."""
    ensure_dirs()
    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV
    csv_path = out_dir / f"{filename}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    return out_dir


def load_full_scores_overlap() -> pd.DataFrame:
    """
    Load the Full Scores sheet from the master XLSX, the only place where
    both raters (K and L) scored the SAME 200 questions.
    This is question-level data (1 row per question, not per option).

    Returns DataFrame with columns like Relevance_LH, Relevance_KC, etc.
    """
    xlsx_path = DATA_DIR / "raw" / "ReQuesta_RealTexts_overlap_LH.KC_12.29.25_reliability.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Master XLSX not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name="Full Scores")
    df = df.dropna(how="all")
    return df


def load_reliability_statistics() -> pd.DataFrame:
    """
    Load the pre-computed Reliability Statistics sheet from the master XLSX.
    Contains inter-rater agreement across multiple calibration rounds.
    """
    xlsx_path = DATA_DIR / "raw" / "ReQuesta_RealTexts_overlap_LH.KC_12.29.25_reliability.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Master XLSX not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name="Reliability Statistics")
    return df


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / f"{prompt_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
