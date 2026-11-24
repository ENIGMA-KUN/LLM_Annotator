"""
Process all raw data files and output clean, experiment-ready files to Data/processed/.

Raw files analyzed:
1. Katerina_Annoted - Katerina-ACL-annotation.csv (360 rows, 15 cols)
2. Linh_Annoted - Linh-ACL-annotation.csv (360 rows, 15 cols)
3. ReQuesta_RealTexts_overlap_LH.KC_12.29.25_reliability.xlsx (14 sheets)
4. Expert Evaluation Option-Level Rubric.pdf (1 page rubric)

Processing steps:
- Forward-fill metadata columns in per-option CSVs
- Clean and validate annotation scores (1-4 scale)
- Extract all xlsx sheets into clean CSVs
- Parse rubric PDF into structured JSON
- Create merged annotator file (Katerina + Linh side-by-side)
- Create experiment-ready dataset
- Generate data quality report

Output files in Data/processed/:
- katerina_annotations.csv        (cleaned Katerina, 360 rows)
- linh_annotations.csv            (cleaned Linh, 360 rows)
- merged_annotations.csv          (both annotators side-by-side, 360 rows)
- full_scores_both_annotators.csv (from xlsx Full Scores sheet, 200 rows)
- finalized_scores.csv            (from xlsx Finalized Scores sheet, 200 rows)
- reliability_statistics.csv      (from xlsx Reliability Statistics sheet)
- subject_scores_sociology.csv    (from xlsx subject sheets)
- subject_scores_history.csv
- subject_scores_lifespan.csv
- subject_scores_anthropology.csv
- rubric.json                     (structured rubric from PDF)
- data_quality_report.json        (quality checks and stats)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metadata import get_metadata_logger

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ANNOTATION_DIMENSIONS = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility"]
Q_DIMENSIONS = ["Q-Relevance", "Q-Clarity", "Q-Ling_Feature", "Q-Sem_Plausibility"]
ALL_SCORE_COLS = ANNOTATION_DIMENSIONS + Q_DIMENSIONS
CONTEXT_COLS = ["question_uuid", "textID", "Source", "text", "answer", "question"]
VALID_SCORES = {1, 2, 3, 4}


def clean_annotator_csv(path: Path, annotator_name: str) -> tuple[pd.DataFrame, dict]:
    """Load and clean an annotator CSV. Returns (cleaned_df, quality_report)."""
    df = pd.read_csv(path)
    report = {
        "annotator": annotator_name,
        "raw_rows": len(df),
        "raw_cols": len(df.columns),
        "columns": df.columns.tolist(),
    }

    # Forward-fill contextual columns
    for col in CONTEXT_COLS:
        if col in df.columns:
            nulls_before = int(df[col].isnull().sum())
            df[col] = df[col].ffill()
            nulls_after = int(df[col].isnull().sum())
            report[f"ffill_{col}"] = {"before": nulls_before, "after": nulls_after}

    # Forward-fill Q-dimension columns (question-level, one per 4 options)
    for col in Q_DIMENSIONS:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Clean score columns
    score_issues = {}
    for col in ALL_SCORE_COLS:
        if col not in df.columns:
            continue
        # Replace textual NA
        df[col] = df[col].replace({"NA": np.nan, "na": np.nan, "N/A": np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Treat 999 as missing
        n_999 = int((df[col] == 999).sum())
        if n_999 > 0:
            df.loc[df[col] == 999, col] = np.nan

        # Validate range
        valid_mask = df[col].isin(VALID_SCORES) | df[col].isnull()
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            invalid_vals = df.loc[~valid_mask, col].unique().tolist()
            score_issues[col] = {"n_invalid": n_invalid, "invalid_values": invalid_vals}
            df.loc[~valid_mask, col] = np.nan

        report[f"nulls_{col}"] = int(df[col].isnull().sum())

    if score_issues:
        report["score_issues"] = score_issues

    # Add option_index (A=1, B=2, C=3, D=4) from option_choice text
    if "option_choice" in df.columns:
        df["option_label"] = df["option_choice"].str.extract(r"^([A-D])\)", expand=False)
        df["option_index"] = df["option_label"].map({"A": 1, "B": 2, "C": 3, "D": 4})

    # Dimension stats
    dim_stats = {}
    for dim in ANNOTATION_DIMENSIONS:
        if dim in df.columns:
            s = df[dim].dropna()
            dim_stats[dim] = {
                "n_valid": int(len(s)),
                "n_missing": int(df[dim].isnull().sum()),
                "mean": round(float(s.mean()), 3) if len(s) > 0 else None,
                "std": round(float(s.std()), 3) if len(s) > 0 else None,
                "distribution": {int(k): int(v) for k, v in s.value_counts().sort_index().items()},
            }
    report["dimension_stats"] = dim_stats

    # Verify expected structure
    n_questions = df["question_uuid"].nunique() if "question_uuid" in df.columns else 0
    report["n_questions"] = n_questions
    report["n_options"] = len(df)
    report["expected_360"] = len(df) == 360
    report["expected_90_questions"] = n_questions == 90

    report["clean_rows"] = len(df)
    return df, report


def process_xlsx(path: Path) -> dict:
    """Extract all sheets from the xlsx file into clean DataFrames."""
    xls = pd.ExcelFile(path)
    sheets = {}
    report = {"file": path.name, "n_sheets": len(xls.sheet_names), "sheets": {}}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        sheets[sheet_name] = df
        report["sheets"][sheet_name] = {
            "rows": len(df),
            "cols": len(df.columns),
            "columns": df.columns.tolist(),
        }

    return sheets, report


def parse_rubric_pdf(path: Path) -> dict:
    """Parse the rubric PDF into structured JSON."""
    # The rubric has 4 dimensions, each with 4 levels (1-4)
    # Based on the PDF text extraction we already saw:
    rubric = {
        "source_file": path.name,
        "parsed_at": datetime.now().isoformat(),
        "scale": {"min": 1, "max": 4, "labels": ["Poor", "Adequate", "Good", "Very Good"]},
        "dimensions": {
            "Relevance": {
                "description": "Response options should address key concepts central to understanding the text",
                "levels": {
                    "1": "Focuses on trivial or unrelated information",
                    "2": "Focuses on a minor detail or a loosely related idea",
                    "3": "Addresses a relevant idea from the text, but less central",
                    "4": "Clearly addresses a central concept in the text",
                },
            },
            "Clarity": {
                "description": "The question stem is worded clearly and concisely",
                "levels": {
                    "1": "Excessively wordy or unclear",
                    "2": "Wordiness that slightly impacts comprehension, mild ambiguity, or awkward syntax",
                    "3": "Mostly clear but contains minor wording issues, or slight ambiguity",
                    "4": "Clearly worded, concise, and unambiguous",
                },
            },
            "Linguistic_Feature": {
                "description": "Distractor is similar to correct answer in linguistic features",
                "levels": {
                    "1": "Distractor is visibly different from correct answer (e.g., length by 12+ characters)",
                    "2": "Distractor has minor differences in length/features",
                    "3": "Distractor has relatively same length/features to correct answer",
                    "4": "Distractor closely matches correct answer in length, structure, and style",
                },
            },
            "Semantic_Plausibility": {
                "description": "Distractor is semantically plausible (no absolute terms, information is not more well known than the correct answer)",
                "levels": {
                    "1": "Distractor is clearly implausible",
                    "2": "Distractor is somewhat implausible",
                    "3": "Distractor is plausible to those who have not read the text",
                    "4": "Distractor is plausible to those who have read but not deeply comprehended the text",
                },
            },
        },
    }

    # Also try to extract raw text from PDF for reference
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() or ""
        rubric["raw_pdf_text"] = raw_text.strip()
    except Exception as e:
        rubric["raw_pdf_text"] = f"Could not extract: {e}"

    return rubric


def create_merged_annotations(df_k: pd.DataFrame, df_l: pd.DataFrame) -> pd.DataFrame:
    """Create a merged DataFrame with both annotators' scores side-by-side."""
    # Use context columns from Katerina as base (both should be identical after ffill)
    base_cols = [c for c in CONTEXT_COLS if c in df_k.columns]
    if "option_choice" in df_k.columns:
        base_cols.append("option_choice")
    if "option_label" in df_k.columns:
        base_cols.append("option_label")
    if "option_index" in df_k.columns:
        base_cols.append("option_index")

    merged = df_k[base_cols].copy()

    # Add Katerina's scores
    for dim in ANNOTATION_DIMENSIONS:
        if dim in df_k.columns:
            merged[f"{dim}_K"] = df_k[dim].values
        if dim in df_l.columns:
            merged[f"{dim}_L"] = df_l[dim].values

    # Add Q-level scores
    for dim in Q_DIMENSIONS:
        if dim in df_k.columns:
            merged[f"{dim}_K"] = df_k[dim].values
        if dim in df_l.columns:
            merged[f"{dim}_L"] = df_l[dim].values

    # Compute agreement metrics per row
    for dim in ANNOTATION_DIMENSIONS:
        k_col = f"{dim}_K"
        l_col = f"{dim}_L"
        if k_col in merged.columns and l_col in merged.columns:
            merged[f"{dim}_agree"] = (merged[k_col] == merged[l_col]).astype(int)
            merged[f"{dim}_diff"] = (merged[k_col] - merged[l_col]).abs()

    return merged


def create_experiment_dataset(df_k: pd.DataFrame) -> pd.DataFrame:
    """Create the primary experiment-ready dataset (360 items for LLM annotation).
    This is the input format the annotator pipeline expects."""
    keep_cols = [c for c in CONTEXT_COLS if c in df_k.columns]
    if "option_choice" in df_k.columns:
        keep_cols.append("option_choice")
    if "option_label" in df_k.columns:
        keep_cols.append("option_label")
    if "option_index" in df_k.columns:
        keep_cols.append("option_index")

    # Include Katerina's scores as ground truth reference
    for dim in ANNOTATION_DIMENSIONS:
        if dim in df_k.columns:
            keep_cols.append(dim)

    df_exp = df_k[keep_cols].copy()
    return df_exp


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    quality_report = {"timestamp": datetime.now().isoformat(), "files": {}}

    # Initialize metadata logger
    meta = get_metadata_logger(run_id=f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    meta.log_environment()
    meta.start_timer("data_processing_total")

    print("=" * 70)
    print("PROCESSING RAW DATA FILES")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Clean annotator CSVs
    # ---------------------------------------------------------------
    print("\n[1/7] Cleaning Katerina annotations...")
    k_path = RAW_DIR / "Katerina_Annoted - Katerina-ACL-annotation.csv"
    df_k, k_report = clean_annotator_csv(k_path, "Katerina")
    df_k.to_csv(PROCESSED_DIR / "katerina_annotations.csv", index=False)
    quality_report["files"]["katerina"] = k_report
    print(f"  -> {len(df_k)} rows, {df_k['question_uuid'].nunique()} questions")
    print(f"  -> Saved: katerina_annotations.csv")

    print("\n[2/7] Cleaning Linh annotations...")
    l_path = RAW_DIR / "Linh_Annoted - Linh-ACL-annotation.csv"
    df_l, l_report = clean_annotator_csv(l_path, "Linh")
    df_l.to_csv(PROCESSED_DIR / "linh_annotations.csv", index=False)
    quality_report["files"]["linh"] = l_report
    print(f"  -> {len(df_l)} rows, {df_l['question_uuid'].nunique()} questions")
    print(f"  -> Saved: linh_annotations.csv")

    # ---------------------------------------------------------------
    # 2. Process XLSX (all 14 sheets)
    # ---------------------------------------------------------------
    print("\n[3/7] Processing XLSX (14 sheets)...")
    xlsx_path = RAW_DIR / "ReQuesta_RealTexts_overlap_LH.KC_12.29.25_reliability.xlsx"
    sheets, xlsx_report = process_xlsx(xlsx_path)
    quality_report["files"]["xlsx"] = xlsx_report

    # Save key sheets as clean CSVs
    sheet_output_map = {
        "Full Scores": "full_scores_both_annotators.csv",
        "Finalized Scores": "finalized_scores.csv",
        "Reliability Statistics": "reliability_statistics.csv",
        "Sociology Revised": "subject_scores_sociology.csv",
        "History": "subject_scores_history.csv",
        "Lifespan Developloment": "subject_scores_lifespan.csv",
        "Anthropology round 2": "subject_scores_anthropology.csv",
        "All": "all_questions_summary.csv",
    }

    for sheet_name, out_name in sheet_output_map.items():
        if sheet_name in sheets:
            df = sheets[sheet_name]
            # Drop fully empty rows
            df = df.dropna(how="all")
            df.to_csv(PROCESSED_DIR / out_name, index=False)
            print(f"  -> {sheet_name}: {len(df)} rows -> {out_name}")

    # ---------------------------------------------------------------
    # 3. Parse rubric PDF
    # ---------------------------------------------------------------
    print("\n[4/7] Parsing rubric PDF...")
    pdf_path = RAW_DIR / "Expert Evaluation Option-Level Rubric.pdf"
    rubric = parse_rubric_pdf(pdf_path)
    rubric_path = PROCESSED_DIR / "rubric.json"
    with open(rubric_path, "w", encoding="utf-8") as f:
        json.dump(rubric, f, indent=2, ensure_ascii=False)
    quality_report["files"]["rubric"] = {
        "dimensions": list(rubric["dimensions"].keys()),
        "scale": rubric["scale"],
    }
    print(f"  -> Extracted 4 dimensions, 4 levels each")
    print(f"  -> Saved: rubric.json")

    # ---------------------------------------------------------------
    # 4. Create merged annotations
    # ---------------------------------------------------------------
    print("\n[5/7] Creating merged annotations (Katerina + Linh side-by-side)...")
    df_merged = create_merged_annotations(df_k, df_l)
    df_merged.to_csv(PROCESSED_DIR / "merged_annotations.csv", index=False)
    print(f"  -> {len(df_merged)} rows, {len(df_merged.columns)} columns")

    # Agreement summary
    for dim in ANNOTATION_DIMENSIONS:
        agree_col = f"{dim}_agree"
        if agree_col in df_merged.columns:
            pct = df_merged[agree_col].mean() * 100
            print(f"  -> {dim} exact agreement: {pct:.1f}%")

    print(f"  -> Saved: merged_annotations.csv")

    # ---------------------------------------------------------------
    # 5. Create experiment-ready dataset
    # ---------------------------------------------------------------
    print("\n[6/7] Creating experiment-ready dataset...")
    df_exp = create_experiment_dataset(df_k)
    df_exp.to_csv(PROCESSED_DIR / "experiment_items.csv", index=False)
    print(f"  -> {len(df_exp)} items ({df_exp['question_uuid'].nunique()} questions x options)")
    print(f"  -> Saved: experiment_items.csv")

    # ---------------------------------------------------------------
    # 6. Save quality report
    # ---------------------------------------------------------------
    print("\n[7/7] Saving data quality report...")

    # Add cross-file consistency checks
    consistency = {}
    # Check that xlsx ACL sheets match the standalone CSVs
    if "Katerina-ACL-annotation" in sheets:
        xlsx_k = sheets["Katerina-ACL-annotation"]
        consistency["katerina_csv_vs_xlsx_rows_match"] = len(df_k) == len(xlsx_k)
    if "Linh-ACL-annotation" in sheets:
        xlsx_l = sheets["Linh-ACL-annotation"]
        consistency["linh_csv_vs_xlsx_rows_match"] = len(df_l) == len(xlsx_l)
    # Check ACL-annotation sheet (should be 720 = 360 + 360)
    if "ACL-annotation" in sheets:
        acl = sheets["ACL-annotation"]
        consistency["acl_annotation_rows"] = len(acl)
        consistency["acl_expected_720"] = len(acl) == 720

    quality_report["consistency_checks"] = consistency
    quality_report["output_files"] = sorted(
        [f.name for f in PROCESSED_DIR.iterdir() if f.is_file()]
    )

    report_path = PROCESSED_DIR / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, default=str)
    print(f"  -> Saved: data_quality_report.json")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {PROCESSED_DIR}")
    print(f"Files created: {len(quality_report['output_files'])}")
    for f in quality_report["output_files"]:
        size = (PROCESSED_DIR / f).stat().st_size
        print(f"  {f:45s} {size:>10,} bytes")

    # Save metadata
    meta.log_processing(
        step_name="raw_data_processing",
        input_files=[
            str(RAW_DIR / "Katerina_Annoted - Katerina-ACL-annotation.csv"),
            str(RAW_DIR / "Linh_Annoted - Linh-ACL-annotation.csv"),
            str(RAW_DIR / "ReQuesta_RealTexts_overlap_LH.KC_12.29.25_reliability.xlsx"),
            str(RAW_DIR / "Expert Evaluation Option-Level Rubric.pdf"),
        ],
        output_files=[str(PROCESSED_DIR / f) for f in quality_report["output_files"]],
        extra={
            "katerina_rows": quality_report["files"]["katerina"]["clean_rows"],
            "linh_rows": quality_report["files"]["linh"]["clean_rows"],
            "xlsx_sheets_processed": len(quality_report["files"].get("xlsx", {}).get("sheets", {})),
            "consistency_checks": quality_report.get("consistency_checks", {}),
        },
    )
    meta.stop_timer("data_processing_total")
    meta.save()


if __name__ == "__main__":
    main()
