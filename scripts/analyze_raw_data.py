"""Analyze all raw data files and report structure, sheets, columns, etc."""
import pandas as pd
import os
import sys

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "raw")

for f in sorted(os.listdir(raw_dir)):
    fp = os.path.join(raw_dir, f)
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"FILE: {f}")
    print(f"SIZE: {os.path.getsize(fp)} bytes")
    print(sep)

    if f.endswith(".csv"):
        df = pd.read_csv(fp)
        print(f"ROWS: {len(df)}, COLS: {len(df.columns)}")
        print(f"COLUMNS: {df.columns.tolist()}")
        print(f"\nDTYPES:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        print(f"\nNULL COUNTS:")
        for col in df.columns:
            nc = df[col].isnull().sum()
            if nc > 0:
                print(f"  {col}: {nc} nulls ({nc/len(df)*100:.1f}%)")
        print(f"\nUNIQUE VALUES:")
        for col in df.columns:
            nuniq = df[col].nunique()
            if nuniq <= 15:
                vals = sorted(df[col].dropna().unique().tolist(), key=str)
                print(f"  {col}: {vals}")
            else:
                print(f"  {col}: {nuniq} unique values")
        print(f"\nFIRST 3 ROWS:")
        print(df.head(3).to_string())
        print(f"\nDIMENSION STATS (if numeric 1-4 columns):")
        dims = ["Relevance", "Clarity", "Linguistic_Feature", "Semantic_Plausibility"]
        for dim in dims:
            if dim in df.columns:
                s = df[dim].dropna()
                print(f"  {dim}: n={len(s)}, mean={s.mean():.2f}, std={s.std():.2f}, "
                      f"dist={s.value_counts().sort_index().to_dict()}")

    elif f.endswith(".xlsx"):
        xls = pd.ExcelFile(fp)
        print(f"SHEET COUNT: {len(xls.sheet_names)}")
        print(f"SHEET NAMES: {xls.sheet_names}")
        for sheet in xls.sheet_names:
            df = pd.read_excel(fp, sheet_name=sheet)
            print(f"\n  --- Sheet: '{sheet}' ---")
            print(f"  ROWS: {len(df)}, COLS: {len(df.columns)}")
            print(f"  COLUMNS: {df.columns.tolist()}")
            print(f"  DTYPES:")
            for col in df.columns:
                print(f"    {col}: {df[col].dtype}")
            print(f"  NULL COUNTS:")
            for col in df.columns:
                nc = df[col].isnull().sum()
                if nc > 0:
                    print(f"    {col}: {nc} nulls ({nc/len(df)*100:.1f}%)")
            print(f"  UNIQUE VALUES:")
            for col in df.columns:
                nuniq = df[col].nunique()
                if nuniq <= 15:
                    vals = sorted(df[col].dropna().unique().tolist(), key=str)
                    print(f"    {col}: {vals}")
                else:
                    print(f"    {col}: {nuniq} unique values")
            print(f"  FIRST 3 ROWS:")
            print(df.head(3).to_string())

    elif f.endswith(".pdf"):
        print("  [PDF file - requires separate extraction]")
        try:
            import pdfplumber
            with pdfplumber.open(fp) as pdf:
                print(f"  PAGES: {len(pdf.pages)}")
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        lines = text.strip().split("\n")
                        print(f"\n  --- Page {i+1} ({len(lines)} lines) ---")
                        for line in lines[:50]:
                            print(f"    {line}")
                        if len(lines) > 50:
                            print(f"    ... ({len(lines)-50} more lines)")
                    tables = page.extract_tables()
                    if tables:
                        for ti, table in enumerate(tables):
                            print(f"\n  TABLE {ti+1} on Page {i+1}: {len(table)} rows x {len(table[0]) if table else 0} cols")
                            for row in table[:5]:
                                print(f"    {row}")
        except ImportError:
            print("  pdfplumber not installed, trying PyPDF2...")
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(fp)
                print(f"  PAGES: {len(reader.pages)}")
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        lines = text.strip().split("\n")
                        print(f"\n  --- Page {i+1} ({len(lines)} lines) ---")
                        for line in lines[:50]:
                            print(f"    {line}")
            except ImportError:
                print("  No PDF library available. Install: pip install pdfplumber")

print("\n\nDONE.")
