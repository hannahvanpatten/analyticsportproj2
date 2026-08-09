# scripts/final_validations.py
"""
Final validation script for datasets before modeling.

Checks:
 - date column parseable and has expected ranges (train_end <= 2024-12-31, test in 2025)
 - target column presence and NaNs
 - duplicates on (productid,date)
 - suspicious columns that likely contain future information (e.g., plan_, contain '2025', 'future', 'next_')
 - prints summary stats

Exit codes:
 - 0 = all good (warnings possible)
 - 1 = critical failures (missing target, duplicates, date parse error)
"""

import argparse
import sys
import pandas as pd
import numpy as np

SUSPICIOUS_KEYWORDS = ["plan_", "plan", "2025", "2026", "future", "next_", "forecast", "target_year"]


def main(path: str, date_col: str = "date", group_col: str = "productid", target_col: str = "sales_units"):
    try:
        df = pd.read_csv(path, parse_dates=[date_col])
    except Exception as e:
        print(f"ERROR: Failed to read {path} or parse {date_col}: {e}", file=sys.stderr)
        return 1

    errs = 0
    warnings = 0

    print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
    # required columns
    for c in (group_col, date_col, target_col):
        if c not in df.columns:
            print(f"ERROR: Required column '{c}' not found.", file=sys.stderr)
            errs += 1

    if errs:
        return 1

    # Check for duplicates
    dup_mask = df.duplicated(subset=[group_col, date_col], keep=False)
    num_dups = int(dup_mask.sum())
    if num_dups > 0:
        print(f"ERROR: Found {num_dups} duplicate rows for (productid,date). Example duplicates:")
        print(df[dup_mask].head(10).to_string(index=False))
        errs += 1
    else:
        print("No duplicate (productid,date) pairs found.")

    # Check date ranges expected for training/test split (heuristic)
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    print(f"Date range: {min_date.date()} -> {max_date.date()}")
    # Expect training up to end of 2024 and test in 2025
    if (df[date_col].dt.year == 2025).any() and (min_date.year <= 2023):
        print("Dataset contains rows for 2025 (holdout) and earlier years. Good — this matches the expected split.")
    else:
        print("WARNING: I did not detect 2025 rows; verify train/test split is as you expect.", file=sys.stderr)
        warnings += 1

    # Check for NaNs in target in training window (conservative: check any NaNs)
    n_target_nans = int(df[target_col].isna().sum())
    if n_target_nans > 0:
        print(f"ERROR: Found {n_target_nans} NaNs in target column '{target_col}'.", file=sys.stderr)
        errs += 1
    else:
        print("No missing values in target column.")

    # Flag suspicious columns that may leak future info
    suspicious = [c for c in df.columns if any(k.lower() in c.lower() for k in SUSPICIOUS_KEYWORDS)]
    if suspicious:
        print("WARNING: The following potentially 'future/planned' columns were detected:")
        for c in suspicious:
            print("  -", c)
        print("If these values would not be available at forecast time, drop or transform them before modeling.", file=sys.stderr)
        warnings += 1
    else:
        print("No suspicious 'future' columns detected by heuristics.")

    # Basic per-product row-count check: expect roughly monthly sequence across date range
    counts = df.groupby(group_col)[date_col].nunique()
    min_count = int(counts.min())
    max_count = int(counts.max())
    print(f"Per-product unique-month counts: min={min_count}, max={max_count}, median={int(counts.median())}")
    if min_count < 12:
        print("WARNING: Some products have fewer than 12 months of history — modeling may be unstable for those SKUs.")
        warnings += 1

    # Print out a small summary of nulls per column (top 10)
    nulls = df.isna().sum().sort_values(ascending=False)
    print("Top columns with missing values:")
    print(nulls.head(10).to_string())

    # Final decision
    if errs:
        print(f"COMPLETED: Found {errs} error(s). Fix errors before modeling.", file=sys.stderr)
        return 1
    elif warnings:
        print(f"COMPLETED: {warnings} warning(s). Review and proceed if intentional.")
        return 0
    else:
        print("COMPLETED: All checks passed.")
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run final dataset validations before modeling.")
    ap.add_argument("path", help="Input CSV path (e.g., dataset_with_derived_metrics.csv)")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--product-col", default="productid")
    ap.add_argument("--target-col", default="sales_units")
    args = ap.parse_args()
    rc = main(args.path, date_col=args.date_col, group_col=args.product_col, target_col=args.target_col)
    sys.exit(rc)
