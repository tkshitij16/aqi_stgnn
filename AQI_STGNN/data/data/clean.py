#!/usr/bin/env python3
"""
Impute blanks/missing values in an AQI timeseries CSV safely for ML training.

Policy
------
- Recognize blanks ("", " ", NA, n/a, null, etc.) as missing.
- Convert mostly-numeric text columns to numeric (heuristic).
- For numeric features:
    * If 'epoch' (datetime) and 'node_id' exist: sort by (node_id, epoch) and
      interpolate within each node over time; then fill remaining NaN with
      per-node medians; then with global medians.
    * Add a binary "<col>__nan" missingness indicator before filling.
- For non-numeric (categorical) features: fill missing with "UNK".
- Never touch the target column if provided via --target (it’s copied through).
- Writes a new CSV with the same columns + missingness flags.

Usage
-----
python impute_aqi.py \
  --in "AQI_STGNN/data/data/stt_master_2024_baclup.csv" \
  --out "AQI_STGNN/data/data/stt_master_2024_imputed.csv" \
  --epoch-col epoch --node-col node_id
# Optional (don’t impute your label):
#  --target "AQI"     (or whatever your label column is)
"""

import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple

NA_STRINGS = ["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null", "-", "--"]

def coerce_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert object columns that are 'mostly numeric' to numeric dtype.
    A column is 'mostly numeric' if ≥90% of its non-NA values parse as numbers.
    """
    numericized = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == "object":
            ser = df[c].astype(str)
            # Try parsing to numeric; non-numeric will become NaN
            parsed = pd.to_numeric(ser, errors="coerce")
            non_na = parsed.notna().sum()
            total = ser.str.len().gt(0).sum()
            if total > 0 and (non_na / total) >= 0.90:
                df[c] = parsed
                numericized.append(c)
    return df, numericized

def add_missing_indicators(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    for c in numeric_cols:
        ind_name = f"{c}__nan"
        # Only add if not already present
        if ind_name not in df.columns:
            df[ind_name] = df[c].isna().astype("int8")
    return df

def interpolate_groupwise_time(df: pd.DataFrame, numeric_cols: List[str],
                               node_col: str, epoch_col: str) -> pd.DataFrame:
    # Ensure correct sorting and time index per group
    df = df.sort_values([node_col, epoch_col]).copy()
    def _interp(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g = g.set_index(epoch_col)
        # time-based interpolation requires a datetime index
        g[numeric_cols] = g[numeric_cols].interpolate(method="time", limit_direction="both")
        return g.reset_index()
    return df.groupby(node_col, group_keys=False).apply(_interp)

def fill_remaining_with_medians(df: pd.DataFrame, numeric_cols: List[str], node_col: str) -> pd.DataFrame:
    # Per-node medians
    if node_col in df.columns:
        med_node = df.groupby(node_col)[numeric_cols].median(numeric_only=True)
        df = df.join(med_node, on=node_col, rsuffix="_node_median")
        for c in numeric_cols:
            node_med = f"{c}_node_median"
            df[c] = df[c].fillna(df[node_med])
            df.drop(columns=[node_med], inplace=True)
    # Global medians
    global_meds = df[numeric_cols].median(numeric_only=True)
    df[numeric_cols] = df[numeric_cols].fillna(global_meds)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    ap.add_argument("--epoch-col", default="epoch", help="Datetime column name (default: epoch)")
    ap.add_argument("--node-col", default="node_id", help="Node/sensor id column (default: node_id)")
    ap.add_argument("--target", default=None, help="Target/label column to carry through (not imputed)")
    args = ap.parse_args()

    # Read with robust NA handling; preserve whitespace-only as NA too
    df = pd.read_csv(
        args.in_path,
        na_values=NA_STRINGS,
        keep_default_na=True
    )
    # Convert pure whitespace strings to NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Parse epoch if present
    has_epoch = args.epoch_col in df.columns
    if has_epoch:
        df[args.epoch_col] = pd.to_datetime(df[args.epoch_col], errors="coerce", utc=False)

    # Heuristic numeric coercion (avoid target)
    exclude = [args.target] if args.target and args.target in df.columns else []
    df, numericized = coerce_numeric_columns(df, exclude=exclude)

    # Identify numeric vs categorical columns (excluding target)
    if exclude:
        work_df = df.drop(columns=exclude)
    else:
        work_df = df

    numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in work_df.columns if c not in numeric_cols]

    # Add missingness indicators for numeric features
    df = add_missing_indicators(df, numeric_cols)

    # Time-aware interpolation within each node if we have node & epoch
    has_node = args.node_col in df.columns
    if has_node and has_epoch:
        df = interpolate_groupwise_time(df, numeric_cols, node_col=args.node_col, epoch_col=args.epoch_col)
    # Fill any leftovers using per-node medians then global medians
    df = fill_remaining_with_medians(df, numeric_cols, node_col=args.node_col if has_node else None)

    # Categorical: fill with sentinel
    for c in categorical_cols:
        if c in df.columns:  # be safe
            df[c] = df[c].fillna("UNK")

    # (Optional) sanity: ensure target untouched except original NaNs
    # If you *do* want to drop rows with missing target, uncomment:
    # if args.target and args.target in df.columns:
    #     df = df[~df[args.target].isna()].copy()

    # Save
    df.to_csv(args.out_path, index=False)

    # Quick report
    num_imputed = sum(df.filter(regex="__nan$").sum().tolist())
    print(f"[OK] Wrote imputed file: {args.out_path}")
    print(f"    Numeric columns treated: {len(numeric_cols)}")
    print(f"    Object→numeric coerced: {numericized}")
    print(f"    Total missing numeric cells originally (approx.): {num_imputed}")

if __name__ == "__main__":
    main()
