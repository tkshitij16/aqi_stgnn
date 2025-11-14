#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normalize (standardize) dynamic & static feature columns using z-scores
computed on the TRAIN split only, as defined in your YAML. This avoids
data leakage and produces a CSV with extra *_z columns.

- Reads config (YAML) to get:
  - master_csv_path
  - train/val/test split boundaries (local Chicago times)
  - dynamic_cols + static_cols
- Computes per-column mean/std on TRAIN rows only (ignoring NaNs).
- Writes normalized copies with suffix `_z` for each dynamic/static column.
- Saves the scalers to NPZ: outputs/norm_scalers.npz (means, scales, column list).

USAGE
-----
python -m src.normalize_data --config configs/chicago_aqi_2024.yaml \
  --out_csv data/stt_master_normalized_z.csv

Notes
-----
* Original columns remain unchanged.
* Angle columns (e.g., theta in radians) are standardized as-is. If you prefer
  sin/cos representation, generate those upstream to avoid changing the model interface.
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    import os
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def to_epoch_local(chicago_str: str) -> int:
    tz = "America/Chicago"
    ts = pd.Timestamp(chicago_str)
    # robust localization across DST transitions
    loc = ts.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)
    return int(loc.tz_convert("UTC").value // 10**9)

def make_train_epochs(cfg) -> Tuple[int,int]:
    tr0, tr1 = cfg["data"]["train_hours"]
    return to_epoch_local(tr0), to_epoch_local(tr1)

def read_master(cfg) -> pd.DataFrame:
    p = cfg["data"]["master_csv_path"]
    tcol = cfg["data"]["time_col"]
    ncol = cfg["data"]["node_col"]
    df = pd.read_csv(p, low_memory=False)
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce").astype("int64")
    df[ncol] = df[ncol].astype(str)
    return df

def fit_stats(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float,float]]:
    """Return per-column (mean, std) computed on TRAIN rows (ignore NaNs)."""
    stats = {}
    for c in cols:
        if c not in train_df.columns:
            stats[c] = (np.nan, np.nan)
            continue
        v = pd.to_numeric(train_df[c], errors="coerce")
        mu = float(np.nanmean(v))
        sd = float(np.nanstd(v, ddof=0))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0  # avoid division by zero; column is constant or empty
        stats[c] = (mu, sd)
    return stats

def apply_stats(full_df: pd.DataFrame, stats: Dict[str, Tuple[float,float]], cols: List[str]) -> pd.DataFrame:
    out = full_df.copy()
    for c in cols:
        if c not in out.columns:
            # create NaN column to keep schema consistent
            out[f"{c}_z"] = np.nan
            continue
        mu, sd = stats[c]
        v = pd.to_numeric(out[c], errors="coerce")
        out[f"{c}_z"] = (v - mu) / sd
    return out

def save_npz_scalers(path: str, stats: Dict[str, Tuple[float,float]]):
    ensure_dir(path)
    cols = list(stats.keys())
    means = np.array([stats[c][0] for c in cols], dtype="float64")
    scales = np.array([stats[c][1] for c in cols], dtype="float64")
    np.savez(path, columns=np.array(cols), means=means, scales=scales)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML (e.g., configs/chicago_aqi_2024.yaml)")
    ap.add_argument("--out_csv", required=True, help="Where to write normalized CSV with *_z columns")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    df = read_master(cfg)

    # identify columns
    dyn_cols = cfg["data"]["dynamic_cols"]
    stat_cols = cfg["data"]["static_cols"]
    all_cols = dyn_cols + stat_cols

    # restrict training rows by epoch
    tcol = cfg["data"]["time_col"]
    tr0, tr1 = make_train_epochs(cfg)
    train_df = df[(df[tcol] >= tr0) & (df[tcol] <= tr1)].copy()

    # fit stats on TRAIN only
    stats = fit_stats(train_df, all_cols)

    # apply to full dataset -> create *_z columns
    out = apply_stats(df, stats, all_cols)

    # Save CSV and scalers
    out_path = args.out_csv
    ensure_dir(out_path)
    out.to_csv(out_path, index=False)
    scalers_path = "outputs/norm_scalers.npz"
    save_npz_scalers(scalers_path, stats)

    print(f"Wrote normalized CSV with *_z columns: {out_path}")
    print(f"Saved scaler stats: {scalers_path}")
    print("Normalized columns:")
    for c in all_cols:
        print("  ", f"{c}_z")

if __name__ == "__main__":
    main()
