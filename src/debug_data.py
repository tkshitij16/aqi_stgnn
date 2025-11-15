#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug / QA script for Chicago AQI STT-GNN pipeline.

This script inspects:
  - The locked master CSV (e.g. data/stt_master_locked_2024.csv)
  - The predictions CSV (e.g. outputs/preds_2024.csv)

It will:
  * Report basic info (rows, columns, time coverage).
  * Check node_id ↔ lat/lon consistency and missing coordinates.
  * Compute per-column NaN fraction, zero fraction, min/max/mean/std.
  * Check how many prediction rows successfully join to master.
  * If aqi_obs is available, compute error metrics between aqi_hat and aqi_obs.

Outputs (under --outdir, default tables/debug):
  - master_columns_summary.csv
  - master_node_coords_issues.csv
  - preds_summary.csv
  - preds_join_coverage.csv
  - preds_error_metrics.csv  (if aqi_obs is available)
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------
def get_logger(name: str = "debug_data") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(levelname)s] %(asctime)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        log.addHandler(ch)
    return log


log = get_logger()


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def summarize_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Compute NaN fraction, zero fraction, min, max, mean, std for numeric columns.
    """
    records: list[dict] = []
    n = len(df)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        nan_frac = s.isna().mean()
        zero_frac = (s == 0).mean()
        rec = {
            "column": c,
            "nan_fraction": float(nan_frac),
            "zero_fraction": float(zero_frac),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std()) if s.notna().any() else np.nan,
            "n": n,
        }
        records.append(rec)
    out = pd.DataFrame(records)
    out = out.sort_values(["nan_fraction", "zero_fraction", "column"])
    return out


def error_metrics(y_hat: np.ndarray, y_true: np.ndarray) -> dict:
    m = np.isfinite(y_hat) & np.isfinite(y_true)
    if m.sum() == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
        }
    yh = y_hat[m]
    yt = y_true[m]
    diff = yh - yt
    mse = np.mean(diff ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {
        "n": int(m.sum()),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--master",
        required=True,
        help="Path to master CSV (e.g. data/stt_master_locked_2024.csv)",
    )
    ap.add_argument(
        "--preds",
        required=True,
        help="Path to predictions CSV (e.g. outputs/preds_2024.csv)",
    )
    ap.add_argument(
        "--outdir",
        default="tables/debug",
        help="Output folder for debug tables (default: tables/debug)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: number of rows to sample from master/preds for quick checks",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --------------------- MASTER CSV ---------------------
    log.info(f"[Master] Loading master CSV: {args.master}")
    dfm = pd.read_csv(args.master, low_memory=False)
    if args.sample:
        dfm = dfm.sample(n=min(args.sample, len(dfm)), random_state=42)

    log.info(f"[Master] Rows={len(dfm):,} | Columns={len(dfm.columns)}")
    log.info(f"[Master] Columns: {list(dfm.columns)}")

    # Epoch coverage
    if "epoch" in dfm.columns:
        ep = pd.to_numeric(dfm["epoch"], errors="coerce")
        t = pd.to_datetime(ep.astype("Int64"), unit="s", utc=True)
        log.info(f"[Master] Epoch range: {ep.min()} .. {ep.max()}")
        log.info(f"[Master] Time range (UTC): {t.min()} .. {t.max()}")
    else:
        log.warning("[Master] No 'epoch' column found.")

    # Node-lat-lon consistency
    coord_issues_path = os.path.join(args.outdir, "master_node_coords_issues.csv")
    if all(c in dfm.columns for c in ["node_id", "lat", "lon"]):
        # make node_id consistently string in master
        dfm["node_id"] = dfm["node_id"].astype(str)

        coord = dfm[["node_id", "lat", "lon"]].dropna()
        # count unique lat/lon per node
        g = coord.groupby("node_id").agg(
            n_rows=("lat", "size"),
            n_unique_lat=("lat", "nunique"),
            n_unique_lon=("lon", "nunique"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
        ).reset_index()
        # nodes with inconsistent coordinates
        issues = g[(g["n_unique_lat"] > 1) | (g["n_unique_lon"] > 1)]
        issues.to_csv(coord_issues_path, index=False)
        log.info(
            f"[Master] Node coord table rows={len(g):,}, issues={len(issues):,} "
            f"→ {coord_issues_path}"
        )

        # nodes missing any coordinates
        miss_coord = (
            dfm[dfm["lat"].isna() | dfm["lon"].isna()]["node_id"]
            .astype(str)
            .unique()
        )
        log.info(f"[Master] Nodes with missing lat/lon: {len(miss_coord)}")
    else:
        log.warning(
            "[Master] 'node_id','lat','lon' not all present; skipping coord check."
        )

    # Column-level summary (numeric)
    numeric_cols = [
        c
        for c in dfm.columns
        if c not in ["node_id"] and pd.api.types.is_numeric_dtype(dfm[c])
    ]
    col_summary = summarize_numeric_columns(dfm, numeric_cols)
    col_summary_path = os.path.join(args.outdir, "master_columns_summary.csv")
    col_summary.to_csv(col_summary_path, index=False)
    log.info(f"[Master] Column summary → {col_summary_path}")

    # --------------------- PREDS CSV ---------------------
    log.info(f"[Preds] Loading predictions CSV: {args.preds}")
    dfp = pd.read_csv(args.preds, low_memory=False)
    if args.sample:
        dfp = dfp.sample(n=min(args.sample, len(dfp)), random_state=42)

    # Ensure node_id is string to match master
    if "node_id" in dfp.columns:
        dfp["node_id"] = dfp["node_id"].astype(str)

    log.info(f"[Preds] Rows={len(dfp):,} | Columns={len(dfp.columns)}")
    log.info(f"[Preds] Columns: {list(dfp.columns)}")

    if "epoch" in dfp.columns:
        ep_p = pd.to_numeric(dfp["epoch"], errors="coerce")
        t_p = pd.to_datetime(ep_p.astype("Int64"), unit="s", utc=True)
        log.info(f"[Preds] Epoch range: {ep_p.min()} .. {ep_p.max()}")
        log.info(f"[Preds] Time range (UTC): {t_p.min()} .. {t_p.max()}")
    else:
        log.warning("[Preds] No 'epoch' column found.")

    # basic preds summary
    preds_summary_cols = [
        c for c in dfp.columns if pd.api.types.is_numeric_dtype(dfp[c])
    ]
    preds_summary = summarize_numeric_columns(dfp, preds_summary_cols)
    preds_summary_path = os.path.join(args.outdir, "preds_summary.csv")
    preds_summary.to_csv(preds_summary_path, index=False)
    log.info(f"[Preds] Numeric column summary → {preds_summary_path}")

    # --------------------- JOIN MASTER & PREDS ---------------------
    if (
        "epoch" in dfm.columns
        and "epoch" in dfp.columns
        and "node_id" in dfm.columns
        and "node_id" in dfp.columns
    ):
        log.info("[Join] Checking coverage between master and preds on (epoch, node_id) …")

        dfm_join = dfm[["epoch", "node_id"]].copy()
        dfm_join["node_id"] = dfm_join["node_id"].astype(str)

        dfp_join = dfp[["epoch", "node_id"]].copy()
        dfp_join["node_id"] = dfp_join["node_id"].astype(str)

        # left join preds -> master
        merged = dfp_join.merge(
            dfm_join.drop_duplicates(["epoch", "node_id"]),
            on=["epoch", "node_id"],
            how="left",
            indicator=True,
        )
        n_total = len(merged)
        n_hit = (merged["_merge"] == "both").sum()
        n_miss = (merged["_merge"] == "left_only").sum()

        join_info = pd.DataFrame(
            [
                {
                    "total_pred_rows": n_total,
                    "matched_in_master": n_hit,
                    "missing_in_master": n_miss,
                    "match_fraction": n_hit / n_total if n_total > 0 else np.nan,
                }
            ]
        )
        join_path = os.path.join(args.outdir, "preds_join_coverage.csv")
        join_info.to_csv(join_path, index=False)
        log.info(
            f"[Join] match={n_hit:,} ({n_hit/n_total:.3%}) | "
            f"missing={n_miss:,} → {join_path}"
        )

        # Error metrics if we have aqi_hat and aqi_obs
        if "aqi_hat" in dfp.columns:
            if "aqi_obs" in dfm.columns:
                dfm_obs = dfm[["epoch", "node_id", "aqi_obs"]].copy()
                dfm_obs["node_id"] = dfm_obs["node_id"].astype(str)

                # Make sure preds side also has node_id as string (defensive)
                dfp_err = dfp.copy()
                dfp_err["node_id"] = dfp_err["node_id"].astype(str)

                dfp_err = dfp_err.merge(
                    dfm_obs,
                    on=["epoch", "node_id"],
                    how="left",
                )
                y_hat = pd.to_numeric(
                    dfp_err["aqi_hat"], errors="coerce"
                ).values
                y_true = pd.to_numeric(
                    dfp_err["aqi_obs"], errors="coerce"
                ).values
                met = error_metrics(y_hat, y_true)

                err_df = pd.DataFrame(
                    [
                        {
                            "n": met["n"],
                            "aqi_rmse": met["rmse"],
                            "aqi_mae": met["mae"],
                            "aqi_r2": met["r2"],
                        }
                    ]
                )
                err_path = os.path.join(args.outdir, "preds_error_metrics.csv")
                err_df.to_csv(err_path, index=False)
                log.info(
                    f"[Error] Overall vs. aqi_obs: "
                    f"RMSE={met['rmse']:.3f}, MAE={met['mae']:.3f}, R2={met['r2']:.3f} "
                    f"(n={met['n']:,}) → {err_path}"
                )
            else:
                log.warning(
                    "[Error] 'aqi_obs' not in master; cannot compute prediction error."
                )
        else:
            log.warning(
                "[Error] 'aqi_hat' not in preds; cannot compute prediction error."
            )
    else:
        log.warning(
            "[Join] Missing 'epoch' or 'node_id' in master/preds; skipping join coverage."
        )

    log.info("[Done] Debug tables written to: %s", args.outdir)


if __name__ == "__main__":
    main()
