#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_yaml, ensure_dir, chicago_epoch_to_local_str, get_logger

log = get_logger("plotting")

def _as_str(df, col="node_id"):
    if col in df.columns:
        df[col] = df[col].astype(str)
    return df

def load_preds(cfg):
    p = cfg["outputs"]["pred_csv_with_time"]
    df = pd.read_csv(p, low_memory=False)
    # ensure node_id as string for merges
    df = _as_str(df, "node_id")
    # ensure readable time if missing
    if "time_local" not in df.columns:
        df["time_local"] = chicago_epoch_to_local_str(df["epoch"])
    # minimal checks
    need = ["epoch", "node_id", "aqi_hat"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"[plotting] Missing columns in predictions: {missing} (file: {p})")
    log.info(f"[Preds] Loaded {p} | rows={len(df):,} | node_id dtype={df['node_id'].dtype}")
    return df

def load_nodes(cfg):
    mpath = cfg["data"]["master_csv_path"]
    node_col = cfg["data"]["node_col"]
    lon_col  = cfg["data"]["lon_col"]
    lat_col  = cfg["data"]["lat_col"]

    m = pd.read_csv(mpath, usecols=[node_col, lon_col, lat_col], low_memory=False)
    m[node_col] = m[node_col].astype(str)
    nodes = m.drop_duplicates(subset=[node_col]).rename(columns={
        node_col: "node_id",
        lon_col: "lon",
        lat_col: "lat",
    })
    nodes = _as_str(nodes, "node_id")
    log.info(f"[Nodes] From {mpath} | unique nodes={len(nodes):,} | node_id dtype={nodes['node_id'].dtype}")
    return nodes

def monthly_maps(cfg):
    preds = load_preds(cfg)
    nodes = load_nodes(cfg)

    # month label (YYYY-MM)
    preds["month"] = preds["time_local"].str.slice(0, 7)

    ensure_dir("figs/dummy.txt")
    months = sorted(preds["month"].unique().tolist())
    if not months:
        log.warning("[Fig] No months present in preds; skipping monthly maps.")
        return

    for m in months:
        pm = preds[preds["month"] == m].groupby("node_id", as_index=False)["aqi_hat"].mean()
        # enforce string on both frames before merge (防 dtype mismatch)
        pm = _as_str(pm, "node_id")
        nodes = _as_str(nodes, "node_id")

        g = pm.merge(nodes, on="node_id", how="left")
        missing_xy = g["lon"].isna().sum()
        if missing_xy:
            log.warning(f"[Fig:{m}] {missing_xy} node(s) missing lon/lat after merge (check node_id consistency).")

        plt.figure(figsize=(6, 6))
        sc = plt.scatter(g["lon"], g["lat"], c=g["aqi_hat"])
        plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title(f"Mean AQI {m}")
        plt.colorbar(sc, label="AQI")
        out = f"figs/monthly_map_AQI_{m}.png"
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
        log.info(f"[Fig] Saved {out}")

    # Save monthly stats table
    stats = preds.groupby("month")["aqi_hat"].agg(["mean", "median", "std", "min", "max"]).reset_index()
    ensure_dir("tables/monthly_stats.csv")
    stats.to_csv("tables/monthly_stats.csv", index=False)
    log.info("[Table] Wrote tables/monthly_stats.csv")

def example_timeseries(cfg, example_nodes=None, start="2024-07-01", end="2024-07-10"):
    preds = load_preds(cfg)
    if example_nodes is None:
        example_nodes = preds["node_id"].drop_duplicates().head(4).tolist()
        log.info(f"[TS] Example nodes: {example_nodes}")

    mask = (preds["time_local"] >= start) & (preds["time_local"] <= end)
    p2 = preds[mask & preds["node_id"].isin(example_nodes)].copy()

    ensure_dir("figs/dummy.txt")
    plt.figure(figsize=(9, 4))
    for nid in example_nodes:
        s = p2[p2["node_id"] == nid].sort_values("epoch")
        if s.empty:
            log.warning(f"[TS] No data in range for node_id={nid}")
            continue
        plt.plot(s["time_local"], s["aqi_hat"], label=str(nid))
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Local time"); plt.ylabel("AQI (pred)")
    plt.title(f"AQI Time Series ({start} to {end})")
    plt.legend()
    out = "figs/timeseries_example_nodes.png"
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    log.info(f"[Fig] Saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    monthly_maps(cfg)
    example_timeseries(cfg)

if __name__ == "__main__":
    main()
