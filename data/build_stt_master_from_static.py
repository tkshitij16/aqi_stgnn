#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

# REQUIRED schema (but we only fill what you actually have)
DYN_ORDER = ["U","theta","T","RH","congestion","PBLH","precip","spd_mean","lane_km_density","v25_share"]
STAT_ORDER = ["ndvi_z","p_impervious","p_dev_intense","p_dev_med","p_dev_low","p_dev_open",
              "p_tree","p_grass","p_shrub","p_water","p_wetland","p_barren","p_cultivated",
              "p_pasture","p_snowice","p_emerging"]

NLCD_GROUPS = {
    "p_tree":       ["Deciduous Forest","Evergreen Forest","Mixed Forest","Tree Canopy","Trees"],
    "p_shrub":      ["Shrub/Scrub","Shrub"],
    "p_grass":      ["Grassland/Herbaceous","Grass","Lawn"],
    "p_pasture":    ["Pasture/Hay","Pasture","Hay"],
    "p_cultivated": ["Cultivated Crops","Crops","Cropland"],
    "p_wetland":    ["Woody Wetlands","Emergent Herbaceous Wetlands","Wetlands"],
    "p_water":      ["Open Water","Water"],
    "p_barren":     ["Barren Land","Barren"],
    "p_dev_open":   ["Developed, Open Space"],
    "p_dev_low":    ["Developed, Low Intensity"],
    "p_dev_med":    ["Developed, Medium Intensity"],
    "p_dev_intense":["Developed, High Intensity"],
    "p_snowice":    ["Snow/Ice","Snow and Ice"],
    "p_emerging":   ["Emergent Herbaceous Wetlands"],
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="features_2024_cleaned.csv")
    ap.add_argument("--out", required=True, help="stt_master_2024.csv")
    ap.add_argument("--theta_unit", default="deg", choices=["deg","rad"], help="wind_dir in deg or rad")
    return ap.parse_args()

def normalize_epoch_seconds(sr):
    sr = pd.to_numeric(sr, errors="coerce")
    if sr.dropna().max() > 1e12:
        sr = (sr // 1000)
    return sr.astype("Int64")

def main():
    args = parse_args()
    df = pd.read_csv(args.src, low_memory=False)

    # --- Keys ---
    if "Station_id" not in df.columns or "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise SystemExit("Expected columns 'Station_id', 'Latitude', 'Longitude' not found.")
    epoch_col = None
    for c in ["epoch","Epoch","EPOCH","Epoch Time","epoch_time","Epoch_Time","EpochTime"]:
        if c in df.columns:
            epoch_col = c; break
    if epoch_col is None:
        raise SystemExit("No epoch column found (e.g., 'Epoch Time').")

    out = pd.DataFrame()
    out["node_id"] = df["Station_id"].astype(str)
    out["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    out["epoch"] = normalize_epoch_seconds(df[epoch_col]).astype("int64")

    # --- Dynamics present in your file ---
    # U
    if "wind_speed" in df.columns:
        out["U"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    # theta
    if "wind_dir" in df.columns:
        th = pd.to_numeric(df["wind_dir"], errors="coerce")
        out["theta"] = np.deg2rad(th) if args.theta_unit=="deg" else th
    # T, RH
    if "temperature_c" in df.columns:
        out["T"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    if "humidity" in df.columns:
        out["RH"] = pd.to_numeric(df["humidity"], errors="coerce")
    # congestion
    if "TrafficWeight" in df.columns:
        out["congestion"] = pd.to_numeric(df["TrafficWeight"], errors="coerce")

    # Ensure absent dynamics exist as NaN (NO zero filling here)
    for c in DYN_ORDER:
        if c not in out.columns:
            out[c] = np.nan

    # --- Targets ---
    if "AQI" in df.columns:
        out["aqi_obs"] = pd.to_numeric(df["AQI"], errors="coerce")
    else:
        out["aqi_obs"] = np.nan
    out["pm25_obs"] = np.nan  # keep as NaN unless you provide it

    # --- Statics ---
    # 1) ndvi_z from NDVI (z-score across nodes)
    if "NDVI" in df.columns:
        ndvi_node = df.groupby(df["Station_id"].astype(str))["NDVI"].mean()
        ndvi_node = (ndvi_node - ndvi_node.mean()) / (ndvi_node.std(ddof=0) + 1e-12)
        out["ndvi_z"] = out["node_id"].map(ndvi_node)
    else:
        out["ndvi_z"] = np.nan

    # 2) NLCD groups -> per-node means, then map
    gby = df.groupby(df["Station_id"].astype(str))
    def per_node_mean(col):
        return gby[col].mean() if col in df.columns else None

    # Developed sums form p_impervious (if available)
    dev_open = per_node_mean("Developed, Open Space")
    dev_low  = per_node_mean("Developed, Low Intensity")
    dev_med  = per_node_mean("Developed, Medium Intensity")
    dev_hi   = per_node_mean("Developed, High Intensity")

    # Initialize all static columns as NaN
    for c in STAT_ORDER:
        if c not in out.columns: out[c] = np.nan

    # Individual dev classes -> map if present
    if dev_open is not None: out["p_dev_open"] = out["node_id"].map(dev_open)
    if dev_low  is not None: out["p_dev_low"]  = out["node_id"].map(dev_low)
    if dev_med  is not None: out["p_dev_med"]  = out["node_id"].map(dev_med)
    if dev_hi   is not None: out["p_dev_intense"] = out["node_id"].map(dev_hi)

    # p_impervious = sum of developed classes (available ones)
    if any(x is not None for x in [dev_open, dev_low, dev_med, dev_hi]):
        s = 0
        if dev_open is not None: s = s + out["node_id"].map(dev_open).fillna(0)
        if dev_low  is not None: s = s + out["node_id"].map(dev_low).fillna(0)
        if dev_med  is not None: s = s + out["node_id"].map(dev_med).fillna(0)
        if dev_hi   is not None: s = s + out["node_id"].map(dev_hi).fillna(0)
        out["p_impervious"] = s

    # Other NLCD groups
    for out_col, cand in NLCD_GROUPS.items():
        if out_col in ["p_dev_open","p_dev_low","p_dev_med","p_dev_intense"]:
            continue  # already handled
        present = [c for c in cand if c in df.columns]
        if present:
            pernode = gby[present].mean().sum(axis=1)
            pernode.index = pernode.index.astype(str)
            out[out_col] = out["node_id"].map(pernode)

    # Order columns
    final_cols = ["epoch","node_id","lon","lat"] + DYN_ORDER + STAT_ORDER + ["pm25_obs","aqi_obs"]
    out = out[final_cols].sort_values(["epoch","node_id"]).reset_index(drop=True)

    # Save
    out.to_csv(args.out, index=False)
    print("Wrote", args.out)
    print("Rows:", len(out), "| Unique nodes:", out["node_id"].nunique())

    # Quick non-null report
    nn = out.notna().sum()
    print("\nNon-null counts (key fields):")
    for k in ["U","theta","T","RH","congestion","aqi_obs","pm25_obs","ndvi_z","p_impervious","p_water","p_wetland","p_grass","p_cultivated","p_pasture","p_barren"]:
        if k in nn.index:
            print(f"{k:14s} : {int(nn[k])}")

if __name__ == "__main__":
    main()
