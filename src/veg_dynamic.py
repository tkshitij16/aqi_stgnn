#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Populate veg_dynamic.csv with 2024 vegetation metrics for each node."""

import argparse
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests

MODIS_NDVI_PRODUCT = "MOD13Q1"
MODIS_NDVI_BAND = "250m_16_days_NDVI"
MODIS_LC_PRODUCT = "MCD12Q1"
MODIS_LC_BAND = "LC_Type1"
MODIS_API = "https://modis.ornl.gov/rst/api/v1"


def get_logger(name: str = "veg_dynamic") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(levelname)s] %(asctime)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


log = get_logger()


def chunked(seq: Iterable, size: int) -> Iterable[List]:
    chunk: List = []
    for item in seq:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def day_of_year_string(year: int, doy: int) -> str:
    return f"A{year}{doy:03d}"


def fetch_modis_subset(url: str, pause: float = 0.2) -> Dict:
    for attempt in range(5):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            log.warning("[MODIS] Request failed (%s): %s", resp.status_code, resp.text[:200])
        except requests.RequestException as exc:  # pragma: no cover - network guard
            log.warning("[MODIS] Exception during request: %s", exc)
        time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Failed to fetch MODIS data after retries: {url}")


def fetch_ndvi_series(
    lat: float, lon: float, year: int, max_chunk: int, pause: float
) -> Dict[datetime.date, float]:
    doys = list(range(1, 366, 16))  # 16-day composite
    ndvi_values: Dict[datetime.date, float] = {}
    for chunk in chunked(doys, max_chunk):
        start = day_of_year_string(year, chunk[0])
        end = day_of_year_string(year, chunk[-1] + 15)
        url = (
            f"{MODIS_API}/{MODIS_NDVI_PRODUCT}/subset?latitude={lat}&longitude={lon}"
            f"&band={MODIS_NDVI_BAND}&startDate={start}&endDate={end}"
            "&kmAboveBelow=0&kmLeftRight=0"
        )
        data = fetch_modis_subset(url, pause=pause)
        scale = float(data.get("scale", 0.0001))
        for entry in data.get("subset", []):
            date = datetime.strptime(entry["calendar_date"], "%Y-%m-%d").date()
            values = entry.get("data", [])
            if not values:
                continue
            value = values[0]
            if value <= -3000:
                ndvi = math.nan
            else:
                ndvi = float(value) * scale
            ndvi_values[date] = ndvi
    return ndvi_values


def fetch_landcover_class(lat: float, lon: float, year: int, pause: float) -> int:
    start = day_of_year_string(year, 1)
    end = day_of_year_string(year, 365)
    url = (
        f"{MODIS_API}/{MODIS_LC_PRODUCT}/subset?latitude={lat}&longitude={lon}"
        f"&band={MODIS_LC_BAND}&startDate={start}&endDate={end}"
        "&kmAboveBelow=0&kmLeftRight=0"
    )
    data = fetch_modis_subset(url, pause=pause)
    subset = data.get("subset", [])
    if not subset:
        return -1
    values = subset[0].get("data", [])
    return int(values[0]) if values else -1


def landcover_fractions(lc_class: int) -> Dict[str, float]:
    frac = {
        "p_impervious": 0.0,
        "p_water": 0.0,
        "p_wetland": 0.0,
        "p_grass": 0.0,
        "p_cultivated": 0.0,
        "p_pasture": 0.0,
        "p_barren": 0.0,
    }

    if lc_class == 13:  # Urban
        frac["p_impervious"] = 1.0
    elif lc_class == 17:  # Water
        frac["p_water"] = 1.0
    elif lc_class == 11:  # Wetlands
        frac["p_wetland"] = 1.0
    elif lc_class in {12, 14}:  # Croplands
        frac["p_cultivated"] = 1.0
    elif lc_class in {8, 9, 10}:  # Savannas / grasslands
        frac["p_grass"] = 0.6
        frac["p_pasture"] = 0.4
    elif lc_class == 16:  # Barren
        frac["p_barren"] = 1.0
    elif lc_class in {1, 2, 3, 4, 5, 6, 7}:  # Forests / shrublands
        frac["p_grass"] = 0.4
        frac["p_pasture"] = 0.2
    else:
        # Unknown class → leave as zeros but avoid all-zero vector
        frac["p_grass"] = 0.1
    return frac


def nearest_ndvi(date: datetime.date, series: Dict[datetime.date, float]) -> float:
    if not series:
        return math.nan
    if date in series:
        return series[date]
    candidates = sorted(series.items(), key=lambda kv: abs((date - kv[0]).days))
    return candidates[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Populate veg_dynamic.csv with MODIS vegetation data")
    ap.add_argument("--input", default="veg_dynamic.csv", help="Input CSV with epoch,node_id,lat,lon and empty vegetation columns")
    ap.add_argument("--output", default="veg_dynamic_filled.csv", help="Where to write the populated CSV")
    ap.add_argument("--year", type=int, default=2024, help="Target calendar year for NDVI composites")
    ap.add_argument(
        "--landcover-year",
        type=int,
        default=2023,
        help="MCD12Q1 reference year for land-cover fractions",
    )
    ap.add_argument("--max-chunk", type=int, default=10, help="Maximum MODIS timesteps per API call (≤10 recommended)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Base pause between retries (seconds)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = ["epoch", "node_id", "lat", "lon"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column '{col}'")

    df["node_id"] = df["node_id"].astype(str)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    results = df.copy()
    results["ndvi_raw"] = np.nan

    node_meta: Dict[str, Dict] = {}
    for node_id, group in results.groupby("node_id"):
        lat = float(group["lat"].iloc[0])
        lon = float(group["lon"].iloc[0])
        ndvi_series = fetch_ndvi_series(lat, lon, args.year, args.max_chunk, args.sleep)
        lc_class = fetch_landcover_class(lat, lon, args.landcover_year, args.sleep)
        fractions = landcover_fractions(lc_class)
        node_meta[node_id] = {
            "ndvi_series": ndvi_series,
            "fractions": fractions,
        }
        log.info(
            "[Node %s] NDVI samples=%d | landcover=%s",
            node_id,
            len(ndvi_series),
            lc_class,
        )

    ndvi_by_node: Dict[str, List[float]] = defaultdict(list)
    for idx, row in results.iterrows():
        node_id = row["node_id"]
        epoch = int(row["epoch"])
        ndvi_series = node_meta[node_id]["ndvi_series"]
        ts = datetime.fromtimestamp(epoch, tz=timezone.utc).date()
        value = nearest_ndvi(ts, ndvi_series)
        results.at[idx, "ndvi_raw"] = value
        ndvi_by_node[node_id].append(value)
        for col, val in node_meta[node_id]["fractions"].items():
            results.at[idx, col] = val

    # Compute z-score per node
    results["ndvi_z"] = np.nan
    for node_id, values in ndvi_by_node.items():
        arr = np.array(values, dtype=float)
        mask = np.isfinite(arr)
        if mask.sum() == 0:
            zvals = np.zeros_like(arr)
        else:
            mu = float(np.nanmean(arr[mask]))
            sigma = float(np.nanstd(arr[mask]))
            if not np.isfinite(sigma) or sigma == 0.0:
                sigma = 1.0
            zvals = (arr - mu) / sigma
        results.loc[results["node_id"] == node_id, "ndvi_z"] = zvals

    results = results.drop(columns=["ndvi_raw"], errors="ignore")

    if "ndvi_z" not in results.columns:
        raise RuntimeError("Failed to compute ndvi_z column")

    results.to_csv(args.output, index=False)
    log.info("[Done] Wrote populated vegetation CSV → %s", args.output)


if __name__ == "__main__":
    main()
