#!/usr/bin/env python
"""Fetch hourly inputs required for stt_master_locked_2024.csv."""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from pandas import Timestamp

from veg_dynamic import (
    fetch_landcover_class,
    fetch_ndvi_series,
    landcover_fractions,
    nearest_ndvi,
)

LOG = logging.getLogger("stt_master_pull")

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/era5"
TRAFFIC_URL = "https://data.cityofchicago.org/resource/sxs8-h27x.json"
OPENAQ_URL = "https://api.openaq.org/v2/measurements"


def configure_logging(level: int = logging.INFO) -> None:
    if LOG.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(asctime)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOG.addHandler(handler)
    LOG.setLevel(level)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull hourly data (meteorology, congestion, vegetation, PM2.5) for "
            "each node and write a CSV compatible with stt_master_locked_2024.csv"
        )
    )
    parser.add_argument(
        "--nodes",
        required=True,
        help="CSV listing node_id,lat,lon columns to seed the fetch",
    )
    parser.add_argument(
        "--output",
        default="stt_master_generated.csv",
        help="Path for the populated CSV",
    )
    parser.add_argument(
        "--start",
        default="2023-01-01T00:00:00",
        help="Start timestamp (America/Chicago local time)",
    )
    parser.add_argument(
        "--end",
        default="2025-11-14T23:00:00",
        help="End timestamp (America/Chicago local time)",
    )
    parser.add_argument(
        "--traffic-radius-m",
        type=int,
        default=800,
        help="Search radius in metres for congestion data",
    )
    parser.add_argument(
        "--openaq-radius-m",
        type=int,
        default=5000,
        help="Search radius in metres for OpenAQ measurements",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Chunk size (days) for Open-Meteo and Socrata queries",
    )
    parser.add_argument(
        "--socrata-token",
        default=os.environ.get("SOCRATA_APP_TOKEN"),
        help="Socrata application token (optional but recommended)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Base back-off when retrying HTTP calls (seconds)",
    )
    return parser.parse_args(argv)


def hourly_range(start_local: str, end_local: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_local, tz="America/Chicago")
    end = pd.Timestamp(end_local, tz="America/Chicago")
    if end < start:
        raise ValueError("End timestamp precedes start timestamp")
    return pd.date_range(start, end, freq="H", tz="America/Chicago").tz_convert("UTC")


def chunk_bounds(
    index: pd.DatetimeIndex, days: int
) -> Iterator[Tuple[Timestamp, Timestamp]]:
    if index.empty:
        return
    start = index[0]
    delta = timedelta(days=days)
    current = start
    while current <= index[-1]:
        nxt = min(current + delta - timedelta(hours=1), index[-1])
        yield current, nxt
        current = nxt + timedelta(hours=1)


def request_with_retries(
    session: requests.Session,
    url: str,
    params: Dict,
    sleep: float,
    max_attempts: int = 4,
) -> Dict:
    for attempt in range(max_attempts):
        try:
            resp = session.get(url, params=params, timeout=90)
            if resp.status_code == 200:
                return resp.json()
            LOG.warning(
                "Request failed %s (%s): %s", url, resp.status_code, resp.text[:200]
            )
        except requests.RequestException as exc:  # pragma: no cover - network guard
            LOG.warning("Exception during request: %s", exc)
        time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {max_attempts} attempts")


def fetch_weather(
    session: requests.Session,
    lat: float,
    lon: float,
    hours: pd.DatetimeIndex,
    chunk_days: int,
    sleep: float,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for chunk_start, chunk_end in chunk_bounds(hours, chunk_days):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": (
                "temperature_2m,relative_humidity_2m,precipitation,"
                "windspeed_10m,winddirection_10m,planetary_boundary_layer_height"
            ),
            "timezone": "UTC",
        }
        data = request_with_retries(session, OPEN_METEO_URL, params, sleep)
        hourly = data.get("hourly", {})
        if not hourly:
            continue
        frame = pd.DataFrame(hourly)
        if frame.empty:
            continue
        frame.rename(
            columns={
                "time": "timestamp",
                "temperature_2m": "T",
                "relative_humidity_2m": "RH",
                "precipitation": "precip",
                "windspeed_10m": "U",
                "winddirection_10m": "theta",
                "planetary_boundary_layer_height": "PBLH",
            },
            inplace=True,
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["timestamp", "U", "theta", "PBLH", "T", "RH", "precip"]
        )
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset="timestamp").sort_values("timestamp")
    result.set_index("timestamp", inplace=True)
    return result


def fetch_congestion(
    session: requests.Session,
    lat: float,
    lon: float,
    hours: pd.DatetimeIndex,
    radius_m: int,
    chunk_days: int,
    sleep: float,
    app_token: Optional[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    headers = {"Accept": "application/json"}
    if app_token:
        headers["X-App-Token"] = app_token
    for chunk_start, chunk_end in chunk_bounds(hours, chunk_days):
        params = {
            "$select": "date_trunc_ymdhm(start_time) as bucket, avg(congestion) as congestion",
            "$where": (
                f"start_time >= '{chunk_start.strftime('%Y-%m-%dT%H:%M:%S')}' AND "
                f"start_time <= '{chunk_end.strftime('%Y-%m-%dT%H:%M:%S')}' AND "
                f"within_circle(the_geom, {lat}, {lon}, {radius_m})"
            ),
            "$group": "bucket",
            "$order": "bucket",
            "$limit": 50000,
        }
        attempt_params = params.copy()
        for attempt in range(4):
            try:
                resp = session.get(
                    TRAFFIC_URL, params=attempt_params, headers=headers, timeout=90
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    break
                LOG.warning(
                    "[traffic] HTTP %s: %s", resp.status_code, resp.text[:200]
                )
            except requests.RequestException as exc:  # pragma: no cover
                LOG.warning("[traffic] exception %s", exc)
            time.sleep(sleep * (attempt + 1))
        else:
            LOG.warning("[traffic] Failed chunk %s - %s", chunk_start, chunk_end)
            continue
        if not payload:
            continue
        frame = pd.DataFrame(payload)
        if "bucket" not in frame.columns:
            continue
        frame.rename(columns={"bucket": "timestamp"}, inplace=True)
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], utc=True, errors="coerce"
        )
        frame["congestion"] = pd.to_numeric(
            frame.get("congestion"), errors="coerce"
        )
        frame = frame.dropna(subset=["timestamp"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "congestion"]).set_index("timestamp")
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset="timestamp").sort_values("timestamp")
    result.set_index("timestamp", inplace=True)
    return result


def fetch_pm25(
    session: requests.Session,
    lat: float,
    lon: float,
    hours: pd.DatetimeIndex,
    radius_m: int,
    chunk_days: int,
    sleep: float,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for chunk_start, chunk_end in chunk_bounds(hours, chunk_days):
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": radius_m,
            "parameter": "pm25",
            "temporal": "hour",
            "date_from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
        }
        for attempt in range(4):
            try:
                resp = session.get(OPENAQ_URL, params=params, timeout=90)
                if resp.status_code == 200:
                    payload = resp.json()
                    break
                LOG.warning(
                    "[openaq] HTTP %s: %s", resp.status_code, resp.text[:200]
                )
            except requests.RequestException as exc:  # pragma: no cover
                LOG.warning("[openaq] exception %s", exc)
            time.sleep(sleep * (attempt + 1))
        else:
            LOG.warning("[openaq] Failed chunk %s - %s", chunk_start, chunk_end)
            continue
        results = payload.get("results", []) if payload else []
        if not results:
            continue
        records = []
        for row in results:
            ts = row.get("date", {}).get("utc") or row.get("date", {}).get("local")
            value = row.get("value")
            if ts is None or value is None:
                continue
            records.append(
                {
                    "timestamp": pd.to_datetime(ts, utc=True),
                    "pm25_obs": float(value),
                }
            )
        if records:
            frame = pd.DataFrame(records)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "pm25_obs"]).set_index("timestamp")
    result = pd.concat(frames, ignore_index=True)
    result = result.groupby("timestamp", as_index=False)["pm25_obs"].mean()
    result.set_index("timestamp", inplace=True)
    return result


def compute_aqi_from_pm25(pm25: pd.Series) -> pd.Series:
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    def aqi_single(val: float) -> float:
        if not math.isfinite(val):
            return math.nan
        for c_low, c_high, a_low, a_high in breakpoints:
            if c_low <= val <= c_high:
                return (a_high - a_low) / (c_high - c_low) * (val - c_low) + a_low
        return 500.0

    return pm25.apply(aqi_single)


@dataclass
class NodeConfig:
    node_id: str
    lat: float
    lon: float


def prepare_base_frame(node: NodeConfig, hours: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index=hours)
    frame["node_id"] = node.node_id
    frame["lat"] = node.lat
    frame["lon"] = node.lon
    frame["epoch"] = frame.index.view("int64") // 10**9
    return frame


def fetch_vegetation(node: NodeConfig, years: Sequence[int]) -> Dict[int, Dict]:
    cache: Dict[int, Dict] = {}
    last_fractions: Optional[Dict[str, float]] = None
    for year in years:
        try:
            ndvi_series = fetch_ndvi_series(
                node.lat,
                node.lon,
                year,
                max_chunk=10,
                pause=0.5,
            )
        except Exception as exc:  # pragma: no cover - network failures
            LOG.warning(
                "[veg] NDVI fetch failed for %s (%s): %s", node.node_id, year, exc
            )
            ndvi_series = {}
        try:
            lc_class = fetch_landcover_class(node.lat, node.lon, year, 0.5)
            fractions = landcover_fractions(lc_class)
            last_fractions = fractions
        except Exception as exc:  # pragma: no cover
            LOG.warning(
                "[veg] Landcover fetch failed for %s (%s): %s",
                node.node_id,
                year,
                exc,
            )
            if last_fractions is not None:
                fractions = last_fractions
            else:
                fractions = {
                    k: 0.0
                    for k in [
                        "p_impervious",
                        "p_water",
                        "p_wetland",
                        "p_grass",
                        "p_cultivated",
                        "p_pasture",
                        "p_barren",
                    ]
                }
        cache[year] = {"ndvi": ndvi_series, "fractions": fractions}
    return cache


def vegetation_features(
    node: NodeConfig,
    frame: pd.DataFrame,
    cache: Dict[int, Dict],
) -> pd.DataFrame:
    ndvi_vals: List[float] = []
    for ts in frame.index:
        year = ts.tz_convert("America/Chicago").year
        if year in cache and cache[year]["ndvi"]:
            ndvi = nearest_ndvi(ts.date(), cache[year]["ndvi"])
        else:
            ndvi = math.nan
        ndvi_vals.append(ndvi)
    frame["ndvi_raw"] = ndvi_vals
    for col in [
        "p_impervious",
        "p_water",
        "p_wetland",
        "p_grass",
        "p_cultivated",
        "p_pasture",
        "p_barren",
    ]:
        frame[col] = 0.0
    # Apply most recent fractions
    latest_year = max(cache.keys()) if cache else None
    if latest_year is not None:
        for col, value in cache[latest_year]["fractions"].items():
            frame[col] = value
    # compute z per node
    arr = frame["ndvi_raw"].to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        frame["ndvi_z"] = 0.0
    else:
        mu = float(np.nanmean(arr[mask]))
        sigma = float(np.nanstd(arr[mask]))
        if not np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0
        frame["ndvi_z"] = (arr - mu) / sigma
    frame.drop(columns=["ndvi_raw"], inplace=True)
    return frame


def build_master_table(args: argparse.Namespace) -> pd.DataFrame:
    nodes_df = pd.read_csv(args.nodes)
    for col in ["node_id", "lat", "lon"]:
        if col not in nodes_df.columns:
            raise ValueError(f"Nodes CSV missing required column '{col}'")
    nodes = [
        NodeConfig(node_id=str(row["node_id"]), lat=float(row["lat"]), lon=float(row["lon"]))
        for _, row in nodes_df.iterrows()
    ]
    hours = hourly_range(args.start, args.end)
    session = requests.Session()
    frames: List[pd.DataFrame] = []
    years = sorted({ts.tz_convert("America/Chicago").year for ts in hours})
    LOG.info("Generating %d hourly steps per node", len(hours))
    for node in nodes:
        LOG.info("[node %s] fetching", node.node_id)
        base = prepare_base_frame(node, hours.copy())
        weather = fetch_weather(session, node.lat, node.lon, hours, args.chunk_days, args.sleep)
        base = base.join(weather, how="left")
        congestion = fetch_congestion(
            session,
            node.lat,
            node.lon,
            hours,
            args.traffic_radius_m,
            args.chunk_days,
            args.sleep,
            args.socrata_token,
        )
        base = base.join(congestion, how="left")
        pm25 = fetch_pm25(
            session,
            node.lat,
            node.lon,
            hours,
            args.openaq_radius_m,
            args.chunk_days,
            args.sleep,
        )
        base = base.join(pm25, how="left")
        base["aqi_obs"] = compute_aqi_from_pm25(base["pm25_obs"])
        veg_cache = fetch_vegetation(node, years)
        base = vegetation_features(node, base, veg_cache)
        frames.append(base.reset_index(drop=True))
    result = pd.concat(frames, ignore_index=True)
    columns = [
        "epoch",
        "node_id",
        "lat",
        "lon",
        "U",
        "theta",
        "PBLH",
        "T",
        "RH",
        "precip",
        "congestion",
        "ndvi_z",
        "p_impervious",
        "p_water",
        "p_wetland",
        "p_grass",
        "p_cultivated",
        "p_pasture",
        "p_barren",
        "pm25_obs",
        "aqi_obs",
    ]
    for col in columns:
        if col not in result.columns:
            result[col] = np.nan
    result = result[columns]
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging()
    table = build_master_table(args)
    LOG.info("Writing %s rows to %s", len(table), args.output)
    table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
