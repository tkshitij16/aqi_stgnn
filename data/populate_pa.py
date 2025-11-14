#!/usr/bin/env python3
import os, json, gzip, argparse
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import yaml

REQUEST_TIMEOUT = 40
MAX_WORKERS = 8
YEAR = 2024
START_DATE = f"{YEAR}-01-01"
END_DATE   = f"{YEAR}-12-31"

OPEN_METEO    = "https://archive-api.open-meteo.com/v1/era5"
OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
SOCRATA_SEG_2024  = "https://data.cityofchicago.org/resource/4g9f-3jbs.json"
SOCRATA_APP_TOKEN = os.getenv("CHICAGO_APP_TOKEN", None)

LANE_RADIUS_M     = 400
TRAFFIC_RADIUS_M  = 1000
CACHE_DIR = ".cache_stt_2024"

def ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)

def round_key(lat: float, lon: float, ndigits=5) -> str:
    return f"{round(float(lat), ndigits)},{round(float(lon), ndigits)}"

def month_ranges(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for m in range(1, 13):
        start = pd.Timestamp(year=year, month=m, day=1, tz="UTC", hour=0)
        end = (pd.Timestamp(year=year, month=m+1, day=1, tz="UTC") - pd.Timedelta(seconds=1)) if m < 12 \
              else pd.Timestamp(year=year, month=12, day=31, tz="UTC", hour=23, minute=59)
        out.append((start, end))
    return out

def haversine_vec(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2_arr); lon2 = np.radians(lon2_arr)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ---------------- ERA5 ----------------
def era5_cache(lat, lon):
    ensure_cache()
    return os.path.join(CACHE_DIR, f"era5_{round(lat,5)}_{round(lon,5)}_{YEAR}.json.gz")

def fetch_era5(lat: float, lon: float) -> pd.DataFrame:
    if os.path.exists(era5_cache(lat, lon)):
        with gzip.open(era5_cache(lat, lon), "rt") as f: data = json.load(f)
    else:
        params = {
            "latitude":  float(lat), "longitude": float(lon),
            "start_date": START_DATE, "end_date": END_DATE,
            "hourly": "boundary_layer_height,precipitation,temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "timezone": "UTC",
        }
        r = requests.get(OPEN_METEO, params=params, timeout=REQUEST_TIMEOUT); r.raise_for_status()
        data = r.json()
        with gzip.open(era5_cache(lat, lon), "wt") as f: json.dump(data, f)

    h = data.get("hourly", {})
    t = pd.to_datetime(h.get("time", []), utc=True)
    if len(t) == 0:
        out = pd.DataFrame(columns=["epoch","PBLH","precip","T","RH","U","theta","node_key"])
        out["node_key"] = round_key(lat, lon); return out

    df = pd.DataFrame({
        "epoch": (t.view("int64") // 10**9).astype("int64"),
        "PBLH":  h.get("boundary_layer_height", [np.nan]*len(t)),
        "precip":h.get("precipitation",         [np.nan]*len(t)),
        "T":     h.get("temperature_2m",        [np.nan]*len(t)),
        "RH":    h.get("relative_humidity_2m",  [np.nan]*len(t)),
        "ws10":  h.get("wind_speed_10m",        [np.nan]*len(t)),
        "wd10":  h.get("wind_direction_10m",    [np.nan]*len(t)),
    })
    theta = np.deg2rad(pd.to_numeric(df["wd10"], errors="coerce"))
    ws = pd.to_numeric(df["ws10"], errors="coerce")
    df["theta"] = theta
    df["U"] = -ws * np.sin(theta)
    df.drop(columns=["ws10","wd10"], inplace=True)
    df["node_key"] = round_key(lat, lon)
    return df

# ---------------- Open-Meteo AQ ----------------
def aq_cache(lat, lon):
    ensure_cache()
    return os.path.join(CACHE_DIR, f"aq_{round(lat,5)}_{round(lon,5)}_{YEAR}.json.gz")

def fetch_openmeteo_aq(lat: float, lon: float) -> pd.DataFrame:
    if os.path.exists(aq_cache(lat, lon)):
        with gzip.open(aq_cache(lat, lon), "rt") as f: data = json.load(f)
    else:
        params = {
            "latitude": float(lat), "longitude": float(lon),
            "start_date": START_DATE, "end_date": END_DATE,
            "hourly": "pm2_5,us_aqi", "timezone": "UTC",
        }
        r = requests.get(OPEN_METEO_AQ, params=params, timeout=REQUEST_TIMEOUT); r.raise_for_status()
        data = r.json()
        with gzip.open(aq_cache(lat, lon), "wt") as f: json.dump(data, f)
    h = data.get("hourly", {})
    t = pd.to_datetime(h.get("time", []), utc=True)
    out = pd.DataFrame({
        "epoch": (t.view("int64") // 10**9).astype("int64") if len(t) else pd.Series([], dtype="int64"),
        "pm25_obs": h.get("pm2_5", []),
        "aqi_obs":  h.get("us_aqi", []),
    })
    out["node_key"] = round_key(lat, lon)
    return out

# ---------------- Traffic ----------------
def socrata_get(params: dict) -> list:
    headers = {"X-App-Token": SOCRATA_APP_TOKEN} if SOCRATA_APP_TOKEN else {}
    for _ in range(3):
        try:
            r = requests.get(SOCRATA_SEG_2024, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200: return r.json()
        except requests.exceptions.RequestException:
            pass
    return []

def traf_cache(key: str, bbox: Tuple[float,float,float,float]) -> str:
    ensure_cache()
    return os.path.join(CACHE_DIR, f"traffic_{key}_{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}.json.gz")

def fetch_month_traffic(bbox, start_ts, end_ts) -> pd.DataFrame:
    key = start_ts.strftime("%Y%m")
    path = traf_cache(key, bbox)
    if os.path.exists(path):
        with gzip.open(path, "rt") as f: data = json.load(f)
    else:
        params = {
            "$select": "start_time,speed,latitude,longitude",
            "$where": (
                f"start_time between '{start_ts.isoformat()}' and '{end_ts.isoformat()}' AND "
                f"latitude between {bbox[0]} and {bbox[2]} AND "
                f"longitude between {bbox[1]} and {bbox[3]}"
            ),
            "$limit": 500000
        }
        data = socrata_get(params)
        with gzip.open(path, "wt") as f: json.dump(data, f)
    if not data:
        return pd.DataFrame(columns=["start_time","speed","latitude","longitude"])
    df = pd.DataFrame.from_records(data)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["speed","latitude","longitude","start_time"])
    return df

def assign_hourly_traffic(df_traf: pd.DataFrame, nodes: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if df_traf.empty:
        return pd.DataFrame(columns=["epoch","node_key","spd_mean","v25_share"])
    df_traf["hour"] = df_traf["start_time"].dt.floor("h")
    out = []
    for hour, g in tqdm(df_traf.groupby("hour"), desc="Traffic→hourly", leave=False):
        arr = g[["latitude","longitude","speed"]].to_numpy()
        for _, r in nodes.iterrows():
            d = haversine_vec(r["lat"], r["lon"], arr[:,0], arr[:,1])
            mask = d <= radius_m
            if not mask.any(): continue
            speeds = arr[mask, 2]
            out.append({
                "epoch": int(pd.Timestamp(hour, tz="UTC").value // 10**9),
                "node_key": r["node_key"],
                "spd_mean": float(np.mean(speeds)),
                "v25_share": float(np.mean(speeds <= 25.0))
            })
    if not out:
        return pd.DataFrame(columns=["epoch","node_key","spd_mean","v25_share"])
    return pd.DataFrame(out)

# ---------------- Lanes ----------------
def compute_lane_km_density(lat: float, lon: float, radius_m: int = LANE_RADIUS_M) -> float:
    try:
        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.timeout = 40
        G = ox.graph_from_point((lat, lon), dist=radius_m, network_type="drive")
        edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        def parse_lanes(v):
            if v is None: return 1.0
            try: return float(v)
            except Exception:
                parts = [p.strip() for p in str(v).split(";")]
                vals = []
                for p in parts:
                    try: vals.append(float(p))
                    except: pass
                return float(np.mean(vals)) if vals else 1.0
        lanes = edges.get("lanes", pd.Series([np.nan]*len(edges))).apply(parse_lanes).fillna(1.0)
        lengths_m = edges.get("length", pd.Series([0.0]*len(edges))).fillna(0.0)
        lane_km = float((lengths_m * lanes).sum() / 1000.0)
        area_km2 = np.pi * (radius_m/1000.0)**2
        return lane_km / area_km2 if area_km2 > 0 else np.nan
    except Exception:
        return np.nan

# ---------------- AQI fallback ----------------
PM25_BREAKPOINTS_2024 = [
    (0.0,   9.0,    0,   50),
    (9.1,   35.4,   51,  100),
    (35.5,  55.4,   101, 150),
    (55.5,  125.4,  151, 200),
    (125.5, 225.4,  201, 300),
    (225.5, 500.4,  301, 500),
]
def aqi_from_pm25_2024(pm25: float) -> float:
    if pd.isna(pm25): return np.nan
    for Cl, Ch, Il, Ih in PM25_BREAKPOINTS_2024:
        if Cl <= pm25 <= Ch:
            return (Ih - Il) / (Ch - Cl) * (pm25 - Cl) + Il
    return 500.0 if pm25 > PM25_BREAKPOINTS_2024[-1][1] else np.nan

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file (schema & paths)")
    ap.add_argument("--input", required=True, help="Input CSV with epoch,node_id,lat,lon")
    ap.add_argument("--skip-lanes", action="store_true", help="Skip OSMnx lane_km_density (fill NaN)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    time_col = data_cfg["time_col"]
    node_col = data_cfg["node_col"]
    lat_col  = data_cfg["lat_col"]
    lon_col  = data_cfg["lon_col"]
    pm25_col = data_cfg["pm25_col"]
    aqi_col  = data_cfg["aqi_col"]
    static_cols = data_cfg["static_cols"]
    out_path = data_cfg["master_csv_path"]

    base = pd.read_csv(args.input)
    for c in [time_col, node_col, lat_col, lon_col]:
        if c not in base.columns:
            raise ValueError(f"Missing required column '{c}' in input CSV")

    base = base.copy()
    base[time_col] = pd.to_numeric(base[time_col], errors="coerce").astype("Int64")
    base = base.dropna(subset=[time_col, node_col, lat_col, lon_col]).copy()
    base[time_col] = base[time_col].astype("int64")
    base["row_id___"] = np.arange(len(base))
    base["node_key"] = base.apply(lambda r: round_key(r[lat_col], r[lon_col]), axis=1)

    # nodes + bbox
    nodes = base[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    nodes = nodes.rename(columns={lat_col: "lat", lon_col: "lon"})
    nodes["node_key"] = nodes.apply(lambda r: round_key(r["lat"], r["lon"]), axis=1)
    min_lat, max_lat = nodes["lat"].min()-0.02, nodes["lat"].max()+0.02
    min_lon, max_lon = nodes["lon"].min()-0.02, nodes["lon"].max()+0.02
    bbox = (float(min_lat), float(min_lon), float(max_lat), float(max_lon))

    # ERA5 per node
    era_frames = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(fetch_era5, float(r["lat"]), float(r["lon"])) for _, r in nodes.iterrows()]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="ERA5 per node"):
            try:
                era = fut.result()
                if not era.empty: era_frames.append(era)
            except Exception: pass
    era_all = pd.concat(era_frames, ignore_index=True) if era_frames else pd.DataFrame(columns=["epoch","node_key"])

    # AQ per node
    aq_frames = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(fetch_openmeteo_aq, float(r["lat"]), float(r["lon"])) for _, r in nodes.iterrows()]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Open-Meteo AQ per node"):
            try:
                dfq = fut.result()
                if not dfq.empty: aq_frames.append(dfq)
            except Exception: pass
    aq_all = pd.concat(aq_frames, ignore_index=True) if aq_frames else pd.DataFrame(columns=["epoch","pm25_obs","aqi_obs","node_key"])

    # Traffic
    traf_months = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 6)) as ex:
        futs = {ex.submit(fetch_month_traffic, bbox, s, e): (s, e) for (s, e) in month_ranges(YEAR)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Traffic monthly"):
            try:
                mdf = fut.result()
                if not mdf.empty: traf_months.append(mdf)
            except Exception: pass
    traf_all = pd.concat(traf_months, ignore_index=True) if traf_months else pd.DataFrame(columns=["start_time","speed","latitude","longitude"])
    traf_nodes = nodes[["lat","lon","node_key"]]
    traf_hourly = assign_hourly_traffic(traf_all, traf_nodes, radius_m=TRAFFIC_RADIUS_M).drop_duplicates(["epoch","node_key"])

    # Lane density (optional)
    if args.skip_lanes:
        lane_map = {}
    else:
        lane_map = {}
        for _, r in tqdm(nodes.iterrows(), total=len(nodes), desc="Lane density (OSMnx)"):
            lane_map[r["node_key"]] = compute_lane_km_density(float(r["lat"]), float(r["lon"]), radius_m=LANE_RADIUS_M)

    # Build working df and TEMPORARILY rename time column to 'epoch' for safe merges
    df = base[[time_col, node_col, lat_col, lon_col, "node_key", "row_id___"]].copy()
    restore_name = None
    if time_col != "epoch":
        df = df.rename(columns={time_col: "epoch"})
        restore_name = time_col

    # Merge ON exact keys (keeps 'epoch')
    if not era_all.empty:
        df = df.merge(era_all, on=["epoch","node_key"], how="left")
    if not aq_all.empty:
        df = df.merge(aq_all, on=["epoch","node_key"], how="left")
    if not traf_hourly.empty:
        df = df.merge(traf_hourly, on=["epoch","node_key"], how="left")
        df["congestion"] = df["v25_share"]

    # lane density map
    df["lane_km_density"] = df["node_key"].map(lane_map) if lane_map else np.nan

    # Ensure all required columns exist
    for c in ["U","theta","PBLH","T","RH","precip","spd_mean","congestion","lane_km_density","v25_share"]:
        if c not in df.columns: df[c] = np.nan
    for s in static_cols:
        if s not in df.columns: df[s] = np.nan

    # AQI fallback if Open-Meteo missing
    if "aqi_obs" not in df.columns:
        df["aqi_obs"] = df.get("pm25_obs", pd.Series(np.nan, index=df.index)).apply(aqi_from_pm25_2024)
    else:
        mask = df["aqi_obs"].isna() & df.get("pm25_obs", pd.Series(np.nan)).notna()
        df.loc[mask, "aqi_obs"] = df.loc[mask, "pm25_obs"].apply(aqi_from_pm25_2024)

    # Restore original time column name if needed
    if restore_name:
        df = df.rename(columns={"epoch": restore_name})
        time_col = restore_name  # keep consistent

    final_cols = [time_col, node_col, lat_col, lon_col] + \
                 ["U","theta","PBLH","T","RH","precip","spd_mean","congestion","lane_km_density","v25_share"] + \
                 static_cols + ["pm25_obs","aqi_obs"]

    out = df[final_cols + ["row_id___","node_key"]].sort_values("row_id___") \
            .drop(columns=["row_id___","node_key"], errors="ignore")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} | rows={len(out):,} | cols={len(out.columns)}")

if __name__ == "__main__":
    main()
