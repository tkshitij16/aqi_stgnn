#!/usr/bin/env python3
"""
Chicago traffic metrics filler (spd_mean, lane_km_density, v25_share) with robust region centroid discovery.

Inputs
------
CSV with columns: node_id, lat, lon, epoch   (epoch = hourly UNIX seconds, CST/CDT for 2024)

Outputs
-------
Adds per-row:
  - spd_mean (mph)         : hourly mean of 10-min speeds
  - v25_share (0..1)       : share of 10-min samples <= 25 mph in that hour
  - lane_km_density        : lane-km per km^2 within radius R (default 500 m)

Sources
-------
- Speeds by Region (historical): kf7e-cur8 (2018-Current)
- Speeds by Segment (historical): 4g9f-3jbs (>= 2024-06-11)
- Regions (for centroids only; multiple fallbacks):
    * t2qc-9pjd (current)
    * sxs8-h27x (2018-current)
    * emtn-qqdi (2013-2018)
- OSM Overpass (lane density)

CLI
---
python traffic.py pa_2024.csv --out pa_2024__with_traffic.csv --source auto --radius_m 500
# optional: --no_osm
# optional: --soda_token TOKEN (or env SODA_APP_TOKEN)
# optional: --region_centroids regions_centroids.csv
# optional: --source segment_only (only fill from 2024-06-11 onward)

Requirements
------------
pip install pandas requests pytz
"""

import os, math, json, time, argparse
from datetime import datetime, timedelta
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------ CONSTANTS ------------------
SODA_DOMAIN = "data.cityofchicago.org"

# Historical speeds
REGION_DS_SPEEDS   = "kf7e-cur8"   # Historical by region (2018+)
SEGMENT_DS_SPEEDS  = "4g9f-3jbs"   # Historical by segment (>= 2024-06-11)
SEGMENT_START = datetime(2024, 6, 11)

# Region centroids fallbacks (coordinate-friendly views)
REGION_DS_PRIORITY = [
    "t2qc-9pjd",  # current regions (preferred)
    "sxs8-h27x",  # regions 2018-current
    "emtn-qqdi",  # regions 2013-2018
]

SODA_TIMEOUT_CONNECT = 10
SODA_TIMEOUT_READ = 60
SODA_MAX_RETRIES = 5
SODA_BACKOFF = 0.5
SODA_PAGE_LIMIT = 50000

OSM_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
OVERPASS_TIMEOUT = 60
OVERPASS_RETRIES = 3
OVERPASS_SLEEP = 1.5

V25_THRESHOLD = 25.0
DEFAULT_RADIUS_M = 500
OSM_HIGHWAYS_REGEX = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"
)

CACHE_DIR = ".traffic_cache"
LANE_CACHE_FILE = os.path.join(CACHE_DIR, "lane_density_cache.json")
REGION_CENTROIDS_FILE = os.path.join(CACHE_DIR, "region_centroids.json")
SEGMENT_CENTROIDS_FILE = os.path.join(CACHE_DIR, "segment_centroids.json")

# Optional embedded token (you can override via --soda_token or env)
EMBEDDED_SODA_APP_TOKEN = None  # set a string if you have one

# ------------------ UTILS ------------------
def log(msg: str):
    print(msg, flush=True)

def ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)

def session_with_retries(token: str | None):
    s = requests.Session()
    retry = Retry(
        total=SODA_MAX_RETRIES,
        backoff_factor=SODA_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"Accept": "application/json"})
    if token:
        s.headers.update({"X-App-Token": token})
    return s

def epoch_to_local_hour(ts, tz_name="America/Chicago"):
    import pytz
    tz = pytz.timezone(tz_name)
    dt = datetime.fromtimestamp(int(ts), tz=tz)
    return dt.replace(minute=0, second=0, microsecond=0)

def haversine_km(p1, p2):
    R = 6371.0088
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def parse_lanes(tag):
    if not tag:
        return 1
    parts = [p.strip() for p in str(tag).replace("|",";").split(";")]
    nums = []
    for p in parts:
        try:
            nums.append(int(float(p)))
        except Exception:
            pass
    return max(nums) if nums else 1

# ------------------ OVERPASS (OSM) ------------------
def overpass_lane_km_density(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}];
    way(around:{radius_m},{lat},{lon})["highway"~"{OSM_HIGHWAYS_REGEX}"];
    out geom tags;
    """
    last_err = None
    for attempt in range(OVERPASS_RETRIES):
        for base in OSM_MIRRORS:
            try:
                r = requests.post(base, data=query, timeout=(10, OVERPASS_TIMEOUT))
                if r.status_code == 200:
                    data = r.json()
                    lane_km = 0.0
                    for el in data.get("elements", []):
                        geom = el.get("geometry") or []
                        lanes = parse_lanes((el.get("tags") or {}).get("lanes"))
                        length_km = 0.0
                        for i in range(1, len(geom)):
                            p1 = (geom[i-1]["lat"], geom[i-1]["lon"])
                            p2 = (geom[i]["lat"], geom[i]["lon"])
                            length_km += haversine_km(p1, p2)
                        lane_km += length_km * lanes
                    area_km2 = math.pi * (radius_m/1000.0)**2
                    return (lane_km / area_km2) if area_km2 > 0 else 0.0
                last_err = RuntimeError(f"{base} HTTP {r.status_code}")
            except Exception as e:
                last_err = e
        time.sleep(OVERPASS_SLEEP * (attempt + 1))
    raise last_err

def load_lane_cache():
    ensure_cache()
    if os.path.exists(LANE_CACHE_FILE):
        try:
            with open(LANE_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_lane_cache(cache):
    ensure_cache()
    with open(LANE_CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ------------------ SOCRATA helpers ------------------
def soda_get_json(session, path, params):
    url = f"https://{SODA_DOMAIN}{path}"
    r = session.get(url, params=params, timeout=(SODA_TIMEOUT_CONNECT, SODA_TIMEOUT_READ))
    r.raise_for_status()
    return r.json()

def socrata_fetch_meta(session, dataset_id: str):
    url = f"https://{SODA_DOMAIN}/api/views/{dataset_id}.json"
    r = session.get(url, timeout=(SODA_TIMEOUT_CONNECT, SODA_TIMEOUT_READ))
    r.raise_for_status()
    return r.json()

def socrata_detect_point_field_from_meta(meta: dict) -> str | None:
    for col in meta.get("columns", []):
        dtype = (col.get("dataTypeName") or "").lower()
        if dtype in ("point", "location"):
            field = col.get("fieldName") or col.get("name")
            if field:
                return field
    return None

def try_fetch_region_centroids_from_dataset(session, dsid: str) -> pd.DataFrame | None:
    """
    Try hard to produce [region_id, lat, lon] from a given regions dataset id.
    Strategy:
      1) Detect a point/location field via metadata -> select region_id + field.latitude/longitude
      2) If metadata unavailable or wrong, sample rows and heuristically find any dict with 'latitude'/'longitude'
      3) Try common explicit field names: latitude/longitude
    """
    # (1) metadata-based
    try:
        meta = socrata_fetch_meta(session, dsid)
        point_field = socrata_detect_point_field_from_meta(meta)
    except Exception:
        point_field = None

    if point_field:
        params = {
            "$select": f"region_id, {point_field} as pt, {point_field}.latitude as lat, {point_field}.longitude as lon",
            "$group": "region_id, pt, lat, lon",
            "$limit": 2000
        }
        try:
            rows = soda_get_json(session, f"/resource/{dsid}.json", params)
            out = []
            for r in rows:
                rid = r.get("region_id")
                lat = r.get("lat")
                lon = r.get("lon")
                if (lat is None or lon is None) and isinstance(r.get("pt"), dict):
                    lat = r["pt"].get("latitude", lat)
                    lon = r["pt"].get("longitude", lon)
                try:
                    out.append({"region_id": int(rid), "lat": float(lat), "lon": float(lon)})
                except Exception:
                    pass
            df = pd.DataFrame(out).dropna()
            if not df.empty:
                return df
        except Exception:
            pass

    # (2) heuristic scan of sample rows
    try:
        sample = soda_get_json(session, f"/resource/{dsid}.json", {"$limit": 100})
        # find candidate dict field with 'latitude'/'longitude'
        dict_cols = set()
        for row in sample:
            for k, v in row.items():
                if isinstance(v, dict) and "latitude" in v and "longitude" in v:
                    dict_cols.add(k)
        for col in dict_cols:
            out = []
            for r in sample:
                rid = r.get("region_id")
                pt = r.get(col)
                if rid is None or not isinstance(pt, dict):
                    continue
                lat, lon = pt.get("latitude"), pt.get("longitude")
                try:
                    out.append({"region_id": int(rid), "lat": float(lat), "lon": float(lon)})
                except Exception:
                    pass
            df = pd.DataFrame(out).dropna().drop_duplicates(subset=["region_id"])
            if not df.empty:
                return df
    except Exception:
        pass

    # (3) explicit latitude/longitude fields
    try:
        rows = soda_get_json(session, f"/resource/{dsid}.json",
                             {"$select": "region_id, latitude, longitude",
                              "$group": "region_id, latitude, longitude",
                              "$limit": 5000})
        out = []
        for r in rows:
            if {"region_id","latitude","longitude"} <= r.keys():
                try:
                    out.append({"region_id": int(r["region_id"]),
                                "lat": float(r["latitude"]),
                                "lon": float(r["longitude"])})
                except Exception:
                    pass
        df = pd.DataFrame(out).dropna()
        if not df.empty:
            return df
    except Exception:
        pass

    return None

def fetch_region_centroids(session, external_csv: str | None = None):
    """
    Robust centroid fetch:
      - If external_csv is provided, use it (columns: region_id,lat,lon)
      - Else try REGION_DS_PRIORITY list until one yields usable lat/lon
      - Cache result
    """
    ensure_cache()
    if external_csv:
        df = pd.read_csv(external_csv)
        req = {"region_id","lat","lon"}
        if not req.issubset(df.columns):
            raise ValueError(f"--region_centroids file must contain columns: {sorted(req)}")
        return df[["region_id","lat","lon"]]

    if os.path.exists(REGION_CENTROIDS_FILE):
        try:
            return pd.read_json(REGION_CENTROIDS_FILE)
        except Exception:
            pass

    for dsid in REGION_DS_PRIORITY:
        log(f"Trying region centroids from dataset {dsid} …")
        df = try_fetch_region_centroids_from_dataset(session, dsid)
        if df is not None and not df.empty:
            df.to_json(REGION_CENTROIDS_FILE, orient="records")
            return df

    raise RuntimeError(
        "Could not derive Region centroids from known datasets. "
        "Pass --region_centroids <csv> with columns region_id,lat,lon; "
        "or run with --source segment_only."
    )

def fetch_segment_centroids(session):
    """
    Centroids for segments (for completeness; not required for segment speeds here).
    Try explicit latitude/longitude fields; else detect a point field and extract.
    Cache to SEGMENT_CENTROIDS_FILE.
    """
    ensure_cache()
    if os.path.exists(SEGMENT_CENTROIDS_FILE):
        try:
            return pd.read_json(SEGMENT_CENTROIDS_FILE)
        except Exception:
            pass

    # Fast path: explicit lat/lon fields (this dataset usually has them)
    try:
        rows = soda_get_json(session, f"/resource/{SEGMENT_DS_SPEEDS}.json",
                             {"$select": "segment_id, latitude, longitude",
                              "$group": "segment_id, latitude, longitude",
                              "$limit": 100000})
        out = []
        for r in rows:
            if "segment_id" in r and "latitude" in r and "longitude" in r:
                try:
                    out.append({
                        "segment_id": int(r["segment_id"]),
                        "lat": float(r["latitude"]),
                        "lon": float(r["longitude"])
                    })
                except Exception:
                    pass
        df = pd.DataFrame(out).dropna()
        if not df.empty:
            df.to_json(SEGMENT_CENTROIDS_FILE, orient="records")
            return df
    except Exception:
        pass

    # Fallback: detect a point/location field via metadata or sampling (similar to regions)
    try:
        meta = socrata_fetch_meta(session, SEGMENT_DS_SPEEDS)
        point_field = socrata_detect_point_field_from_meta(meta)
    except Exception:
        point_field = None

    if point_field:
        params = {
            "$select": f"segment_id, {point_field} as pt, {point_field}.latitude as lat, {point_field}.longitude as lon",
            "$group": "segment_id, pt, lat, lon",
            "$limit": 100000
        }
        try:
            rows = soda_get_json(session, f"/resource/{SEGMENT_DS_SPEEDS}.json", params)
            out = []
            for r in rows:
                sid = r.get("segment_id")
                lat = r.get("lat")
                lon = r.get("lon")
                if (lat is None or lon is None) and isinstance(r.get("pt"), dict):
                    lat = r["pt"].get("latitude", lat)
                    lon = r["pt"].get("longitude", lon)
                try:
                    out.append({"segment_id": int(sid), "lat": float(lat), "lon": float(lon)})
                except Exception:
                    pass
            df = pd.DataFrame(out).dropna()
            if not df.empty:
                df.to_json(SEGMENT_CENTROIDS_FILE, orient="records")
                return df
        except Exception:
            pass

    # Last attempt: sample rows; look for dict columns with lat/lon
    try:
        sample = soda_get_json(session, f"/resource/{SEGMENT_DS_SPEEDS}.json", {"$limit": 200})
        dict_cols = set()
        for row in sample:
            for k, v in row.items():
                if isinstance(v, dict) and "latitude" in v and "longitude" in v:
                    dict_cols.add(k)
        for col in dict_cols:
            out = []
            for r in sample:
                sid = r.get("segment_id")
                pt = r.get(col)
                if sid is None or not isinstance(pt, dict):
                    continue
                lat, lon = pt.get("latitude"), pt.get("longitude")
                try:
                    out.append({"segment_id": int(sid), "lat": float(lat), "lon": float(lon)})
                except Exception:
                    pass
            df = pd.DataFrame(out).dropna().drop_duplicates(subset=["segment_id"])
            if not df.empty:
                df.to_json(SEGMENT_CENTROIDS_FILE, orient="records")
                return df
    except Exception:
        pass

    raise RuntimeError("Could not derive Segment centroids.")

def nearest_id(lat, lon, df, id_col):
    d2 = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return int(df.loc[d2.idxmin(), id_col])

def daterange_days(start_dt, end_dt):
    d = start_dt
    while d < end_dt:
        yield d
        d += timedelta(days=1)

def day_bounds(dt):
    start = datetime(dt.year, dt.month, dt.day)
    end = start + timedelta(days=1)
    return start, end

def fetch_speeds_bydays(session, dataset, entity_id_field, entity_id, start_dt, end_dt):
    """
    Pull 10-min records for one region/segment in [start_dt, end_dt) by DAY with paging.
    Returns DataFrame with columns: time(UTC tz-aware), speed(float), entity_id (int)
    """
    time_fields = ["time", "TIME", "measurement_tstamp", "measurement_timestamp"]
    speed_fields = ["speed", "SPEED"]
    all_rows = []

    for day in daterange_days(start_dt, end_dt):
        d0, d1 = day_bounds(day)
        log(f"  {dataset}:{entity_id} {d0.date()} ..")
        fetched_any = False
        for tf in time_fields:
            for sf in speed_fields:
                offset = 0
                while True:
                    params = {
                        "$select": f"{tf} as t, {sf} as s",
                        "$where": f"{entity_id_field}={entity_id} AND {tf} between '{d0:%Y-%m-%dT%H:%M:%S}' and '{d1:%Y-%m-%dT%H:%M:%S}'",
                        "$order": "t",
                        "$limit": SODA_PAGE_LIMIT,
                        "$offset": offset
                    }
                    try:
                        rows = soda_get_json(session, f"/resource/{dataset}.json", params)
                    except Exception as e:
                        log(f"    ! error day page: {e}")
                        break
                    if not rows:
                        break
                    fetched_any = True
                    all_rows.extend(rows)
                    offset += SODA_PAGE_LIMIT
                if fetched_any:
                    break
            if fetched_any:
                break
        if not fetched_any:
            log("    (no rows)")
    if not all_rows:
        return pd.DataFrame(columns=["time","speed",entity_id_field])
    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["t"], errors="coerce", utc=True)
    df["speed"] = pd.to_numeric(df["s"], errors="coerce")
    df = df.dropna(subset=["time", "speed"])
    df[entity_id_field] = entity_id
    return df[["time","speed",entity_id_field]]

def hourly_agg_chi(df_utc):
    import pytz
    if df_utc.empty:
        return pd.DataFrame(columns=["hour_dt","spd_mean","v25_share"])
    central = pytz.timezone("America/Chicago")
    tmp = df_utc.copy()
    tmp["time_local"] = tmp["time"].dt.tz_convert(central)
    tmp["hour_dt"] = tmp["time_local"].dt.floor("H")
    g = tmp.groupby("hour_dt", as_index=False)
    out = g.agg(
        spd_mean=("speed","mean"),
        v25_share=("speed", lambda x: (x <= V25_THRESHOLD).mean())
    )
    return out

# ------------------ PIPELINE ------------------
def populate(input_csv, output_csv=None, soda_token=None, radius_m=DEFAULT_RADIUS_M,
             source="auto", no_osm=False, region_centroids_csv=None):

    ensure_cache()
    token = soda_token or os.getenv("SODA_APP_TOKEN") or EMBEDDED_SODA_APP_TOKEN
    sess = session_with_retries(token)

    df = pd.read_csv(input_csv)
    req = {"node_id","lat","lon","epoch"}
    if not req.issubset(df.columns):
        raise ValueError(f"Input must include columns: {sorted(req)}")

    # normalize epoch -> local hour (tz-aware)
    df["hour_local"] = df["epoch"].apply(epoch_to_local_hour)
    hours_local = pd.to_datetime(df["hour_local"])

    # Unique nodes
    nodes = df[["node_id","lat","lon"]].drop_duplicates().reset_index(drop=True)

    # ---- OSM lane density with cache ----
    lane_cache = load_lane_cache()
    lane_map = {}
    if no_osm:
        log("Skipping OSM lane density (--no_osm).")
    else:
        for _, r in nodes.iterrows():
            nid, la, lo = r["node_id"], float(r["lat"]), float(r["lon"])
            key = f"{round(la,6)},{round(lo,6)},{radius_m}"
            if key not in lane_cache:
                try:
                    log(f"OSM lane density for node {nid} at ({la:.6f},{lo:.6f}) r={radius_m}m")
                    lane_cache[key] = overpass_lane_km_density(la, lo, radius_m)
                except Exception as e:
                    log(f"  OSM error: {e} -> setting None")
                    lane_cache[key] = None
                save_lane_cache(lane_cache)
                time.sleep(OVERPASS_SLEEP)
            lane_map[nid] = lane_cache[key]

    # ---- Choose source(s) ----
    if source not in ("auto","region","segment","segment_only"):
        raise ValueError("--source must be auto|region|segment|segment_only")

    region_df = None
    if source in ("auto","region"):
        log("Fetching region centroids…")
        region_df = fetch_region_centroids(sess, external_csv=region_centroids_csv)

    node_to_region = {}
    if region_df is not None:
        for _, r in nodes.iterrows():
            la, lo = float(r["lat"]), float(r["lon"])
            nid = r["node_id"]
            node_to_region[nid] = nearest_id(la, lo, region_df, "region_id")

    # ---- Build pulls ----
    min_hour = hours_local.min().to_pydatetime()
    max_hour = (hours_local.max() + pd.Timedelta(hours=1)).to_pydatetime()
    pulls = []

    if source == "region":
        for nid in df["node_id"].unique():
            rid = node_to_region[nid]
            pulls.append((REGION_DS_SPEEDS, "region_id", rid, min_hour, max_hour))
    elif source == "segment":
        # For completeness: if you want strictly segments for whole range (will be sparse pre-2024-06-11)
        # we map each node to nearest segment centroid to define a segment_id anchor, then pull that segment
        seg_df = fetch_segment_centroids(sess)
        node_to_segment = {}
        for _, r in nodes.iterrows():
            nid, la, lo = r["node_id"], float(r["lat"]), float(r["lon"])
            node_to_segment[nid] = nearest_id(la, lo, seg_df, "segment_id")
        for nid in df["node_id"].unique():
            sid = node_to_segment[nid]
            pulls.append((SEGMENT_DS_SPEEDS, "segment_id", sid, min_hour, max_hour))
    elif source == "segment_only":
        # Only fill >= 2024-06-11; earlier hours will remain NA for speed fields
        seg_df = fetch_segment_centroids(sess)
        node_to_segment = {}
        for _, r in nodes.iterrows():
            nid, la, lo = r["node_id"], float(r["lat"]), float(r["lon"])
            node_to_segment[nid] = nearest_id(la, lo, seg_df, "segment_id")
        s = max(min_hour, SEGMENT_START)
        if s < max_hour:
            for nid in df["node_id"].unique():
                sid = node_to_segment[nid]
                pulls.append((SEGMENT_DS_SPEEDS, "segment_id", sid, s, max_hour))
    else:  # auto
        t_split = SEGMENT_START
        if min_hour < t_split:
            for nid in df["node_id"].unique():
                rid = node_to_region[nid]
                pulls.append((REGION_DS_SPEEDS, "region_id", rid, min_hour, min(max_hour, t_split)))
        if max_hour > t_split:
            seg_df = fetch_segment_centroids(sess)
            node_to_segment = {}
            for _, r in nodes.iterrows():
                nid, la, lo = r["node_id"], float(r["lat"]), float(r["lon"])
                node_to_segment[nid] = nearest_id(la, lo, seg_df, "segment_id")
            for nid in df["node_id"].unique():
                sid = node_to_segment[nid]
                pulls.append((SEGMENT_DS_SPEEDS, "segment_id", sid, max(min_hour, t_split), max_hour))

    # de-duplicate pulls
    pulls_unique = {}
    for ds, fld, eid, s, e in pulls:
        key = (ds, fld, eid, s.date(), e.date())
        pulls_unique[key] = (ds, fld, eid, s, e)
    pulls = list(pulls_unique.values())

    # ---- Fetch & aggregate hourly ----
    lookups = {}  # (dataset, eid, hour_dt) -> (spd_mean, v25_share)
    for (ds, fld, eid, s, e) in pulls:
        log(f"Pull {ds} {fld}={eid} {s.date()}..{(e - timedelta(days=1)).date()}")
        raw = fetch_speeds_bydays(sess, ds, fld, eid, s, e)
        hourly = hourly_agg_chi(raw)
        for _, row in hourly.iterrows():
            lookups[(ds, eid, row["hour_dt"])] = (row["spd_mean"], row["v25_share"])

    # ---- Attach + PRINT each row ----
    spd_mean_list, v25_share_list, lane_density_list = [], [], []
    log("\nPopulating final values for each row:\n")
    for _, r in df.iterrows():
        nid = r["node_id"]
        la, lo = float(r["lat"]), float(r["lon"])
        epoch = int(r["epoch"])
        hour = pd.to_datetime(r["hour_local"])

        # choose the dataset anchor per hour
        if source == "region":
            ds = REGION_DS_SPEEDS
            key_id = node_to_region[nid]
        elif source == "segment":
            ds = SEGMENT_DS_SPEEDS
            key_id = node_to_segment[nid]
        elif source == "segment_only":
            if hour >= SEGMENT_START.replace(tzinfo=hour.tzinfo):
                ds = SEGMENT_DS_SPEEDS
                key_id = node_to_segment[nid]
            else:
                ds = None
                key_id = None
        else:  # auto
            if hour >= SEGMENT_START.replace(tzinfo=hour.tzinfo):
                ds = SEGMENT_DS_SPEEDS
                key_id = node_to_segment[nid]
            else:
                ds = REGION_DS_SPEEDS
                key_id = node_to_region[nid]

        if ds is None:
            spd, v25 = None, None
        else:
            spd, v25 = lookups.get((ds, key_id, hour), (None, None))

        lane = lane_map.get(nid)

        spd_mean_list.append(spd)
        v25_share_list.append(v25)
        lane_density_list.append(lane)

        print(
            f"epoch={epoch} node={nid} lat={la:.5f} lon={lo:.5f} "
            f"spd_mean={spd if spd is not None else 'NA'} "
            f"lane_km_density={lane if lane is not None else 'NA'} "
            f"v25_share={v25 if v25 is not None else 'NA'}",
            flush=True
        )

    # ---- Write output ----
    out = df.copy()
    out["spd_mean"] = spd_mean_list
    out["lane_km_density"] = lane_density_list
    out["v25_share"] = v25_share_list

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}__with_traffic.csv"
    out.to_csv(output_csv, index=False)
    log(f"\n✅ Saved: {output_csv}\n")

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(
        description="Chicago traffic metrics (spd_mean, lane_km_density, v25_share) with robust region centroid discovery and per-row prints"
    )
    ap.add_argument("input_csv", help="CSV with columns node_id,lat,lon,epoch")
    ap.add_argument("--out", help="Output CSV path (default: <input>__with_traffic.csv)")
    ap.add_argument("--soda_token", help="Socrata app token (overrides embedded token/env)")
    ap.add_argument("--radius_m", type=int, default=DEFAULT_RADIUS_M, help="OSM buffer radius in meters (default 500)")
    ap.add_argument("--source", choices=["auto","region","segment","segment_only"], default="auto",
                    help="auto: regions before 2024-06-11 and segments after; segment_only fills only >= 2024-06-11")
    ap.add_argument("--no_osm", action="store_true", help="Skip OSM lane density")
    ap.add_argument("--region_centroids", help="Path to CSV with columns region_id,lat,lon to override API centroid discovery")
    args = ap.parse_args()

    populate(args.input_csv, args.out, args.soda_token, args.radius_m, args.source, args.no_osm, args.region_centroids)

if __name__ == "__main__":
    main()
