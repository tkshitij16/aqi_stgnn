#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Weekly Chicago AQI maps + "drivers" panel.

Expected predictions schema (default):
  epoch,node_id,aqi_hat,pm25_hat

What it does
------------
- Uses --metric (default: aqi_hat). If you pass 'aqi_pred', it auto-maps to 'aqi_hat'.
- If no lon/lat, joins from --nodes-csv (or master) by node_id.
- Builds uniform hex grid over Chicago boundary and aggregates weekly means.
- For each week, saves:
    figs/weekly_maps/weekly_map_YYYY-Www.png  (map + correlation bars)
    tables/weekly_maps/hex_YYYY-Www.csv       (hex centers + weekly mean AQI)

Run (PowerShell):
  python -m src.weekly_maps `
    --pred-csv outputs/preds_2024.csv `
    --nodes-csv data/stt_master_locked_2024.csv `
    --master-csv data/stt_master_locked_2024.csv `
    --boundary data/chicago_boundary.geojson `
    --hex-edge-m 600 `
    --metric aqi_hat `
    --outdir figs/weekly_maps `
    --tables-dir tables/weekly_maps
"""

import argparse, os, logging
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

def get_logger(name="weekly_maps"):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        log.addHandler(ch)
    return log

log = get_logger()

# ------------- helpers -------------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def to_metric(gdf, epsg="EPSG:26916"):  # UTM 16N
    return gdf.to_crs(epsg)

def load_boundary(path):
    g = gpd.read_file(path)
    if g.crs is None: g = g.set_crs("EPSG:4326", allow_override=True)
    g = g.to_crs("EPSG:4326")
    return gpd.GeoDataFrame(geometry=[unary_union(g.geometry)], crs="EPSG:4326")

def week_key_local(epoch_series, tz="America/Chicago"):
    t = pd.to_datetime(epoch_series.astype("int64"), unit="s", utc=True).dt.tz_convert(tz)
    iso = t.dt.isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2))

def hexagon(cx, cy, edge):
    ang = np.deg2rad([0,60,120,180,240,300])
    xs = cx + edge*np.cos(ang); ys = cy + edge*np.sin(ang)
    return Polygon(zip(xs, ys))

def hex_grid_shapely(boundary_gdf, hex_edge_m=600):
    bnd_m = to_metric(boundary_gdf)
    minx, miny, maxx, maxy = bnd_m.total_bounds
    dx = (3**0.5)*hex_edge_m; dy = 1.5*hex_edge_m
    cols = int(np.ceil((maxx-minx)/dx))+2; rows = int(np.ceil((maxy-miny)/dy))+2
    hexes=[]
    for r in range(rows):
        for c in range(cols):
            cx = minx + c*dx + (0.5*dx if r%2 else 0)
            cy = miny + r*dy
            hexes.append(hexagon(cx, cy, hex_edge_m))
    grid_m = gpd.GeoDataFrame(geometry=hexes, crs="EPSG:26916")
    grid_m = gpd.overlay(grid_m, bnd_m, how="intersection", keep_geom_type=True)
    grid_m = grid_m[~grid_m.geometry.is_empty].copy()
    return grid_m.to_crs("EPSG:4326")

def read_nodes(nodes_csv):
    cols = ["node_id","lon","lat"]
    df = pd.read_csv(nodes_csv, usecols=cols, low_memory=False)
    df["node_id"] = df["node_id"].astype(str)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon","lat"]).drop_duplicates(subset=["node_id"])
    return df

def attach_lonlat(df, nodes_csv_or_master):
    if ("lon" in df.columns) and ("lat" in df.columns):
        return df
    if "node_id" not in df.columns:
        raise ValueError("Need lon/lat or node_id to map coordinates.")
    nodes = read_nodes(nodes_csv_or_master)
    out = df.copy()
    out["node_id"] = out["node_id"].astype(str)
    out = out.merge(nodes, on="node_id", how="left")
    out = out.dropna(subset=["lon","lat"]).copy()
    return out

def maybe_alias_metric(df, name):
    aliases = {
        "aqi_pred": "aqi_hat",
        "aqi": "aqi_hat",
        "pm25_pred": "pm25_hat",
        "pm25": "pm25_hat",
    }
    if name in df.columns:
        return name
    if name in aliases and aliases[name] in df.columns:
        log.warning(f"[Metric] '{name}' not found; using '{aliases[name]}'")
        return aliases[name]
    return name

def agg_points_to_hex(points_gdf, hex_gdf, val_col):
    pts = points_gdf[["geometry", val_col]].copy()
    pts_m = to_metric(pts); hex_m = to_metric(hex_gdf)
    join = gpd.sjoin(pts_m, hex_m, predicate="within", how="left")
    grp = join.groupby(join.index_right)[val_col].mean()
    hex_m[val_col] = grp.reindex(range(len(hex_m))).astype(float).values
    return hex_m.to_crs("EPSG:4326")

def corr_bar(ax, aqi, features_dict):
    labels=[]; vals=[]
    for k, v in features_dict.items():
        a = np.asarray(aqi, dtype=float)
        b = np.asarray(v, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 5:
            r = np.nan
        else:
            r = np.corrcoef(a[mask], b[mask])[0,1]
        labels.append(k); vals.append(r)
    ax.barh(np.arange(len(labels)), vals)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Correlation (r)")
    ax.set_title("Drivers of spatial variation (weekly)")
    ax.set_xlim(-1, 1)

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--nodes-csv", default="data/stt_master_locked_2024.csv",
                    help="For lon/lat if predictions lack them")
    ap.add_argument("--master-csv", default="data/stt_master_locked_2024.csv",
                    help="Locked master with features (U, PBLH, congestion, ndvi_z, p_impervious)")
    ap.add_argument("--boundary", required=True, help="Chicago boundary GeoJSON/Shapefile")
    ap.add_argument("--hex-edge-m", type=float, default=600.0)
    ap.add_argument("--metric", default="aqi_hat", help="aqi_hat (default) or pm25_hat or aqi_obs")
    ap.add_argument("--start-week", default=None)
    ap.add_argument("--end-week", default=None)
    ap.add_argument("--outdir", default="figs/weekly_maps")
    ap.add_argument("--tables-dir", default="tables/weekly_maps")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.tables_dir, exist_ok=True)

    # Load predictions
    df = pd.read_csv(args.pred_csv, low_memory=False)
    if "epoch" not in df.columns: raise ValueError("Missing 'epoch' in predictions.")
    if "node_id" not in df.columns: raise ValueError("Missing 'node_id' in predictions.")
    df["node_id"] = df["node_id"].astype(str)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")

    # Metric aliasing
    args.metric = maybe_alias_metric(df, args.metric)

    # If user asked for 'aqi_obs' but it's not in predictions, we will source from master later
    need_obs_from_master = (args.metric == "aqi_obs" and "aqi_obs" not in df.columns)

    # Attach lon/lat if missing
    src_nodes = args.nodes_csv if args.nodes_csv else args.master_csv
    if ("lon" not in df.columns) or ("lat" not in df.columns):
        df = attach_lonlat(df, src_nodes)

    # Build week label
    df["week"] = week_key_local(df["epoch"])

    # Load boundary and hex grid
    bnd = load_boundary(args.boundary)
    hex_grid = hex_grid_shapely(bnd, hex_edge_m=args.hex_edge_m)

    # Load features from master for correlation panel
    feat_cols = ["epoch","node_id","U","PBLH","congestion","ndvi_z","p_impervious","lon","lat"]
    dfm = pd.read_csv(args.master_csv, usecols=[c for c in feat_cols if c in pd.read_csv(args.master_csv, nrows=0).columns],
                      low_memory=False)
    for c in ["epoch","node_id"]: 
        if c not in dfm.columns: raise ValueError(f"Column '{c}' missing in master CSV.")
    dfm["node_id"] = dfm["node_id"].astype(str)
    dfm["epoch"] = pd.to_numeric(dfm["epoch"], errors="coerce").astype("Int64")
    if "U" not in dfm.columns or "PBLH" not in dfm.columns:
        log.warning("[Master] Missing U or PBLH; ventilation will be NaN.")
        dfm["U"] = pd.to_numeric(dfm.get("U", np.nan))
        dfm["PBLH"] = pd.to_numeric(dfm.get("PBLH", np.nan))
    dfm["VC"] = pd.to_numeric(dfm.get("U"), errors="coerce") * pd.to_numeric(dfm.get("PBLH"), errors="coerce")
    # week for master
    dfm["week"] = week_key_local(dfm["epoch"])

    # If metric is 'aqi_obs' and not in predictions, bring it from master
    if need_obs_from_master:
        obs = dfm[["epoch","node_id","week","aqi_obs"]].dropna(subset=["aqi_obs"]) if "aqi_obs" in dfm.columns else None
        if obs is None:
            raise ValueError("Requested metric 'aqi_obs' but master lacks 'aqi_obs'.")
        df = df.merge(obs[["epoch","node_id","aqi_obs"]], on=["epoch","node_id"], how="left")
        if df["aqi_obs"].isna().all():
            raise ValueError("Could not attach observed AQI from master.")
        # leave args.metric as 'aqi_obs'

    # Weekly means per node for metric
    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not present after aliasing/join.")
    pred_week = df[["week","node_id","lon","lat", args.metric]].groupby(
        ["week","node_id","lon","lat"], as_index=False).mean()

    # Weekly means per node for features
    feat_week = dfm[["week","node_id","VC","congestion","ndvi_z","p_impervious"]].groupby(
        ["week","node_id"], as_index=False).mean()

    merged = pred_week.merge(feat_week, on=["week","node_id"], how="left")

    # Iterate weeks
    weeks = sorted(merged["week"].unique())
    if args.start_week: weeks = [w for w in weeks if w >= args.start_week]
    if args.end_week:   weeks = [w for w in weeks if w <= args.end_week]

    for wk in weeks:
        g = merged[merged["week"]==wk].dropna(subset=["lon","lat", args.metric]).copy()
        if g.empty:
            log.warning(f"[Week {wk}] No data; skip.")
            continue

        pts = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326")
        hex_aqi = agg_points_to_hex(pts, hex_grid, args.metric)

        # Aggregate features to hex
        feat_hex = {}
        for col in ["VC","congestion","ndvi_z","p_impervious"]:
            if col in g.columns:
                gg = g[["lon","lat", col]].dropna()
                if gg.empty:
                    feat_hex[col] = np.full(len(hex_aqi), np.nan)
                else:
                    tmp = gpd.GeoDataFrame(gg, geometry=gpd.points_from_xy(gg["lon"], gg["lat"]), crs="EPSG:4326")
                    tmp_hex = agg_points_to_hex(tmp, hex_grid, col)
                    feat_hex[col] = tmp_hex[col].values
            else:
                feat_hex[col] = np.full(len(hex_aqi), np.nan)

        # Save table with hex centroids + weekly values
        table = hex_aqi.copy()
        cent_m = to_metric(table).geometry.centroid
        cent_ll = gpd.GeoSeries(cent_m, crs="EPSG:26916").to_crs("EPSG:4326")
        table["lon"] = cent_ll.x.values; table["lat"] = cent_ll.y.values
        table_path = os.path.join(args.tables_dir, f"hex_{wk}.csv")
        table[[args.metric,"lon","lat"]].to_csv(table_path, index=False)

        # Figure: map + correlations
        fig_path = os.path.join(args.outdir, f"weekly_map_{wk}.png")
        import matplotlib as mpl
        v = hex_aqi[args.metric].values
        vmin = np.nanpercentile(v, 2); vmax = np.nanpercentile(v, 98)
        cmap = mpl.cm.viridis

        fig = plt.figure(figsize=(10,5))
        ax_map = fig.add_subplot(1,2,1)
        bnd.boundary.plot(ax=ax_map, linewidth=1.0, color="black")
        hex_aqi.plot(ax=ax_map, column=args.metric, vmin=vmin, vmax=vmax, cmap=cmap, legend=True)
        ax_map.set_title(f"AQI (weekly mean) — {wk}")
        ax_map.set_xlabel("Longitude"); ax_map.set_ylabel("Latitude")

        ax_bar = fig.add_subplot(1,2,2)
        corr_bar(ax_bar, hex_aqi[args.metric].values, {
            "Ventilation (U×PBLH)": feat_hex["VC"],
            "Traffic proxy (congestion)": feat_hex["congestion"],
            "Vegetation (NDVI z)": feat_hex["ndvi_z"],
            "Impervious surface": feat_hex["p_impervious"]
        })
        fig.tight_layout(); fig.savefig(fig_path, dpi=300); plt.close(fig)
        log.info(f"[Week {wk}] Saved → {fig_path} | {table_path}")

if __name__ == "__main__":
    main()
