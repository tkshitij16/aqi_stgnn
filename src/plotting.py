#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Daily AQI heatmaps on a Chicago map from predicted CSV.

Now supports predictions CSVs that DO NOT contain lon/lat:
- Provide --nodes-csv (defaults to data/stt_master_locked_2024.csv) with node_id, lon, lat
- The script merges on node_id to attach coordinates.

Inputs
------
Predicted CSV with at least:
- epoch (UTC seconds)
- node_id (required if lon/lat absent)
- aqi_pred (default metric) or aqi_obs (fallback)
Optional:
- lon, lat (if present, we don't need nodes-csv)

Outputs
-------
- One PNG per local (America/Chicago) day under --outdir (default: figs/daily_maps)
- Optional per-day hex tables under tables/daily_hex (mean AQI per hex)
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Optional deps
try:
    import h3
    _H3_OK = True
except Exception:
    _H3_OK = False

try:
    import contextily as cx
    _CX_OK = True
except Exception:
    _CX_OK = False

# Reuse utils if available; fallback otherwise
try:
    from src.utils import get_logger, ensure_dir
except Exception:
    def get_logger(name="plotting"):
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
    def ensure_dir(path):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

log = get_logger("plotting")

# ---------------- CRS helpers ----------------
def to_metric(gdf, epsg="EPSG:26916"):  # UTM 16N (meters) for Chicago
    return gdf.to_crs(epsg)

def load_boundary(path):
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326", allow_override=True)
    g = g.to_crs("EPSG:4326")
    geom = unary_union(g.geometry)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

def auto_bbox(points_gdf, buffer_deg=0.05):
    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    xmin -= buffer_deg; ymin -= buffer_deg
    xmax += buffer_deg; ymax += buffer_deg
    poly = Polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

# ---------------- Hex grid builders ----------------
def hexagon(cx, cy, edge):
    ang = np.deg2rad([0,60,120,180,240,300])
    xs = cx + edge*np.cos(ang)
    ys = cy + edge*np.sin(ang)
    return Polygon(zip(xs, ys))

def hex_grid_shapely(boundary_gdf, hex_edge_m=600):
    bnd_m = to_metric(boundary_gdf)
    minx, miny, maxx, maxy = bnd_m.total_bounds

    dx = (3**0.5) * hex_edge_m
    dy = 1.5 * hex_edge_m
    cols = int(np.ceil((maxx-minx)/dx)) + 2
    rows = int(np.ceil((maxy-miny)/dy)) + 2

    hexes = []
    for r in range(rows):
        for c in range(cols):
            cx = minx + c*dx + (0.5*dx if r % 2 else 0)
            cy = miny + r*dy
            hexes.append(hexagon(cx, cy, hex_edge_m))
    grid_m = gpd.GeoDataFrame(geometry=hexes, crs="EPSG:26916")
    grid_m = gpd.overlay(grid_m, bnd_m, how="intersection", keep_geom_type=True)
    grid_m = grid_m[~grid_m.geometry.is_empty].copy()
    return grid_m.to_crs("EPSG:4326")

def _poly_to_geojson_lonlat(poly):
    ext = [(x, y) for (x, y) in poly.exterior.coords]
    holes = [[(x, y) for (x, y) in ring.coords] for ring in poly.interiors]
    return {"type": "Polygon", "coordinates": [ext] + holes}

def hex_grid_h3(boundary_gdf, res=8):
    if not _H3_OK:
        raise RuntimeError("h3 not installed (pip install h3)")
    geom = boundary_gdf.geometry.iloc[0]
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    idxs = set()
    for p in polys:
        gj = _poly_to_geojson_lonlat(p)
        try:
            idxs |= set(h3.polyfill(gj, res, geo_json_conformant=True))
        except TypeError:
            idxs |= set(h3.polyfill(gj, res))
    cells = []
    for h in sorted(idxs):
        b = h3.h3_to_geo_boundary(h, geo_json=True)  # [[lat,lon],...]
        cells.append(Polygon([(lon, lat) for (lat, lon) in b]))
    return gpd.GeoDataFrame({"h3_index": list(sorted(idxs))}, geometry=cells, crs="EPSG:4326")

# ---------------- Data IO ----------------
def read_nodes(nodes_csv, nodes_node_col="node_id", nodes_lon_col="lon", nodes_lat_col="lat"):
    """Read unique node_id → (lon,lat) mapping from a large CSV quickly."""
    usecols = [nodes_node_col, nodes_lon_col, nodes_lat_col]
    df = pd.read_csv(nodes_csv, usecols=usecols, low_memory=False)
    df = df.rename(columns={
        nodes_node_col: "node_id",
        nodes_lon_col: "lon",
        nodes_lat_col: "lat",
    })
    # Coerce types
    df["node_id"] = df["node_id"].astype(str)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    # Drop NA and duplicates
    before = len(df)
    df = df.dropna(subset=["lon","lat"])
    df = df.drop_duplicates(subset=["node_id"])
    log.info(f"[Nodes] Loaded mapping: {len(df):,} unique nodes (from {before:,} rows)")
    return df[["node_id","lon","lat"]]

def attach_lonlat(pred_df, nodes_csv, node_col_pred="node_id",
                  nodes_node_col="node_id", nodes_lon_col="lon", nodes_lat_col="lat"):
    """Ensure pred_df has lon/lat. If missing, merge using nodes_csv mapping."""
    has_lonlat = ("lon" in pred_df.columns) and ("lat" in pred_df.columns)
    if has_lonlat:
        # Coerce types, drop bad rows
        pred_df["lon"] = pd.to_numeric(pred_df["lon"], errors="coerce")
        pred_df["lat"] = pd.to_numeric(pred_df["lat"], errors="coerce")
        bad = pred_df["lon"].isna() | pred_df["lat"].isna()
        if bad.any():
            log.warning(f"[Pred] Dropping {bad.sum()} rows with invalid lon/lat in predictions.")
            pred_df = pred_df[~bad].copy()
        return pred_df

    # Need nodes mapping
    if nodes_csv is None:
        raise ValueError("Predictions lack lon/lat and --nodes-csv was not provided.")
    if node_col_pred not in pred_df.columns:
        raise ValueError(f"Predictions lack '{node_col_pred}' for joining to nodes.")

    nodes = read_nodes(nodes_csv, nodes_node_col, nodes_lon_col, nodes_lat_col)
    df = pred_df.copy()
    df[node_col_pred] = df[node_col_pred].astype(str)
    nodes["node_id"] = nodes["node_id"].astype(str)

    before = len(df)
    df = df.merge(nodes, left_on=node_col_pred, right_on="node_id", how="left")
    missing = df["lon"].isna() | df["lat"].isna()
    if missing.any():
        miss_ids = df.loc[missing, node_col_pred].astype(str).nunique()
        log.warning(f"[Join] {missing.sum()} rows ({miss_ids} unique node_id) missing lon/lat after join — dropping.")
        df = df[~missing].copy()

    after = len(df)
    log.info(f"[Join] Predictions mapped to coordinates: {after:,}/{before:,} rows")
    return df

# ---------------- Core plotting helpers ----------------
def publish_style():
    plt.rcParams.update({
        "figure.dpi": 220,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
    })

def read_predictions(path, metric="aqi_pred",
                     nodes_csv=None, node_col_pred="node_id",
                     nodes_node_col="node_id", nodes_lon_col="lon", nodes_lat_col="lat"):
    df = pd.read_csv(path, low_memory=False)
    if "epoch" not in df.columns:
        raise ValueError(f"Missing required column 'epoch' in {path}")
    # Pick metric
    if metric not in df.columns:
        fallback = "aqi_obs"
        if fallback not in df.columns:
            raise ValueError(f"Metric '{metric}' not found and fallback '{fallback}' also missing.")
        log.warning(f"[Data] '{metric}' not in CSV; using '{fallback}'")
        metric = fallback

    # Keep only needed cols (node col may or may not exist)
    need = ["epoch", metric]
    if "lon" in df.columns: need.append("lon")
    if "lat" in df.columns: need.append("lat")
    if node_col_pred in df.columns: need.append(node_col_pred)
    df = df[need].copy().rename(columns={metric:"metric"})

    # Attach lon/lat if missing
    df = attach_lonlat(df, nodes_csv, node_col_pred, nodes_node_col, nodes_lon_col, nodes_lat_col)

    # Final coercions
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df["metric"] = pd.to_numeric(df["metric"], errors="coerce")
    df = df.dropna(subset=["epoch","lon","lat","metric"]).copy()
    return df

def add_local_day(df, tz="America/Chicago"):
    t = pd.to_datetime(df["epoch"].astype("int64"), unit="s", utc=True).dt.tz_convert(tz)
    df["local_day"] = t.dt.strftime("%Y-%m-%d")
    return df

def per_day_groups(df, start=None, end=None):
    days = sorted(df["local_day"].unique())
    if start: days = [d for d in days if d >= start]
    if end:   days = [d for d in days if d <= end]
    for d in days:
        yield d, df[df["local_day"] == d].copy()

def build_hex(boundary_gdf, backend="shapely", hex_edge_m=600, h3_res=8):
    if backend == "h3":
        log.info(f"[Hex] H3 uniform cover res={h3_res}")
        return hex_grid_h3(boundary_gdf, res=h3_res)
    else:
        log.info(f"[Hex] Shapely uniform cover edge={hex_edge_m} m")
        return hex_grid_shapely(boundary_gdf, hex_edge_m=hex_edge_m)

def agg_points_to_hex(points_gdf, hex_gdf):
    # spatial join -> groupby mean of 'metric'
    pts = points_gdf[["metric","geometry"]].copy()
    pts_m = to_metric(pts)
    hex_m = to_metric(hex_gdf)
    join = gpd.sjoin(pts_m, hex_m, predicate="within", how="left")
    grp = join.groupby(join.index_right)["metric"].mean()
    hex_m["aqi_mean"] = grp.reindex(range(len(hex_m))).astype(float).values
    # bring to 4326
    out = hex_m.to_crs("EPSG:4326")
    return out

def plot_one_day_hex(day, hex_chor, boundary_gdf, out_png, vmin, vmax, cmap="viridis", use_basemap=False):
    publish_style()
    ensure_dir(out_png)
    fig, ax = plt.subplots(figsize=(6.8,6.8))
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black", zorder=2)
    # colorbar via ScalarMappable for consistent scale
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    vals = hex_chor["aqi_mean"].values
    colors = sm.to_rgba(vals)
    hex_chor.plot(ax=ax, color=colors, edgecolor="none", alpha=0.95, zorder=1)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01); cbar.set_label("AQI")
    ax.set_title(f"AQI heatmap — {day}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    if use_basemap and _CX_OK:
        ax.set_axis_off()
        hex_web = hex_chor.to_crs(epsg=3857)
        bnd_web = boundary_gdf.to_crs(epsg=3857) if boundary_gdf is not None else None
        fig, ax = plt.subplots(figsize=(6.8,6.8))
        if bnd_web is not None:
            bnd_web.boundary.plot(ax=ax, linewidth=1.0, color="black", zorder=2)
        hex_web.plot(ax=ax, color=colors, edgecolor="none", alpha=0.95, zorder=1)
        cx.add_basemap(ax, crs=hex_web.crs, source=cx.providers.Stamen.TonerLite)
        ax.set_title(f"AQI heatmap — {day}")
        plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01).set_label("AQI")
        plt.savefig(out_png, dpi=300)
    else:
        plt.savefig(out_png, dpi=300)
    plt.close()

def plot_one_day_points(day, points_gdf, boundary_gdf, out_png, vmin, vmax, cmap="viridis", use_basemap=False):
    publish_style()
    ensure_dir(out_png)
    fig, ax = plt.subplots(figsize=(6.8,6.8))
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    # colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(points_gdf.geometry.x, points_gdf.geometry.y,
                    c=points_gdf["metric"].values, s=18, linewidths=0.3,
                    edgecolors="black", cmap=cmap, norm=norm, alpha=0.95)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01); cbar.set_label("AQI")
    ax.set_title(f"AQI points — {day}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    if use_basemap and _CX_OK:
        ax.set_axis_off()
        g_web = points_gdf.set_crs(4326).to_crs(3857)
        bnd_web = boundary_gdf.to_crs(3857) if boundary_gdf is not None else None
        fig, ax = plt.subplots(figsize=(6.8,6.8))
        if bnd_web is not None:
            bnd_web.boundary.plot(ax=ax, linewidth=1.0, color="black")
        sc = ax.scatter(g_web.geometry.x, g_web.geometry.y,
                        c=points_gdf["metric"].values, s=18, linewidths=0.3,
                        edgecolors="black", cmap=cmap, norm=norm, alpha=0.95)
        cx.add_basemap(ax, crs=g_web.crs, source=cx.providers.Stamen.TonerLite)
        ax.set_title(f"AQI points — {day}")
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("AQI")
        plt.savefig(out_png, dpi=300)
    else:
        plt.savefig(out_png, dpi=300)
    plt.close()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Predicted CSV (epoch, node_id and/or lon/lat, aqi_pred|aqi_obs)")
    ap.add_argument("--metric", default="aqi_pred", help="Column to plot; default aqi_pred (fallback aqi_obs)")
    # Node mapping (used if lon/lat absent in predictions)
    ap.add_argument("--nodes-csv", default="data/stt_master_locked_2024.csv",
                    help="CSV with node_id, lon, lat (default uses locked master)")
    ap.add_argument("--node-col-pred", default="node_id", help="Node column in predictions")
    ap.add_argument("--nodes-node-col", default="node_id", help="Node column in nodes CSV")
    ap.add_argument("--nodes-lon-col", default="lon", help="Longitude column in nodes CSV")
    ap.add_argument("--nodes-lat-col", default="lat", help="Latitude column in nodes CSV")
    # Map options
    ap.add_argument("--boundary", default=None, help="City boundary GeoJSON/Shapefile (recommended)")
    ap.add_argument("--mode", choices=["hex","points"], default="hex", help="Heatmap type")
    ap.add_argument("--hex-backend", choices=["shapely","h3"], default="shapely", help="Hex engine for --mode hex")
    ap.add_argument("--hex-edge-m", type=float, default=600.0, help="Hex edge length (m) for shapely backend")
    ap.add_argument("--h3-res", type=int, default=8, help="H3 resolution if hex-backend=h3")
    ap.add_argument("--vmin", type=float, default=0.0, help="Colorbar min (AQI)")
    ap.add_argument("--vmax", type=float, default=200.0, help="Colorbar max (AQI)")
    ap.add_argument("--start", default=None, help="First day (YYYY-MM-DD) to plot")
    ap.add_argument("--end", default=None, help="Last day (YYYY-MM-DD) to plot")
    ap.add_argument("--outdir", default="figs/daily_maps", help="Output directory for PNGs")
    ap.add_argument("--tables-dir", default="tables/daily_hex", help="Output dir for per-day hex tables")
    ap.add_argument("--use-basemap", action="store_true", help="Overlay web tiles (requires contextily, internet)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.mode == "hex":
        os.makedirs(args.tables_dir, exist_ok=True)

    # Load and prep data
    df = read_predictions(
        args.pred_csv, metric=args.metric,
        nodes_csv=args.nodes_csv, node_col_pred=args.node_col_pred,
        nodes_node_col=args.nodes_node_col, nodes_lon_col=args.nodes_lon_col, nodes_lat_col=args.nodes_lat_col
    )
    df = add_local_day(df, tz="America/Chicago")
    log.info(f"[Data] Loaded {len(df):,} rows; days={df['local_day'].nunique()} from {df['local_day'].min()} to {df['local_day'].max()}")

    # Color scale
    vmin, vmax = args.vmin, args.vmax
    if vmin >= vmax:
        vmin = df["metric"].quantile(0.02)
        vmax = df["metric"].quantile(0.98)
        log.info(f"[Scale] Auto vmin/vmax = {vmin:.1f}/{vmax:.1f}")

    # Points GDF (all rows)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")

    # Boundary
    if args.boundary:
        bnd = load_boundary(args.boundary)
    else:
        bnd = auto_bbox(pts, buffer_deg=0.05)
        log.warning("[Boundary] No boundary provided; using auto bbox.")

    # Hex grid (built once)
    if args.mode == "hex":
        try:
            hex_grid = build_hex(bnd, backend=args.hex_backend, hex_edge_m=args.hex_edge_m, h3_res=args.h3_res)
        except Exception as e:
            log.warning(f"[Hex] Backend failed ({e}); falling back to shapely.")
            hex_grid = build_hex(bnd, backend="shapely", hex_edge_m=args.hex_edge_m)

    # Iterate days
    n_png = 0
    for day in sorted(df["local_day"].unique()):
        df_day = df[df["local_day"] == day].copy()
        if df_day.empty:
            continue

        if args.mode == "hex":
            g_day = gpd.GeoDataFrame(df_day, geometry=gpd.points_from_xy(df_day["lon"], df_day["lat"]), crs="EPSG:4326")
            hex_chor = agg_points_to_hex(g_day, hex_grid)

            # Write per-day table with hex centroids (in meters -> back to 4326)
            table_path = os.path.join(args.tables_dir, f"hex_{day}.csv")
            hex_out = hex_chor.copy()
            cent_m = to_metric(hex_out).geometry.centroid
            cent_ll = gpd.GeoSeries(cent_m, crs="EPSG:26916").to_crs("EPSG:4326")
            hex_out["lon"] = cent_ll.x.values
            hex_out["lat"] = cent_ll.y.values
            cols = ["aqi_mean","lon","lat"]
            if "h3_index" in hex_out.columns: cols = ["h3_index"] + cols
            hex_out[cols].to_csv(table_path, index=False)

            out_png = os.path.join(args.outdir, f"aqi_heatmap_{day}.png")
            plot_one_day_hex(day, hex_chor, bnd, out_png, vmin=vmin, vmax=vmax, cmap="viridis", use_basemap=args.use_basemap)
            log.info(f"[Day] {day} → {out_png} (hex cells={len(hex_chor)})")

        else:
            g_day = gpd.GeoDataFrame(df_day, geometry=gpd.points_from_xy(df_day["lon"], df_day["lat"]), crs="EPSG:4326")
            out_png = os.path.join(args.outdir, f"aqi_points_{day}.png")
            plot_one_day_points(day, g_day, bnd, out_png, vmin=vmin, vmax=vmax, cmap="viridis", use_basemap=args.use_basemap)
            log.info(f"[Day] {day} → {out_png} (points={len(g_day)})")

        n_png += 1

    log.info(f"[Done] Wrote {n_png} daily map(s) to {args.outdir}")

if __name__ == "__main__":
    main()
