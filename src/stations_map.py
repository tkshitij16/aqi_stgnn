#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

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

from sklearn.neighbors import NearestNeighbors

# ---------- logging ----------
def get_logger():
    log = logging.getLogger("stations_map")
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

# ---------- utils ----------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_nodes_from_master(master_csv, lon_col="lon", lat_col="lat", node_col="node_id"):
    df = pd.read_csv(master_csv, usecols=[node_col, lon_col, lat_col])
    df[node_col] = df[node_col].astype(str)
    nodes = df.drop_duplicates(subset=[node_col]).copy()
    nodes = nodes.rename(columns={lon_col: "lon", lat_col: "lat"})
    gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes["lon"], nodes["lat"]),
                           crs="EPSG:4326")
    log.info(f"[Data] Unique stations: {len(gdf)}")
    return gdf

def load_boundary(boundary_path):
    g = gpd.read_file(boundary_path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326", allow_override=True)
    g = g.to_crs("EPSG:4326")
    geom = unary_union(g.geometry)
    log.info(f"[Boundary] Loaded: {boundary_path}")
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

def auto_bbox_polygon(points_gdf, buffer_deg=0.05):
    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    xmin -= buffer_deg; ymin -= buffer_deg
    xmax += buffer_deg; ymax += buffer_deg
    poly = Polygon([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)])
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

def to_metric(gdf, epsg="EPSG:26916"):  # UTM zone 16N (meters)
    return gdf.to_crs(epsg)

def hexagon(cx, cy, edge):
    angles = np.deg2rad([0,60,120,180,240,300])
    xs = cx + edge * np.cos(angles)
    ys = cy + edge * np.sin(angles)
    return Polygon(zip(xs, ys))

def hex_grid_shapely(boundary_gdf, hex_edge_m=500):
    bnd_m = to_metric(boundary_gdf)
    minx, miny, maxx, maxy = bnd_m.total_bounds
    dx = (3**0.5) * hex_edge_m
    dy = 1.5 * hex_edge_m
    cols = int(np.ceil((maxx - minx) / dx)) + 2
    rows = int(np.ceil((maxy - miny) / dy)) + 2
    hexes = []
    for r in range(rows):
        for c in range(cols):
            cx = minx + c * dx + (0.5 * dx if r % 2 else 0)
            cy = miny + r * dy
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
        raise RuntimeError("h3 is not installed. pip install h3")
    geom = boundary_gdf.geometry.iloc[0]
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    idxs = set()
    for p in polys:
        gj = _poly_to_geojson_lonlat(p)
        try:
            new_idxs = h3.polyfill(gj, res, geo_json_conformant=True)
        except TypeError:
            new_idxs = h3.polyfill(gj, res)
        idxs |= set(new_idxs)
    hex_polys = []
    for h in sorted(idxs):
        b = h3.h3_to_geo_boundary(h, geo_json=True)  # [[lat,lon],...]
        hex_polys.append(Polygon([(lon, lat) for (lat, lon) in b]))
    grid = gpd.GeoDataFrame({"h3_index": list(sorted(idxs))}, geometry=hex_polys, crs="EPSG:4326")
    return grid

def count_points_in_polys(points_gdf, polys_gdf):
    pts_m = to_metric(points_gdf)
    polys_m = to_metric(polys_gdf)
    join = gpd.sjoin(pts_m, polys_m, predicate="within", how="left")
    counts = join.groupby(join.index_right).size()
    polys_m["count"] = counts.reindex(range(len(polys_m))).fillna(0).astype(int).values
    return polys_m.to_crs("EPSG:4326")

def nearest_distance_m(centroids_m_df, stations_m_df):
    if centroids_m_df.geometry.iloc[0].geom_type != "Point":
        centroids_m_df = centroids_m_df.copy()
        centroids_m_df["geometry"] = centroids_m_df.geometry.centroid
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    Xs = np.c_[stations_m_df.geometry.x.values, stations_m_df.geometry.y.values]
    nbrs.fit(Xs)
    Xc = np.c_[centroids_m_df.geometry.x.values, centroids_m_df.geometry.y.values]
    dists, _ = nbrs.kneighbors(Xc, n_neighbors=1)
    return dists[:,0]

def idw_score(centroids_m_df, stations_m_df, power=2.0, k=8, radius_m=None, eps=1.0):
    if centroids_m_df.geometry.iloc[0].geom_type != "Point":
        centroids_m_df = centroids_m_df.copy()
        centroids_m_df["geometry"] = centroids_m_df.geometry.centroid
    n_st = len(stations_m_df)
    if n_st == 0:
        return np.zeros(len(centroids_m_df), dtype=float)
    k = min(k, n_st)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
    Xs = np.c_[stations_m_df.geometry.x.values, stations_m_df.geometry.y.values]
    nbrs.fit(Xs)
    Xc = np.c_[centroids_m_df.geometry.x.values, centroids_m_df.geometry.y.values]
    dists, _ = nbrs.kneighbors(Xc, n_neighbors=k)
    if radius_m is not None:
        mask = dists <= radius_m
    else:
        mask = np.ones_like(dists, dtype=bool)
    dists = np.where(mask, dists, np.inf)
    dists = np.maximum(dists, eps)
    w = 1.0 / (dists ** power)
    w[~np.isfinite(w)] = 0.0
    return w.sum(axis=1)

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

# ---------- plotting ----------
def plot_hex(points_gdf, boundary_gdf, backend="h3", h3_res=8, hex_edge_m=500,
             uniform=False, shade="count",
             idw_power=2.0, idw_k=8, idw_radius_m=None,
             out_png="figs/stations_hex.png",
             out_cells_csv="tables/hex_cells.csv",
             out_cells_geo="tables/hex_cells.geojson",
             add_points=True, use_basemap=False):
    publish_style()
    ensure_dir(out_png); ensure_dir(out_cells_csv); ensure_dir(out_cells_geo)

    if uniform and (boundary_gdf is None or boundary_gdf.empty):
        raise ValueError("Uniform cover requires --boundary (GeoJSON/Shapefile).")

    if uniform:
        if backend == "h3":
            if not _H3_OK:
                log.warning("h3 not installed; falling back to shapely hex backend.")
                backend = "shapely"
            else:
                log.info(f"[Hex] H3 uniform polyfill res={h3_res}")
                grid = hex_grid_h3(boundary_gdf, res=h3_res)
        if backend == "shapely":
            log.info(f"[Hex] Shapely uniform hex grid edge={hex_edge_m} m")
            grid = hex_grid_shapely(boundary_gdf, hex_edge_m=hex_edge_m)
    else:
        if boundary_gdf is None:
            boundary_gdf = auto_bbox_polygon(points_gdf, buffer_deg=0.05)
            log.info("[Hex] Using auto bbox (station-touching grid).")
        if backend == "h3" and _H3_OK:
            log.info(f"[Hex] H3 station-touching res={h3_res}")
            idxs = [h3.geo_to_h3(lat, lon, h3_res) for lat, lon in zip(points_gdf["lat"], points_gdf["lon"])]
            idxs = sorted(set(idxs))
            polys = []
            for hi in idxs:
                b = h3.h3_to_geo_boundary(hi, geo_json=True)
                polys.append(Polygon([(lon, lat) for (lat, lon) in b]))
            grid = gpd.GeoDataFrame({"h3_index": idxs}, geometry=polys, crs="EPSG:4326")
            grid = gpd.overlay(grid, boundary_gdf, how="intersection", keep_geom_type=True)
        else:
            grid = hex_grid_shapely(boundary_gdf, hex_edge_m=hex_edge_m)

    # counts
    grid = count_points_in_polys(points_gdf, grid)

    # metric CRS for centroid-based metrics
    grid_m = to_metric(grid)
    pts_m  = to_metric(points_gdf)
    centroids_m = grid_m.copy()
    centroids_m["geometry"] = grid_m.geometry.centroid

    # nearest + IDW
    grid_m["nearest_m"] = nearest_distance_m(centroids_m, pts_m)
    grid_m["idw"] = idw_score(centroids_m, pts_m, power=idw_power, k=idw_k, radius_m=idw_radius_m, eps=1.0)
    nz = grid_m["idw"].values
    p95 = np.percentile(nz[nz>0], 95) if (nz>0).any() else 1.0
    p95 = p95 if np.isfinite(p95) and p95>0 else 1.0
    grid_m["idw_norm"] = np.clip(grid_m["idw"]/p95, 0.0, 1.0)

    # back to 4326
    grid = grid_m.to_crs("EPSG:4326")

    # ---- Save tables (centroids computed in metric CRS, then projected) ----
    centroids_wgs = centroids_m.to_crs("EPSG:4326")
    grid["lon"] = centroids_wgs.geometry.x.values
    grid["lat"] = centroids_wgs.geometry.y.values

    cols = ["count","nearest_m","idw","idw_norm","lon","lat"]
    if "h3_index" in grid.columns: cols = ["h3_index"] + cols
    grid[cols].to_csv(out_cells_csv, index=False)
    grid.to_file(out_cells_geo, driver="GeoJSON")
    log.info(f"[Hex] Wrote {out_cells_csv} and {out_cells_geo}")

    # shading
    s = shade.lower()
    if s == "nearest":
        column = "nearest_m"; cmap = "viridis_r"; ttl = "Hex cover — distance to nearest station (m)"
    elif s == "idw":
        column = "idw_norm"; cmap = "viridis"; ttl = "Hex cover — IDW coverage (normalized)"
    else:
        column = "count"; cmap = "viridis"; ttl = "Hex cover — station count"

    # plot
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black", zorder=2)
    grid.plot(ax=ax, column=column, cmap=cmap, edgecolor="none", legend=True, alpha=0.95, zorder=1)
    if add_points:
        points_gdf.plot(ax=ax, markersize=8, color="white", edgecolor="black", linewidth=0.3, alpha=0.9, zorder=3)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(ttl)

    if use_basemap and _CX_OK:
        ax.set_axis_off()
        grid_web = grid.to_crs(epsg=3857)
        if boundary_gdf is not None:
            bnd_web  = boundary_gdf.to_crs(epsg=3857)
        pts_web  = points_gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=(6.8,6.8))
        if boundary_gdf is not None:
            bnd_web.boundary.plot(ax=ax, linewidth=1.0, color="black", zorder=2)
        grid_web.plot(ax=ax, column=column, cmap=cmap, edgecolor="none", legend=True, alpha=0.95, zorder=1)
        if add_points:
            pts_web.plot(ax=ax, markersize=8, color="white", edgecolor="black", linewidth=0.3, zorder=3)
        cx.add_basemap(ax, crs=grid_web.crs, source=cx.providers.Stamen.TonerLite)
        ax.set_title(ttl)
        plt.savefig(out_png, dpi=300)
    else:
        plt.savefig(out_png, dpi=300)
    plt.close()
    log.info(f"[Hex] Figure saved → {out_png}")

def plot_pins(points_gdf, boundary_gdf=None, out_png="figs/stations_pins.png", use_basemap=False):
    ensure_dir(out_png)
    publish_style()
    if boundary_gdf is None:
        boundary_gdf = auto_bbox_polygon(points_gdf, buffer_deg=0.05)
        log.info("[Pins] Using auto bbox.")
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    boundary_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    points_gdf.plot(ax=ax, markersize=12, color="tab:red", edgecolor="white", linewidth=0.6, alpha=0.95)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Station Locations (Pins)")
    if use_basemap and _CX_OK:
        pts = points_gdf.to_crs(epsg=3857)
        bnd = boundary_gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=(6.8,6.8))
        bnd.boundary.plot(ax=ax, linewidth=1.0, color="black")
        pts.plot(ax=ax, markersize=12, color="tab:red", edgecolor="white", linewidth=0.6)
        cx.add_basemap(ax, crs=bnd.crs, source=cx.providers.Stamen.TonerLite)
        ax.set_title("Station Locations (Pins)")
        plt.savefig(out_png, dpi=300)
    else:
        plt.savefig(out_png, dpi=300)
    plt.close()
    log.info(f"[Pins] Figure saved → {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--mode", choices=["hex","pins"], default="hex")
    ap.add_argument("--boundary", required=False, default=None)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--hex-backend", choices=["h3","shapely"], default="h3")
    ap.add_argument("--h3-res", type=int, default=8)
    ap.add_argument("--hex-edge-m", type=float, default=500.0)
    ap.add_argument("--shade", choices=["count","nearest","idw"], default="count")
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--idw-k", type=int, default=8)
    ap.add_argument("--idw-radius-m", type=float, default=None)
    ap.add_argument("--out", default="figs/stations_hex.png")
    ap.add_argument("--use-basemap", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out)
    pts = read_nodes_from_master(args.master)
    bnd = load_boundary(args.boundary) if args.boundary else None

    if args.mode == "hex":
        if args.uniform and bnd is None:
            raise SystemExit("Uniform cover requested but --boundary not provided.")
        if not args.uniform and bnd is None:
            bnd = auto_bbox_polygon(pts, buffer_deg=0.05)
            log.info("[Hex] Using auto bbox (station-touching grid).")
        plot_hex(
            points_gdf=pts,
            boundary_gdf=bnd,
            backend=args.hex_backend,
            h3_res=args.h3_res,
            hex_edge_m=args.hex_edge_m,
            uniform=args.uniform,
            shade=args.shade,
            idw_power=args.idw_power,
            idw_k=args.idw_k,
            idw_radius_m=args.idw_radius_m,
            out_png=args.out,
            out_cells_csv="tables/hex_cells.csv",
            out_cells_geo="tables/hex_cells.geojson",
            add_points=True,
            use_basemap=args.use_basemap
        )
    else:
        if bnd is None:
            bnd = auto_bbox_polygon(pts, buffer_deg=0.05)
        plot_pins(points_gdf=pts, boundary_gdf=bnd, out_png=args.out, use_basemap=args.use_basemap)

if __name__ == "__main__":
    main()
