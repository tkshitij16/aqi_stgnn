#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local (street-level) AQI dashboards for four key Chicago regions.

Regions (fixed bounding boxes in lat/lon):
1. Lakefront Downtown (Loop + lakefront)
2. Southeast Industrial / Calumet Corridor
3. High-vegetation area (Schiller Woods)
4. Illinois Medical District (IMD)

Inputs
------
Predictions CSV (e.g. outputs/preds_2024.csv):
    epoch, node_id, aqi_hat, pm25_hat

Master CSV (locked, e.g. data/stt_master_locked_2024.csv):
    epoch,node_id,lat,lon,U,theta,PBLH,T,RH,precip,congestion,
    ndvi_z,p_impervious,p_water,p_wetland,p_grass,
    p_cultivated,p_pasture,p_barren,pm25_obs,aqi_obs

Boundary (for context; whole Chicago outline):
    data/chicago_boundary.geojson

Outputs
-------
- Individual PDFs for each region+week (temporary)
- One merged PDF per run (all requested weeks × 4 regions):
    <outdir>/local_dashboard_merged.pdf

Usage (PowerShell)
------------------
# All weeks present in 2024
python -m src.local_dashboards `
  --pred-csv outputs/preds_2024.csv `
  --master-csv data/stt_master_locked_2024.csv `
  --boundary data/chicago_boundary.geojson `
  --metric aqi_hat `
  --outdir figs/local_dashboards

# Single week, e.g., 2024-W30
python -m src.local_dashboards `
  --pred-csv outputs/preds_2024.csv `
  --master-csv data/stt_master_locked_2024.csv `
  --boundary data/chicago_boundary.geojson `
  --metric aqi_hat `
  --week 2024-W30 `
  --outdir figs/local_dashboards
"""

import argparse
import os
import logging
import tempfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

# Optional basemap + PDF merger
try:
    import contextily as cx
    _CX_OK = True
except Exception:
    _CX_OK = False

try:
    from PyPDF2 import PdfMerger
    _PDF_OK = True
except Exception:
    _PDF_OK = False


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
def get_logger(name="local_dashboards"):
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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def to_metric(gdf, epsg="EPSG:3857"):
    """Project to WebMercator for contextily basemap."""
    return gdf.to_crs(epsg)


def load_boundary(path):
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326", allow_override=True)
    g = g.to_crs("EPSG:4326")
    geom = unary_union(g.geometry)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


def week_key_local(epoch_series, tz="America/Chicago"):
    t = (
        pd.to_datetime(epoch_series.astype("int64"), unit="s", utc=True)
        .dt.tz_convert(tz)
    )
    iso = t.dt.isocalendar()
    week_str = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    return week_str, t


def maybe_alias_metric(df, name: str) -> str:
    """Return the first available column compatible with *name*."""
    alias_graph = {
        "aqi_hat": ["aqi_pred", "aqi"],
        "aqi_pred": ["aqi_hat", "aqi"],
        "aqi": ["aqi_hat", "aqi_pred"],
        "pm25_hat": ["pm25_pred", "pm25"],
        "pm25_pred": ["pm25_hat", "pm25"],
        "pm25": ["pm25_hat", "pm25_pred"],
    }

    seen = set()
    queue = [name]
    while queue:
        candidate = queue.pop(0)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in df.columns:
            if candidate != name:
                log.warning(
                    "[Metric] '%s' unavailable; using '%s' instead.", name, candidate
                )
            return candidate
        for nxt in alias_graph.get(candidate, []):
            if nxt not in seen:
                queue.append(nxt)
    return name


def aqi_category_cmap():
    bounds = [0, 50, 100, 150, 200, 300, 500]
    colors = [
        "#00e400",  # Good
        "#ffff00",  # Moderate
        "#ff7e00",  # USG
        "#ff0000",  # Unhealthy
        "#8f3f97",  # Very Unhealthy
        "#7e0023",  # Hazardous
    ]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm


def aqi_category_legend(ax):
    cats = [
        ("Good", 0, 50, "#00e400"),
        ("Moderate", 51, 100, "#ffff00"),
        ("USG", 101, 150, "#ff7e00"),
        ("Unhealthy", 151, 200, "#ff0000"),
        ("Very Unhealthy", 201, 300, "#8f3f97"),
        ("Hazardous", 301, 500, "#7e0023"),
    ]
    patches = [
        mpatches.Patch(color=c, label=f"{name} ({lo}-{hi})")
        for (name, lo, hi, c) in cats
    ]
    ax.legend(handles=patches, loc="center", frameon=False, ncol=1, fontsize=8)
    ax.axis("off")


def week_dates_from_iso(wk: str):
    from datetime import datetime

    y = int(wk.split("-W")[0])
    w = int(wk.split("-W")[1])
    start = datetime.fromisocalendar(y, w, 1).date()  # Monday
    end = datetime.fromisocalendar(y, w, 7).date()    # Sunday
    return start, end


def merge_pdfs(pdf_paths, out_path: str):
    if not _PDF_OK:
        log.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        return False
    merger = PdfMerger()
    for p in pdf_paths:
        try:
            merger.append(p)
        except Exception as e:
            log.warning(f"[Merge] Skipping '{p}' due to: {e}")
    ensure_dir(out_path)
    with open(out_path, "wb") as f:
        merger.write(f)
    merger.close()
    log.info(f"[Merge] Merged PDF → {out_path}")
    return True


def describe_driver(driver_label: str, r: float) -> str:
    """Climate-based explanation of correlation strength & sign."""
    if r is None or not np.isfinite(r):
        return f"{driver_label}: data were insufficient to assess spatial influence in this neighbourhood."

    mag = abs(r)
    if mag < 0.10:
        strength = "negligible"
    elif mag < 0.30:
        strength = "weak"
    elif mag < 0.60:
        strength = "moderate"
    else:
        strength = "strong"

    sign = "positive" if r > 0 else "negative"
    base = f"{driver_label}: {strength} {sign} correlation (r≈{r:.2f}). "
    key = driver_label.lower()

    if "ventilation" in key:
        if r < 0:
            return base + (
                "Higher AQI tended to occur on weaker-ventilation links "
                "(lower wind speed × shallower PBL), consistent with local stagnation."
            )
        else:
            return base + (
                "Higher AQI co-occurred with stronger ventilation, suggesting advection "
                "of pollution from upwind corridors outweighed local dilution."
            )
    if "traffic" in key or "congestion" in key:
        if r > 0:
            return base + (
                "Streets with heavier traffic generally showed higher AQI, indicating a "
                "clear roadway emission influence at the street scale."
            )
        else:
            return base + (
                "AQI did not systematically increase with congestion, implying regional "
                "background or meteorology dominated over local traffic this week."
            )
    if "vegetation" in key or "ndvi" in key:
        if r < 0:
            return base + (
                "Greener blocks tended to have cleaner air, consistent with vegetated "
                "corridors buffering emissions and enhancing dispersion."
            )
        else:
            return base + (
                "Greener cells also exhibited slightly higher AQI, likely due to a "
                "widespread regional haze that affected both vegetated and built-up areas."
            )
    if "impervious" in key:
        if r > 0:
            return base + (
                "More impervious, built-up surfaces coincided with elevated AQI, aligning "
                "with dense emission sources and reduced near-surface mixing."
            )
        else:
            return base + (
                "Impervious cells did not show systematically higher AQI, suggesting a more "
                "homogeneous pollution field across urban fabrics."
            )
    if "temperature" in key:
        if r > 0:
            return base + (
                "Warmer street segments tended to have higher AQI, consistent with "
                "photochemical enhancement and urban heat-island effects."
            )
        else:
            return base + (
                "Cooler areas showed marginally higher AQI, pointing to stagnation-dominated "
                "episodes rather than heat-driven photochemistry."
            )
    if "humidity" in key:
        if r > 0:
            return base + (
                "More humid pockets were associated with higher AQI, consistent with moist, "
                "stable boundary layers trapping pollutants near the ground."
            )
        else:
            return base + (
                "Drier pockets exhibited slightly higher AQI, indicating that turbulent "
                "mixing in drier air masses dominated over moist stagnation."
            )
    if "wind speed" in key:
        if r < 0:
            return base + (
                "Higher wind speeds were linked to cleaner air, reflecting the role of "
                "ventilation in flushing pollutants from street canyons."
            )
        else:
            return base + (
                "Higher wind speeds coincided with higher AQI, suggesting transport from "
                "upwind source regions into this neighbourhood."
            )

    return base + "Its influence appears secondary compared to other drivers this week."


# -------------------------------------------------------------------
# Region definitions (approximate bounding boxes)
# -------------------------------------------------------------------
REGIONS = {
    "lakefront_downtown": {
        "name": "Lakefront Downtown",
        "min_lat": 41.870,
        "max_lat": 41.895,
        "min_lon": -87.640,
        "max_lon": -87.610,
    },
    "industrial_calumet": {
        "name": "Southeast Industrial / Calumet Corridor",
        "min_lat": 41.600,
        "max_lat": 41.670,
        "min_lon": -87.660,
        "max_lon": -87.530,
    },
    "schiller_woods": {
        "name": "High-vegetation Area (Schiller Woods)",
        "min_lat": 41.930,
        "max_lat": 41.975,
        "min_lon": -87.875,
        "max_lon": -87.805,
    },
    "illinois_medical_district": {
        "name": "Illinois Medical District",
        "min_lat": 41.865,
        "max_lat": 41.880,
        "min_lon": -87.690,
        "max_lon": -87.655,
    },
}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-csv",
        default=None,
        help="Predictions CSV. If omitted and master contains aqi_pred/pm25_pred,"
        " those columns are used instead.",
    )
    ap.add_argument("--master-csv", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument(
        "--metric",
        default="aqi_hat",
        help="aqi_hat (default) | pm25_hat | aqi_obs",
    )
    ap.add_argument(
        "--week",
        default=None,
        help="ISO week e.g. 2024-W30; if omitted, all weeks present in predictions",
    )
    ap.add_argument("--outdir", default="figs/local_dashboards")
    ap.add_argument(
        "--merged-name",
        default="local_dashboard_merged.pdf",
        help="Merged PDF filename",
    )
    ap.add_argument(
        "--keep-singles",
        action="store_true",
        help="Keep per-region PDFs instead of deleting temporary files",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Master with drivers & lat/lon ----
    head = pd.read_csv(args.master_csv, nrows=0)
    usecols = [
        c
        for c in [
            "epoch",
            "node_id",
            "lat",
            "lon",
            "U",
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
            "aqi_obs",
            "aqi_pred",
            "pm25_pred",
        ]
        if c in head.columns
    ]
    dfm = pd.read_csv(args.master_csv, usecols=usecols, low_memory=False)
    dfm["node_id"] = dfm["node_id"].astype(str)
    dfm["epoch"] = pd.to_numeric(dfm["epoch"], errors="coerce").astype("Int64")
    dfm["VC"] = (
        pd.to_numeric(dfm.get("U"), errors="coerce")
        * pd.to_numeric(dfm.get("PBLH"), errors="coerce")
    )

    # Node-level static lat/lon lookup (one row per node)
    nodes_ll = dfm[["node_id", "lat", "lon"]].dropna(subset=["lat", "lon"]).drop_duplicates("node_id")

    # ---- Predictions ----
    has_master_preds = {"aqi_pred", "pm25_pred"}.issubset(dfm.columns)
    if has_master_preds:
        log.info("[Predictions] Using master CSV columns aqi_pred/pm25_pred for dashboards.")
        dfp = dfm[["epoch", "node_id", "aqi_pred", "pm25_pred", "lat", "lon"]].copy()
        dfp = dfp.rename(columns={"aqi_pred": "aqi_hat", "pm25_pred": "pm25_hat"})
    else:
        if not args.pred_csv:
            raise ValueError(
                "Master CSV lacks aqi_pred/pm25_pred columns and --pred-csv was not provided."
            )
        dfp = pd.read_csv(args.pred_csv, low_memory=False)
        for col in ["epoch", "node_id"]:
            if col not in dfp.columns:
                raise ValueError(f"Missing '{col}' in predictions.")
        dfp["node_id"] = dfp["node_id"].astype(str)
        dfp["epoch"] = pd.to_numeric(dfp["epoch"], errors="coerce").astype("Int64")
        dfp = dfp.merge(nodes_ll, on="node_id", how="left")
        missing_ll = dfp["lat"].isna() | dfp["lon"].isna()
        if missing_ll.any():
            log.warning(
                "[Predictions] Dropping %d rows without lat/lon after merge.",
                int(missing_ll.sum()),
            )
            dfp = dfp[~missing_ll].copy()

    args.metric = maybe_alias_metric(dfp, args.metric)

    # Add aqi_obs to predictions if not present
    if "aqi_obs" in dfm.columns and "aqi_obs" not in dfp.columns:
        dfp = dfp.merge(
            dfm[["epoch", "node_id", "aqi_obs"]],
            on=["epoch", "node_id"],
            how="left",
        )

    # Week labels
    dfp["week"], _ = week_key_local(dfp["epoch"])
    dfm["week"], _ = week_key_local(dfm["epoch"])

    weeks_all = sorted(dfp["week"].dropna().unique())
    weeks = [args.week] if args.week else weeks_all
    if not weeks:
        log.error("No weeks found in predictions.")
        return

    # Boundary (for context)
    bnd = load_boundary(args.boundary)
    cmap_cat, norm_cat = aqi_category_cmap()

    # Temp dir
    tmpdir = tempfile.mkdtemp(prefix="local_pages_")
    generated = []

    for wk in weeks:
        wk_start, wk_end = week_dates_from_iso(wk)
        g_week = dfp[dfp["week"] == wk].copy()
        if g_week.empty:
            log.warning(f"[{wk}] No prediction rows; skipping week.")
            continue

        pred_col = args.metric
        if pred_col not in g_week.columns and pred_col == "aqi_obs":
            raise ValueError("Requested 'aqi_obs' but not present in predictions/master.")
        g_week[pred_col] = pd.to_numeric(g_week[pred_col], errors="coerce")

        # --- Node-level weekly mean predictions ---
        node_week = (
            g_week[["node_id", pred_col]]
            .groupby("node_id", as_index=False)
            .mean()
        )
        # Attach lat/lon from master at node level (THIS IS THE KEY FIX)
        node_week = node_week.merge(nodes_ll, on="node_id", how="left")
        miss_ll = node_week["lat"].isna() | node_week["lon"].isna()
        if miss_ll.any():
            log.warning(
                f"[{wk}] Dropping {miss_ll.sum()} node(s) without lat/lon after join."
            )
            node_week = node_week[~miss_ll].copy()

        if node_week.empty:
            log.warning(f"[{wk}] No nodes with lat/lon after join; skipping week.")
            continue

        # Master subset for this week (for dynamic features)
        m_week = dfm[dfm["week"] == wk].copy()
        feature_cols = ["VC", "congestion", "ndvi_z", "p_impervious", "T", "RH", "U"]

        # Weekly mean features per node
        f_week = (
            m_week[["node_id"] + feature_cols]
            .groupby("node_id", as_index=False)
            .mean()
        )

        for region_id, reg in REGIONS.items():
            name = reg["name"]
            min_lat, max_lat = reg["min_lat"], reg["max_lat"]
            min_lon, max_lon = reg["min_lon"], reg["max_lon"]

            sub_nodes = node_week[
                (node_week["lat"] >= min_lat)
                & (node_week["lat"] <= max_lat)
                & (node_week["lon"] >= min_lon)
                & (node_week["lon"] <= max_lon)
            ].copy()
            if sub_nodes.empty:
                log.warning(f"[{wk} | {name}] No nodes in bounding box; skipping.")
                continue

            # Attach features for this region
            reg_df = sub_nodes.merge(f_week, on="node_id", how="left")

            # Driver correlations
            vals = reg_df[pred_col].values
            driver_info = []
            label_map = {
                "VC": "Ventilation (U×PBLH)",
                "congestion": "Traffic congestion",
                "ndvi_z": "Vegetation (NDVI z)",
                "p_impervious": "Impervious surface",
                "T": "Temperature",
                "RH": "Relative humidity",
                "U": "Wind speed",
            }
            for key in feature_cols:
                if key not in reg_df.columns:
                    r = np.nan
                else:
                    a = np.asarray(vals, float)
                    b = np.asarray(reg_df[key].values, float)
                    m = np.isfinite(a) & np.isfinite(b)
                    r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() >= 5 else np.nan
                driver_info.append((key, label_map[key], r))

            # Local climate summary
            median = np.nanmedian(vals)
            p10 = np.nanpercentile(vals, 10) if np.isfinite(vals).any() else np.nan
            p90 = np.nanpercentile(vals, 90) if np.isfinite(vals).any() else np.nan

            T_mean = reg_df["T"].mean() if "T" in reg_df.columns else np.nan
            RH_mean = reg_df["RH"].mean() if "RH" in reg_df.columns else np.nan
            U_mean = reg_df["U"].mean() if "U" in reg_df.columns else np.nan

            climate_summary = (
                f"{name}, week {wk} ({wk_start}–{wk_end}): "
                f"street-level weekly AQI median ≈ {median:.0f} "
                f"(P10≈{p10:.0f}, P90≈{p90:.0f}). "
            )
            bits = []
            if np.isfinite(T_mean):
                bits.append(f"T̄≈{T_mean:.1f} °C")
            if np.isfinite(RH_mean):
                bits.append(f"RH̄≈{RH_mean:.0f}%")
            if np.isfinite(U_mean):
                bits.append(f"Ū≈{U_mean:.1f} m/s")
            climate_dyn = "Local mean conditions: " + ", ".join(bits) + "." if bits else ""

            driver_sentences = [
                describe_driver(label, r) for (_, label, r) in driver_info
            ]

            # --- Figure layout ---
            plt.rcParams.update(
                {
                    "figure.dpi": 220,
                    "savefig.bbox": "tight",
                    "axes.titlesize": 11,
                    "axes.labelsize": 9,
                    "xtick.labelsize": 8,
                    "ytick.labelsize": 8,
                    "legend.fontsize": 8,
                }
            )

            fig = plt.figure(figsize=(11.2, 8.5))
            gs = gridspec.GridSpec(
                4,
                12,
                height_ratios=[0.6, 0.9, 3.6, 2.8],
                width_ratios=[1, 1, 1, 1, 1, 0.7, 0.7, 1, 1, 1, 1, 1],
                hspace=0.32,
                wspace=0.25,
            )

            # Title row
            axTitle = fig.add_subplot(gs[0, :])
            axTitle.axis("off")
            title_txt = f"{name} — Street-level AQI dashboard | {wk_start} to {wk_end}"
            axTitle.text(
                0.01,
                0.65,
                title_txt,
                ha="left",
                va="center",
                fontsize=13,
                weight="bold",
            )

            # Weekly inference row
            axInf = fig.add_subplot(gs[1, :])
            axInf.axis("off")
            axInf.text(
                0.01,
                0.95,
                "Weekly inference:",
                fontsize=11,
                weight="bold",
                va="top",
            )
            inf_y = 0.72
            axInf.text(
                0.02,
                inf_y,
                climate_summary,
                fontsize=9.6,
                va="top",
                wrap=True,
            )
            if climate_dyn:
                axInf.text(
                    0.02,
                    inf_y - 0.32,
                    climate_dyn,
                    fontsize=9.3,
                    va="top",
                    wrap=True,
                )

            # Core visual row: map, legend, drivers
            axMap = fig.add_subplot(gs[2, 0:6])
            axLeg = fig.add_subplot(gs[2, 6:8])
            axDrv = fig.add_subplot(gs[2, 8:12])

            sub_gdf = gpd.GeoDataFrame(
                reg_df,
                geometry=gpd.points_from_xy(reg_df["lon"], reg_df["lat"]),
                crs="EPSG:4326",
            )

            # Bounding box polygon for framing the basemap nicely
            region_poly = gpd.GeoDataFrame(
                geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326"
            )

            if _CX_OK:
                sub_web = to_metric(sub_gdf)
                bnd_web = to_metric(bnd)
                reg_web = to_metric(region_poly)
                bnd_web.boundary.plot(ax=axMap, color="#cccccc", linewidth=0.6, zorder=1)
                cx.add_basemap(
                    axMap,
                    crs=sub_web.crs,
                    source=cx.providers.CartoDB.Positron,
                    attribution=False,
                    alpha=0.98,
                )
                cmap_cat_local, norm_cat_local = aqi_category_cmap()
                axMap.scatter(
                    sub_web.geometry.x,
                    sub_web.geometry.y,
                    c=sub_gdf[pred_col].values,
                    cmap=cmap_cat_local,
                    norm=norm_cat_local,
                    s=40,
                    edgecolors="black",
                    linewidths=0.35,
                    alpha=0.95,
                    zorder=5,
                )
                bounds = reg_web.total_bounds
                axMap.set_xlim(bounds[0], bounds[2])
                axMap.set_ylim(bounds[1], bounds[3])
                axMap.set_axis_off()
            else:
                bnd.boundary.plot(ax=axMap, color="#cccccc", linewidth=0.8, zorder=1)
                sc = axMap.scatter(
                    sub_gdf.geometry.x,
                    sub_gdf.geometry.y,
                    c=sub_gdf[pred_col].values,
                    cmap=cmap_cat,
                    norm=norm_cat,
                    s=40,
                    edgecolors="black",
                    linewidths=0.35,
                    alpha=0.95,
                )
                axMap.set_xlim(min_lon, max_lon)
                axMap.set_ylim(min_lat, max_lat)
                axMap.set_xlabel("Longitude")
                axMap.set_ylabel("Latitude")

            axMap.set_title("Weekly mean AQI at nodes (street scale)")

            aqi_category_legend(axLeg)
            axLeg.set_title("AQI colour key", fontsize=9.5)

            labels_drv = [lab for _, lab, _ in driver_info]
            vals_drv = [r for _, _, r in driver_info]
            y_pos = np.arange(len(labels_drv))
            axDrv.barh(y_pos, vals_drv, height=0.6, color="#5c6bc0")
            axDrv.set_yticks(y_pos)
            axDrv.set_yticklabels(labels_drv, fontsize=7.5)
            axDrv.set_xlim(-1.0, 1.0)
            axDrv.axvline(0.0, color="#444444", linewidth=0.6)
            axDrv.set_xlabel("Corr (r)")
            axDrv.set_title("Drivers of spatial variation")
            axDrv.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.8)

            # Driver-wise interpretation panel
            axTxt = fig.add_subplot(gs[3, :])
            axTxt.axis("off")
            axTxt.text(
                0.01,
                0.94,
                "Driver-wise interpretation:",
                fontsize=10.5,
                weight="bold",
                va="top",
            )
            bullets = [f"• {sent}" for sent in driver_sentences if sent]
            body = "\n\n".join(bullets) if bullets else "No statistically robust driver signals this week."
            axTxt.text(
                0.02,
                0.88,
                body,
                fontsize=9.2,
                va="top",
                wrap=True,
            )

            out_pdf = os.path.join(tmpdir, f"local_{wk}_{region_id}.pdf")
            fig.savefig(out_pdf, dpi=300)
            plt.close(fig)
            generated.append(out_pdf)
            log.info(f"[{wk} | {name}] Page → {out_pdf}")

    # Merge all pages
    if not generated:
        log.error("No local dashboards generated; nothing to merge.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    merged_path = os.path.join(args.outdir, args.merged_name)
    if len(generated) == 1:
        ensure_dir(merged_path)
        shutil.copyfile(generated[0], merged_path)
        log.info(f"[Merge] Single page → copied to {merged_path}")
    else:
        if not _PDF_OK:
            log.error("PyPDF2 not installed; cannot merge. Install PyPDF2 to merge.")
        else:
            merge_pdfs(sorted(generated), merged_path)

    if not args.keep_singles:
        shutil.rmtree(tmpdir, ignore_errors=True)
        log.info("[Cleanup] Removed individual local pages (kept merged only).")
    else:
        log.info(f"[Keep] Individual pages kept in: {tmpdir}")


if __name__ == "__main__":
    main()
