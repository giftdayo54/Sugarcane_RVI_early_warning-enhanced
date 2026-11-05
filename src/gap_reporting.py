"""
src/gap_reporting.py

Persist gap-analysis results in the same `outputs/` layout the early-warning pipeline uses, so the Home
dashboard (and QGIS) can read them next to `block_report.csv` / `block_status.geojson`.

    outputs/reports/gap_report.csv          one row per block (latest analysis wins)
    outputs/vectors/gap_polygons.geojson    gap patches (EPSG:4326), bare vs weed-covered
    outputs/rasters/gaps/<block>_<date>.tif class raster with embedded palette
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .gap_detection import GapResult, expected_cane_height, gap_polygons, write_class_geotiff
from .imagery import Scene

DEFAULT_SEVERITY = {"low": 5.0, "moderate": 10.0, "high": 20.0}


def severity(gap_pct: float, thresholds: dict | None = None) -> str:
    t = {**DEFAULT_SEVERITY, **(thresholds or {})}
    if gap_pct is None or not np.isfinite(gap_pct):
        return "Unknown"
    if gap_pct < t["low"]:
        return "Low"
    if gap_pct < t["moderate"]:
        return "Moderate"
    if gap_pct < t["high"]:
        return "High"
    return "Severe"


def height_for_age(cfg: dict | None, age_days: float) -> float:
    """Suggested average cane height (m) from the config's logistic curve (override with measured values)."""
    g = (cfg or {}).get("gaps", {})
    curve = g.get("height_curve", {}) or {}
    return round(expected_cane_height(age_days, **{k: v for k, v in curve.items() if k in ("h_max", "k", "t0")}), 2)


def save_result(scene: Scene, result: GapResult, out_dir: str | Path = "outputs", cfg: dict | None = None) -> dict:
    """Write/merge this field's results into outputs/. Returns the paths written."""
    out = Path(out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    (out / "rasters" / "gaps").mkdir(parents=True, exist_ok=True)

    row = dict(result.stats)
    row["gap_severity"] = severity(row["gap_pct"], ((cfg or {}).get("gaps", {}) or {}).get("severity_pct"))
    rep_path = out / "reports" / "gap_report.csv"
    new = pd.DataFrame([row])
    if rep_path.exists():
        old = pd.read_csv(rep_path)
        old = old[old["block_id"].astype(str) != str(row["block_id"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(rep_path, index=False)

    poly_path = out / "vectors" / "gap_polygons.geojson"
    polys = gap_polygons(result.class_map, scene.grid, scene.field_id, scene.date)
    if poly_path.exists():
        try:
            old_p = gpd.read_file(poly_path)
            old_p = old_p[old_p["block_id"].astype(str) != str(scene.field_id)]
            polys = pd.concat([old_p, polys], ignore_index=True) if len(old_p) else polys
        except Exception:
            pass
    gpd.GeoDataFrame(polys, geometry="geometry", crs=4326).to_file(poly_path, driver="GeoJSON")

    tif_path = out / "rasters" / "gaps" / f"{scene.field_id}_{scene.date}.tif"
    write_class_geotiff(tif_path, result.class_map, scene.grid)
    return {"report": str(rep_path), "polygons": str(poly_path), "raster": str(tif_path)}


def load_gap_report(out_dir: str | Path = "outputs") -> pd.DataFrame | None:
    p = Path(out_dir) / "reports" / "gap_report.csv"
    return pd.read_csv(p) if p.exists() else None
