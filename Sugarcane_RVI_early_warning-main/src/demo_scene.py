"""
src/demo_scene.py

Synthetic Sentinel-2 + Sentinel-1 scenes with a KNOWN gap layout, so the Imagery
and Gap Analysis pages can be exercised without Sentinel Hub credentials.

THIS IS SYNTHETIC DATA. Spectra and backscatter are plausible textbook-style
values, not measurements; any accuracy computed on it only shows that the
pipeline runs end to end -- it says nothing about real-world accuracy.

Truth codes: 0 healthy cane, 1 bare-soil gap, 2 weed-covered gap, 3 thin/stunted canopy.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import box

import geopandas as gpd

from .imagery import Grid, Scene, erode_mask, polygon_mask
from .raster_calc import S2_BANDS

# mean surface reflectance per class, in S2_BANDS order
#                  B01   B02   B03   B04   B05   B06   B07   B08   B8A   B09   B11   B12
_SPECTRA = {
    0: [0.030, 0.030, 0.060, 0.035, 0.090, 0.300, 0.380, 0.420, 0.440, 0.450, 0.200, 0.090],  # closed cane canopy
    1: [0.100, 0.100, 0.130, 0.170, 0.190, 0.210, 0.230, 0.250, 0.260, 0.260, 0.360, 0.300],  # bare soil
    2: [0.040, 0.040, 0.080, 0.060, 0.120, 0.270, 0.330, 0.360, 0.380, 0.390, 0.250, 0.130],  # green weeds
    3: [0.045, 0.045, 0.075, 0.060, 0.110, 0.240, 0.300, 0.330, 0.350, 0.360, 0.240, 0.120],  # thin / stunted
}
# Sentinel-1 composite backscatter (dB): (VV, VH)
_S1_DB = {0: (-8.5, -14.5), 1: (-12.5, -22.0), 2: (-11.0, -19.5), 3: (-10.0, -17.0)}

CLASS_NAMES = {0: "Healthy cane", 1: "Bare-soil gap", 2: "Weed-covered gap", 3: "Thin / stunted canopy"}


def demo_field_geometry(lon: float = 34.80, lat: float = -15.80, width_m: float = 750, height_m: float = 600):
    """A rectangular demo field (WGS84) around the repo's demo estate (Malawi lowveld)."""
    dlon = width_m / (111_320 * np.cos(np.radians(lat)))
    dlat = height_m / 110_574
    return box(lon - dlon / 2, lat - dlat / 2, lon + dlon / 2, lat + dlat / 2)


def _blob(shape, cy, cx, ry, rx, rng, rough=0.25):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    d = d + ndi.gaussian_filter(rng.normal(0, 1, shape), 2.0) * rough
    return d <= 1.0


def make_demo_scene(
    seed: int = 7, field_id: str = "DEMO-01", date: str = "2026-10-28",
    cloud_fraction: float = 0.0, with_s1: bool = True, s1_noise_db: float = 0.9,
) -> Scene:
    rng = np.random.default_rng(seed)
    geom_wgs = demo_field_geometry()
    series = gpd.GeoSeries([geom_wgs], crs="EPSG:4326")
    utm = series.estimate_utm_crs()
    geom_utm = series.to_crs(utm).iloc[0]
    minx, miny, maxx, maxy = geom_utm.bounds
    r = 10.0
    minx, miny = np.floor(minx / r) * r - r, np.floor(miny / r) * r - r
    maxx, maxy = np.ceil(maxx / r) * r + r, np.ceil(maxy / r) * r + r
    grid = Grid(utm.to_epsg(), (minx, miny, maxx, maxy), int(round((maxx - minx) / r)), int(round((maxy - miny) / r)), r)
    shape = grid.shape
    fmask = polygon_mask(geom_utm, grid)

    # --- truth layout (fixed layout, jittered by seed) -------------------------------------------
    truth = np.zeros(shape, dtype=np.uint8)
    H, W = shape
    thin = _blob(shape, H * 0.80, W * 0.20, H * 0.12, W * 0.16, rng)
    truth[thin] = 3
    for cy, cx, ry, rx in [(.22, .25, 4, 7), (.40, .70, 6, 4), (.70, .55, 3, 9), (.15, .82, 5, 5)]:
        truth[_blob(shape, H * cy + rng.normal(0, 1), W * cx + rng.normal(0, 1), ry, rx, rng)] = 1
    for cy, cx, ry, rx in [(.30, .50, 5, 8), (.58, .22, 7, 5), (.82, .75, 4, 6), (.50, .88, 6, 4)]:
        truth[_blob(shape, H * cy + rng.normal(0, 1), W * cx + rng.normal(0, 1), ry, rx, rng)] = 2
    truth[~fmask] = 0

    # --- mixed-pixel fractions: blur one-hot so edges are not perfectly sharp ---------------------
    frac = np.stack([ndi.gaussian_filter((truth == k).astype(np.float32), 0.7) for k in range(4)], axis=0)
    frac /= np.maximum(frac.sum(axis=0, keepdims=True), 1e-6)

    bands = {}
    for bi, b in enumerate(S2_BANDS):
        refl = sum(frac[k] * _SPECTRA[k][bi] for k in range(4))
        refl = refl * (1 + rng.normal(0, 0.03, shape)) + rng.normal(0, 0.004, shape)  # texture + sensor noise
        bands[b] = np.clip(refl, 0.001, 1.0).astype(np.float32)

    # --- optional cloud + shadow -----------------------------------------------------------------
    scl = np.where(fmask, 4, 0).astype(np.uint8)
    scl[(truth == 1) & fmask] = 5
    if cloud_fraction > 0:
        cloud = _blob(shape, H * 0.3, W * 0.65, H * np.sqrt(cloud_fraction) * 0.9, W * np.sqrt(cloud_fraction) * 0.9, rng, 0.4)
        scl[cloud] = 9
        for b in S2_BANDS:
            bands[b][cloud] = np.nan
    valid_data = fmask.copy()
    for b in S2_BANDS:
        bands[b] = np.where(valid_data, bands[b], np.nan).astype(np.float32)
    clear = fmask & ~np.isin(scl, (3, 8, 9, 10))

    scene = Scene(field_id=field_id, date=date, grid=grid, bands=bands, field_mask=fmask, clear_mask=clear,
                  scl=scl, boundary_utm=geom_utm, cloud_catalog=round(cloud_fraction * 100, 1), synthetic=True,
                  truth=truth, meta={"source": "SYNTHETIC demo data", "geometry_wgs84": geom_wgs})

    if with_s1:
        scene.s1 = make_demo_s1(truth, fmask, rng, s1_noise_db)
    return scene


def make_demo_s1(truth, fmask, rng, noise_db: float = 0.9) -> dict:
    frac = np.stack([ndi.gaussian_filter((truth == k).astype(np.float32), 0.7) for k in range(4)], axis=0)
    frac /= np.maximum(frac.sum(axis=0, keepdims=True), 1e-6)
    vv_db = sum(frac[k] * _S1_DB[k][0] for k in range(4)) + rng.normal(0, noise_db, truth.shape)
    vh_db = sum(frac[k] * _S1_DB[k][1] for k in range(4)) + rng.normal(0, noise_db, truth.shape)
    vv = np.where(fmask, 10 ** (vv_db / 10), np.nan).astype(np.float32)
    vh = np.where(fmask, 10 ** (vh_db / 10), np.nan).astype(np.float32)
    return {"VV": vv, "VH": vh, "n_scenes": 3, "dates": ["synthetic"] * 3, "direction": "ASCENDING", "speckle_px": 3}


def make_demo_history(n: int = 5, end_date: str = "2026-10-28", step_days: int = 5) -> list[Scene]:
    """A few demo dates (newest first); one is cloudy so 'latest clear scene' logic is demonstrable."""
    import pandas as pd
    end = pd.Timestamp(end_date)
    out = []
    for i in range(n):
        d = (end - pd.Timedelta(days=step_days * i)).strftime("%Y-%m-%d")
        cloud = 0.35 if i == 1 else (0.05 if i == 3 else 0.0)
        out.append(make_demo_scene(seed=7 + i, date=d, cloud_fraction=cloud, with_s1=True))
    return out


def inner_mask(scene: Scene, buffer_m: float = 10.0) -> np.ndarray:
    return erode_mask(scene.field_mask, buffer_m, scene.grid.resolution)
