"""
src/imagery.py

Pixel-level Sentinel-2 L2A + Sentinel-1 GRD retrieval for single fields, built
on the same Sentinel Hub Process API / OAuth setup as `src/s1_growth_curves.py`.

Where `s1_growth_curves.py` reduces each acquisition to one field-mean number,
this module keeps the *pixels*, so the dashboard can show band images, compute
indices with the raster calculator, and classify canopy gaps.

Key design points
-----------------
* Sentinel-2 and Sentinel-1 are requested on the SAME UTM grid (bounds snapped to
  the 10 m pixel lattice), so S1 and S2 are pixel-registered with no resampling.
* Collections are re-defined with `service_url=config.sh_base_url`. The stock
  `DataCollection.SENTINEL2_L2A` constants pin services.sentinel-hub.com and the
  library lets that override the config, which would send Copernicus Data Space
  (CDSE) credentials to the wrong server.
* Request builders are pure functions (no network) so they can be unit-tested.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from scipy import ndimage as ndi
from shapely.geometry import shape as shapely_shape

from sentinelhub import BBox, CRS, DataCollection, Geometry, MimeType, SentinelHubCatalog, SentinelHubRequest, SHConfig

from .raster_calc import S2_BANDS
from .s1_growth_curves import cdse_collection, download_tolerant

logger = logging.getLogger("imagery")

RESOLUTION_M = 10
TIME_BUFFER_MINUTES = 5
MAX_DIM = 2500
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)  # nodata, saturated, cloud shadow, cloud (med/high), cirrus, snow

EVALSCRIPT_S2 = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","SCL","dataMask"] }],
    output: { id: "default", bands: 14, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B01,s.B02,s.B03,s.B04,s.B05,s.B06,s.B07,s.B08,s.B8A,s.B09,s.B11,s.B12,s.SCL,s.dataMask];
}
"""

EVALSCRIPT_S1 = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"] }],
    output: { id: "default", bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) { return [s.VV, s.VH, s.dataMask]; }
"""


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class Grid:
    """A north-up projected raster grid shared by S2, S1 and all derived layers."""
    epsg: int
    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy (projected metres)
    width: int
    height: int
    resolution: float = RESOLUTION_M
    resolution_y: float | None = None   # only set for fine (display/export) grids; analysis grids are square

    @property
    def transform(self) -> Affine:
        return from_bounds(*self.bounds, self.width, self.height)

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_area_ha(self) -> float:
        return self.resolution * (self.resolution_y or self.resolution) / 10_000.0


@dataclass
class Scene:
    """One field, one Sentinel-2 date, optional Sentinel-1 composite on the same grid."""
    field_id: str
    date: str
    grid: Grid
    bands: dict[str, np.ndarray]          # B01..B12 reflectance, NaN where invalid
    field_mask: np.ndarray                # inside the field polygon
    clear_mask: np.ndarray                # inside field AND cloud/shadow free AND valid data
    scl: np.ndarray | None = None
    s1: dict | None = None                # {"VV","VH" (linear), "n_scenes", "dates"}
    boundary_utm: object | None = None    # shapely geometry in grid CRS
    cloud_catalog: float | None = None
    synthetic: bool = False
    truth: np.ndarray | None = None       # only for synthetic demo scenes
    meta: dict = field(default_factory=dict)

    @property
    def clear_fraction(self) -> float:
        n = int(self.field_mask.sum())
        return float(self.clear_mask.sum() / n) if n else 0.0

    @property
    def field_area_ha(self) -> float:
        return float(self.field_mask.sum() * self.grid.pixel_area_ha)


# --------------------------------------------------------------------------
# Collections / grid / masks
# --------------------------------------------------------------------------

def s2_collection(config: SHConfig) -> DataCollection:
    return cdse_collection(DataCollection.SENTINEL2_L2A, config, "s2_l2a_user_endpoint")


def s1_collection(config: SHConfig, direction: str = "ASCENDING") -> DataCollection:
    base = DataCollection.SENTINEL1_IW_ASC if direction.upper().startswith("ASC") else DataCollection.SENTINEL1_IW_DES
    return cdse_collection(base, config, f"s1_{direction[:3].lower()}_user_endpoint")


def make_grid(geom_wgs84, resolution: float = RESOLUTION_M, pad_px: int = 1) -> tuple[Grid, object]:
    """Grid covering ``geom_wgs84`` in its UTM zone, snapped to the resolution lattice.

    Snapping to multiples of 10 m lines the grid up with Sentinel-2's native pixel
    boundaries (S2 tiles are origin-aligned to 10 m in UTM), avoiding sub-pixel shifts.
    Returns (grid, geometry_in_grid_crs).
    """
    series = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326")
    utm = series.estimate_utm_crs()
    geom_utm = series.to_crs(utm).iloc[0]
    minx, miny, maxx, maxy = geom_utm.bounds
    r = resolution
    minx = np.floor(minx / r) * r - pad_px * r
    miny = np.floor(miny / r) * r - pad_px * r
    maxx = np.ceil(maxx / r) * r + pad_px * r
    maxy = np.ceil(maxy / r) * r + pad_px * r
    width, height = int(round((maxx - minx) / r)), int(round((maxy - miny) / r))
    if width < 3 or height < 3:
        raise ValueError("Field is too small to resolve at 10 m.")
    if width > MAX_DIM or height > MAX_DIM:
        raise ValueError(f"Field is too large for one Process API request ({width}x{height} px; limit {MAX_DIM}).")
    return Grid(utm.to_epsg(), (minx, miny, maxx, maxy), width, height, r), geom_utm


def apply_affine(t: Affine, x: float, y: float) -> tuple[float, float]:
    """Apply an Affine to a point (works across `affine` versions; `t * (x, y)` is deprecated in 3.x)."""
    return t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f


def polygon_mask(geom_utm, grid: Grid, all_touched: bool = False) -> np.ndarray:
    return geometry_mask([geom_utm], grid.shape, grid.transform, invert=True, all_touched=all_touched)


def erode_mask(mask: np.ndarray, buffer_m: float, resolution: float = RESOLUTION_M) -> np.ndarray:
    """Shrink a mask by ``buffer_m`` (removes mixed boundary / headland pixels)."""
    it = int(round(buffer_m / resolution))
    if it <= 0:
        return mask.copy()
    return ndi.binary_erosion(mask, iterations=it, border_value=0)


def clear_from_scl(scl: np.ndarray | None, shape) -> np.ndarray:
    if scl is None:
        return np.ones(shape, dtype=bool)
    return ~np.isin(scl.astype(np.int16), SCL_BAD)


# --------------------------------------------------------------------------
# Request builders (pure)
# --------------------------------------------------------------------------

def _time_window(dt: pd.Timestamp) -> tuple[str, str]:
    d = pd.Timestamp(dt)
    return ((d - pd.Timedelta(minutes=TIME_BUFFER_MINUTES)).isoformat(),
            (d + pd.Timedelta(minutes=TIME_BUFFER_MINUTES)).isoformat())


def build_s2_request(grid: Grid, geom_utm, acquisition_dt, config: SHConfig, upsampling: str = "BILINEAR") -> SentinelHubRequest:
    t0, t1 = _time_window(acquisition_dt)
    crs = CRS(grid.epsg)
    return SentinelHubRequest(
        evalscript=EVALSCRIPT_S2,
        input_data=[SentinelHubRequest.input_data(
            data_collection=s2_collection(config), time_interval=(t0, t1),
            other_args={"processing": {"upsampling": upsampling, "downsampling": "NEAREST"}},
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox(grid.bounds, crs=crs), geometry=Geometry(geom_utm, crs=crs),
        size=(grid.width, grid.height), config=config,
    )


def build_s1_request(grid: Grid, geom_utm, acquisition_dt, config: SHConfig, direction: str = "ASCENDING") -> SentinelHubRequest:
    """Same processing options as the existing growth-curve engine (sigma0 ellipsoid, orthorectified)."""
    t0, t1 = _time_window(acquisition_dt)
    crs = CRS(grid.epsg)
    return SentinelHubRequest(
        evalscript=EVALSCRIPT_S1,
        input_data=[SentinelHubRequest.input_data(
            data_collection=s1_collection(config, direction), time_interval=(t0, t1),
            other_args={"processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True}},
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox(grid.bounds, crs=crs), geometry=Geometry(geom_utm, crs=crs),
        size=(grid.width, grid.height), config=config,
    )


# --------------------------------------------------------------------------
# Catalog search
# --------------------------------------------------------------------------

def _search(collection, bounds_wgs84, start: str, end: str, config: SHConfig, fields_extra: list[str]) -> list[dict]:
    catalog = SentinelHubCatalog(config=config)
    bbox = BBox(bbox=tuple(bounds_wgs84), crs=CRS.WGS84)
    try:
        return list(catalog.search(
            collection=collection, bbox=bbox, time=(start, end),
            fields={"include": ["id", "properties.datetime", "geometry", *fields_extra], "exclude": []},
        ))
    except Exception as exc:
        raise RuntimeError(f"Sentinel Hub catalog search failed: {exc}") from exc


def search_s2_scenes(geom_wgs84, start: str, end: str, config: SHConfig, max_cloud: float = 100.0) -> pd.DataFrame:
    """Sentinel-2 L2A acquisitions intersecting the field, newest first (one row per date)."""
    items = _search(s2_collection(config), geom_wgs84.bounds, start, end, config, ["properties.eo:cloud_cover"])
    rows = []
    for it in items:
        if not shapely_shape(it["geometry"]).intersects(geom_wgs84):
            continue
        cc = it["properties"].get("eo:cloud_cover")
        if cc is not None and cc > max_cloud:
            continue
        dt = pd.to_datetime(it["properties"]["datetime"])
        rows.append({"id": it["id"], "datetime": dt, "date": dt.strftime("%Y-%m-%d"), "cloud_cover_pct": cc})
    df = pd.DataFrame(rows, columns=["id", "datetime", "date", "cloud_cover_pct"])
    if df.empty:
        return df
    return df.sort_values("datetime", ascending=False).drop_duplicates("date").reset_index(drop=True)


def search_s1_scenes(geom_wgs84, start: str, end: str, config: SHConfig, direction: str = "ASCENDING") -> pd.DataFrame:
    items = _search(s1_collection(config, direction), geom_wgs84.bounds, start, end, config, [])
    rows = [{"id": it["id"], "datetime": pd.to_datetime(it["properties"]["datetime"])}
            for it in items if shapely_shape(it["geometry"]).intersects(geom_wgs84)]
    df = pd.DataFrame(rows, columns=["id", "datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


# --------------------------------------------------------------------------
# Download + decode
# --------------------------------------------------------------------------

def _download(requests: list[SentinelHubRequest], config: SHConfig, max_threads: int = 4) -> list:
    """Run Process API requests; failed ones come back as None (see s1_growth_curves.download_tolerant)."""
    try:
        return download_tolerant([r.download_list[0] for r in requests], config, max_threads=max_threads)
    except Exception as exc:
        raise RuntimeError(f"Failed downloading from Sentinel Hub: {exc}") from exc


def decode_s2_array(arr: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """(H,W,14) Process API output -> (bands, SCL uint8, dataMask bool). Invalid pixels become NaN."""
    if arr.ndim != 3 or arr.shape[2] != 14:
        raise ValueError(f"Unexpected Sentinel-2 response shape {arr.shape}; expected (H, W, 14).")
    data_mask = arr[:, :, 13] > 0.5
    bands = {}
    for i, name in enumerate(S2_BANDS):
        b = arr[:, :, i].astype(np.float32)
        bands[name] = np.where(data_mask & np.isfinite(b), b, np.nan).astype(np.float32)
    scl = np.where(data_mask, arr[:, :, 12], 0).astype(np.uint8)
    return bands, scl, data_mask


def fetch_s2_scene(field_id: str, geom_wgs84, acquisition_dt, config: SHConfig, cloud_catalog: float | None = None) -> Scene:
    grid, geom_utm = make_grid(geom_wgs84)
    req = build_s2_request(grid, geom_utm, acquisition_dt, config)
    arr = _download([req], config, 1)[0]
    if arr is None:
        raise RuntimeError(f"No Sentinel-2 data returned for {field_id} on {pd.Timestamp(acquisition_dt).date()}.")
    bands, scl, data_mask = decode_s2_array(arr)
    fmask = polygon_mask(geom_utm, grid)
    clear = fmask & data_mask & clear_from_scl(scl, grid.shape)
    return Scene(field_id=str(field_id), date=pd.Timestamp(acquisition_dt).strftime("%Y-%m-%d"), grid=grid,
                 bands=bands, field_mask=fmask, clear_mask=clear, scl=scl, boundary_utm=geom_utm,
                 cloud_catalog=cloud_catalog, meta={"source": "Sentinel-2 L2A (Sentinel Hub Process API)",
                                                    "geometry_wgs84": geom_wgs84, "acquisition_dt": pd.Timestamp(acquisition_dt)})


def find_latest_clear_scene(
    field_id: str, geom_wgs84, scenes_df: pd.DataFrame, config: SHConfig,
    min_clear_fraction: float = 0.9, max_tries: int = 6,
    progress: Callable[[str], None] | None = None,
) -> Scene:
    """Walk newest->oldest until a scene has >= ``min_clear_fraction`` clear pixels *inside this field*.

    Catalog cloud cover is tile-wide; a field can be clear under a 60 % cloudy tile (and vice versa),
    so the decision is made on the field's own SCL mask.
    """
    best: Scene | None = None
    last_error: Exception | None = None
    for _, row in scenes_df.head(max_tries).iterrows():
        if progress:
            progress(f"Checking {row['date']} ...")
        try:
            sc = fetch_s2_scene(field_id, geom_wgs84, row["datetime"], config, row.get("cloud_cover_pct"))
        except Exception as exc:  # a single bad date shouldn't abort the search
            logger.warning("Skipping %s: %s", row["date"], exc)
            last_error = exc
            continue
        if best is None or sc.clear_fraction > best.clear_fraction:
            best = sc
        if sc.clear_fraction >= min_clear_fraction:
            return sc
    if best is None:
        raise RuntimeError("Could not download any Sentinel-2 scene for this field/date range."
                           + (f" Last error: {last_error}" if last_error else ""))
    best.meta["warning"] = (f"No scene reached {min_clear_fraction:.0%} clear pixels inside the field; "
                            f"using the clearest ({best.date}: {best.clear_fraction:.0%} clear).")
    return best


# --------------------------------------------------------------------------
# Sentinel-1 composite
# --------------------------------------------------------------------------

def nan_boxcar(a: np.ndarray, size: int) -> np.ndarray:
    """Mean filter that ignores NaNs (so field edges / masked pixels don't bleed in)."""
    if size <= 1:
        return a
    valid = np.isfinite(a).astype(np.float32)
    num = ndi.uniform_filter(np.where(valid > 0, a, 0).astype(np.float32), size, mode="nearest")
    den = ndi.uniform_filter(valid, size, mode="nearest")
    with np.errstate(all="ignore"):
        out = np.where(den >= 0.34, num / den, np.nan)
    return np.where(valid > 0, out, np.nan).astype(np.float32)  # keep original footprint


def composite_s1(stack_vv: list[np.ndarray], stack_vh: list[np.ndarray], speckle_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Temporal median (linear power) + NaN-aware boxcar. Both reduce speckle without leaking class edges much."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices = pixels outside the field
        vv = np.nanmedian(np.stack(stack_vv), axis=0)
        vh = np.nanmedian(np.stack(stack_vh), axis=0)
    return nan_boxcar(vv, speckle_px), nan_boxcar(vh, speckle_px)


def fetch_s1_composite(
    scene: Scene, config: SHConfig, window_days: int = 12, direction: str = "ASCENDING",
    max_scenes: int = 4, speckle_px: int = 3, min_coverage: float = 0.5, max_threads: int = 4,
) -> dict:
    """Pull Sentinel-1 acquisitions within +/- ``window_days`` of the S2 date and composite them."""
    geom = scene.meta.get("geometry_wgs84")
    if geom is None:
        raise ValueError("Scene has no WGS84 geometry; cannot search Sentinel-1.")
    d0 = pd.Timestamp(scene.date)
    cat = search_s1_scenes(geom, (d0 - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d"),
                           (d0 + pd.Timedelta(days=window_days + 1)).strftime("%Y-%m-%d"), config, direction)
    if cat.empty:
        raise ValueError(f"No Sentinel-1 {direction.lower()} acquisitions within +/-{window_days} days of {scene.date}.")
    cat["dt_abs"] = (cat["datetime"].dt.tz_localize(None) - d0).abs()
    cat = cat.sort_values("dt_abs").head(max_scenes)

    reqs = [build_s1_request(scene.grid, scene.boundary_utm, row["datetime"], config, direction) for _, row in cat.iterrows()]
    arrays = _download(reqs, config, max_threads)

    vvs, vhs, used = [], [], []
    for (_, row), arr in zip(cat.iterrows(), arrays):
        if arr is None:
            continue
        mask = arr[:, :, 2] > 0.5
        if scene.field_mask.sum() == 0 or (mask & scene.field_mask).sum() / scene.field_mask.sum() < min_coverage:
            continue
        vv, vh = arr[:, :, 0].astype(np.float32), arr[:, :, 1].astype(np.float32)
        ok = mask & (vv > 0) & (vh > 0)
        vvs.append(np.where(ok, vv, np.nan)); vhs.append(np.where(ok, vh, np.nan))
        used.append(row["datetime"].strftime("%Y-%m-%d"))
    if not vvs:
        raise ValueError("Sentinel-1 scenes were found but none covered enough of the field.")
    vv, vh = composite_s1(vvs, vhs, speckle_px)
    return {"VV": vv, "VH": vh, "n_scenes": len(vvs), "dates": used, "direction": direction, "speckle_px": speckle_px}


# --------------------------------------------------------------------------
# Fine-resolution (display / export) rasters - Copernicus-Browser style "custom resolution"
# --------------------------------------------------------------------------
# IMPORTANT: Sentinel-2 is 10 m (B02/B03/B04/B08), 20 m (red-edge, SWIR) or 60 m (B01/B09). Asking the Process API for
# 1 m/px RESAMPLES those pixels (nearest / bilinear / bicubic). The picture is smoother and easier to read, and exports
# match what the Copernicus Browser produces, but no new ground detail is created. Gap detection always runs on the
# native 10 m grid, never on these interpolated rasters.

UPSAMPLING_METHODS = ("BICUBIC", "BILINEAR", "NEAREST")


@dataclass
class FineRaster:
    grid: Grid
    bands: dict[str, np.ndarray]     # float32 reflectance, NaN where no data
    valid: np.ndarray                # inside the field polygon and data present (fine resolution)
    clear: np.ndarray                # valid AND cloud/shadow free (10 m SCL mask resampled by nearest)
    res_x: float
    res_y: float
    upsampling: str
    capped: bool = False             # requested resolution was coarsened to respect the 2,500 px limit
    synthetic: bool = False
    date: str = ""


def plan_fine_grid(grid: Grid, res_x: float, res_y: float | None = None, max_dim: int = MAX_DIM) -> tuple[Grid, float, float, bool]:
    """Same bounds as ``grid`` at ``res_x`` x ``res_y`` metres/pixel, coarsened if it would exceed the Process API size limit."""
    res_y = res_y or res_x
    if res_x <= 0 or res_y <= 0:
        raise ValueError("Resolution must be > 0 m/px.")
    minx, miny, maxx, maxy = grid.bounds
    wm, hm = maxx - minx, maxy - miny
    rx, ry = max(res_x, wm / max_dim), max(res_y, hm / max_dim)
    rx, ry = float(np.ceil(rx * 100) / 100), float(np.ceil(ry * 100) / 100)
    fine = Grid(grid.epsg, grid.bounds, int(round(wm / rx)), int(round(hm / ry)), rx, ry)
    return fine, rx, ry, (rx > res_x + 1e-9 or ry > res_y + 1e-9)


def evalscript_for_bands(bands: list[str]) -> str:
    for b in bands:
        if b not in S2_BANDS:
            raise ValueError(f"Unknown Sentinel-2 band {b}.")
    quoted = ",".join(f'"{b}"' for b in bands)
    returned = ",".join(f"s.{b}" for b in bands)
    return (f'//VERSION=3\nfunction setup() {{\n  return {{ input: [{{ bands: [{quoted},"dataMask"] }}],\n'
            f'    output: {{ id: "default", bands: {len(bands) + 1}, sampleType: "FLOAT32" }} }};\n}}\n'
            f'function evaluatePixel(s) {{ return [{returned},s.dataMask]; }}\n')


def _acq_window(scene: "Scene") -> tuple[str, str]:
    dt = scene.meta.get("acquisition_dt")
    if dt is not None:
        return _time_window(dt)
    d = pd.Timestamp(scene.date)
    return d.isoformat(), (d + pd.Timedelta(hours=23, minutes=59, seconds=59)).isoformat()


def build_fine_request(scene: "Scene", fine_grid: Grid, config: SHConfig, bands: list[str], upsampling: str = "BICUBIC") -> SentinelHubRequest:
    if upsampling not in UPSAMPLING_METHODS:
        raise ValueError(f"upsampling must be one of {UPSAMPLING_METHODS}")
    t0, t1 = _acq_window(scene)
    crs = CRS(fine_grid.epsg)
    return SentinelHubRequest(
        evalscript=evalscript_for_bands(bands),
        input_data=[SentinelHubRequest.input_data(
            data_collection=s2_collection(config), time_interval=(t0, t1),
            other_args={"processing": {"upsampling": upsampling, "downsampling": "NEAREST"}})],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox(fine_grid.bounds, crs=crs), geometry=Geometry(scene.boundary_utm, crs=crs),
        size=(fine_grid.width, fine_grid.height), config=config)


def resample_mask_nearest(mask: np.ndarray, coarse: Grid, fine: Grid) -> np.ndarray:
    """Nearest-neighbour a coarse (10 m) boolean mask onto the fine grid (works for non-integer ratios)."""
    rows = np.clip(((np.arange(fine.height) + 0.5) * (fine.resolution_y or fine.resolution) / (coarse.resolution_y or coarse.resolution)).astype(int), 0, coarse.height - 1)
    cols = np.clip(((np.arange(fine.width) + 0.5) * fine.resolution / coarse.resolution).astype(int), 0, coarse.width - 1)
    return mask[np.ix_(rows, cols)]


def fetch_s2_fine(scene: "Scene", config: SHConfig, bands: list[str], res_x: float = 1.0, res_y: float | None = None,
                  upsampling: str = "BICUBIC") -> FineRaster:
    """Download ``bands`` for the scene's date at ``res_x`` x ``res_y`` m/px (resampled from native resolution)."""
    fine_grid, rx, ry, capped = plan_fine_grid(scene.grid, res_x, res_y)
    arr = _download([build_fine_request(scene, fine_grid, config, bands, upsampling)], config, 1)[0]
    if arr is None:
        raise RuntimeError(f"No Sentinel-2 data returned for {scene.field_id} on {scene.date} at {rx:g} m/px.")
    if arr.ndim != 3 or arr.shape[2] != len(bands) + 1:
        raise ValueError(f"Unexpected response shape {arr.shape}; expected (H, W, {len(bands) + 1}).")
    data_mask = arr[:, :, -1] > 0.5
    out = {b: np.where(data_mask & np.isfinite(arr[:, :, i]), arr[:, :, i], np.nan).astype(np.float32) for i, b in enumerate(bands)}
    inside = polygon_mask(scene.boundary_utm, fine_grid)
    valid = inside & data_mask
    clear = valid & resample_mask_nearest(scene.clear_mask | ~scene.field_mask, scene.grid, fine_grid)
    return FineRaster(fine_grid, out, valid, clear, rx, ry, upsampling, capped, False, scene.date)


def write_bands_geotiff(fine: FineRaster, band_names: list[str], crs: str = "UTM", nodata: float = -9999.0,
                        extra_tags: dict | None = None) -> bytes:
    """Multi-band 32-bit float GeoTIFF ("Analytical" download). ``crs``: "UTM" (metres) or "EPSG:4326" (WGS 84)."""
    import rasterio
    from rasterio.io import MemoryFile
    g = fine.grid
    data = np.stack([np.where(fine.valid, fine.bands[b], np.nan) for b in band_names]).astype(np.float32)
    tags = {"source": "SYNTHETIC demo data" if fine.synthetic else "Sentinel-2 L2A via Sentinel Hub Process API",
            "native_resolution": "B02/B03/B04/B08 10 m; B05-B07/B8A/B11/B12 20 m; B01/B09 60 m",
            "resampling": fine.upsampling, "requested_resolution_m": f"{fine.res_x:g} x {fine.res_y:g}",
            "note": "Resampled from native resolution: smoother, not more detailed.", "date": fine.date, **(extra_tags or {})}
    if crs.upper().startswith("EPSG:4326") or crs.upper().startswith("WGS"):
        minx, miny, maxx, maxy = g.bounds
        lon_min, lat_min, lon_max, lat_max = transform_bounds(g.crs, "EPSG:4326", minx, miny, maxx, maxy)
        lat_c = (lat_min + lat_max) / 2
        res = (fine.res_x / (111_320 * np.cos(np.radians(lat_c))), fine.res_y / 110_574)
        dst_t, dw, dh = calculate_default_transform(g.crs, "EPSG:4326", g.width, g.height, minx, miny, maxx, maxy, resolution=res)
        out = np.full((len(band_names), dh, dw), np.nan, dtype=np.float32)
        for i in range(len(band_names)):
            reproject(data[i], out[i], src_transform=g.transform, src_crs=g.crs, dst_transform=dst_t, dst_crs="EPSG:4326",
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        arr, transform, crs_out, h, w = out, dst_t, "EPSG:4326", dh, dw
    else:
        arr, transform, crs_out, h, w = data, g.transform, g.crs, g.height, g.width
    arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=h, width=w, count=len(band_names), dtype="float32", crs=crs_out, transform=transform,
                      nodata=nodata, compress="deflate", predictor=3, tiled=False) as dst:
            dst.write(arr)
            for i, b in enumerate(band_names, 1):
                dst.set_band_description(i, b)
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
        return mem.read()
