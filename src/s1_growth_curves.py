"""
src/s1_growth_curves.py

Field-level Sentinel-1 (GRD, IW, ASCENDING) VH/VV growth-curve engine,
built on the Sentinel Hub Process API. This is the importable version
used by `pages/1_S1_Growth_Curves.py` — no argparse/CLI/plt.show() here,
just functions that return data and figures.

See that Streamlit page for how it's wired up to the dashboard.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress
from shapely.geometry import shape

from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    Geometry,
    MimeType,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

SCALE_M = 10  # metres
MIN_COVERAGE = 0.5
SG_WINDOW = 7
SG_POLYORDER = 2
GROWTH_WINDOW_DAYS = 90
MAX_PROCESS_API_DIM = 2500  # hard Process API limit, pixels per side
TIME_BUFFER_MINUTES = 5

EVALSCRIPT_S1 = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"] }],
    output: { id: "default", bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH, sample.dataMask];
}
"""

logger = logging.getLogger("s1_growth_curves")


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def authenticate_sentinel_hub(client_id: str, client_secret: str, base_url: str, token_url: str) -> SHConfig:
    """Build and validate an SHConfig via OAuth client-credentials. Raises RuntimeError on failure."""
    if not client_id or not client_secret:
        raise RuntimeError("Sentinel Hub client ID/secret are required.")

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = base_url
    config.sh_token_url = token_url

    try:
        SentinelHubCatalog(config=config).get_collections()
    except Exception as exc:
        raise RuntimeError(f"could not authenticate with Sentinel Hub: {exc}") from exc

    return config


# --------------------------------------------------------------------------
# Field loading / validation
# --------------------------------------------------------------------------

ID_CANDIDATES = ["field_id", "block_id", "id", "ID", "Id", "name", "Name", "field_name", "FID", "fid"]
DEFAULT_CROP_TYPE = "unspecified"


def load_fields(geojson_path: str) -> gpd.GeoDataFrame:
    """Load, validate, and clean field boundaries from a GeoJSON file.

    Only a valid geometry is strictly required. field_id is auto-detected
    from common candidate property names (or auto-generated if none match),
    and crop_type defaults to "unspecified" if absent — both are filled in
    rather than rejected, since crop_type is purely descriptive here (it
    doesn't gate the Sentinel-1 fetch itself, only downstream RVI baseline
    lookups elsewhere in the app).
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as exc:
        raise ValueError(f"failed to parse GeoJSON file '{geojson_path}': {exc}") from exc

    if gdf.crs is None:
        logger.warning("GeoJSON has no CRS defined; assuming EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        logger.warning(f"GeoJSON CRS is '{gdf.crs}', not EPSG:4326; reprojecting automatically.")
        gdf = gdf.to_crs(epsg=4326)

    n_before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if len(gdf) < n_before:
        logger.warning(f"Dropped {n_before - len(gdf)} field(s) with empty/missing geometries.")

    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    if gdf.empty:
        raise ValueError("no valid field geometries remain after cleaning.")
    gdf = gdf.reset_index(drop=True)

    id_col = next((c for c in ID_CANDIDATES if c in gdf.columns), None)
    if id_col:
        gdf["field_id"] = gdf[id_col].astype(str)
        if gdf["field_id"].isna().any() or (gdf["field_id"] == "").any() or (gdf["field_id"] == "nan").any():
            gdf["field_id"] = [
                fid if fid not in (None, "", "nan") else f"field_{i + 1}" for i, fid in enumerate(gdf["field_id"])
            ]
    else:
        gdf["field_id"] = [f"field_{i + 1}" for i in range(len(gdf))]
        logger.warning("No name/ID property found in the GeoJSON — auto-generated field_id values were assigned.")

    if "crop_type" not in gdf.columns:
        gdf["crop_type"] = DEFAULT_CROP_TYPE
        logger.warning(f"No crop_type property found in the GeoJSON — defaulted to '{DEFAULT_CROP_TYPE}'.")
    else:
        gdf["crop_type"] = gdf["crop_type"].fillna(DEFAULT_CROP_TYPE).replace("", DEFAULT_CROP_TYPE)

    equal_area = gdf.to_crs("EPSG:6933")
    gdf["area_m2"] = equal_area.geometry.area

    return gdf.reset_index(drop=True)


# --------------------------------------------------------------------------
# Sentinel-1 catalog search
# --------------------------------------------------------------------------

def search_sentinel1_catalog(fields_gdf: gpd.GeoDataFrame, start_date: str, end_date: str, config: SHConfig) -> list:
    """One Catalog search over the whole AOI for Sentinel-1 IW/ASCENDING footprints."""
    catalog = SentinelHubCatalog(config=config)
    minx, miny, maxx, maxy = fields_gdf.total_bounds
    search_bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    try:
        items = list(
            catalog.search(
                collection=DataCollection.SENTINEL1_IW_ASC,
                bbox=search_bbox,
                time=(start_date, end_date),
                fields={"include": ["id", "properties.datetime", "geometry"], "exclude": []},
            )
        )
    except Exception as exc:
        raise RuntimeError(f"Sentinel Hub catalog search failed: {exc}") from exc

    if not items:
        raise ValueError("no Sentinel-1 (IW, ASCENDING) products found for the given date range/region.")

    return [
        {"id": item["id"], "datetime": pd.to_datetime(item["properties"]["datetime"]), "footprint": shape(item["geometry"])}
        for item in items
    ]


def match_field_acquisitions(fields_gdf: gpd.GeoDataFrame, acquisitions: list) -> dict:
    """Keep only acquisitions whose footprint actually intersects each field."""
    field_acqs = {}
    for _, row in fields_gdf.iterrows():
        matches = sorted(
            (a for a in acquisitions if a["footprint"].intersects(row.geometry)),
            key=lambda a: a["datetime"],
        )
        field_acqs[row["field_id"]] = matches
    return field_acqs


# --------------------------------------------------------------------------
# Process API request construction
# --------------------------------------------------------------------------

def build_process_requests(fields_gdf: gpd.GeoDataFrame, field_acqs: dict, config: SHConfig, scale: float = SCALE_M) -> list:
    """Build one clipped Process API request per (field, acquisition) pair."""
    requests_info = []

    for _, row in fields_gdf.iterrows():
        field_id, crop_type, area_m2 = row["field_id"], row["crop_type"], row["area_m2"]
        acqs = field_acqs.get(field_id, [])
        if not acqs:
            continue

        field_series = gpd.GeoSeries([row.geometry], crs="EPSG:4326")
        utm_crs = field_series.estimate_utm_crs()
        geom_utm = field_series.to_crs(utm_crs).iloc[0]

        bbox = BBox(geom_utm.bounds, crs=CRS(utm_crs.to_epsg()))
        width, height = bbox_to_dimensions(bbox, resolution=scale)

        if width == 0 or height == 0:
            logger.warning(f"Field {field_id} is too small to resolve at {scale} m resolution; skipping.")
            continue
        if width > MAX_PROCESS_API_DIM or height > MAX_PROCESS_API_DIM:
            logger.warning(f"Field {field_id} exceeds the Process API size limit ({width}x{height}); skipping.")
            continue

        geometry = Geometry(geom_utm, crs=CRS(utm_crs.to_epsg()))

        for acq in acqs:
            dt = acq["datetime"]
            t_from = (dt - pd.Timedelta(minutes=TIME_BUFFER_MINUTES)).isoformat()
            t_to = (dt + pd.Timedelta(minutes=TIME_BUFFER_MINUTES)).isoformat()

            request = SentinelHubRequest(
                evalscript=EVALSCRIPT_S1,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL1_IW_ASC,
                        time_interval=(t_from, t_to),
                        other_args={"processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True}},
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                geometry=geometry,
                size=(width, height),
                config=config,
            )
            requests_info.append(
                {"field_id": field_id, "crop_type": crop_type, "area_m2": area_m2, "date": dt.strftime("%Y-%m-%d"), "request": request}
            )

    return requests_info


# --------------------------------------------------------------------------
# Coverage + extraction
# --------------------------------------------------------------------------

def calculate_coverage(mask: np.ndarray, scale: float, field_area_m2: float) -> float:
    """Fraction of the field's true area covered by valid (in-swath, in-polygon) pixels."""
    if field_area_m2 <= 0:
        return 0.0
    return (int(mask.sum()) * scale * scale) / field_area_m2


def extract_timeseries(
    requests_info: list,
    config: SHConfig,
    scale: float = SCALE_M,
    min_coverage: float = MIN_COVERAGE,
    max_threads: int = 5,
    progress_callback=None,
) -> pd.DataFrame:
    """Download all Process API requests in parallel and compute per-observation VH/VV stats.

    progress_callback, if given, is called once with the total request count
    before downloading starts (useful for a Streamlit progress bar/status).
    """
    if not requests_info:
        raise ValueError("no Process API requests to execute (no fields matched any acquisitions).")

    if progress_callback:
        progress_callback(len(requests_info))

    download_requests = [info["request"].download_list[0] for info in requests_info]
    client = SentinelHubDownloadClient(config=config)
    try:
        arrays = client.download(download_requests, max_threads=max_threads, show_progress=False, raise_download_errors=False)
    except Exception as exc:
        raise RuntimeError(f"failed downloading Process API responses from Sentinel Hub: {exc}") from exc

    records = []
    for info, array in zip(requests_info, arrays):
        field_id, crop_type, date_str, area_m2 = info["field_id"], info["crop_type"], info["date"], info["area_m2"]

        if array is None:
            logger.warning(f"No data returned for field {field_id} on {date_str}; skipping.")
            continue

        vv, vh, mask = array[:, :, 0].astype(float), array[:, :, 1].astype(float), array[:, :, 2].astype(bool)

        coverage = calculate_coverage(mask, scale, area_m2)
        if coverage < min_coverage:
            continue

        vv_valid = vv[mask]
        vh_valid = vh[mask]
        vv_valid = vv_valid[vv_valid > 0]
        vh_valid = vh_valid[vh_valid > 0]
        if vv_valid.size == 0 or vh_valid.size == 0:
            logger.warning(f"Missing/invalid VV or VH values for field {field_id} on {date_str}; skipping.")
            continue

        VV_dB = float(np.mean(10 * np.log10(vv_valid)))
        VH_dB = float(np.mean(10 * np.log10(vh_valid)))

        vh_lin = 10 ** (VH_dB / 10.0)
        vv_lin = 10 ** (VV_dB / 10.0)
        ratio = vh_lin / vv_lin if vv_lin != 0 else np.nan
        rvi = 4 * vh_lin / (vh_lin + vv_lin) if (vh_lin + vv_lin) != 0 else np.nan

        records.append(
            {
                "field_id": field_id, "crop_type": crop_type, "date": date_str,
                "VH_dB": VH_dB, "VV_dB": VV_dB, "VH_VV_ratio": ratio, "RVI": rvi, "coverage": coverage,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("no valid observations remained after filtering (date range, geometries, or coverage threshold).")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["field_id", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Smoothing + summary metrics
# --------------------------------------------------------------------------

def apply_smoothing(df: pd.DataFrame, window_length: int = SG_WINDOW, polyorder: int = SG_POLYORDER) -> pd.DataFrame:
    """Savitzky-Golay smoothing of VH_dB per field; fields with too few points keep their raw series."""
    frames = []
    for field_id, group in df.groupby("field_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        vh = group["VH_dB"].to_numpy(dtype=float)
        if len(vh) >= window_length:
            try:
                smoothed = savgol_filter(vh, window_length=window_length, polyorder=polyorder, mode="interp")
            except Exception as exc:
                logger.warning(f"Smoothing failed for field {field_id}: {exc}. Using raw VH.")
                smoothed = vh
        else:
            smoothed = vh
        group["VH_dB_smoothed"] = smoothed
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def calculate_summary_metrics(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Per-field date_of_peak_VH, max_VH, and the first-90-day OLS growth slope."""
    start_dt = pd.to_datetime(start_date)
    horizon = start_dt + pd.Timedelta(days=GROWTH_WINDOW_DAYS)

    summaries = []
    for field_id, group in df.groupby("field_id", sort=False):
        group = group.sort_values("date")
        crop_type = group["crop_type"].iloc[0]

        peak_idx = group["VH_dB_smoothed"].idxmax()
        date_of_peak_vh = group.loc[peak_idx, "date"].date()
        max_vh = group.loc[peak_idx, "VH_dB_smoothed"]

        window = group[(group["date"] >= start_dt) & (group["date"] <= horizon)]
        if len(window) >= 3:
            x = (window["date"] - start_dt).dt.days.to_numpy(dtype=float)
            y = window["VH_dB_smoothed"].to_numpy(dtype=float)
            slope = linregress(x, y).slope
        else:
            slope = np.nan

        summaries.append(
            {
                "field_id": field_id, "crop_type": crop_type, "date_of_peak_VH": date_of_peak_vh,
                "max_VH": max_vh, "growth_rate_slope_first_90_days": slope,
            }
        )

    return pd.DataFrame(summaries)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def build_growth_curves_figure(df: pd.DataFrame) -> plt.Figure:
    """One subplot per field: raw VH (grey markers) + smoothed VH (blue line). Returns the Figure."""
    field_ids = list(df["field_id"].unique())
    n = len(field_ids)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, field_id in zip(axes_flat, field_ids):
        group = df[df["field_id"] == field_id].sort_values("date")
        ax.plot(group["date"], group["VH_dB"], linestyle="None", marker="o", markersize=4, color="lightgrey", label="Raw VH")
        ax.plot(group["date"], group["VH_dB_smoothed"], color="blue", linewidth=1.8, label="Smoothed VH")

        ax.set_title(f"Field {field_id} ({group['crop_type'].iloc[0]})")
        ax.set_xlabel("Date")
        ax.set_ylabel("VH backscatter (dB)")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Sentinel-1 VH Growth Curves by Field", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
