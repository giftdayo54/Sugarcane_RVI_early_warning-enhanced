"""
src/ui_common.py

Streamlit helpers shared by pages/3_Satellite_Imagery.py and pages/4_Gap_Analysis.py:
Sentinel Hub credentials, field-boundary source selection (shared across pages through
session_state), live-vs-demo scene loading, and small download helpers.

Streamlit drops widget state when you navigate between pages, so anything that must
survive (boundaries, credentials, the loaded scene) is kept under plain session_state keys.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from .demo_scene import demo_field_geometry, make_demo_history
from .gap_detection import write_float_geotiff
from .imagery import (Scene, fetch_s1_composite, fetch_s2_scene, find_latest_clear_scene, search_s2_scenes)
from .s1_growth_curves import authenticate_sentinel_hub, load_fields

REPO_BLOCKS = Path("data/vectors/blocks.geojson")
SRC_UPLOAD, SRC_REPO, SRC_DEMO = "Upload GeoJSON", "Repo blocks (data/vectors/blocks.geojson)", "Synthetic demo field (no credentials)"
CDSE_BASE = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
MAX_CACHED_SCENES = 6


def secret(key: str, default: str = "") -> str:
    from .secrets_util import get_secret
    return get_secret(key, default)


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def credentials_sidebar() -> dict:
    """Sidebar credentials. Secrets (Streamlit secrets / env vars) are used automatically; the text boxes are only an
    optional override and are never pre-filled with a stored secret. Returns a plain dict."""
    from .secrets_util import diagnostics, find_secret
    ss = st.session_state
    sec_id, src_id = find_secret("SH_CLIENT_ID")
    sec_pw, src_pw = find_secret("SH_CLIENT_SECRET")
    base_default, _ = find_secret("SH_BASE_URL", CDSE_BASE)
    token_default, _ = find_secret("SH_TOKEN_URL", CDSE_TOKEN)

    st.sidebar.subheader("Sentinel Hub credentials")
    have_secrets = bool(sec_id and sec_pw)
    if have_secrets:
        st.sidebar.success(f"Using credentials from {src_id}.")
    else:
        st.sidebar.info("No credentials found in Streamlit secrets or environment - enter them below "
                        "(or fill .streamlit/secrets.toml).")
    with st.sidebar.expander("Enter / override credentials", expanded=not have_secrets):
        typed_id = st.text_input("Client ID", value=ss.get("_sh_id", ""), type="password",
                                 placeholder="from secrets" if sec_id else "sh-xxxxxxxx-...")
        typed_pw = st.text_input("Client secret", value=ss.get("_sh_secret", ""), type="password",
                                 placeholder="from secrets" if sec_pw else "")
    with st.sidebar.expander("Advanced"):
        base = st.text_input("Base URL", value=ss.get("_sh_base") or base_default)
        token = st.text_input("Token URL", value=ss.get("_sh_token") or token_default)
        threads = st.slider("Parallel downloads", 1, 8, 4)
    with st.sidebar.expander("My secrets aren't loading - diagnose"):
        d = diagnostics()
        st.caption("Names only - values are never shown.")
        st.write({"top-level keys seen": d["top_level_keys"] or "none", "sections seen": d["sections"] or "none",
                  "SH_CLIENT_ID": d["found"]["SH_CLIENT_ID"], "SH_CLIENT_SECRET": d["found"]["SH_CLIENT_SECRET"]})
        st.markdown("- Keys must be named `SH_CLIENT_ID` / `SH_CLIENT_SECRET` (top level), or sit in a `[sentinelhub]`, `[cdse]` "
                    "or `[copernicus]` section.\n- Locally the file is `.streamlit/secrets.toml` (next to where you run `streamlit run`).\n"
                    "- On Streamlit Cloud paste the same text in *Settings → Secrets* and **reboot the app**.\n"
                    "- Blank or placeholder values (`your-...`) are ignored.")
    ss.update({"_sh_id": typed_id, "_sh_secret": typed_pw, "_sh_base": base, "_sh_token": token})
    return {"id": typed_id or sec_id, "secret": typed_pw or sec_pw, "base": base, "token": token, "threads": threads}


def get_config(creds: dict):
    """Authenticate once per credential set (cached in session_state)."""
    if not creds["id"] or not creds["secret"]:
        raise RuntimeError("Enter Sentinel Hub client ID and secret in the sidebar (or add them to Streamlit secrets).")
    key = (creds["id"], creds["secret"], creds["base"], creds["token"])
    cached = st.session_state.get("_sh_cfg")
    if cached and cached[0] == key:
        return cached[1]
    cfg = authenticate_sentinel_hub(creds["id"], creds["secret"], creds["base"], creds["token"])
    st.session_state["_sh_cfg"] = (key, cfg)
    return cfg


# --------------------------------------------------------------------------
# Field source
# --------------------------------------------------------------------------

def _demo_fields() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"field_id": ["DEMO-01"], "crop_type": ["plant"], "area_m2": [0.0],
                             "planting_date": ["2026-03-01"]}, geometry=[demo_field_geometry()], crs="EPSG:4326")


def field_source_sidebar() -> tuple[gpd.GeoDataFrame | None, bool]:
    """Pick where field boundaries come from. Returns (fields_gdf or None, is_demo).

    The parsed boundaries are shared through session_state so the second page needn't ask again.
    """
    ss = st.session_state
    st.sidebar.header("Fields")
    default_idx = [SRC_UPLOAD, SRC_REPO, SRC_DEMO].index(ss.get("_field_source", SRC_UPLOAD))
    source = st.sidebar.radio("Field boundaries", [SRC_UPLOAD, SRC_REPO, SRC_DEMO], index=default_idx, key="in_field_source")
    ss["_field_source"] = source

    if source == SRC_DEMO:
        st.sidebar.warning("SYNTHETIC DATA - for trying the tools only. Nothing here is a real measurement.")
        return _demo_fields(), True

    fields = None
    if source == SRC_REPO:
        if REPO_BLOCKS.exists():
            fields = _load_path(str(REPO_BLOCKS))
        else:
            st.sidebar.error(f"{REPO_BLOCKS} not found.")
    else:
        up = st.sidebar.file_uploader("Field boundaries (GeoJSON)", type=["geojson", "json"], key="in_field_upload")
        if up is not None:
            key = f"{up.name}:{up.size}"
            if ss.get("_upload_key") != key:
                with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                    tmp.write(up.getvalue()); path = tmp.name
                try:
                    ss["_uploaded_fields"] = _load_path(path)
                    ss["_upload_key"] = key
                except Exception as exc:
                    st.sidebar.error(f"Couldn't read that file: {exc}")
                finally:
                    Path(path).unlink(missing_ok=True)
        fields = ss.get("_uploaded_fields")
        if fields is not None and up is None:
            st.sidebar.caption("Using the boundaries uploaded earlier in this session.")
    if fields is not None:
        st.sidebar.caption(f"{len(fields)} field(s) loaded.")
    return fields, False


def _load_path(path: str) -> gpd.GeoDataFrame:
    return load_fields(path)


def date_range_sidebar(default_days: int = 45) -> tuple[str, str]:
    c1, c2 = st.sidebar.columns(2)
    start = c1.date_input("From", value=date.today() - timedelta(days=default_days), key="in_start")
    end = c2.date_input("To", value=date.today(), key="in_end")
    return str(start), str(end)


# --------------------------------------------------------------------------
# Scene loading (live or demo) + small cache
# --------------------------------------------------------------------------

def _scene_cache() -> dict:
    return st.session_state.setdefault("_scene_cache", {})


def remember_scene(scene: Scene) -> None:
    cache = _scene_cache()
    cache[(scene.field_id, scene.date, scene.synthetic)] = scene
    while len(cache) > MAX_CACHED_SCENES:
        cache.pop(next(iter(cache)))
    st.session_state["scene"] = scene


def current_scene(field_id: str, demo: bool) -> Scene | None:
    sc = st.session_state.get("scene")
    return sc if sc is not None and sc.field_id == field_id and sc.synthetic == demo else None


def demo_history() -> list[Scene]:
    if "_demo_history" not in st.session_state:
        st.session_state["_demo_history"] = make_demo_history(5)
    return st.session_state["_demo_history"]


def list_scenes(demo: bool, geom, start: str, end: str, creds: dict, max_tile_cloud: float) -> pd.DataFrame:
    if demo:
        return pd.DataFrame({"date": [s.date for s in demo_history()], "cloud_cover_pct": [s.cloud_catalog for s in demo_history()],
                             "datetime": [pd.Timestamp(s.date) for s in demo_history()], "id": ["synthetic"] * len(demo_history())})
    return search_s2_scenes(geom, start, end, get_config(creds), max_tile_cloud)


def load_scene_for_date(demo: bool, field_id: str, geom, row: pd.Series, creds: dict, set_current: bool = True) -> Scene:
    if demo:
        sc = next(s for s in demo_history() if s.date == row["date"])
    else:
        cached = _scene_cache().get((field_id, row["date"], False))
        sc = cached or fetch_s2_scene(field_id, geom, row["datetime"], get_config(creds), row.get("cloud_cover_pct"))
    if set_current:
        remember_scene(sc)
    else:  # keep it cached for reuse without switching the page's current scene
        _scene_cache()[(sc.field_id, sc.date, sc.synthetic)] = sc
    return sc


def load_latest_clear(demo: bool, field_id: str, geom, scenes_df: pd.DataFrame, creds: dict,
                      min_clear: float, progress=None) -> Scene:
    if demo:
        for s in demo_history():  # newest first
            if s.clear_fraction >= min_clear:
                remember_scene(s)
                return s
        s = max(demo_history(), key=lambda x: x.clear_fraction); remember_scene(s); return s
    sc = find_latest_clear_scene(field_id, geom, scenes_df, get_config(creds), min_clear, progress=progress)
    remember_scene(sc)
    return sc


def attach_s1(scene: Scene, demo: bool, creds: dict, window_days: int, direction: str, max_scenes: int, speckle_px: int) -> Scene:
    if demo and scene.s1 is not None:
        return scene
    scene.s1 = fetch_s1_composite(scene, get_config(creds), window_days=window_days, direction=direction,
                                  max_scenes=max_scenes, speckle_px=speckle_px, max_threads=creds["threads"])
    return scene


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------

def float_tif_bytes(arr, grid, tags: dict | None = None) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        path = tmp.name
    try:
        write_float_geotiff(path, arr, grid, tags=tags)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def class_tif_bytes(class_map, grid) -> bytes:
    from .gap_detection import write_class_geotiff
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        path = tmp.name
    try:
        write_class_geotiff(path, class_map, grid)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def read_vector_upload(uploaded_file) -> gpd.GeoDataFrame:
    """Read an uploaded GeoJSON/JSON into a GeoDataFrame (CRS defaults to EPSG:4326 if missing)."""
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue()); path = tmp.name
    try:
        gdf = gpd.read_file(path)
    finally:
        Path(path).unlink(missing_ok=True)
    return gdf.set_crs(4326) if gdf.crs is None else gdf


def read_raster_to_grid(uploaded_file, grid) -> "np.ndarray":
    """Read an uploaded single-band GeoTIFF (e.g. a UAV canopy-height model) and resample it onto the scene grid."""
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue()); path = tmp.name
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("The raster has no CRS; georeference it first.")
            dst = np.full(grid.shape, np.nan, dtype="float32")
            reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=grid.transform, dst_crs=grid.crs, resampling=Resampling.average,
                      src_nodata=src.nodata, dst_nodata=np.nan)
        return dst
    finally:
        Path(path).unlink(missing_ok=True)
