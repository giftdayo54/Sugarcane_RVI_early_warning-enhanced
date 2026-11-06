"""
src/map_view.py

Interactive web map (zoom / pan / hover) of the loaded field boundaries over a basemap, with an optional raster
overlay (true colour, an index, or the gap classes) reprojected to lon/lat so it lines up with the tiles.

Uses Plotly's MapLibre `map` layout - no Mapbox token needed. Tile sources:
  * Esri World Imagery (sub-metre satellite in most agricultural areas)  - attribution: Esri, Maxar, Earthstar Geographics
  * OpenStreetMap, Carto Positron / Dark Matter, OpenTopoMap
Tiles are loaded by the browser, so the machine viewing the app needs internet access to them.
"""
from __future__ import annotations

import base64
import io
import json

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from .imagery import Grid, apply_affine

_ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
_TOPO = "https://tile.opentopomap.org/{z}/{x}/{y}.png"

# name -> (plotly map style, extra raster layers, attribution)
BASEMAPS: dict[str, tuple[str, list[dict], str]] = {
    "Satellite (Esri World Imagery)": ("white-bg", [dict(below="traces", sourcetype="raster", source=[_ESRI], sourceattribution="Esri, Maxar, Earthstar Geographics")], "Esri"),
    "Street map (OpenStreetMap)": ("open-street-map", [], "OpenStreetMap contributors"),
    "Light (Carto Positron)": ("carto-positron", [], "Carto"),
    "Dark (Carto Dark Matter)": ("carto-darkmatter", [], "Carto"),
    "Topographic (OpenTopoMap)": ("white-bg", [dict(below="traces", sourcetype="raster", source=[_TOPO], sourceattribution="OpenTopoMap (CC-BY-SA)")], "OpenTopoMap"),
}
DEFAULT_BASEMAP = "Satellite (Esri World Imagery)"


def center_and_zoom(bounds: tuple[float, float, float, float], width_px: int = 760, height_px: int = 480) -> tuple[dict, float]:
    """Centre + zoom so the lon/lat bounds fill roughly the viewport (Web-Mercator maths)."""
    minx, miny, maxx, maxy = bounds
    lon_span, lat_span = max(maxx - minx, 1e-6), max(maxy - miny, 1e-6)
    lat_c = (miny + maxy) / 2
    z_lon = np.log2(width_px * 360.0 / (256.0 * lon_span))
    z_lat = np.log2(height_px * 360.0 * np.cos(np.radians(lat_c)) / (256.0 * lat_span))
    zoom = float(np.clip(min(z_lon, z_lat) - 0.35, 1, 19))  # small margin
    return {"lon": (minx + maxx) / 2, "lat": lat_c}, zoom


def _hover_text(row, id_col: str) -> str:
    parts = [f"<b>{row[id_col]}</b>"]
    for col in ("variety", "crop_type", "planting_date", "estate"):
        if col in row.index and str(row[col]) not in ("nan", "None", "NaT", ""):
            parts.append(f"{col.replace('_', ' ')}: {row[col]}")
    if "area_ha" in row.index:
        parts.append(f"area: {float(row['area_ha']):.1f} ha")
    return "<br>".join(parts)


def _prepare(fields: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    g = fields.copy()
    g = g.to_crs(4326) if g.crs is not None else g.set_crs(4326)
    g[id_col] = g[id_col].astype(str)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].reset_index(drop=True)
    area = g.to_crs(g.estimate_utm_crs()).geometry.area / 10_000.0
    g["area_ha"] = area.values
    return g


def raster_overlay_layer(rgba: np.ndarray, grid: Grid, opacity: float = 0.85, smooth: bool = True) -> tuple[dict, tuple[float, float, float, float]]:
    """Reproject a (H,W,4) uint8 RGBA image from the grid CRS to EPSG:4326 and return a plotly map `layer` + its lon/lat bounds."""
    from PIL import Image
    h, w = rgba.shape[:2]
    src_t: Affine = grid.transform
    left, bottom, right, top = grid.bounds
    dst_t, dw, dh = calculate_default_transform(grid.crs, "EPSG:4326", w, h, left=left, bottom=bottom, right=right, top=top)
    # keep the overlay light-weight but sharp enough
    scale = min(1.0, 2000 / max(dw, dh))
    if scale < 1.0:
        dst_t, dw, dh = calculate_default_transform(grid.crs, "EPSG:4326", w, h, left=left, bottom=bottom, right=right, top=top,
                                                    dst_width=int(dw * scale), dst_height=int(dh * scale))
    out = np.zeros((4, dh, dw), dtype=np.uint8)
    for b in range(4):
        reproject(np.ascontiguousarray(rgba[:, :, b]), out[b], src_transform=src_t, src_crs=grid.crs, dst_transform=dst_t,
                  dst_crs="EPSG:4326", resampling=Resampling.bilinear if smooth else Resampling.nearest)
    img = np.transpose(out, (1, 2, 0))
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, format="PNG", optimize=True)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    lon0, lat0 = apply_affine(dst_t, 0, 0)
    lon1, lat1 = apply_affine(dst_t, dw, dh)
    coords = [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]]
    return dict(sourcetype="image", source=uri, coordinates=coords, opacity=float(opacity), below="traces"), (lon0, lat1, lon1, lat0)


def rgba_from_rgb(rgb_u8: np.ndarray, alpha_mask: np.ndarray, alpha: int = 255) -> np.ndarray:
    out = np.zeros(rgb_u8.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = rgb_u8
    out[..., 3] = np.where(alpha_mask, alpha, 0)
    return out


def boundary_map(
    fields: gpd.GeoDataFrame, id_col: str = "field_id", selected: str | list[str] | None = None, basemap: str = DEFAULT_BASEMAP,
    overlay: tuple[np.ndarray, Grid] | None = None, overlay_opacity: float = 0.85, overlay_smooth: bool = True,
    title: str | None = None, height: int = 520, zoom_to_selected: bool = True, extra_polygons: gpd.GeoDataFrame | None = None,
) -> go.Figure:
    """Boundaries (all fields thin yellow, selected thick red) over a basemap, with optional RGBA raster overlay.

    ``overlay`` = (rgba_uint8, grid). ``extra_polygons`` (EPSG:4326, with a 'gap_type' column) draws gap outlines.
    """
    g = _prepare(fields, id_col)
    sel = [selected] if isinstance(selected, str) else list(selected or [])
    style, layers, _ = BASEMAPS.get(basemap, BASEMAPS[DEFAULT_BASEMAP])
    map_layers = [dict(layer) for layer in layers]

    fig = go.Figure()
    others, chosen = g[~g[id_col].isin(sel)], g[g[id_col].isin(sel)]

    def add_poly(gdf, name, line_color, width, fill_opacity):
        if gdf.empty:
            return
        gj = json.loads(gdf.to_json())
        for i, f in enumerate(gj["features"]):
            f["id"] = str(i)
        fig.add_trace(go.Choroplethmap(
            geojson=gj, locations=[str(i) for i in range(len(gdf))], z=[1] * len(gdf), colorscale=[[0, line_color], [1, line_color]],
            showscale=False, marker=dict(opacity=fill_opacity, line=dict(color=line_color, width=width)),
            hovertext=[_hover_text(r, id_col) for _, r in gdf.iterrows()], hovertemplate="%{hovertext}<extra></extra>", name=name))

    add_poly(others, "Fields", "#ffd54f", 1.5, 0.08)
    add_poly(chosen, "Selected field", "#ff1744", 3.0, 0.12)

    if extra_polygons is not None and len(extra_polygons):
        for gtype, color, nm in (("bare_soil", "#c0392b", "Bare-soil gaps"), ("weed_covered", "#f9a825", "Weed-covered gaps")):
            sub = extra_polygons[extra_polygons["gap_type"] == gtype]
            if len(sub):
                gj = json.loads(sub.to_json())
                for i, f in enumerate(gj["features"]):
                    f["id"] = str(i)
                fig.add_trace(go.Choroplethmap(
                    geojson=gj, locations=[str(i) for i in range(len(sub))], z=[1] * len(sub), colorscale=[[0, color], [1, color]], showscale=False,
                    marker=dict(opacity=0.35, line=dict(color=color, width=1.5)),
                    hovertext=[f"{nm}<br>{a:.2f} ha" for a in sub["area_ha"]], hovertemplate="%{hovertext}<extra></extra>", name=nm))

    focus = chosen if (zoom_to_selected and not chosen.empty) else g
    bounds = tuple(focus.total_bounds)
    if overlay is not None:
        layer, ob = raster_overlay_layer(overlay[0], overlay[1], overlay_opacity, overlay_smooth)
        map_layers.append(layer)
        bounds = ob if zoom_to_selected else tuple(g.total_bounds)
    center, zoom = center_and_zoom(bounds)
    fig.update_layout(map=dict(style=style, center=center, zoom=zoom, layers=map_layers), height=height,
                      margin=dict(l=0, r=0, t=36 if title else 0, b=0), title=title, showlegend=False, uirevision="keep")
    attribution = {"Satellite (Esri World Imagery)": "Imagery © Esri, Maxar, Earthstar Geographics"}.get(basemap)
    if attribution:
        fig.add_annotation(text=attribution, xref="paper", yref="paper", x=1, y=0, xanchor="right", yanchor="bottom", showarrow=False,
                           font=dict(size=10, color="white"), bgcolor="rgba(0,0,0,0.45)")
    return fig


MAP_CONFIG = {"scrollZoom": True, "displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}
