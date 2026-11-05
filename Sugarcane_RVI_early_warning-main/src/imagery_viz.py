"""
src/imagery_viz.py

Rendering helpers: percentile stretches, RGB composites, colour-mapped single bands
(as PNG-ready arrays), interactive Plotly maps with hover values, histograms and
the gap class map. Everything works in the scene's projected grid (no basemap), with
the field boundary drawn on top.
"""
from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps

from .gap_detection import CLASS_COLORS, CLASS_LABELS
from .imagery import Grid, apply_affine

COMPOSITES = {
    "True colour (B04-B03-B02)": ("B04", "B03", "B02"),
    "False colour - vegetation (B08-B04-B03)": ("B08", "B04", "B03"),
    "Agriculture (B11-B08-B02)": ("B11", "B08", "B02"),
    "SWIR (B12-B8A-B04)": ("B12", "B8A", "B04"),
    "Red-edge (B8A-B05-B04)": ("B8A", "B05", "B04"),
}

MAX_DISPLAY_PX = 700


def stretch(a: np.ndarray, lo: float = 2, hi: float = 98, vmin=None, vmax=None) -> np.ndarray:
    """Scale to 0-1 with percentile (or explicit) limits; NaN stays NaN."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.full(a.shape, np.nan, dtype=np.float32)
    if vmin is None or vmax is None:
        p_lo, p_hi = np.percentile(v, [lo, hi])
        vmin = p_lo if vmin is None else vmin
        vmax = p_hi if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.clip((a - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def rgb_composite(bands: dict[str, np.ndarray], triplet: tuple[str, str, str], lo: float = 2, hi: float = 98,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """Float RGB (H,W,3) 0-1; pixels outside ``mask`` or with NaN become white."""
    chans = []
    ref = None
    for b in triplet:
        arr = bands[b]
        ref = arr if mask is None else np.where(mask, arr, np.nan)
        chans.append(ref)
    # one shared stretch per channel computed on valid pixels
    out = np.stack([stretch(c, lo, hi) for c in chans], axis=-1)
    bad = ~np.all(np.isfinite(out), axis=-1)
    out = np.where(bad[..., None], 1.0, out)
    return out.astype(np.float32)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def colorize(a: np.ndarray, cmap: str = "viridis", vmin=None, vmax=None, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Single band -> uint8 RGB with a colormap; NaN -> white."""
    s = stretch(a, lo, hi, vmin, vmax)
    rgba = colormaps[cmap](np.nan_to_num(s, nan=0.0))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(a)] = 255
    return rgb


def png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _step(shape) -> int:
    return max(1, int(np.ceil(max(shape[:2]) / MAX_DISPLAY_PX)))


def _boundary_trace(boundary, grid: Grid, step: int):
    if boundary is None:
        return []
    inv = ~grid.transform
    traces = []
    geoms = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
    for g in geoms:
        xs, ys = g.exterior.xy
        cols, rows = zip(*[apply_affine(inv, x, y) for x, y in zip(xs, ys)])
        traces.append(go.Scatter(x=[(c - 0.5) / step for c in cols], y=[(r - 0.5) / step for r in rows], mode="lines",
                                 line=dict(color="black", width=2), hoverinfo="skip", showlegend=False))
    return traces


def _layout(fig: go.Figure, shape, title: str, step: int, height: int = 520):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=45, b=10))
    fig.update_xaxes(visible=False, range=[-0.5, shape[1] / step - 0.5], constrain="domain")
    fig.update_yaxes(visible=False, range=[shape[0] / step - 0.5, -0.5], scaleanchor="x", scaleratio=1)
    return fig


def image_figure(rgb_u8: np.ndarray, grid: Grid, boundary=None, title: str = "") -> go.Figure:
    step = _step(rgb_u8.shape)
    small = rgb_u8[::step, ::step]
    fig = go.Figure(go.Image(z=small, hoverinfo="skip"))
    for t in _boundary_trace(boundary, grid, step):
        fig.add_trace(t)
    return _layout(fig, rgb_u8.shape, title, step)


def raster_figure(a: np.ndarray, grid: Grid, boundary=None, title: str = "", cmap: str = "viridis",
                  vmin=None, vmax=None, colorbar_title: str = "") -> go.Figure:
    """Colour-mapped single band as an interactive heatmap (hover shows the value)."""
    step = _step(a.shape)
    small = a[::step, ::step]
    s = small[np.isfinite(small)]
    if vmin is None and s.size:
        vmin = float(np.percentile(s, 2))
    if vmax is None and s.size:
        vmax = float(np.percentile(s, 98))
    fig = go.Figure(go.Heatmap(z=small, zmin=vmin, zmax=vmax, colorscale=_plotly_scale(cmap),
                               colorbar=dict(title=colorbar_title), hovertemplate="value %{z:.3f}<extra></extra>",
                               zsmooth=False))
    for t in _boundary_trace(boundary, grid, step):  # axes are in downsampled pixel units
        fig.add_trace(t)
    return _layout(fig, a.shape, title, step)


def _plotly_scale(cmap: str, n: int = 32):
    cm = colormaps[cmap]
    return [[i / (n - 1), "rgb({},{},{})".format(*(int(255 * c) for c in cm(i / (n - 1))[:3]))] for i in range(n)]


def class_figure(class_map: np.ndarray, grid: Grid, boundary=None, title: str = "Gap classification") -> go.Figure:
    step = _step(class_map.shape)
    small = class_map[::step, ::step].astype(float)
    keys = sorted(CLASS_LABELS)
    # discrete colour scale: each code occupies an equal band
    scale = []
    for i, k in enumerate(keys):
        scale += [[i / len(keys), CLASS_COLORS[k]], [(i + 1) / len(keys), CLASS_COLORS[k]]]
    z = np.vectorize({k: i for i, k in enumerate(keys)}.get)(small.astype(int)).astype(float)
    text = np.vectorize(CLASS_LABELS.get)(small.astype(int))
    z = np.where(small == 0, np.nan, z)  # nodata transparent
    fig = go.Figure(go.Heatmap(z=z, zmin=-0.5, zmax=len(keys) - 0.5, colorscale=scale, showscale=False,
                               text=text, hovertemplate="%{text}<extra></extra>", zsmooth=False))
    for k in keys[1:]:  # legend proxies
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=CLASS_COLORS[k], symbol="square"),
                                 name=CLASS_LABELS[k]))
    for t in _boundary_trace(boundary, grid, step):
        fig.add_trace(t)
    _layout(fig, class_map.shape, title, step)
    fig.update_layout(legend=dict(orientation="h", y=-0.02))
    return fig


def histogram_figure(a: np.ndarray, title: str, lines: dict[str, float] | None = None, nbins: int = 60) -> go.Figure:
    v = a[np.isfinite(a)]
    fig = go.Figure(go.Histogram(x=v, nbinsx=nbins, marker_color="#546e7a"))
    for label, x in (lines or {}).items():
        fig.add_vline(x=x, line_dash="dash", line_color="#c0392b", annotation_text=label, annotation_position="top")
    fig.update_layout(title=title, height=260, margin=dict(l=10, r=10, t=40, b=10), bargap=0.02, showlegend=False)
    return fig


def class_png(class_map: np.ndarray) -> bytes:
    rgb = np.full(class_map.shape + (3,), 255, dtype=np.uint8)
    for k, hexcol in CLASS_COLORS.items():
        if k == 0:
            continue
        rgb[class_map == k] = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
    return png_bytes(rgb)
