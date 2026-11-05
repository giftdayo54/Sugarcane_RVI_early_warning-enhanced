"""
src/gap_detection.py

Canopy-gap detection for sugarcane using Sentinel-2 (optical) + Sentinel-1 (radar),
with a height-based test that separates

    * bare-soil gaps          - low NDVI, short/no canopy
    * WEED-COVERED gaps       - HIGH NDVI (look green) but canopy far shorter than cane
    * healthy cane            - high NDVI and canopy near the expected cane height
    * thin / stunted canopy   - everything else that is vegetated but not at full height

Why two sensors?
----------------
NDVI cannot tell cane from weeds: both are green. A gap full of weeds is therefore
invisible to optical-only methods and is under-counted. Canopy *structure* differs:
cane is several metres tall, weeds typically < 1 m. C-band cross-pol backscatter (VH)
responds to canopy volume/structure, so it carries a height-like signal.

Height proxy (important caveat)
-------------------------------
Sentinel-1 does not measure height. We build a *relative* canopy index per field:

    rel = (S1 - S1_soil) / (S1_cane - S1_soil)         clipped to [0, 1.3]
    height_m = rel * H_ref

where H_ref is the average common cane height you supply for that crop age,
S1_cane is the field's own healthy-cane level (a robust "core" where NDVI and
S1 agree the canopy is full), and S1_soil is the level over bare pixels in the
same field (or a default contrast when the field has none). Thresholds are
fractions of H_ref ("shorter than half the normal cane height"). If you have a
real canopy-height raster (UAV DSM-DTM, LiDAR), pass it as ``chm`` and it is
used instead of the S1 proxy.

This is a transparent heuristic. Validate it on your own reference polygons
(see `validate_against_reference`) before using the numbers operationally.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize, shapes
from scipy import ndimage as ndi
from shapely.geometry import shape as shapely_shape

from .imagery import Grid, Scene, apply_affine, erode_mask
from .raster_calc import build_layers, evaluate

# ---- class codes ------------------------------------------------------------------------------
NODATA, HEALTHY, BARE_GAP, WEEDY_GAP, THIN = 0, 1, 2, 3, 4
CLASS_LABELS = {NODATA: "No data / excluded", HEALTHY: "Healthy cane", BARE_GAP: "Gap - bare soil",
                WEEDY_GAP: "Gap - covered by weeds", THIN: "Thin / stunted canopy"}
CLASS_COLORS = {NODATA: "#bdbdbd", HEALTHY: "#2e7d32", BARE_GAP: "#c0392b", WEEDY_GAP: "#f9a825", THIN: "#aed581"}
GAP_CLASSES = (BARE_GAP, WEEDY_GAP)

SUPERVISED_LABELS = {"healthy": HEALTHY, "bare_gap": BARE_GAP, "weedy_gap": WEEDY_GAP, "thin": THIN}


# ---- parameters -------------------------------------------------------------------------------
@dataclass(frozen=True)
class GapParams:
    h_ref_m: float = 3.0               # average common cane height for this crop age (m)
    ndvi_bare: float = 0.35            # NDVI below this = no green cover
    h_gap_frac: float = 0.50           # canopy shorter than this fraction of h_ref = gap-height
    h_ok_frac: float = 0.75            # canopy at/above this fraction of h_ref = full canopy
    s1_metric: str = "VH_dB"           # "VH_dB" or "RVI"
    soil_contrast_db: float = 6.0      # default cane-minus-soil VH contrast when no bare pixels exist
    soil_contrast_rvi: float = 0.35    # same, for RVI
    edge_buffer_m: float = 10.0        # ignore pixels this close to the boundary (mixed pixels, headlands)
    min_patch_px: int = 4              # gap patches smaller than this are treated as noise (4 px = 0.04 ha)
    min_age_days: int = 120            # below this the canopy has not closed - gaps are not meaningful
    use_s1: bool = True

    def validate(self) -> None:
        if self.h_ref_m <= 0:
            raise ValueError("Average cane height must be > 0 m.")
        if not (0 < self.h_gap_frac < self.h_ok_frac <= 1.5):
            raise ValueError("Need 0 < gap-height fraction < full-canopy fraction.")
        if self.s1_metric not in ("VH_dB", "RVI"):
            raise ValueError("s1_metric must be 'VH_dB' or 'RVI'.")


def params_from_config(cfg: dict | None, **overrides) -> GapParams:
    """Build GapParams from the ``gaps:`` block of config.yaml (missing keys fall back to defaults)."""
    g = dict((cfg or {}).get("gaps", {}))
    allowed = set(GapParams.__dataclass_fields__)
    kwargs = {k: v for k, v in g.items() if k in allowed}
    kwargs.update({k: v for k, v in overrides.items() if k in allowed and v is not None})
    return GapParams(**kwargs)


def expected_cane_height(age_days: float, h_max: float = 3.6, k: float = 0.025, t0: float = 150.0) -> float:
    """Generic logistic height-for-age curve (m). A STARTING POINT only: replace with the average
    height you actually measure for the variety/crop class/estate at that age."""
    return float(h_max / (1.0 + np.exp(-k * (float(age_days) - t0))))


# ---- result container -------------------------------------------------------------------------
@dataclass
class GapResult:
    class_map: np.ndarray
    features: dict[str, np.ndarray]
    ndvi_only_gap: np.ndarray
    valid_mask: np.ndarray
    anchors: dict
    stats: dict
    warnings: list[str] = field(default_factory=list)
    patches: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: GapParams | None = None
    method: str = "rules"


# ---- features ---------------------------------------------------------------------------------
def _s1_metric(vv, vh, metric: str) -> np.ndarray:
    with np.errstate(all="ignore"):
        if metric == "RVI":
            return np.where((vv + vh) > 0, 4 * vh / (vv + vh), np.nan).astype(np.float32)
        return np.where(vh > 0, 10 * np.log10(vh), np.nan).astype(np.float32)


def compute_features(scene: Scene, params: GapParams) -> dict[str, np.ndarray]:
    layers = build_layers(scene.bands, scene.s1 if params.use_s1 else None)
    f = {
        "ndvi": evaluate("(B08 - B04) / (B08 + B04)", layers),
        "ndre": evaluate("(B8A - B05) / (B8A + B05)", layers),
        "ndmi": evaluate("(B08 - B11) / (B08 + B11)", layers),
        "bsi": evaluate("((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))", layers),
    }
    if params.use_s1 and scene.s1 is not None:
        vv, vh = np.asarray(scene.s1["VV"], np.float32), np.asarray(scene.s1["VH"], np.float32)
        f["vh_db"] = np.where(vh > 0, 10 * np.log10(vh), np.nan).astype(np.float32)
        f["vv_db"] = np.where(vv > 0, 10 * np.log10(vv), np.nan).astype(np.float32)
        f["rvi"] = _s1_metric(vv, vh, "RVI")
        f["s1_metric"] = _s1_metric(vv, vh, params.s1_metric)
    return f


def _align_check(arr: np.ndarray, grid: Grid, what: str) -> np.ndarray:
    if arr.shape != grid.shape:
        raise ValueError(f"{what} shape {arr.shape} does not match the scene grid {grid.shape}.")
    return arr.astype(np.float32)


def relative_height(
    x: np.ndarray, ndvi: np.ndarray, valid: np.ndarray, params: GapParams,
) -> tuple[np.ndarray, dict, list[str]]:
    """S1-derived relative canopy index (1.0 = field's own healthy-cane level, 0 = bare soil)."""
    warns: list[str] = []
    v = valid & np.isfinite(x) & np.isfinite(ndvi)
    if v.sum() < 30:
        raise ValueError("Too few valid pixels to calibrate the canopy-height proxy.")
    q_nd, q_x = np.percentile(ndvi[v], 60), np.percentile(x[v], 60)
    core = v & (ndvi >= q_nd) & (x >= q_x)
    if core.sum() < 30:  # fall back to the upper quartile of S1
        core = v & (x >= np.percentile(x[v], 75))
        warns.append("Healthy-cane core was small; used the upper S1 quartile as the cane reference.")
    x_cane = float(np.median(x[core]))

    contrast = params.soil_contrast_db if params.s1_metric == "VH_dB" else params.soil_contrast_rvi
    bare = v & (ndvi < params.ndvi_bare)
    if bare.sum() >= 20:
        x_soil = float(np.median(x[bare]))
        soil_src = "measured on bare pixels in this field"
        if x_cane - x_soil < 0.35 * contrast:  # implausibly small contrast -> don't trust it
            x_soil, soil_src = x_cane - contrast, "default contrast (field's bare pixels gave implausibly low contrast)"
            warns.append("Bare-pixel S1 level was too close to cane level; used the default cane-soil contrast.")
    else:
        x_soil, soil_src = x_cane - contrast, "default contrast (no/few bare pixels in this field)"
    denom = max(x_cane - x_soil, 1e-6)
    rel = np.clip((x - x_soil) / denom, 0.0, 1.3).astype(np.float32)
    rel = np.where(np.isfinite(x), rel, np.nan).astype(np.float32)
    anchors = {"s1_cane_level": x_cane, "s1_soil_level": x_soil, "s1_soil_source": soil_src,
               "core_pixels": int(core.sum()), "core_fraction": float(core.sum() / v.sum()),
               "s1_metric": params.s1_metric}
    if anchors["core_fraction"] < 0.15:
        warns.append("Less than 15% of the field looks like confident healthy cane - the cane reference may be unreliable.")
    return rel, anchors, warns


# ---- classification ---------------------------------------------------------------------------
def _remove_small_patches(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 1 or not mask.any():
        return mask
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[labels]


def classify_rules(feat: dict[str, np.ndarray], valid: np.ndarray, rel: np.ndarray | None, p: GapParams) -> np.ndarray:
    nd = feat["ndvi"]
    ok = valid & np.isfinite(nd)
    cls = np.zeros(nd.shape, dtype=np.uint8)
    green = nd >= p.ndvi_bare
    if rel is None:  # optical-only: weeds cannot be separated from cane
        cls[ok & green] = HEALTHY
        cls[ok & ~green] = BARE_GAP
    else:
        ok &= np.isfinite(rel)
        short, tall = rel < p.h_gap_frac, rel >= p.h_ok_frac
        cls[ok & green & tall] = HEALTHY
        cls[ok & ~green & short] = BARE_GAP
        cls[ok & green & short] = WEEDY_GAP
        cls[ok & (cls == NODATA)] = THIN  # vegetated but not at full height / low NDVI but not short
    # patch-size filter: tiny gap specks are within speckle/mixed-pixel noise at 10 m
    gap = np.isin(cls, GAP_CLASSES)
    keep = _remove_small_patches(gap, p.min_patch_px)
    cls[gap & ~keep] = HEALTHY if rel is None else THIN  # specks revert to the neighbouring canopy state
    return cls


def ndvi_only_gaps(feat: dict[str, np.ndarray], valid: np.ndarray, p: GapParams) -> np.ndarray:
    """What an optical-only method would call a gap (same threshold, same patch filter)."""
    g = valid & np.isfinite(feat["ndvi"]) & (feat["ndvi"] < p.ndvi_bare)
    return _remove_small_patches(g, p.min_patch_px)


def gap_patch_table(class_map: np.ndarray, grid: Grid) -> pd.DataFrame:
    gap = np.isin(class_map, GAP_CLASSES)
    labels, n = ndi.label(gap, structure=np.ones((3, 3)))
    rows = []
    pa = grid.pixel_area_ha
    for i in range(1, n + 1):
        m = labels == i
        npx = int(m.sum())
        weedy = int((class_map[m] == WEEDY_GAP).sum())
        cy, cx = ndi.center_of_mass(m)
        x, y = apply_affine(grid.transform, cx + 0.5, cy + 0.5)
        rows.append({"patch_id": i, "area_ha": npx * pa, "pixels": npx, "weedy_share_pct": 100 * weedy / npx,
                     "dominant": "weed-covered" if weedy / npx >= 0.5 else "bare soil", "x": x, "y": y})
    return pd.DataFrame(rows, columns=["patch_id", "area_ha", "pixels", "weedy_share_pct", "dominant", "x", "y"]) \
        .sort_values("area_ha", ascending=False).reset_index(drop=True)


def summarize_gaps(scene: Scene, cls: np.ndarray, ndvi_gap: np.ndarray, valid: np.ndarray, patches: pd.DataFrame,
                   anchors: dict, p: GapParams, mode: str, method: str, warns: list[str], feat: dict) -> dict:
    pa = scene.grid.pixel_area_ha
    ha = lambda m: float(np.count_nonzero(m) * pa)
    valid_ha = ha(cls != NODATA)
    bare, weedy = ha(cls == BARE_GAP), ha(cls == WEEDY_GAP)
    pct = lambda v: (100.0 * v / valid_ha) if valid_ha > 0 else float("nan")
    ndvi_gap_ha = ha(ndvi_gap)
    nd = feat["ndvi"]
    row = {
        "block_id": scene.field_id, "date": scene.date, "mode": mode, "method": method,
        "field_ha": scene.field_area_ha, "analysed_ha": valid_ha, "clear_fraction_pct": 100 * scene.clear_fraction,
        "healthy_ha": ha(cls == HEALTHY), "thin_ha": ha(cls == THIN),
        "bare_gap_ha": bare, "weedy_gap_ha": weedy, "total_gap_ha": bare + weedy,
        "gap_pct": pct(bare + weedy), "bare_gap_pct": pct(bare), "weedy_gap_pct": pct(weedy),
        "ndvi_only_gap_ha": ndvi_gap_ha, "ndvi_only_gap_pct": pct(ndvi_gap_ha),
        "gap_missed_by_ndvi_ha": max(0.0, weedy),  # green-but-short pixels NDVI alone calls "fine"
        "weed_covered_share_of_gap_pct": (100 * weedy / (bare + weedy)) if (bare + weedy) > 0 else 0.0,
        "n_gap_patches": int(len(patches)),
        "largest_gap_ha": float(patches["area_ha"].max()) if len(patches) else 0.0,
        "mean_gap_patch_ha": float(patches["area_ha"].mean()) if len(patches) else 0.0,
        "median_ndvi": float(np.nanmedian(nd[cls != NODATA])) if valid_ha > 0 else float("nan"),
        "h_ref_m": p.h_ref_m, "s1_scenes": int(scene.s1["n_scenes"]) if (scene.s1 and p.use_s1) else 0,
        "synthetic": bool(scene.synthetic), "notes": " | ".join(warns),
    }
    row.update({k: v for k, v in anchors.items() if k in ("s1_cane_level", "s1_soil_level")})
    return row


def analyze_scene(
    scene: Scene, params: GapParams, chm: np.ndarray | None = None, age_days: float | None = None,
) -> GapResult:
    """Run the full S1+S2 gap analysis on one scene."""
    params.validate()
    warns: list[str] = []
    if age_days is not None and age_days < params.min_age_days:
        warns.append(f"Crop is {age_days:.0f} days old (< {params.min_age_days}); the canopy has probably not closed, "
                     "so bare ground is expected and 'gaps' are not meaningful yet.")
    if scene.clear_fraction < 0.5:
        warns.append(f"Only {scene.clear_fraction:.0%} of the field is cloud-free; results cover that part only.")
    if scene.field_area_ha < 0.3:
        warns.append("Field is very small for 10 m imagery; treat results with caution.")

    valid = erode_mask(scene.clear_mask, params.edge_buffer_m, scene.grid.resolution)
    feat = compute_features(scene, params)
    for k in ("ndvi",):
        feat[k] = np.where(valid, feat[k], np.nan).astype(np.float32)

    rel, anchors, height = None, {}, None
    if chm is not None:
        height = _align_check(chm, scene.grid, "Canopy height raster")
        rel = np.clip(height / params.h_ref_m, 0, 1.3).astype(np.float32)
        anchors = {"s1_soil_source": "n/a - measured canopy-height raster supplied", "core_fraction": float("nan")}
        mode = "S2 + measured canopy height"
    elif params.use_s1 and "s1_metric" in feat:
        rel, anchors, w = relative_height(feat["s1_metric"], feat["ndvi"], valid, params)
        warns += w
        height = (rel * params.h_ref_m).astype(np.float32)
        mode = "S2 + S1 height proxy"
    else:
        mode = "S2 only"
        warns.append("No Sentinel-1 data: weed-covered gaps cannot be separated from cane, so they are counted as healthy.")
    if rel is not None:
        feat["rel_height"], feat["height_m"] = rel, height

    cls = classify_rules(feat, valid, rel, params)
    ndvi_gap = ndvi_only_gaps(feat, valid, params)
    patches = gap_patch_table(cls, scene.grid)
    stats = summarize_gaps(scene, cls, ndvi_gap, valid, patches, anchors, params, mode, "rules", warns, feat)
    return GapResult(cls, feat, ndvi_gap, valid, anchors, stats, warns, patches, params, "rules")


# ---- optional supervised classifier -----------------------------------------------------------
FEATURE_KEYS = ["ndvi", "ndre", "ndmi", "bsi", "vh_db", "vv_db", "rel_height"]


def sample_features_at_points(result: GapResult, scene: Scene, points: gpd.GeoDataFrame, label_col: str = "label") -> pd.DataFrame:
    """Sample the feature stack at labelled points (labels: healthy / bare_gap / weedy_gap / thin)."""
    if label_col not in points.columns:
        raise ValueError(f"Training points need a '{label_col}' property (healthy / bare_gap / weedy_gap / thin).")
    pts = points.to_crs(scene.grid.epsg)
    inv = ~scene.grid.transform
    keys = [k for k in FEATURE_KEYS if k in result.features]
    rows = []
    for geom, lab in zip(pts.geometry, pts[label_col].astype(str).str.lower().str.strip()):
        if geom is None or geom.is_empty or lab not in SUPERVISED_LABELS:
            continue
        col, row = apply_affine(inv, geom.x, geom.y)
        r, c = int(np.floor(row)), int(np.floor(col))
        if not (0 <= r < scene.grid.height and 0 <= c < scene.grid.width) or not result.valid_mask[r, c]:
            continue
        vals = {k: float(result.features[k][r, c]) for k in keys}
        if all(np.isfinite(list(vals.values()))):
            rows.append({**vals, "label": lab, "y": SUPERVISED_LABELS[lab]})
    return pd.DataFrame(rows)


def classify_supervised(result: GapResult, scene: Scene, training: pd.DataFrame, seed: int = 42) -> tuple[GapResult, dict]:
    """Random Forest on the same features, trained on the user's labelled points.

    Returns a new GapResult plus CV diagnostics. With few points, cross-validated scores are optimistic
    because nearby points are spatially autocorrelated -- validate on separate fields/polygons.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score, accuracy_score

    keys = [k for k in FEATURE_KEYS if k in result.features and k in training.columns]
    counts = training["y"].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError("Need at least 2 classes with >= 3 labelled points each to train the classifier.")
    X, y = training[keys].to_numpy(), training["y"].to_numpy()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    diag = {"n_points": int(len(training)), "class_counts": {CLASS_LABELS[int(k)]: int(v) for k, v in counts.items()},
            "features": keys}
    n_splits = int(min(5, counts.min()))
    if n_splits >= 2:
        pred = cross_val_predict(rf, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
        diag.update({"cv_accuracy": float(accuracy_score(y, pred)), "cv_macro_f1": float(f1_score(y, pred, average="macro"))})
    rf.fit(X, y)

    stack = np.stack([result.features[k] for k in keys], axis=-1)
    ok = result.valid_mask & np.all(np.isfinite(stack), axis=-1)
    cls = np.zeros(result.class_map.shape, dtype=np.uint8)
    cls[ok] = rf.predict(stack[ok]).astype(np.uint8)
    p = result.params
    gap = np.isin(cls, GAP_CLASSES)
    cls[gap & ~_remove_small_patches(gap, p.min_patch_px)] = THIN
    patches = gap_patch_table(cls, scene.grid)
    stats = summarize_gaps(scene, cls, result.ndvi_only_gap, result.valid_mask, patches, result.anchors, p,
                           result.stats["mode"], "random forest (user-labelled points)", result.warnings, result.features)
    diag["feature_importance"] = dict(zip(keys, map(float, rf.feature_importances_)))
    return replace(result, class_map=cls, stats=stats, patches=patches, method="random forest"), diag


# ---- outputs ----------------------------------------------------------------------------------
def gap_polygons(class_map: np.ndarray, grid: Grid, field_id: str = "", date: str = "") -> gpd.GeoDataFrame:
    """Vectorise gap classes into polygons (EPSG:4326) with class + area attributes."""
    mask = np.isin(class_map, GAP_CLASSES)
    recs = []
    if mask.any():
        for geom, val in shapes(class_map.astype(np.uint8), mask=mask, transform=grid.transform, connectivity=8):
            recs.append({"block_id": field_id, "date": date, "gap_class": CLASS_LABELS[int(val)],
                         "gap_type": "weed_covered" if int(val) == WEEDY_GAP else "bare_soil",
                         "geometry": shapely_shape(geom)})
    gdf = gpd.GeoDataFrame(recs, columns=["block_id", "date", "gap_class", "gap_type", "geometry"], geometry="geometry", crs=grid.crs)
    if len(gdf):
        gdf["area_ha"] = gdf.geometry.area / 10_000.0
    else:
        gdf["area_ha"] = []
    return gdf.to_crs(4326)


def write_class_geotiff(path, class_map: np.ndarray, grid: Grid) -> None:
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=1, dtype="uint8",
                       crs=grid.crs, transform=grid.transform, nodata=0, compress="deflate") as dst:
        dst.write(class_map.astype(np.uint8), 1)
        dst.write_colormap(1, {k: tuple(int(CLASS_COLORS[k][i:i + 2], 16) for i in (1, 3, 5)) + (255,) for k in CLASS_LABELS})
        dst.update_tags(classes="; ".join(f"{k}={v}" for k, v in CLASS_LABELS.items()))


def write_float_geotiff(path, arr: np.ndarray, grid: Grid, nodata: float = -9999.0, tags: dict | None = None) -> None:
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=1, dtype="float32",
                       crs=grid.crs, transform=grid.transform, nodata=nodata, compress="deflate") as dst:
        dst.write(np.where(np.isfinite(arr), arr, nodata).astype(np.float32), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


# ---- validation against reference polygons ----------------------------------------------------
def validate_against_reference(result: GapResult, scene: Scene, reference: gpd.GeoDataFrame, type_col: str = "gap_type") -> dict:
    """Compare detected gaps with field-verified gap polygons (e.g. UAV-digitised).

    Only evaluates pixels inside the analysed area (valid mask). Reports the fused result and the
    NDVI-only baseline so you can see whether Sentinel-1 + height actually helps *on your data*.
    """
    ref = reference.to_crs(scene.grid.epsg)
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
    if ref.empty:
        raise ValueError("Reference layer has no usable polygons.")
    ref_gap = rasterize([(g, 1) for g in ref.geometry], out_shape=scene.grid.shape, transform=scene.grid.transform,
                        fill=0, dtype="uint8").astype(bool)
    area = result.valid_mask

    def metrics(pred: np.ndarray) -> dict:
        p, r = pred & area, ref_gap & area
        tp, fp, fn = int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        pa = scene.grid.pixel_area_ha
        return {"precision": prec, "recall": rec, "f1": f1, "iou": iou,
                "detected_ha": float(p.sum() * pa), "reference_ha": float(r.sum() * pa),
                "area_error_ha": float((p.sum() - r.sum()) * pa),
                "area_error_pct": float(100 * (p.sum() - r.sum()) / r.sum()) if r.sum() else float("nan")}

    out = {"fused": metrics(np.isin(result.class_map, GAP_CLASSES)), "ndvi_only": metrics(result.ndvi_only_gap)}
    if type_col in ref.columns:
        weedy_ref = rasterize([(g, 1) for g, t in zip(ref.geometry, ref[type_col].astype(str).str.lower())
                               if "weed" in t], out_shape=scene.grid.shape, transform=scene.grid.transform,
                              fill=0, dtype="uint8").astype(bool) & area
        if weedy_ref.any():
            out["weed_covered_recall_fused"] = float((np.isin(result.class_map, (WEEDY_GAP,)) & weedy_ref).sum() / weedy_ref.sum())
            out["weed_covered_recall_ndvi_only"] = float((result.ndvi_only_gap & weedy_ref).sum() / weedy_ref.sum())
    return out


def validate_against_truth_raster(result: GapResult, truth: np.ndarray) -> dict:
    """Pixel agreement against the synthetic demo truth (0 healthy, 1 bare, 2 weedy, 3 thin). Demo only."""
    m = result.valid_mask
    mapping = {0: HEALTHY, 1: BARE_GAP, 2: WEEDY_GAP, 3: THIN}
    t = np.zeros_like(result.class_map)
    for k, v in mapping.items():
        t[truth == k] = v
    t_gap, p_gap = np.isin(t, GAP_CLASSES) & m, np.isin(result.class_map, GAP_CLASSES) & m
    tp = int((t_gap & p_gap).sum())
    return {"gap_recall": tp / max(int(t_gap.sum()), 1), "gap_precision": tp / max(int(p_gap.sum()), 1),
            "weedy_recall": float(((result.class_map == WEEDY_GAP) & (t == WEEDY_GAP) & m).sum() / max(int(((t == WEEDY_GAP) & m).sum()), 1)),
            "ndvi_only_weedy_recall": float((result.ndvi_only_gap & (t == WEEDY_GAP) & m).sum() / max(int(((t == WEEDY_GAP) & m).sum()), 1)),
            "true_gap_ha": float(t_gap.sum() * 0.01), "detected_gap_ha": float(p_gap.sum() * 0.01),
            "ndvi_only_gap_ha": float((result.ndvi_only_gap & m).sum() * 0.01)}
