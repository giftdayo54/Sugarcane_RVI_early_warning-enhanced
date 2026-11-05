"""Pipeline sanity tests on SYNTHETIC scenes. They show the logic behaves as designed; they do NOT
measure real-world accuracy (that needs your own field-verified reference polygons)."""
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src.demo_scene import make_demo_scene
from src.imagery import apply_affine
from src.gap_detection import (BARE_GAP, GAP_CLASSES, HEALTHY, NODATA, THIN, WEEDY_GAP, GapParams, analyze_scene,
                               classify_supervised, expected_cane_height, gap_polygons, params_from_config,
                               sample_features_at_points, validate_against_reference, validate_against_truth_raster)


@pytest.fixture(scope="module")
def scene():
    return make_demo_scene(seed=7)


@pytest.fixture(scope="module")
def result(scene):
    return analyze_scene(scene, GapParams(h_ref_m=3.0), age_days=240)


def test_area_bookkeeping_adds_up(result):
    s = result.stats
    assert s["healthy_ha"] + s["thin_ha"] + s["bare_gap_ha"] + s["weedy_gap_ha"] == pytest.approx(s["analysed_ha"])
    assert s["total_gap_ha"] == pytest.approx(s["bare_gap_ha"] + s["weedy_gap_ha"])
    assert s["gap_pct"] == pytest.approx(100 * s["total_gap_ha"] / s["analysed_ha"])
    assert s["analysed_ha"] < s["field_ha"]            # edge buffer removed
    assert (result.class_map[~result.valid_mask] == NODATA).all()


def test_s1_height_separates_weedy_gaps_that_ndvi_misses(scene, result):
    v = validate_against_truth_raster(result, scene.truth)
    assert v["ndvi_only_weedy_recall"] < 0.05           # NDVI alone cannot see green gaps
    assert v["weedy_recall"] > 0.6                      # fused method recovers most of them (synthetic)
    assert v["gap_precision"] > 0.8
    assert result.stats["total_gap_ha"] > result.stats["ndvi_only_gap_ha"] * 1.5


def test_weedy_pixels_are_green_and_short(result):
    w = result.class_map == WEEDY_GAP
    assert np.nanmedian(result.features["ndvi"][w]) >= 0.35
    assert np.nanmedian(result.features["height_m"][w]) < 0.5 * 3.0
    b = result.class_map == BARE_GAP
    assert np.nanmedian(result.features["ndvi"][b]) < 0.35
    h = result.class_map == HEALTHY
    assert np.nanmedian(result.features["height_m"][h]) > 0.75 * 3.0 * 0.95


def test_s2_only_fallback_flags_limitation(scene):
    r = analyze_scene(scene, GapParams(h_ref_m=3.0, use_s1=False))
    assert r.stats["mode"] == "S2 only" and r.stats["weedy_gap_ha"] == 0
    assert any("weed" in w.lower() for w in r.warnings)
    assert r.stats["total_gap_ha"] == pytest.approx(r.stats["ndvi_only_gap_ha"])


def test_young_crop_warns(scene):
    r = analyze_scene(scene, GapParams(h_ref_m=3.0), age_days=60)
    assert any("not closed" in w for w in r.warnings)


def test_cloud_reduces_analysed_area_and_warns():
    sc = make_demo_scene(seed=3, cloud_fraction=0.6)
    r = analyze_scene(sc, GapParams(h_ref_m=3.0))
    full = analyze_scene(make_demo_scene(seed=3), GapParams(h_ref_m=3.0))
    assert r.stats["analysed_ha"] < 0.8 * full.stats["analysed_ha"]
    assert any("cloud-free" in w for w in r.warnings)


def test_reference_height_scales_thresholds(scene):
    # a much taller "average cane" makes the same S1 level look shorter -> more height-gap pixels, never fewer
    tall = analyze_scene(scene, GapParams(h_ref_m=3.0, h_gap_frac=0.7))
    base = analyze_scene(scene, GapParams(h_ref_m=3.0, h_gap_frac=0.5))
    assert tall.stats["weedy_gap_ha"] >= base.stats["weedy_gap_ha"]


def test_measured_chm_overrides_s1(scene):
    chm = np.where(scene.truth == 0, 3.0, np.where(scene.truth == 2, 0.4, np.where(scene.truth == 1, 0.0, 2.0))).astype("float32")
    r = analyze_scene(scene, GapParams(h_ref_m=3.0), chm=chm)
    assert r.stats["mode"] == "S2 + measured canopy height"
    v = validate_against_truth_raster(r, scene.truth)
    assert v["weedy_recall"] > 0.8
    with pytest.raises(ValueError, match="shape"):
        analyze_scene(scene, GapParams(h_ref_m=3.0), chm=np.zeros((3, 3)))


def test_polygons_area_matches_raster_area(scene, result):
    gdf = gap_polygons(result.class_map, scene.grid, "DEMO-01", scene.date)
    assert gdf.crs.to_epsg() == 4326 and set(gdf["gap_type"]) == {"bare_soil", "weed_covered"}
    assert gdf["area_ha"].sum() == pytest.approx(result.stats["total_gap_ha"], rel=1e-6)


def test_validation_perfect_reference_scores_one(scene, result):
    ref = gap_polygons(result.class_map, scene.grid)
    v = validate_against_reference(result, scene, ref)
    assert v["fused"]["f1"] == pytest.approx(1.0) and v["fused"]["area_error_ha"] == pytest.approx(0.0)
    assert v["ndvi_only"]["recall"] < v["fused"]["recall"]
    assert "weed_covered_recall_fused" in v and v["weed_covered_recall_ndvi_only"] < v["weed_covered_recall_fused"]


def test_supervised_rf_runs_and_respects_valid_mask(scene, result):
    rng = np.random.default_rng(1)
    pts, labs = [], []
    for code, lab in [(0, "healthy"), (1, "bare_gap"), (2, "weedy_gap")]:
        rr, cc = np.where((scene.truth == code) & result.valid_mask)
        for i in rng.choice(len(rr), 25, replace=False):
            x, y = apply_affine(scene.grid.transform, cc[i] + 0.5, rr[i] + 0.5)
            pts.append(Point(x, y)); labs.append(lab)
    gdf = gpd.GeoDataFrame({"label": labs}, geometry=pts, crs=scene.grid.crs)
    train = sample_features_at_points(result, scene, gdf)
    assert len(train) == 75
    res2, diag = classify_supervised(result, scene, train)
    assert diag["cv_accuracy"] > 0.8 and res2.method == "random forest"
    assert (res2.class_map[~result.valid_mask] == NODATA).all()
    with pytest.raises(ValueError):
        classify_supervised(result, scene, train.iloc[:2])


def test_param_validation_and_config_loading():
    with pytest.raises(ValueError):
        GapParams(h_ref_m=0).validate()
    with pytest.raises(ValueError):
        GapParams(h_gap_frac=0.8, h_ok_frac=0.6).validate()
    p = params_from_config({"gaps": {"ndvi_bare": 0.4, "bogus": 1}}, h_ref_m=2.5, min_patch_px=None)
    assert p.ndvi_bare == 0.4 and p.h_ref_m == 2.5 and p.min_patch_px == 4


def test_height_curve_monotonic_and_bounded():
    h = [expected_cane_height(a) for a in (30, 90, 150, 240, 360)]
    assert all(b > a for a, b in zip(h, h[1:])) and h[-1] < 3.6 and h[0] < 0.3
