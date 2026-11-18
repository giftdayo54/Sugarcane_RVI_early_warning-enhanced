import pathlib

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.io import MemoryFile
from sentinelhub import SHConfig

import src.imagery as im
from src.demo_scene import make_demo_fine, make_demo_scene

ROOT = pathlib.Path(__file__).resolve().parents[1]


def cfg():
    c = SHConfig(); c.sh_client_id, c.sh_client_secret = "x", "y"; c.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return c


@pytest.fixture(scope="module")
def scene():
    sc = make_demo_scene(seed=5)
    sc.meta["acquisition_dt"] = pd.Timestamp("2026-10-28T08:20:00Z")
    return sc


def test_plan_fine_grid_exact_at_1m(scene):
    g, rx, ry, capped = im.plan_fine_grid(scene.grid, 1.0)
    assert (g.width, g.height) == (scene.grid.width * 10, scene.grid.height * 10) and (rx, ry) == (1.0, 1.0) and not capped
    assert g.bounds == scene.grid.bounds and g.transform.a == pytest.approx(1.0)


def test_separate_x_and_y_resolution(scene):
    g, rx, ry, _ = im.plan_fine_grid(scene.grid, 2.0, 1.0)
    assert (rx, ry) == (2.0, 1.0) and g.width == scene.grid.width * 5 and g.height == scene.grid.height * 10
    assert g.transform.a == pytest.approx(2.0) and abs(g.transform.e) == pytest.approx(1.0)


def test_large_field_is_coarsened_not_rejected():
    big = im.Grid(32736, (500000, 8000000, 505000, 8004000), 500, 400)        # 5 km x 4 km
    g, rx, ry, capped = im.plan_fine_grid(big, 1.0)
    assert capped and g.width <= 2500 and g.height <= 2500 and rx >= 2.0
    with pytest.raises(ValueError):
        im.plan_fine_grid(big, 0)


def test_request_payload_matches_copernicus_style_custom_resolution(scene):
    g, *_ = im.plan_fine_grid(scene.grid, 1.0)
    req = im.build_fine_request(scene, g, cfg(), ["B04", "B03", "B02", "B08"], "BICUBIC")
    p = req.payload
    assert (p["output"]["width"], p["output"]["height"]) == (g.width, g.height)
    assert p["input"]["data"][0]["processing"]["upsampling"] == "BICUBIC"
    assert "bands: 5" in p["evalscript"] and "s.B08" in p["evalscript"] and "dataMask" in p["evalscript"]
    assert req.download_list[0].url.startswith("https://sh.dataspace.copernicus.eu/")
    assert p["input"]["bounds"]["bbox"] == list(g.bounds)
    with pytest.raises(ValueError):
        im.build_fine_request(scene, g, cfg(), ["B04"], "LANCZOS")
    with pytest.raises(ValueError):
        im.evalscript_for_bands(["B99"])


def test_fetch_fine_decodes_masks_and_clear(scene, monkeypatch):
    def fake(reqs, config, threads=1):
        w, h = reqs[0].payload["output"]["width"], reqs[0].payload["output"]["height"]
        arr = np.full((h, w, 4), 0.2, "float32"); arr[:, :, 3] = 1; arr[:5, :5, 3] = 0
        return [arr]
    monkeypatch.setattr(im, "_download", fake)
    f = im.fetch_s2_fine(scene, cfg(), ["B04", "B03", "B02"], 1.0)
    assert f.grid.shape == (scene.grid.height * 10, scene.grid.width * 10)
    assert np.isnan(f.bands["B04"][:5, :5]).all() and f.valid.sum() > 0 and f.clear.sum() <= f.valid.sum()
    assert f.valid.sum() == pytest.approx(scene.field_mask.sum() * 100, rel=0.06)       # polygon at 1 m ~ 100 px per 10 m px


def test_clear_mask_resampling_keeps_cloud_footprint():
    cloudy = make_demo_scene(seed=3, cloud_fraction=0.5)
    f = make_demo_fine(cloudy, ["B04", "B03", "B02"], 1.0)
    frac_coarse = cloudy.clear_mask.sum() / cloudy.field_mask.sum()
    frac_fine = f.clear.sum() / f.valid.sum()
    assert frac_fine == pytest.approx(frac_coarse, abs=0.03)


def test_demo_fine_is_smooth_interpolation_of_native(scene):
    f = make_demo_fine(scene, ["B08"], 1.0)
    coarse_mean = np.nanmean(scene.bands["B08"][scene.field_mask])
    assert np.nanmean(f.bands["B08"][f.valid]) == pytest.approx(coarse_mean, rel=0.03)
    from scipy import ndimage as ndi
    interior = ndi.binary_erosion(f.valid, iterations=5)                      # ignore the polygon edge itself
    gy, gx = np.gradient(np.nan_to_num(f.bands["B08"]))
    fine_step = np.hypot(gx, gy)[interior].max()                              # reflectance change per 1 m pixel
    native = scene.bands["B08"]
    cy, cx = np.gradient(np.nan_to_num(native))
    native_step = np.hypot(cx, cy)[ndi.binary_erosion(scene.field_mask, iterations=1)].max()   # per 10 m pixel
    assert fine_step < 0.5 * native_step and f.synthetic                      # far smoother than blocky 10 m steps


@pytest.mark.parametrize("crs", ["UTM", "EPSG:4326"])
def test_analytical_geotiff_export(scene, crs):
    f = make_demo_fine(scene, ["B04", "B03", "B02", "B08"], 2.0, 1.0)
    data = im.write_bands_geotiff(f, ["B04", "B03", "B02", "B08"], crs)
    with MemoryFile(data) as mem, mem.open() as src:
        assert src.count == 4 and src.dtypes[0] == "float32" and src.descriptions == ("B04", "B03", "B02", "B08")
        assert "resampling" in src.tags() and "smoother" in src.tags()["note"]
        if crs == "UTM":
            assert src.crs.to_epsg() == scene.grid.epsg and (src.res[0], src.res[1]) == pytest.approx((2.0, 1.0))
        else:
            assert src.crs.to_epsg() == 4326
            assert src.res[0] == pytest.approx(2.0 / 111320, rel=0.05) and src.res[1] == pytest.approx(1.0 / 110574, rel=0.05)
            lon0, lat0, lon1, lat1 = src.bounds
            minx, miny, maxx, maxy = scene.meta["geometry_wgs84"].bounds
            assert lon0 <= minx + 1e-4 and lon1 >= maxx - 1e-4 and lat0 <= miny + 1e-4 and lat1 >= maxy - 1e-4
        band = src.read(4)
        assert (band == -9999).any() and np.nanmax(band) < 1.5 and src.nodata == -9999


def test_sharpening_radius_scales_with_resolution_gap():
    from src.imagery_viz import hd_render, natural_rgb
    sc = make_demo_scene(seed=5)
    f = make_demo_fine(sc, ["B04", "B03", "B02"], 1.0)
    rgb = natural_rgb(f.bands, mask=f.clear)
    soft, _, _ = hd_render(rgb, "As fetched")
    crisp, _, _ = hd_render(rgb, "Sharpen (recommended)", native_res_m=10, pixel_m=1.0)
    edge = lambda a: np.abs(np.diff(a.astype(float), axis=1)).mean()
    assert crisp.shape == soft.shape == f.grid.shape + (3,) and edge(crisp) > edge(soft) * 1.05


def test_page_download_flow_end_to_end(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "pages" / "3_Satellite_Imagery.py"), default_timeout=120)
    at.session_state["_field_source"] = "Synthetic demo field (no credentials)"
    at.run()
    [b for b in at.button if "latest clear" in b.label][0].click(); at.run()
    assert not at.exception, [e.value for e in at.exception]
    # default display resolution is 1 m/px and the page says what that means
    assert any("1 m/px" in sb.label or "1 m/px" in str(sb.value) for sb in at.sidebar.selectbox) or at.session_state["_disp_res"] == 1.0
    texts = [m.value for m in at.sidebar.markdown] + [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.sidebar.caption]
    assert any("no new ground detail" in t for t in texts)
    [b for b in at.button if b.label == "Prepare GeoTIFF"][0].click(); at.run()
    assert not at.exception, [e.value for e in at.exception]
    data, fname = at.session_state["_dl_bytes"]
    assert fname.endswith("_1m.tif")
    with MemoryFile(data) as mem, mem.open() as src:
        assert src.count == 4 and src.res == pytest.approx((1.0, 1.0))
