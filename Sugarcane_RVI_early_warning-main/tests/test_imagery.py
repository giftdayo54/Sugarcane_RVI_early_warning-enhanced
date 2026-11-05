import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from sentinelhub import SHConfig

from src.imagery import (Grid, build_s1_request, build_s2_request, clear_from_scl, composite_s1, decode_s2_array,
                         erode_mask, make_grid, nan_boxcar, polygon_mask)
from src.raster_calc import S2_BANDS


def cdse_config():
    c = SHConfig()
    c.sh_client_id, c.sh_client_secret = "x", "y"
    c.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return c


FIELD = box(34.80, -15.80, 34.806, -15.795)  # ~0.6 x 0.55 km


def test_grid_snaps_to_10m_lattice_and_covers_field():
    grid, g = make_grid(FIELD)
    minx, miny, maxx, maxy = grid.bounds
    assert minx % 10 == 0 and miny % 10 == 0 and maxx % 10 == 0 and maxy % 10 == 0
    assert (maxx - minx) / grid.width == pytest.approx(10) and (maxy - miny) / grid.height == pytest.approx(10)
    assert minx < g.bounds[0] and maxx > g.bounds[2] and miny < g.bounds[1] and maxy > g.bounds[3]
    # polygon mask area is close to true polygon area
    m = polygon_mask(g, grid)
    assert m.sum() * 100 == pytest.approx(g.area, rel=0.03)


def test_oversize_and_tiny_fields_rejected():
    with pytest.raises(ValueError, match="too large"):
        make_grid(box(34.0, -16.0, 34.5, -15.5))
    with pytest.raises(ValueError, match="too small"):
        make_grid(box(34.8, -15.8, 34.80001, -15.80001), pad_px=0)


def test_requests_target_configured_endpoint_not_commercial():
    """Regression: stock DataCollection constants pin services.sentinel-hub.com and override config."""
    grid, g = make_grid(FIELD)
    for req in (build_s2_request(grid, g, "2026-10-28T08:20:00Z", cdse_config()),
                build_s1_request(grid, g, "2026-10-28T03:30:00Z", cdse_config())):
        assert req.download_list[0].url.startswith("https://sh.dataspace.copernicus.eu/")


def test_s2_request_payload():
    grid, g = make_grid(FIELD)
    req = build_s2_request(grid, g, pd.Timestamp("2026-10-28T08:20:00Z"), cdse_config())
    p = req.payload
    assert p["input"]["data"][0]["type"] == "sentinel-2-l2a"
    assert p["output"]["width"] == grid.width and p["output"]["height"] == grid.height
    for b in S2_BANDS + ["SCL", "dataMask"]:
        assert f'"{b}"' in p["evalscript"]
    assert "bands: 14" in p["evalscript"]
    assert p["input"]["bounds"]["properties"]["crs"].endswith(str(grid.epsg))
    assert p["input"]["data"][0]["processing"]["upsampling"] == "BILINEAR"


def test_s1_request_matches_existing_pipeline_processing():
    grid, g = make_grid(FIELD)
    p = build_s1_request(grid, g, pd.Timestamp("2026-10-28T03:30:00Z"), cdse_config(), "DESCENDING").payload
    d = p["input"]["data"][0]
    assert d["type"] == "sentinel-1-grd"
    assert d["processing"] == {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True}
    assert d["dataFilter"]["orbitDirection"] == "DESCENDING"


def test_decode_s2_array_masks_nodata_and_validates_shape():
    arr = np.random.rand(5, 6, 14).astype("float32")
    arr[:, :, 13] = 1; arr[0, 0, 13] = 0; arr[:, :, 12] = 4
    bands, scl, dm = decode_s2_array(arr)
    assert list(bands) == S2_BANDS and np.isnan(bands["B04"][0, 0]) and not dm[0, 0] and scl[1, 1] == 4
    with pytest.raises(ValueError):
        decode_s2_array(np.zeros((5, 6, 3)))


def test_clear_mask_from_scl():
    scl = np.array([[4, 5, 8, 9], [3, 10, 6, 0]], dtype="uint8")
    assert clear_from_scl(scl, scl.shape).tolist() == [[True, True, False, False], [False, False, True, False]]


def test_erode_mask():
    m = np.zeros((9, 9), bool); m[1:8, 1:8] = True
    e = erode_mask(m, 10, 10)
    assert e.sum() == 25 and erode_mask(m, 0).sum() == m.sum()


def test_nan_boxcar_ignores_nans_and_keeps_footprint():
    a = np.full((6, 6), 2.0, "float32"); a[0, :] = np.nan; a[3, 3] = 20
    out = nan_boxcar(a, 3)
    assert np.isnan(out[0]).all()                      # footprint preserved
    assert np.isfinite(out[1]).all() and out[1, 1] == pytest.approx(2.0)  # no NaN bleed-in
    assert out[3, 3] < 20


def test_composite_s1_median_reduces_outlier():
    base = np.full((4, 4), 0.1, "float32")
    spike = base.copy(); spike[1, 1] = 5.0
    vv, vh = composite_s1([base, base, spike], [base * .2] * 3, speckle_px=1)
    assert vv[1, 1] == pytest.approx(0.1)
