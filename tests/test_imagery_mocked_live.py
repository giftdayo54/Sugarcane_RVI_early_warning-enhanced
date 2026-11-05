"""Live-path logic with the Sentinel Hub network layer mocked (no credentials needed)."""
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, mapping
from sentinelhub import SHConfig

import src.imagery as im
from src.demo_scene import demo_field_geometry, make_demo_scene
from src.raster_calc import S2_BANDS

GEOM = demo_field_geometry()


def cfg():
    c = SHConfig(); c.sh_client_id, c.sh_client_secret = "x", "y"; c.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return c


def s2_array(clear: bool):
    sc = make_demo_scene(seed=1)
    arr = np.zeros(sc.grid.shape + (14,), "float32")
    for i, b in enumerate(S2_BANDS):
        arr[:, :, i] = np.nan_to_num(sc.bands[b])
    arr[:, :, 12] = 4 if clear else 9          # SCL: vegetation vs cloud
    arr[:, :, 13] = sc.field_mask
    return arr


def test_search_s2_filters_cloud_dedupes_and_sorts(monkeypatch):
    foot = mapping(box(34.5, -16.0, 35.0, -15.5))
    items = [{"id": "a", "geometry": foot, "properties": {"datetime": "2026-10-20T08:20:00Z", "eo:cloud_cover": 10}},
             {"id": "b", "geometry": foot, "properties": {"datetime": "2026-10-28T08:20:00Z", "eo:cloud_cover": 5}},
             {"id": "b2", "geometry": foot, "properties": {"datetime": "2026-10-28T08:20:05Z", "eo:cloud_cover": 5}},
             {"id": "c", "geometry": foot, "properties": {"datetime": "2026-10-25T08:20:00Z", "eo:cloud_cover": 95}},
             {"id": "far", "geometry": mapping(box(10, 10, 11, 11)), "properties": {"datetime": "2026-10-30T08:20:00Z", "eo:cloud_cover": 1}}]
    monkeypatch.setattr(im, "_search", lambda *a, **k: items)
    df = im.search_s2_scenes(GEOM, "2026-10-01", "2026-11-01", cfg(), max_cloud=80)
    assert df["date"].tolist() == ["2026-10-28", "2026-10-20"]


def test_latest_clear_walks_back_until_field_is_clear(monkeypatch):
    calls = []

    def fake_dl(reqs, config, threads=1):
        calls.append(1)
        return [s2_array(clear=len(calls) >= 3)]   # first two dates cloudy, third clear
    monkeypatch.setattr(im, "_download", fake_dl)
    sdf = pd.DataFrame({"datetime": pd.to_datetime(["2026-10-28", "2026-10-23", "2026-10-18", "2026-10-13"]),
                        "date": ["2026-10-28", "2026-10-23", "2026-10-18", "2026-10-13"], "cloud_cover_pct": [60, 50, 5, 2]})
    sc = im.find_latest_clear_scene("F1", GEOM, sdf, cfg(), min_clear_fraction=0.9)
    assert sc.date == "2026-10-18" and sc.clear_fraction > 0.99 and len(calls) == 3


def test_latest_clear_falls_back_to_clearest_with_warning(monkeypatch):
    monkeypatch.setattr(im, "_download", lambda *a, **k: [s2_array(clear=False)])
    sdf = pd.DataFrame({"datetime": pd.to_datetime(["2026-10-28"]), "date": ["2026-10-28"], "cloud_cover_pct": [60]})
    sc = im.find_latest_clear_scene("F1", GEOM, sdf, cfg())
    assert "clearest" in sc.meta["warning"]


def test_s1_composite_uses_nearest_dates_and_skips_low_coverage(monkeypatch):
    sc = make_demo_scene(seed=2, with_s1=False)
    sc.date = "2026-10-28"
    cat = pd.DataFrame({"id": list("abcde"), "datetime": pd.to_datetime(
        ["2026-10-16T03:30Z", "2026-10-22T03:30Z", "2026-10-27T03:30Z", "2026-11-02T03:30Z", "2026-11-07T03:30Z"])})
    monkeypatch.setattr(im, "search_s1_scenes", lambda *a, **k: cat)
    seen = {}

    def fake_dl(reqs, config, threads=4):
        seen["n"] = len(reqs)
        out = []
        for i in range(len(reqs)):
            arr = np.zeros(sc.grid.shape + (3,), "float32")
            arr[:, :, 0], arr[:, :, 1] = 0.12, 0.03
            arr[:, :, 2] = sc.field_mask if i != 1 else 0   # second one has no coverage
            out.append(arr)
        return out
    monkeypatch.setattr(im, "_download", fake_dl)
    s1 = im.fetch_s1_composite(sc, cfg(), window_days=12, max_scenes=3)
    assert seen["n"] == 3 and s1["n_scenes"] == 2
    assert np.nanmedian(s1["VH"][sc.field_mask]) == pytest.approx(0.03, rel=1e-3)
    assert s1["dates"][0] == "2026-10-27"


def test_s1_composite_errors_clearly(monkeypatch):
    sc = make_demo_scene(with_s1=False)
    monkeypatch.setattr(im, "search_s1_scenes", lambda *a, **k: pd.DataFrame(columns=["id", "datetime"]))
    with pytest.raises(ValueError, match="No Sentinel-1"):
        im.fetch_s1_composite(sc, cfg())
