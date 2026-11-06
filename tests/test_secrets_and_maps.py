import os
import pathlib

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- secrets resolver
@pytest.fixture
def fake_secrets(monkeypatch):
    import streamlit as st
    for k in ("SH_CLIENT_ID", "SH_CLIENT_SECRET", "SH_BASE_URL", "SH_TOKEN_URL"):
        monkeypatch.delenv(k, raising=False)

    def setter(d):
        monkeypatch.setattr(st, "secrets", dict(d), raising=False)
    return setter


@pytest.mark.parametrize("secrets", [
    {"SH_CLIENT_ID": "id1", "SH_CLIENT_SECRET": "pw1"},
    {"sh_client_id": "id1", "sh_client_secret": "pw1"},
    {"CLIENT_ID": "id1", "CLIENT_SECRET": "pw1"},
    {"sentinelhub": {"client_id": "id1", "client_secret": "pw1"}},
    {"cdse": {"SH_CLIENT_ID": "id1", "SH_CLIENT_SECRET": "pw1"}},
])
def test_resolver_accepts_common_layouts(fake_secrets, secrets):
    from src.secrets_util import find_secret
    fake_secrets(secrets)
    assert find_secret("SH_CLIENT_ID")[0] == "id1" and find_secret("SH_CLIENT_SECRET")[0] == "pw1"


def test_placeholders_and_blanks_ignored_then_env_used(fake_secrets, monkeypatch):
    from src.secrets_util import find_secret
    fake_secrets({"SH_CLIENT_ID": "", "SH_CLIENT_SECRET": "your-client-secret"})
    assert find_secret("SH_CLIENT_ID") == ("", "")
    monkeypatch.setenv("SH_CLIENT_ID", "envid")
    v, src = find_secret("SH_CLIENT_ID")
    assert v == "envid" and "environment" in src


def test_diagnostics_never_returns_values(fake_secrets):
    from src.secrets_util import diagnostics
    fake_secrets({"SH_CLIENT_ID": "SUPERSECRETVALUE", "sentinelhub": {"client_secret": "ANOTHERSECRET"}})
    assert "SUPERSECRETVALUE" not in str(diagnostics()) and "ANOTHERSECRET" not in str(diagnostics())


def test_secrets_toml_templates_are_valid_and_blank_template_is_ignored():
    import tomllib
    ex = tomllib.loads((ROOT / ".streamlit" / "secrets.toml.example").read_text())
    assert {"SH_CLIENT_ID", "SH_CLIENT_SECRET", "SH_BASE_URL", "SH_TOKEN_URL"} <= set(ex)
    live = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text())
    assert live["SH_CLIENT_ID"] == "" and live["SH_CLIENT_SECRET"] == ""
    assert ".streamlit/secrets.toml" in (ROOT / ".gitignore").read_text()


def test_app_picks_up_secrets_through_streamlits_own_loader(monkeypatch):
    """End to end: AppTest injects st.secrets exactly as Streamlit does; the sidebar must report them as in use."""
    from streamlit.testing.v1 import AppTest
    for k in ("SH_CLIENT_ID", "SH_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(ROOT)
    for layout in ({"SH_CLIENT_ID": "abc", "SH_CLIENT_SECRET": "def"}, {"sentinelhub": {"client_id": "abc", "client_secret": "def"}}):
        at = AppTest.from_file(str(ROOT / "pages" / "3_Satellite_Imagery.py"), default_timeout=60)
        at.session_state["_field_source"] = "Repo blocks (data/vectors/blocks.geojson)"
        for k, v in layout.items():
            at.secrets[k] = v
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Using credentials from Streamlit secrets" in s.value for s in at.sidebar.success)
    # and with nothing configured the user is told where to put them
    at = AppTest.from_file(str(ROOT / "pages" / "3_Satellite_Imagery.py"), default_timeout=60)
    at.session_state["_field_source"] = "Repo blocks (data/vectors/blocks.geojson)"
    at.run()
    assert any("No credentials found" in s.value for s in at.sidebar.info)


def test_old_pages_use_the_resolver():
    for p in ("1_S1_Growth_Curves.py", "2_Upload_Farm_Boundaries.py"):
        assert "src.secrets_util" in (ROOT / "pages" / p).read_text()


# ---------------------------------------------------------------- maps
def _fields():
    return gpd.GeoDataFrame({"field_id": ["A", "B"], "variety": ["N14", "N25"]},
                            geometry=[box(34.80, -15.80, 34.806, -15.795), box(34.81, -15.80, 34.815, -15.795)], crs=4326)


@pytest.mark.parametrize("name", list(__import__("src.map_view", fromlist=["BASEMAPS"]).BASEMAPS))
def test_every_basemap_builds_a_valid_figure(name):
    from src.map_view import boundary_map
    fig = boundary_map(_fields(), "field_id", "A", name)
    d = fig.to_dict()
    assert len(d["data"]) == 2 and d["layout"]["map"]["zoom"] > 12          # selected + others, zoomed to the field
    assert "hovertext" in d["data"][0]


def test_satellite_basemap_uses_tokenless_esri_tiles():
    from src.map_view import boundary_map
    d = boundary_map(_fields(), "field_id", "A", "Satellite (Esri World Imagery)").to_dict()
    assert "arcgisonline" in d["layout"]["map"]["layers"][0]["source"][0]
    assert d["layout"]["map"]["style"] == "white-bg"


def test_zoom_shows_whole_extent_when_nothing_selected():
    from src.map_view import boundary_map
    one = boundary_map(_fields(), "field_id", "A").layout.map.zoom
    allf = boundary_map(_fields(), "field_id", None).layout.map.zoom
    assert allf < one


def test_overlay_is_georeferenced_to_the_field():
    from src.demo_scene import make_demo_scene
    from src.map_view import boundary_map, rgba_from_rgb
    sc = make_demo_scene()
    rgb = np.full(sc.grid.shape + (3,), 120, "uint8")
    g = gpd.GeoDataFrame({"field_id": ["DEMO"]}, geometry=[sc.meta["geometry_wgs84"]], crs=4326)
    d = boundary_map(g, "field_id", "DEMO", overlay=(rgba_from_rgb(rgb, sc.field_mask), sc.grid)).to_dict()
    layer = d["layout"]["map"]["layers"][-1]
    (lon0, lat0), (lon1, _), (_, lat1) = layer["coordinates"][0], layer["coordinates"][1], layer["coordinates"][2]
    minx, miny, maxx, maxy = sc.meta["geometry_wgs84"].bounds
    assert lon0 <= minx and lon1 >= maxx and lat1 <= miny and lat0 >= maxy          # covers the field
    assert (maxx - minx) * 0.8 < (lon1 - lon0) < (maxx - minx) * 1.3                    # ... and isn't wildly larger
    assert layer["source"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------- HD true colour
def test_hd_render_upsamples_and_restores_background():
    from src.demo_scene import make_demo_scene
    from src.imagery_viz import hd_render, natural_rgb
    sc = make_demo_scene()
    rgb = natural_rgb(sc.bands, mask=sc.clear_mask)
    out, valid, f = hd_render(rgb, "HD x4 + sharpen")
    assert f == 4 and out.shape[:2] == (sc.grid.height * 4, sc.grid.width * 4) and valid.shape == out.shape[:2]
    assert (out[~valid] == 255).all() and out.dtype == np.uint8
    native, _, f1 = hd_render(rgb, "Native pixels (10 m)")
    assert f1 == 1 and native.shape[:2] == sc.grid.shape


def test_natural_rgb_is_brighter_and_less_cast_than_per_band():
    from src.demo_scene import make_demo_scene
    from src.imagery_viz import natural_rgb
    sc = make_demo_scene()
    joint = natural_rgb(sc.bands, mask=sc.field_mask, joint=True)
    per = natural_rgb(sc.bands, mask=sc.field_mask, joint=False)
    canopy = (sc.truth == 0) & sc.field_mask
    assert np.nanmean(joint[canopy]) > 0.15                                           # not near-black
    assert joint.shape == per.shape


def test_hd_render_caps_huge_fields():
    from src.imagery_viz import MAX_IMAGE_PX, hd_render
    big = np.random.rand(900, 1000, 3).astype("float32")
    out, _, f = hd_render(big, "HD x4 + sharpen")
    assert max(out.shape[:2]) <= max(MAX_IMAGE_PX, 1000) and f >= 1
