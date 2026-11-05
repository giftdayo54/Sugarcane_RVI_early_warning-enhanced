"""Regression: current sentinelhub-py has no `raise_download_errors`. These tests patch only the lowest-level
DownloadClient.download (with its REAL signature), so an unsupported keyword reproduces the user's TypeError."""
import pathlib
import re

import numpy as np
import pytest
from sentinelhub import DownloadClient, SHConfig

import src.imagery as im
from src.s1_growth_curves import download_tolerant


def cfg():
    c = SHConfig(); c.sh_client_id, c.sh_client_secret = "x", "y"; c.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return c


class Req:  # stands in for a DownloadRequest
    def __init__(self, ok=True): self.ok = ok


def real_signature_download(calls):
    def download(self, download_requests, max_threads=None, decode_data=True, show_progress=False):
        reqs = list(download_requests)
        calls.append(len(reqs))
        if any(not r.ok for r in reqs):
            raise RuntimeError("401 Unauthorized" if all(not r.ok for r in reqs) else "400 bad request")
        return [np.ones((2, 2, 3), "float32") for _ in reqs]
    return download


def test_batch_download_uses_only_supported_kwargs(monkeypatch):
    calls = []
    monkeypatch.setattr(DownloadClient, "download", real_signature_download(calls))
    out = download_tolerant([Req(), Req()], cfg())
    assert len(out) == 2 and all(o is not None for o in out)


def test_one_bad_request_does_not_lose_the_good_ones(monkeypatch):
    calls = []
    monkeypatch.setattr(DownloadClient, "download", real_signature_download(calls))
    out = download_tolerant([Req(), Req(ok=False), Req()], cfg())
    assert out[0] is not None and out[1] is None and out[2] is not None


def test_everything_failing_fails_fast_with_the_real_error(monkeypatch):
    calls = []
    monkeypatch.setattr(DownloadClient, "download", real_signature_download(calls))
    with pytest.raises(RuntimeError, match="401"):
        download_tolerant([Req(ok=False) for _ in range(50)], cfg())
    assert len(calls) <= 5          # 1 batch + 3 singles, not 51 calls


def test_single_request_error_propagates(monkeypatch):
    monkeypatch.setattr(DownloadClient, "download", real_signature_download([]))
    with pytest.raises(RuntimeError, match="401"):
        download_tolerant([Req(ok=False)], cfg())


def test_imagery_download_wrapper_reports_cause(monkeypatch):
    monkeypatch.setattr(DownloadClient, "download", real_signature_download([]))

    class R:  # minimal SentinelHubRequest stand-in
        download_list = [Req(ok=False)]
    with pytest.raises(RuntimeError, match="Failed downloading from Sentinel Hub: 401"):
        im._download([R()], cfg())


def test_latest_clear_reports_last_error_instead_of_generic_message(monkeypatch):
    import pandas as pd
    from src.demo_scene import demo_field_geometry
    monkeypatch.setattr(im, "_download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Failed downloading: 401 Unauthorized")))
    sdf = pd.DataFrame({"datetime": pd.to_datetime(["2026-10-28"]), "date": ["2026-10-28"], "cloud_cover_pct": [5]})
    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        im.find_latest_clear_scene("F", demo_field_geometry(), sdf, cfg())


def test_no_module_passes_the_removed_keyword():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = [str(p) for p in list((root / "src").glob("*.py")) + list((root / "pages").glob("*.py")) + [root / "s1_growth_curves.py"]
                 if re.search(r"raise_download_errors\s*=", p.read_text())]
    assert not offenders, offenders


def test_existing_growth_curve_requests_target_configured_endpoint():
    from src.s1_growth_curves import build_process_requests
    import geopandas as gpd
    from shapely.geometry import box
    import pandas as pd
    fields = gpd.GeoDataFrame({"field_id": ["A"], "crop_type": ["plant"], "area_m2": [300000.0]}, geometry=[box(34.80, -15.80, 34.803, -15.797)], crs=4326)
    acq = {"A": [{"datetime": pd.Timestamp("2026-10-20T03:30:00Z"), "date": "2026-10-20"}]}
    infos = build_process_requests(fields, acq, cfg())
    assert infos and infos[0]["request"].download_list[0].url.startswith("https://sh.dataspace.copernicus.eu/")
