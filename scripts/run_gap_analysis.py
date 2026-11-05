"""
Batch gap analysis for every block in data/vectors/blocks.geojson (or --blocks), writing into outputs/ so the
Home dashboard picks the results up. Run after (or independently of) `python main.py`.

    export SH_CLIENT_ID=... SH_CLIENT_SECRET=...
    python scripts/run_gap_analysis.py --lookback-days 60
    python scripts/run_gap_analysis.py --demo            # synthetic data, no credentials

Cane height per block: property `avg_cane_height_m` in the GeoJSON, else the config growth curve from
`planting_date` (blocks.geojson property, or the earliest date in observations.csv), else --default-height.
"""
import argparse, os, sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gap_detection import analyze_scene, params_from_config  # noqa: E402
from src.gap_reporting import height_for_age, save_result  # noqa: E402
from src.imagery import fetch_s1_composite, find_latest_clear_scene, search_s2_scenes  # noqa: E402
from src.s1_growth_curves import authenticate_sentinel_hub, load_fields  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--blocks"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--lookback-days", type=int); ap.add_argument("--default-height", type=float)
    ap.add_argument("--only", nargs="*", help="restrict to these block_ids")
    a = ap.parse_args()
    cfg, log = load_config(a.config), get_logger("gap_batch")
    G = cfg.get("gaps", {}) or {}
    out = cfg["paths"]["outputs"]
    params = params_from_config(cfg)

    if a.demo:
        from src.demo_scene import make_demo_scene
        scene = make_demo_scene()
        res = analyze_scene(scene, replace(params, h_ref_m=a.default_height or 3.0), age_days=240)
        print(save_result(scene, res, out, cfg)); return

    cid, sec = os.environ.get("SH_CLIENT_ID"), os.environ.get("SH_CLIENT_SECRET")
    if not cid or not sec:
        sys.exit("Set SH_CLIENT_ID and SH_CLIENT_SECRET (Copernicus Data Space OAuth client).")
    sh = authenticate_sentinel_hub(cid, sec, os.environ.get("SH_BASE_URL", "https://sh.dataspace.copernicus.eu"),
                                   os.environ.get("SH_TOKEN_URL", "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"))
    fields = load_fields(a.blocks or cfg["paths"]["blocks"])
    if a.only: fields = fields[fields.field_id.isin(a.only)]
    obs = pd.read_csv(cfg["paths"]["observations"], parse_dates=["planting_date"]) if Path(cfg["paths"]["observations"]).exists() else None
    s1d, s2d = G.get("s1", {}) or {}, G.get("s2", {}) or {}
    end = date.today(); start = end - timedelta(days=a.lookback_days or int(s2d.get("lookback_days", 60)))
    for _, f in fields.iterrows():
        fid = f.field_id
        try:
            sdf = search_s2_scenes(f.geometry, str(start), str(end), sh, float(s2d.get("max_tile_cloud", 80)))
            if sdf.empty: log.warning("%s: no Sentinel-2 scenes", fid); continue
            scene = find_latest_clear_scene(fid, f.geometry, sdf, sh, float(s2d.get("min_clear_fraction", 0.85)))
            p = params
            try:
                scene.s1 = fetch_s1_composite(scene, sh, int(s1d.get("window_days", 12)), s1d.get("orbit", "ASCENDING"),
                                              int(s1d.get("max_scenes", 4)), int(s1d.get("speckle_px", 3)))
            except Exception as exc:
                log.warning("%s: Sentinel-1 unavailable (%s) -> optical-only", fid, exc); p = replace(p, use_s1=False)
            plant = pd.to_datetime(f.get("planting_date"), errors="coerce") if "planting_date" in f.index else pd.NaT
            if pd.isna(plant) and obs is not None and fid in set(obs.block_id.astype(str)):
                plant = obs.loc[obs.block_id.astype(str) == fid, "planting_date"].min()
            age = (pd.Timestamp(scene.date) - plant).days if pd.notna(plant) else None
            h = f.get("avg_cane_height_m") if "avg_cane_height_m" in f.index else None
            h = float(h) if h is not None and pd.notna(h) else (height_for_age(cfg, age) if age is not None else a.default_height)
            if h is None: log.warning("%s: no cane height (add avg_cane_height_m / planting_date or --default-height)", fid); continue
            res = analyze_scene(scene, replace(p, h_ref_m=h), age_days=age)
            save_result(scene, res, out, cfg)
            log.info("%s %s: gap %.1f%% (bare %.1f, weedy %.1f; NDVI-only %.1f) mode=%s", fid, scene.date, res.stats["gap_pct"],
                     res.stats["bare_gap_pct"], res.stats["weedy_gap_pct"], res.stats["ndvi_only_gap_pct"], res.stats["mode"])
        except Exception as exc:
            log.error("%s failed: %s", fid, exc)


if __name__ == "__main__":
    main()
