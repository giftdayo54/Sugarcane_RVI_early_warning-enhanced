"""
pages/4_Gap_Analysis.py

Detect and quantify canopy gaps in sugarcane fields from Sentinel-2 + Sentinel-1, and separate
gaps that are BARE from gaps that are covered by (green) weeds, using the average common cane
height as the reference for how tall a healthy canopy should be. See src/gap_detection.py for
the method and its caveats.
"""
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import ui_common as ui
from src.gap_detection import (expected_cane_height, BARE_GAP, CLASS_COLORS, CLASS_LABELS, GAP_CLASSES, HEALTHY, THIN, WEEDY_GAP, GapParams,
                               analyze_scene, classify_supervised, gap_polygons, params_from_config, sample_features_at_points,
                               validate_against_reference, validate_against_truth_raster)
from src.gap_reporting import height_for_age, save_result, severity
from src.imagery_viz import (RENDER_MODES, class_figure, class_png, colorize, hd_render, histogram_figure, image_figure, natural_rgb, png_bytes,
                             raster_figure)
from src.map_view import BASEMAPS, DEFAULT_BASEMAP, MAP_CONFIG, boundary_map, rgba_from_rgb
from src.utils import load_config

st.set_page_config(page_title="Gap Analysis", layout="wide")
st.title("🌱 Canopy gap analysis · Sentinel-1 + Sentinel-2")
st.caption("Quantify gaps per field and tell **bare-soil gaps** from **weed-covered gaps** (green, but far shorter than cane).")

try:
    CFG = load_config("config/config.yaml")
except Exception:
    CFG = {}
G = CFG.get("gaps", {}) or {}
S1D, S2D = G.get("s1", {}) or {}, G.get("s2", {}) or {}

# ------------------------------------------------------------------ sidebar
fields, demo = ui.field_source_sidebar()
if fields is None:
    st.info("Upload a field-boundary GeoJSON in the sidebar (or choose the repo blocks / the synthetic demo field).")
    st.stop()
ids = sorted(fields["field_id"].astype(str))

scope = "Single field"
if not demo and len(ids) > 1:
    scope = st.sidebar.radio("Scope", ["Single field", "All fields (batch)"], horizontal=True)
sel_ids = ids if scope.startswith("All") else [st.sidebar.selectbox("Field", ids)]

st.sidebar.header("Sentinel-2 search")
start, end = ui.date_range_sidebar(int(S2D.get("lookback_days", 60)))
min_clear = st.sidebar.slider("Min. cloud-free share of the field", 0.5, 1.0, float(S2D.get("min_clear_fraction", 0.85)), 0.05)
max_tile_cloud = 100 if demo else st.sidebar.slider("Skip scenes with tile cloud cover above (%)", 10, 100, int(S2D.get("max_tile_cloud", 80)), 5)

st.sidebar.header("Sentinel-1")
orbit_opts = ["ASCENDING", "DESCENDING"]
s1_orbit = st.sidebar.radio("Orbit", orbit_opts, index=orbit_opts.index(S1D.get("orbit", "ASCENDING")), horizontal=True,
                            help="Match the orbit used for your RVI baseline.")
s1_window = st.sidebar.slider("± days around the optical image", 3, 30, int(S1D.get("window_days", 12)))
s1_max = st.sidebar.slider("Max scenes to composite", 1, 6, int(S1D.get("max_scenes", 4)))
s1_speckle = st.sidebar.select_slider("Speckle filter (px)", [1, 3, 5], value=int(S1D.get("speckle_px", 3)))
creds = ui.credentials_sidebar() if not demo else {"threads": 4}

if demo:
    st.warning("**Synthetic demo data.** The field, spectra and gaps are simulated so you can try the workflow without credentials. "
               "Results here say nothing about real accuracy.")

with st.expander("How weed-covered gaps are separated from cane", expanded=False):
    st.markdown(
        "- **NDVI alone** cannot tell cane from weeds - both are green - so a gap full of weeds looks like healthy canopy.\n"
        "- **Sentinel-1 cross-pol (VH)** responds to canopy structure. We turn it into a *relative canopy-height index* for each field: "
        "1.0 = that field's own healthy-cane level, 0 = bare soil, then multiply by the **average common cane height** you enter, "
        "giving a height in metres.\n"
        "- A pixel that is **green (NDVI ≥ threshold) but shorter than ~½ the cane height** is classed as a **weed-covered gap**. "
        "Low NDVI + short = **bare gap**. Green + near full height = **healthy cane**. Everything else vegetated = **thin / stunted canopy**.\n"
        "- **Caveat:** Sentinel-1 does not measure height. The proxy assumes most of the field is healthy cane and that VH rises with canopy "
        "volume; it saturates in dense canopy and is affected by soil moisture and incidence angle. If you have a UAV/LiDAR canopy-height "
        "model, upload it and it replaces the proxy. Always check results against field-verified polygons (validation panel below).\n"
        "- At 10 m resolution, gaps narrower than ~1–2 pixels are not resolvable; the numbers are *detectable gap area*, not a stool count.")

# ------------------------------------------------------------------ crop & height inputs
st.subheader("1 · Crop age and average cane height")
st.caption("The height reference drives the weed test. Enter the **average common cane height (m)** you measure/expect for this crop "
           "class and age. If blank, the config's generic growth curve is suggested (replace it with your own numbers).")


def _prop(row, names):
    for n in names:
        if n in row.index and pd.notna(row[n]) and str(row[n]).strip() not in ("", "nan", "None"):
            return row[n]
    return None


rows = []
for fid in sel_ids:
    r = fields[fields["field_id"].astype(str) == fid].iloc[0]
    pd_raw = _prop(r, ["planting_date", "harvest_date", "last_harvest_date"])
    pdt = pd.to_datetime(pd_raw, errors="coerce") if pd_raw is not None else pd.NaT
    h_raw = _prop(r, ["avg_cane_height_m", "cane_height_m"])
    if h_raw is not None:
        h = float(h_raw)
    elif pd.notna(pdt):
        h = height_for_age(CFG, (pd.Timestamp(date.today()) - pdt).days)
    else:
        h = None
    rows.append({"field_id": fid, "planting_or_last_harvest_date": pdt, "avg_cane_height_m": h})
crop_df = pd.DataFrame(rows)
crop_df["planting_or_last_harvest_date"] = pd.to_datetime(crop_df["planting_or_last_harvest_date"])
edited = st.data_editor(
    crop_df, hide_index=True, use_container_width=True, key=f"crop_editor_{'|'.join(sel_ids)[:80]}",
    column_config={
        "field_id": st.column_config.TextColumn("Field", disabled=True),
        "planting_or_last_harvest_date": st.column_config.DateColumn("Planting / last harvest date (for crop age)"),
        "avg_cane_height_m": st.column_config.NumberColumn("Average common cane height (m)", min_value=0.1, max_value=8.0, step=0.1, format="%.2f"),
    })

with st.expander("🗺️ Field map - your boundaries over a basemap", expanded=False):
    gm1, gm2 = st.columns([2, 1])
    _bms = list(BASEMAPS)
    gbase = gm1.selectbox("Basemap", _bms, index=_bms.index(st.session_state.get("_basemap", DEFAULT_BASEMAP)), key="fm_basemap_4")
    st.session_state["_basemap"] = gbase
    gall = gm2.checkbox("Show all fields", value=True, key="fm_all_4")
    st.plotly_chart(boundary_map(fields if gall else fields[fields["field_id"].astype(str).isin(sel_ids)], "field_id", sel_ids, gbase),
                    use_container_width=True, config=MAP_CONFIG)

base = params_from_config(CFG)
with st.expander("📏 How to read the heights on this page (all in metres)", expanded=False):
    cur_cfg = (G.get("height_curve") or {})
    curve_kw = {k: cur_cfg[k] for k in ("h_max", "k", "t0") if k in cur_cfg}
    ages = np.arange(0, 421, 5)
    curve = pd.DataFrame({"Crop age (days)": ages, "Expected height (m)": [expected_cane_height(a, **curve_kw) for a in ages]})
    figc = px.line(curve, x="Crop age (days)", y="Expected height (m)", title="Generic expected cane height vs crop age (config curve)")
    pts = []
    for _, r in edited.iterrows():
        if pd.notna(r["planting_or_last_harvest_date"]) and pd.notna(r["avg_cane_height_m"]):
            pts.append({"field": r["field_id"], "age": (pd.Timestamp(date.today()) - pd.Timestamp(r["planting_or_last_harvest_date"])).days,
                        "h": float(r["avg_cane_height_m"])})
    if pts:
        pdf = pd.DataFrame(pts)
        figc.add_scatter(x=pdf["age"], y=pdf["h"], mode="markers+text", text=pdf["field"], textposition="top center", name="Your fields (today)",
                         marker=dict(size=11, color="#d84315"))
    figc.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(figc, use_container_width=True)
    st.markdown(
        "**Units:** every height on this page is in **metres (m)** - the crop is measured from the ground to the top of the canopy.\n\n"
        "**1. Average common cane height (the number you type)** - what a *healthy* canopy of this crop class and age should measure in the field. "
        "It is the yardstick. The curve above is only a generic starting guess (logistic, max ≈ 3.6 m, set in `config.yaml → gaps.height_curve`); "
        "replace it with heights you actually measure for your varieties and ages. Orange dots show your fields' ages vs the heights entered.\n\n"
        "**2. Height proxy map (m)** - *not a measured height.* Sentinel-1 VH is converted to a relative canopy index "
        "(0 = bare-soil radar level, 1 = this field's own healthy-cane level) and multiplied by the reference height. Read it as "
        "*'how cane-like is the canopy structure, expressed in metres of cane'*: ~0 m = bare ground, ≈ reference = full cane canopy, "
        "above the reference (up to 1.3×) = denser than the field's typical cane. A weed patch showing 0.8 m means its radar signal is about "
        "a quarter of cane's - the weeds are short, but 0.8 m is not a tape-measure value.\n\n"
        "**3. Thresholds in metres** - gap-height and full-canopy limits are fractions of the reference (table below). "
        "**4. Measured canopy-height raster** - if you upload a UAV/LiDAR model (Detection parameters), heights become true metres.\n\n"
        "**Not to be confused with** the *Growth Curves* page: its y-axis is Sentinel-1 VH backscatter in **dB**, not metres.")
    ex = []
    for _, r in edited.iterrows():
        if pd.notna(r["avg_cane_height_m"]):
            h0 = float(r["avg_cane_height_m"])
            ex.append({"Field": r["field_id"], "Reference height (m)": round(h0, 2),
                       f"Gap-height: shorter than (m)": round(h0 * base.h_gap_frac, 2),
                       "Full canopy: at least (m)": round(h0 * base.h_ok_frac, 2),
                       "Thin / stunted: in between": f"{h0 * base.h_gap_frac:.2f}-{h0 * base.h_ok_frac:.2f} m"})
    if ex:
        st.dataframe(pd.DataFrame(ex), hide_index=True, use_container_width=True)
        st.caption("Table uses the default fractions from config; the sliders in *Detection parameters* change them.")

# ------------------------------------------------------------------ detection parameters
with st.expander("2 · Detection parameters", expanded=False):
    base = params_from_config(CFG)
    p1, p2, p3 = st.columns(3)
    ndvi_bare = p1.slider("NDVI below this = no green cover", 0.10, 0.60, float(base.ndvi_bare), 0.01)
    h_gap = p2.slider("Height-gap: shorter than this × cane height", 0.20, 0.80, float(base.h_gap_frac), 0.05)
    h_ok = p3.slider("Full canopy: at least this × cane height", 0.50, 1.00, float(base.h_ok_frac), 0.05)
    q1, q2, q3, q4 = st.columns(4)
    metric = q1.radio("S1 metric for the height proxy", ["VH_dB", "RVI"], index=["VH_dB", "RVI"].index(base.s1_metric),
                      help="RVI is the index used by this repo's early-warning pipeline.")
    edge = q2.number_input("Edge buffer (m)", 0.0, 50.0, float(base.edge_buffer_m), 5.0, help="Ignores boundary/headland pixels.")
    min_px = q3.number_input("Min. gap patch (pixels)", 1, 50, int(base.min_patch_px), 1, help="1 pixel = 0.01 ha. Smaller patches are treated as noise.")
    min_age = q4.number_input("Canopy closes at age (days)", 30, 365, int(base.min_age_days), 10)
    use_s1_opt = st.checkbox("Use Sentinel-1 (needed to separate weed-covered gaps)", value=True)
    chm_file = None
    if scope == "Single field":
        chm_file = st.file_uploader("Optional: measured canopy-height raster (GeoTIFF, metres) - replaces the S1 proxy",
                                    type=["tif", "tiff"], key="chm_up")
    reuse = False
    cur = ui.current_scene(sel_ids[0], demo) if scope == "Single field" else None
    if cur is not None:
        reuse = st.checkbox(f"Use the scene already loaded on the Imagery page ({cur.date})", value=True)

params_base = replace(base, ndvi_bare=ndvi_bare, h_gap_frac=h_gap, h_ok_frac=h_ok, s1_metric=metric, edge_buffer_m=float(edge),
                      min_patch_px=int(min_px), min_age_days=int(min_age), use_s1=use_s1_opt)


# ------------------------------------------------------------------ runner
def run_one(fid: str, height: float, planting, status) -> tuple:
    geom = fields.loc[fields["field_id"].astype(str) == fid, "geometry"].iloc[0]
    if reuse and cur is not None and fid == cur.field_id:
        scene = cur
    else:
        status.write(f"**{fid}** · searching Sentinel-2...")
        sdf = ui.list_scenes(demo, geom, start, end, creds, max_tile_cloud)
        if sdf.empty:
            raise RuntimeError("no Sentinel-2 scenes in the date range")
        scene = ui.load_latest_clear(demo, fid, geom, sdf, creds, min_clear, progress=lambda m: status.write(f"**{fid}** · {m}"))
    p = params_base
    notes = []
    if p.use_s1 and scene.s1 is None:
        status.write(f"**{fid}** · compositing Sentinel-1 around {scene.date}...")
        try:
            ui.attach_s1(scene, demo, creds, s1_window, s1_orbit, s1_max, s1_speckle)
        except Exception as exc:
            notes.append(f"Sentinel-1 unavailable ({exc}); ran optical-only.")
            p = replace(p, use_s1=False)
    age = (pd.Timestamp(scene.date) - pd.Timestamp(planting)).days if planting is not None and pd.notna(planting) else None
    chm = None
    if chm_file is not None:
        chm = ui.read_raster_to_grid(chm_file, scene.grid)
    status.write(f"**{fid}** · classifying gaps...")
    res = analyze_scene(scene, replace(p, h_ref_m=float(height)), chm=chm, age_days=age)
    res.warnings = notes + res.warnings
    if notes:
        res.stats["notes"] = " | ".join(res.warnings)
    res.stats["crop_age_days"] = age
    return scene, res


def _inputs_ok() -> bool:
    bad = edited[edited["avg_cane_height_m"].isna()]["field_id"].tolist()
    if bad:
        st.error("Enter the average cane height for: " + ", ".join(bad))
        return False
    return True


st.subheader("3 · Run")
btn_label = "▶ Run gap analysis" if scope == "Single field" else f"▶ Run for all {len(sel_ids)} fields"
if st.button(btn_label, type="primary") and _inputs_ok():
    with st.status("Running gap analysis...", expanded=True) as status:
        try:
            if scope == "Single field":
                r0 = edited.iloc[0]
                scene, res = run_one(r0["field_id"], r0["avg_cane_height_m"], r0["planting_or_last_harvest_date"], status)
                st.session_state["gap_run"] = {"scene": scene, "result": res, "demo": demo}
                st.session_state.pop("gap_batch", None)
                st.session_state.pop("gap_rf", None)
            else:
                out_rows, out_polys, failures = [], [], []
                for i, r in edited.iterrows():
                    try:
                        scene, res = run_one(r["field_id"], r["avg_cane_height_m"], r["planting_or_last_harvest_date"], status)
                        s = dict(res.stats)
                        s["gap_severity"] = severity(s["gap_pct"], G.get("severity_pct"))
                        out_rows.append(s)
                        out_polys.append(gap_polygons(res.class_map, scene.grid, scene.field_id, scene.date))
                        st.session_state.setdefault("_batch_objs", {})[scene.field_id] = (scene, res)
                    except Exception as exc:
                        failures.append({"field_id": r["field_id"], "error": str(exc)})
                st.session_state["gap_batch"] = {"rows": out_rows, "polys": out_polys, "failures": failures}
                st.session_state.pop("gap_run", None)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Gap analysis failed: {exc}")


# ------------------------------------------------------------------ batch results
def show_batch(batch):
    if batch["failures"]:
        st.warning("Some fields failed: " + "; ".join(f"{f['field_id']} ({f['error']})" for f in batch["failures"]))
    if not batch["rows"]:
        return
    df = pd.DataFrame(batch["rows"])
    st.subheader("Results · all fields")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fields analysed", len(df))
    k2.metric("Total gap area", f"{df['total_gap_ha'].sum():.1f} ha")
    k3.metric("Area-weighted gap", f"{100 * df['total_gap_ha'].sum() / df['analysed_ha'].sum():.1f}%")
    k4.metric("Green gaps missed by NDVI", f"{df['weedy_gap_ha'].sum():.1f} ha")
    long = df.melt(id_vars="block_id", value_vars=["bare_gap_pct", "weedy_gap_pct"], var_name="type", value_name="% of field")
    long["type"] = long["type"].map({"bare_gap_pct": "Bare soil", "weedy_gap_pct": "Weed-covered"})
    fig = px.bar(long, x="block_id", y="% of field", color="type", title="Gap % by field",
                 color_discrete_map={"Bare soil": CLASS_COLORS[BARE_GAP], "Weed-covered": CLASS_COLORS[WEEDY_GAP]})
    st.plotly_chart(fig, use_container_width=True)
    show_cols = ["block_id", "date", "mode", "analysed_ha", "gap_pct", "bare_gap_pct", "weedy_gap_pct", "ndvi_only_gap_pct",
                 "total_gap_ha", "n_gap_patches", "largest_gap_ha", "gap_severity", "clear_fraction_pct", "notes"]
    st.dataframe(df[[c for c in show_cols if c in df.columns]].round(2).sort_values("gap_pct", ascending=False),
                 use_container_width=True, hide_index=True)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download gap report (CSV)", df.to_csv(index=False).encode(), "gap_report.csv", "text/csv")
    polys = pd.concat(batch["polys"], ignore_index=True) if batch["polys"] else None
    if polys is not None and len(polys):
        import geopandas as gpd
        d2.download_button("Download gap polygons (GeoJSON)", gpd.GeoDataFrame(polys, crs=4326).to_json().encode(), "gap_polygons.geojson", "application/geo+json")
    if d3.button("💾 Save into pipeline outputs/"):
        written = None
        for fid, (scene, res) in st.session_state.get("_batch_objs", {}).items():
            if fid in set(df["block_id"].astype(str)):
                written = save_result(scene, res, "outputs", CFG)
        st.success("Saved to outputs/reports/gap_report.csv and outputs/vectors/gap_polygons.geojson - the Home page will now show gap results." if written else "Nothing to save.")
    st.caption("Single-field mode shows the maps, threshold inspector and validation for one field.")


# ------------------------------------------------------------------ single-field results
def show_single(run):
    scene, result0, is_demo = run["scene"], run["result"], run["demo"]
    rf = st.session_state.get("gap_rf")
    result = result0
    if rf is not None and rf["field_id"] == scene.field_id:
        which = st.radio("Classifier shown", ["Rule-based (transparent thresholds)", "Random forest (trained on your points)"], horizontal=True)
        if which.startswith("Random"):
            result = rf["result"]
    s, p = result.stats, result.params
    boundary = scene.boundary_utm

    st.subheader(f"Results · {scene.field_id} · image {scene.date}")
    for w in result.warnings:
        st.warning(w)
    sev = severity(s["gap_pct"], G.get("severity_pct"))
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Analysed area", f"{s['analysed_ha']:.1f} ha", help="Cloud-free, inside the field, minus the edge buffer.")
    k2.metric("Total gap", f"{s['total_gap_ha']:.2f} ha", f"{s['gap_pct']:.1f}% · {sev}", delta_color="off")
    k3.metric("Bare-soil gap", f"{s['bare_gap_ha']:.2f} ha", f"{s['bare_gap_pct']:.1f}%", delta_color="off")
    k4.metric("Weed-covered gap", f"{s['weedy_gap_ha']:.2f} ha", f"{s['weedy_gap_pct']:.1f}%", delta_color="off")
    k5.metric("NDVI-only would say", f"{s['ndvi_only_gap_pct']:.1f}%", help="Gap % from optical NDVI alone, same threshold and patch filter.")
    k6.metric("Gap patches", f"{s['n_gap_patches']}", f"largest {s['largest_gap_ha']:.2f} ha", delta_color="off")
    if s["weedy_gap_ha"] > 0:
        st.info(f"**{s['weedy_gap_ha']:.2f} ha** ({s['weed_covered_share_of_gap_pct']:.0f}% of the detected gap area) is green but far shorter than cane - "
                f"an NDVI-only map would count it as healthy canopy. Mode: *{s['mode']}* · reference cane height {p.h_ref_m:.2f} m.")
    elif s["mode"] == "S2 only":
        st.info("Optical-only mode: weed-covered gaps cannot be separated from cane. Add Sentinel-1 or a canopy-height raster.")

    t_map, t_base, t_why, t_cmp, t_thr, t_patch, t_val, t_ml = st.tabs(
        ["🗺️ Map", "🛰️ On basemap", "🔬 Why NDVI misses weeds", "↔️ NDVI-only vs fused", "🎚️ Thresholds", "📋 Patches", "✅ Validation", "🤖 Train on my points"])

    with t_map:
        rmode = st.selectbox("True-colour rendering", list(RENDER_MODES), index=list(RENDER_MODES).index("HD x4 + sharpen"), key="gap_render")
        a, b = st.columns(2)
        hd_img, _, hd_scale = hd_render(natural_rgb(scene.bands, mask=scene.clear_mask), rmode)
        a.plotly_chart(image_figure(hd_img, scene.grid, boundary, f"True colour · {scene.date}", scale=hd_scale, height=520), use_container_width=True, config=MAP_CONFIG)
        b.plotly_chart(class_figure(result.class_map, scene.grid, boundary, "Gap classification"), use_container_width=True)
        st.caption("Amber = green gap that is short (weeds). Red = bare soil. Light green = vegetated but thin/stunted. Grey/blank = excluded (cloud, edge buffer, outside).")

    with t_base:
        bb1, bb2, bb3 = st.columns([2, 2, 1])
        _bm = list(BASEMAPS)
        gb = bb1.selectbox("Basemap", _bm, index=_bm.index(st.session_state.get("_basemap", DEFAULT_BASEMAP)), key="gap_basemap")
        st.session_state["_basemap"] = gb
        what = bb2.selectbox("Draped image", ["Gap classification", "True colour (HD)", "Gap outlines only"])
        op = bb3.slider("Opacity", 0.1, 1.0, 0.75, 0.05, key="gap_op")
        ov, outlines, smooth = None, None, False
        if what == "Gap classification":
            rgb_c = np.full(result.class_map.shape + (3,), 255, dtype=np.uint8)
            for k_, hx in CLASS_COLORS.items():
                rgb_c[result.class_map == k_] = [int(hx[i:i + 2], 16) for i in (1, 3, 5)]
            ov = (rgba_from_rgb(rgb_c, result.class_map != 0), scene.grid)
        elif what == "True colour (HD)":
            hd2, v2, _ = hd_render(natural_rgb(scene.bands, mask=scene.clear_mask), "HD x4 + sharpen")
            ov, smooth = (rgba_from_rgb(hd2, v2), scene.grid), True
        if what == "Gap outlines only" or what == "True colour (HD)":
            outlines = gap_polygons(result.class_map, scene.grid, scene.field_id, scene.date)
        st.plotly_chart(boundary_map(fields, "field_id", scene.field_id, gb, overlay=ov, overlay_opacity=op, overlay_smooth=smooth,
                                     title=f"{scene.field_id} · {scene.date} · {what}", height=640, extra_polygons=outlines),
                        use_container_width=True, config=MAP_CONFIG)
        st.caption("Gap pixels draped on real-world imagery - zoom in on the Esri satellite tiles to check whether a flagged patch is "
                   "bare ground or a weedy/short-canopy area before sending a team.")

    with t_why:
        f = result.features
        cols = st.columns(3)
        cols[0].plotly_chart(raster_figure(f["ndvi"], scene.grid, boundary, "NDVI (optical)", "RdYlGn", 0, 0.95), use_container_width=True)
        if "height_m" in f:
            cols[1].plotly_chart(raster_figure(f["height_m"], scene.grid, boundary, f"Canopy height proxy (m) - ref {p.h_ref_m:.1f} m", "YlGn", 0, p.h_ref_m * 1.2, "m"),
                                 use_container_width=True)
            cols[1].caption(f"Metres of *cane-equivalent* canopy: ≈{p.h_ref_m:.1f} m = full cane canopy, 0 m = bare-soil radar level. "
                            f"Below {p.h_gap_frac * p.h_ref_m:.1f} m = gap-height; ≥ {p.h_ok_frac * p.h_ref_m:.1f} m = full canopy. "
                            "A proxy from radar structure, not a tape-measure height (see the heights guide above).")
        if "vh_db" in f:
            cols[2].plotly_chart(raster_figure(f["vh_db"], scene.grid, boundary, "Sentinel-1 VH (dB)", "viridis"), use_container_width=True)
        if "height_m" in f:
            m = result.valid_mask & np.isfinite(f["ndvi"]) & np.isfinite(f["height_m"])
            idx = np.flatnonzero(m.ravel())
            if idx.size:
                idx = np.random.default_rng(0).choice(idx, min(4000, idx.size), replace=False)
                sample = pd.DataFrame({"NDVI": f["ndvi"].ravel()[idx], "Height proxy (m)": f["height_m"].ravel()[idx],
                                       "class": [CLASS_LABELS[int(c)] for c in result.class_map.ravel()[idx]]})
                fig = px.scatter(sample, x="NDVI", y="Height proxy (m)", color="class", opacity=0.6, title="Every pixel: greenness vs canopy height",
                                 color_discrete_map={CLASS_LABELS[k]: v for k, v in CLASS_COLORS.items()})
                fig.add_vline(x=p.ndvi_bare, line_dash="dash"); fig.add_hline(y=p.h_gap_frac * p.h_ref_m, line_dash="dash")
                fig.add_hline(y=p.h_ok_frac * p.h_ref_m, line_dash="dot")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Weed-covered gaps sit at the top-left of the cane cloud's greenness but **below** the dashed height line: green, yet short. "
                           "NDVI alone only sees the vertical dashed line.")

    with t_cmp:
        a, b = st.columns(2)
        nd_only = np.where(result.valid_mask, np.where(result.ndvi_only_gap, 2, 1), 0).astype(np.uint8)
        a.plotly_chart(class_figure(nd_only, scene.grid, boundary, f"NDVI-only gaps · {s['ndvi_only_gap_pct']:.1f}%"), use_container_width=True)
        b.plotly_chart(class_figure(result.class_map, scene.grid, boundary, f"Fused S1+S2 gaps · {s['gap_pct']:.1f}%"), use_container_width=True)
        st.caption(f"Difference = {max(s['total_gap_ha'] - s['ndvi_only_gap_ha'], 0):.2f} ha that optical-only analysis does not flag.")

    with t_thr:
        f = result.features
        st.caption("Tune the thresholds in *2 · Detection parameters* and re-run. The dashed lines show where they currently cut.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(histogram_figure(f["ndvi"], "NDVI", {"bare/green": p.ndvi_bare}), use_container_width=True)
        if "height_m" in f:
            c2.plotly_chart(histogram_figure(f["height_m"], "Canopy height proxy (m)",
                                             {"gap-height": p.h_gap_frac * p.h_ref_m, "full canopy": p.h_ok_frac * p.h_ref_m}), use_container_width=True)
        an = result.anchors
        if an:
            st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in an.items()})

    with t_patch:
        if result.patches.empty:
            st.success("No gap patches above the minimum size.")
        else:
            show = result.patches.drop(columns=["x", "y"]).round(3)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.plotly_chart(px.histogram(result.patches, x="area_ha", nbins=20, title="Gap patch sizes (ha)"), use_container_width=True)

    with t_val:
        if is_demo and scene.truth is not None:
            st.markdown("**Synthetic demo only** - agreement with the simulated layout (proves the pipeline runs, not real accuracy):")
            v = validate_against_truth_raster(result, scene.truth)
            st.dataframe(pd.DataFrame([{"gap recall": v["gap_recall"], "gap precision": v["gap_precision"], "weed-gap recall (fused)": v["weedy_recall"],
                                        "weed-gap recall (NDVI only)": v["ndvi_only_weedy_recall"], "true gap ha": v["true_gap_ha"],
                                        "detected gap ha": v["detected_gap_ha"], "NDVI-only gap ha": v["ndvi_only_gap_ha"]}]).round(3),
                         hide_index=True, use_container_width=True)
        st.markdown("**Validate against your own field-verified gaps** (e.g. UAV-digitised polygons). Optional property `gap_type` "
                    "(`bare` / `weed_covered`) enables a per-type check.")
        ref_up = st.file_uploader("Reference gap polygons (GeoJSON)", type=["geojson", "json"], key="ref_up")
        if ref_up is not None:
            try:
                m = validate_against_reference(result, scene, ui.read_vector_upload(ref_up))
                tbl = pd.DataFrame(m["fused"], index=["Fused S1+S2"])
                tbl.loc["NDVI only"] = m["ndvi_only"]
                st.dataframe(tbl.round(3), use_container_width=True)
                if "weed_covered_recall_fused" in m:
                    st.write(f"Weed-covered gap recall - fused: **{m['weed_covered_recall_fused']:.2f}**, NDVI only: **{m['weed_covered_recall_ndvi_only']:.2f}**")
                st.caption("Evaluated only on pixels analysed (cloud-free, inside the edge buffer). Reference polygons should be from a date close to the image.")
            except Exception as exc:
                st.error(f"Validation failed: {exc}")

    with t_ml:
        st.markdown("Optional supervised classifier. Upload **points** with a `label` property: `healthy`, `bare_gap`, `weedy_gap`, `thin`. "
                    "A Random Forest is trained on the same pixel features (NDVI, NDRE, NDMI, BSI, VH, VV, height proxy).")
        pts_up = st.file_uploader("Labelled points (GeoJSON)", type=["geojson", "json"], key="pts_up")
        if pts_up is not None and st.button("Train and classify"):
            try:
                train = sample_features_at_points(result0, scene, ui.read_vector_upload(pts_up))
                res_rf, diag = classify_supervised(result0, scene, train)
                st.session_state["gap_rf"] = {"field_id": scene.field_id, "result": res_rf, "diag": diag}
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if rf is not None and rf["field_id"] == scene.field_id:
            d = rf["diag"]
            st.write(f"Trained on {d['n_points']} points: {d['class_counts']}")
            if "cv_accuracy" in d:
                st.write(f"Cross-validated accuracy {d['cv_accuracy']:.2f}, macro-F1 {d['cv_macro_f1']:.2f}. "
                         "These are optimistic when points are close together (spatial autocorrelation) - validate on a separate field.")
            st.bar_chart(pd.Series(d["feature_importance"], name="importance"))

    st.subheader("Export")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.download_button("Class raster (GeoTIFF)", ui.class_tif_bytes(result.class_map, scene.grid), f"{scene.field_id}_{scene.date}_gap_classes.tif", "image/tiff")
    polys = gap_polygons(result.class_map, scene.grid, scene.field_id, scene.date)
    e2.download_button("Gap polygons (GeoJSON)", polys.to_json().encode(), f"{scene.field_id}_{scene.date}_gaps.geojson", "application/geo+json")
    e3.download_button("Summary (CSV)", pd.DataFrame([s]).to_csv(index=False).encode(), f"{scene.field_id}_{scene.date}_gap_summary.csv", "text/csv")
    e4.download_button("Class map (PNG)", class_png(result.class_map), f"{scene.field_id}_{scene.date}_gap_classes.png", "image/png")
    if e5.button("💾 Save into pipeline outputs/"):
        paths = save_result(scene, result, "outputs", CFG)
        st.success("Saved. The Home page now lists this field's gap results.")
        st.caption(f"{paths['report']} · {paths['polygons']} · {paths['raster']}")


if st.session_state.get("gap_batch"):
    show_batch(st.session_state["gap_batch"])
elif st.session_state.get("gap_run") and (scope == "Single field"):
    run = st.session_state["gap_run"]
    if run["demo"] == demo and run["scene"].field_id in sel_ids:
        show_single(run)
    else:
        st.info("Press **Run gap analysis** to analyse the selected field.")
else:
    st.info("Set the cane height above and press **Run gap analysis**.")
