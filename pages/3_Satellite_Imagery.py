"""
pages/3_Satellite_Imagery.py

Pull recent Sentinel-2 imagery for a field, look at every band, build composites, and compute
any index with a typed band-maths expression (NDVI, NDRE, NDWI, NDMI, EVI, custom ...).
Sentinel-1 VV/VH/RVI can be added to the same grid. Independent of (but compatible with) the
RVI early-warning pipeline: it uses the same boundaries, credentials and Sentinel Hub client.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import ui_common as ui
from src.imagery_viz import COMPOSITES, colorize, histogram_figure, image_figure, png_bytes, raster_figure, rgb_composite, to_uint8
from src.raster_calc import (ALIASES, FUNCTIONS, PRESETS, S2_BAND_INFO, S2_BANDS, available_presets, build_layers, evaluate,
                             referenced_names, summarize, validate_expression, ExpressionError)

st.set_page_config(page_title="Satellite Imagery", layout="wide")
st.title("🛰️ Satellite imagery & raster calculator")
st.caption("Pull the latest clear Sentinel-2 image for a field, inspect every band, and compute indices from band expressions.")

# ------------------------------------------------------------------ sidebar
fields, demo = ui.field_source_sidebar()
if fields is None:
    st.info("Upload a field-boundary GeoJSON in the sidebar (or choose the repo blocks / the synthetic demo field).")
    st.stop()

field_id = st.sidebar.selectbox("Field", sorted(fields["field_id"].astype(str)))
geom = fields.loc[fields["field_id"].astype(str) == field_id, "geometry"].iloc[0]

st.sidebar.header("Imagery search")
start, end = ui.date_range_sidebar(45)
min_clear = st.sidebar.slider("Min. cloud-free share of the field", 0.5, 1.0, 0.85, 0.05,
                              help="'Latest clear image' keeps going back in time until the field itself is at least this clear.")
max_tile_cloud = st.sidebar.slider("Skip scenes with tile cloud cover above (%)", 10, 100, 80, 5) if not demo else 100
creds = ui.credentials_sidebar() if not demo else {"threads": 4}

if demo:
    st.warning("**Synthetic demo data.** This field, its spectra and its gaps are simulated to let you try the tools without credentials.")

# ------------------------------------------------------------------ scene selection
st.subheader("1 · Get imagery")
c1, c2, c3 = st.columns([1, 1, 2])
find_clicked = c1.button("🔎 List available scenes", use_container_width=True)
latest_clicked = c2.button("⚡ Pull latest clear image", type="primary", use_container_width=True)

if find_clicked or latest_clicked or "img_scenes_df" not in st.session_state or st.session_state.get("img_scenes_key") != (field_id, demo, start, end):
    if find_clicked or latest_clicked:
        try:
            with st.spinner("Searching the Sentinel-2 catalog..."):
                df = ui.list_scenes(demo, geom, start, end, creds, max_tile_cloud)
            st.session_state["img_scenes_df"], st.session_state["img_scenes_key"] = df, (field_id, demo, start, end)
        except Exception as exc:
            st.error(f"Catalog search failed: {exc}")
            st.stop()

scenes_df = st.session_state.get("img_scenes_df")
if scenes_df is not None and st.session_state.get("img_scenes_key") == (field_id, demo, start, end):
    if scenes_df.empty:
        st.warning("No Sentinel-2 scenes found for this field and date range. Widen the dates or relax the cloud filter.")
        st.stop()
    if latest_clicked:
        status = st.empty()
        try:
            ui.load_latest_clear(demo, field_id, geom, scenes_df, creds, min_clear, progress=lambda m: status.info(m))
            status.empty()
        except Exception as exc:
            status.empty(); st.error(f"Could not pull imagery: {exc}")
    with c3:
        pick = st.selectbox("…or choose a specific date", scenes_df["date"].tolist(),
                            format_func=lambda d: f"{d}   (tile cloud {scenes_df.loc[scenes_df['date'] == d, 'cloud_cover_pct'].iloc[0]:.0f}%)"
                            if pd.notna(scenes_df.loc[scenes_df['date'] == d, 'cloud_cover_pct'].iloc[0]) else d)
        if st.button("Load selected date"):
            try:
                with st.spinner(f"Downloading {pick}..."):
                    ui.load_scene_for_date(demo, field_id, geom, scenes_df[scenes_df["date"] == pick].iloc[0], creds)
            except Exception as exc:
                st.error(f"Download failed: {exc}")
    with st.expander(f"{len(scenes_df)} scene(s) found", expanded=False):
        st.dataframe(scenes_df[["date", "cloud_cover_pct"]], use_container_width=True, hide_index=True)

scene = ui.current_scene(field_id, demo)
if scene is None:
    st.info("Click **Pull latest clear image** (or list scenes and load a date) to begin.")
    st.stop()
if scene.meta.get("warning"):
    st.warning(scene.meta["warning"])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Image date", scene.date)
m2.metric("Cloud-free (this field)", f"{scene.clear_fraction:.0%}")
m3.metric("Tile cloud cover", "n/a" if scene.cloud_catalog is None else f"{scene.cloud_catalog:.0f}%")
m4.metric("Field area", f"{scene.field_area_ha:.1f} ha" if scene.field_area_ha else "n/a")
m5.metric("Sentinel-1", f"{scene.s1['n_scenes']} scene(s)" if scene.s1 else "not loaded")

tab_comp, tab_bands, tab_calc, tab_hist, tab_s1 = st.tabs(
    ["🖼️ Composites", "📚 Bands", "🧮 Raster calculator", "🕒 Recent scenes", "📡 Sentinel-1"])

boundary = scene.boundary_utm

# ------------------------------------------------------------------ composites
with tab_comp:
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    choice = cc1.selectbox("Composite", list(COMPOSITES) + ["Custom (pick 3 bands)"])
    if choice.startswith("Custom"):
        r, g, b = st.columns(3)
        trip = (r.selectbox("Red channel", S2_BANDS, index=S2_BANDS.index("B08")),
                g.selectbox("Green channel", S2_BANDS, index=S2_BANDS.index("B04")),
                b.selectbox("Blue channel", S2_BANDS, index=S2_BANDS.index("B03")))
    else:
        trip = COMPOSITES[choice]
    lo, hi = cc2.slider("Stretch percentiles", 0, 100, (2, 98))
    mask_clouds = cc3.checkbox("Hide clouds / shadow", value=True)
    rgb = rgb_composite(scene.bands, trip, lo, hi, scene.clear_mask if mask_clouds else scene.field_mask)
    st.plotly_chart(image_figure(to_uint8(rgb), scene.grid, boundary, f"{choice} · {scene.date}"), use_container_width=True)
    st.download_button("Download composite PNG", png_bytes(to_uint8(rgb)), f"{field_id}_{scene.date}_composite.png", "image/png")
    st.caption("Black line = field boundary. White = outside the field, masked, or no data. "
               "In false colour, healthy cane is bright red; bare soil is cyan/grey; clouds are white.")

# ------------------------------------------------------------------ bands
with tab_bands:
    st.caption("Sentinel-2 L2A surface reflectance. B01/B09 (60 m) and the 20 m bands are resampled to the 10 m grid.")
    bc1, bc2, bc3 = st.columns(3)
    band_cmap = bc1.selectbox("Colour map", ["gray", "viridis", "magma", "YlGn"], key="band_cmap")
    plo, phi = bc2.slider("Stretch percentiles", 0, 100, (2, 98), key="band_stretch")
    shared = bc3.checkbox("Same scale for all bands", value=False, help="Use one reflectance range so brightness is comparable between bands.")
    mask_b = scene.clear_mask
    vmin = vmax = None
    if shared:
        allv = [scene.bands[b][mask_b] for b in S2_BANDS]
        allv = pd.Series([x for a in allv for x in a[:: max(1, len(a) // 2000)] if x == x])
        vmin, vmax = float(allv.quantile(plo / 100)), float(allv.quantile(phi / 100))
    cols = st.columns(4)
    rows = []
    for i, b in enumerate(S2_BANDS):
        name, wl, res = S2_BAND_INFO[b]
        arr = scene.bands[b]
        shown = colorize(arr, band_cmap, vmin, vmax, plo, phi)
        cols[i % 4].image(shown, caption=f"{b} · {name} · {wl} nm · {res} m", use_container_width=True)
        v = arr[mask_b & (arr == arr)]
        rows.append({"band": b, "name": name, "wavelength_nm": wl, "native_res_m": res,
                     "mean": float(v.mean()) if v.size else float("nan"), "median": float(pd.Series(v).median()) if v.size else float("nan"),
                     "std": float(v.std()) if v.size else float("nan")})
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
    st.download_button("Download band statistics (CSV)", pd.DataFrame(rows).to_csv(index=False).encode(), f"{field_id}_{scene.date}_band_stats.csv", "text/csv")
    insp = st.selectbox("Inspect one band (hover for values)", S2_BANDS, index=S2_BANDS.index("B08"))
    st.plotly_chart(raster_figure(scene.bands[insp], scene.grid, boundary, f"{insp} reflectance", band_cmap, colorbar_title="reflectance"),
                    use_container_width=True)

# ------------------------------------------------------------------ raster calculator
with tab_calc:
    layers = build_layers(scene.bands, scene.s1)
    avail = available_presets(layers)
    left, right = st.columns([1, 2])
    preset = left.selectbox("Start from an index", ["(custom)"] + avail, index=1 if "NDVI" in avail else 0)
    default_expr = PRESETS[preset].formula if preset != "(custom)" else "(B08 - B04) / (B08 + B04)"
    if preset != "(custom)":
        left.caption(PRESETS[preset].description)
    expr = right.text_area("Expression (edit freely)", value=default_expr, key=f"expr_{preset}", height=90,
                           help="Use band names (B04, B08 …) or aliases (RED, NIR, SWIR1 …). Functions: sqrt, abs, log, exp, min, max, clip, where. "
                                "Comparisons (>, <) give 1/0 masks; combine with & and |.")
    ok, msg = validate_expression(expr, layers)
    if ok:
        used = sorted(referenced_names(expr))
        right.success(f"Valid · uses: {', '.join(used) if used else 'constants only'}")
    else:
        right.error(msg)

    o1, o2, o3, o4 = st.columns(4)
    p = PRESETS.get(preset)
    cmap = o1.selectbox("Colour map", ["RdYlGn", "viridis", "magma", "YlGn", "Blues", "BrBG", "YlOrBr", "Spectral", "gray"],
                        index=["RdYlGn", "viridis", "magma", "YlGn", "Blues", "BrBG", "YlOrBr", "Spectral", "gray"].index(p.cmap) if p and p.cmap in
                        ["RdYlGn", "viridis", "magma", "YlGn", "Blues", "BrBG", "YlOrBr", "Spectral", "gray"] else 0)
    auto = o2.checkbox("Auto range (2–98 %)", value=p is None)
    vmin_in = o3.number_input("Min", value=float(p.vmin) if p else 0.0, disabled=auto, format="%.3f")
    vmax_in = o4.number_input("Max", value=float(p.vmax) if p else 1.0, disabled=auto, format="%.3f")
    mask_calc = st.checkbox("Mask clouds / shadow and outside-field pixels", value=True, key="calc_mask")

    if ok:
        try:
            result = evaluate(expr, layers, scene.clear_mask if mask_calc else scene.field_mask)
        except ExpressionError as exc:
            st.error(str(exc)); result = None
        if result is not None:
            title = f"{preset if preset != '(custom)' else 'Custom'} · {scene.date}"
            st.plotly_chart(raster_figure(result, scene.grid, boundary, title, cmap, None if auto else vmin_in, None if auto else vmax_in),
                            use_container_width=True)
            stats = summarize(result)
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Mean", f"{stats['mean']:.3f}"); sc2.metric("Median", f"{stats['median']:.3f}")
            sc3.metric("Min", f"{stats['min']:.3f}"); sc4.metric("Max", f"{stats['max']:.3f}"); sc5.metric("Std", f"{stats['std']:.3f}")
            st.plotly_chart(histogram_figure(result, "Distribution of values in the field"), use_container_width=True)
            d1, d2, d3 = st.columns(3)
            d1.download_button("Download GeoTIFF", ui.float_tif_bytes(result, scene.grid, {"expression": expr, "date": scene.date, "field": field_id}),
                               f"{field_id}_{scene.date}_calc.tif", "image/tiff")
            vmn, vmx = (None, None) if auto else (vmin_in, vmax_in)
            d2.download_button("Download PNG", png_bytes(colorize(result, cmap, vmn, vmx)), f"{field_id}_{scene.date}_calc.png", "image/png")
            d3.download_button("Download stats (CSV)", pd.DataFrame([{**stats, "expression": expr, "date": scene.date, "field": field_id}]).to_csv(index=False).encode(),
                               f"{field_id}_{scene.date}_calc_stats.csv", "text/csv")

    with st.expander("Calculator cheat-sheet"):
        st.markdown(
            "**Bands** (reflectance 0–1): " + ", ".join(S2_BANDS) + "  \n"
            "**Aliases:** " + ", ".join(f"{a}={b}" for a, b in ALIASES.items()) + "  \n"
            "**Sentinel-1** (after pulling a composite): `VV`, `VH` (linear), `VV_dB`, `VH_dB`  \n"
            "**Functions:** " + ", ".join(sorted(FUNCTIONS)) + "  \n"
            "**Operators:** `+ - * / **`, comparisons `< <= > >= == !=`, masks `&` `|` `~`  \n"
            "**Examples:** `(NIR - RED) / (NIR + RED)` · `(B08 - B11) / (B08 + B11)` · `where(NDVI > 0.6, 1, 0)` · "
            "`(B08 - B04) / (B08 + B04) * (B8A - B05) / (B8A + B05)` · `VH_dB - VV_dB`  \n"
            "**Note on NDWI:** two definitions are in common use. `NDWI` here is McFeeters (green/NIR, open water). "
            "Gao's NDWI (canopy water content) is `NDMI` = (B08 − B11)/(B08 + B11).")
        st.dataframe(pd.DataFrame([{"index": k, "formula": v.formula, "what it shows": v.description} for k, v in PRESETS.items()]),
                     hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ recent scenes
with tab_hist:
    st.caption("How the field has changed over the most recent images - thumbnails and field-mean trend of one index.")
    h1, h2 = st.columns(2)
    n_recent = h1.slider("Number of recent scenes", 2, 8, 4)
    hist_idx = h2.selectbox("Index", [k for k in available_presets(build_layers(scene.bands)) if k not in ("RVI", "VH_VV_DB")], index=0)
    if st.button("Build recent-scenes panel", type="primary"):
        sdf = st.session_state.get("img_scenes_df")
        panel = []
        prog = st.progress(0.0, text="Downloading recent scenes...")
        for i, (_, row) in enumerate(sdf.head(n_recent).iterrows()):
            try:
                sc = ui.load_scene_for_date(demo, field_id, geom, row, creds, set_current=False)
                idx = evaluate(PRESETS[hist_idx].formula, build_layers(sc.bands), sc.clear_mask)
                panel.append((sc, idx))
            except Exception as exc:
                st.warning(f"{row['date']}: {exc}")
            prog.progress((i + 1) / max(1, min(n_recent, len(sdf))), text=f"Processed {i + 1} scene(s)")
        prog.empty()
        st.session_state["hist_panel"] = (field_id, hist_idx, [(sc.date, sc.clear_fraction, idx) for sc, idx in panel], scene.grid)
    hp = st.session_state.get("hist_panel")
    if hp and hp[0] == field_id and hp[1] == hist_idx:
        _, _, items, grid_h = hp
        pr = PRESETS[hist_idx]
        cols = st.columns(min(4, max(1, len(items))))
        for i, (d, cf, idx) in enumerate(items):
            cols[i % len(cols)].image(colorize(idx, pr.cmap, pr.vmin, pr.vmax), use_container_width=True,
                                      caption=f"{d} · {hist_idx} mean {float(pd.Series(idx.ravel()).mean()):.2f} · {cf:.0%} clear")
        trend = pd.DataFrame([{"date": d, hist_idx: float(pd.Series(idx.ravel()).mean()), "clear_%": cf * 100} for d, cf, idx in items]).sort_values("date")
        trend = trend[trend["clear_%"] >= 30]
        if len(trend) >= 2:
            st.plotly_chart(px.line(trend, x="date", y=hist_idx, markers=True, title=f"Field-mean {hist_idx} (scenes with ≥30 % clear pixels)"),
                            use_container_width=True)

# ------------------------------------------------------------------ sentinel-1
with tab_s1:
    if scene.s1 is None:
        st.info("Sentinel-1 radar sees through cloud and responds to canopy structure - needed for the gap/weed analysis.")
        s1a, s1b, s1c, s1d = st.columns(4)
        direction = s1a.radio("Orbit", ["ASCENDING", "DESCENDING"], help="Keep this consistent with your RVI baseline (ascending by default in this repo).")
        window = s1b.slider("± days around the image", 3, 30, 12)
        n_s1 = s1c.slider("Max scenes to composite", 1, 6, 4)
        spk = s1d.select_slider("Speckle filter (px)", [1, 3, 5], value=3)
        if st.button("Pull Sentinel-1 composite", type="primary"):
            try:
                with st.spinner("Downloading Sentinel-1..."):
                    ui.attach_s1(scene, demo, creds, window, direction, n_s1, spk)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        s1 = scene.s1
        layers_s1 = build_layers(scene.bands, s1)
        st.caption(f"{s1['n_scenes']} {s1.get('direction', '')} scene(s) · temporal median + {s1.get('speckle_px', 3)}×{s1.get('speckle_px', 3)} speckle filter · "
                   f"dates: {', '.join(s1['dates'])}")
        mask = scene.field_mask
        vals = {"VH (dB)": (evaluate("VH_dB", layers_s1, mask), "viridis"), "VV (dB)": (evaluate("VV_dB", layers_s1, mask), "viridis"),
                "RVI = 4·VH/(VV+VH)": (evaluate("RVI", layers_s1, mask), "YlGn")}
        cols = st.columns(3)
        for col, (title, (arr, cm)) in zip(cols, vals.items()):
            col.plotly_chart(raster_figure(arr, scene.grid, boundary, title, cm), use_container_width=True)
        if st.button("Remove Sentinel-1 from this scene") and not scene.synthetic:
            scene.s1 = None; st.rerun()
