"""
pages/2_Upload_Farm_Boundaries.py

Third task: bring farm/block boundaries into the RVI early-warning system
by uploading a GeoJSON in the browser (as an alternative to committing
data/vectors/blocks.geojson to the repo and rerunning main.py).

Only an identifying property is required (block_id/id/name/field_id — the
first one found is used; if none exist, IDs are auto-generated). Uploading
a file with just boundaries + an ID is enough to see them on the map.

variety / crop_type / planting_date are optional in the file itself.
crop_type here means the sugarcane growth STAGE (e.g. "ratoon" vs "plant"
cane) — used to pick the right comparison baseline — not crop species, so
it can't be safely defaulted to "sugarcane". They're only needed if you
run live analysis on a brand-new farm, and if missing, you fill them in
via an editable table in-app instead of the file.

Behavior:
- A farm whose ID already has rows in data/metadata/observations.csv is
  "Established" -> matched directly to its existing row in
  outputs/reports/block_report.csv (no new satellite calls; this also
  lets you re-map an existing block onto an updated/corrected boundary
  shape, since the uploaded geometry is what gets displayed).
- A farm whose ID has never been seen is "New" -> once variety/crop stage/
  planting date are filled in, Sentinel-1 VH/VV is fetched live via
  Sentinel Hub for its date range, converted to RVI with the same formula
  the rest of the pipeline uses (src.rvi_calculation.calculate_rvi), and
  scored against the EXISTING healthy baseline
  (outputs/reports/healthy_baseline.csv) for its variety/crop stage. This
  is a preliminary, rule-based-only assessment — no Isolation Forest,
  since that model was trained on the full historical population and
  refitting it on a handful of new points would be meaningless. It stays
  "preliminary" until the farm accumulates its own healthy-season history
  and is folded into a full `python main.py` rerun (a ready-to-commit
  observations.csv is offered as a download for exactly that).
"""
import tempfile
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

from src.alerts import build_alerts
from src.anomaly_detection import score_rules
from src.baseline_model import attach_baseline
from src.rvi_calculation import calculate_rvi
from src.s1_growth_curves import (
    authenticate_sentinel_hub,
    build_process_requests,
    extract_timeseries,
    load_fields,
    match_field_acquisitions,
    search_sentinel1_catalog,
)

st.set_page_config(page_title="Upload Farm Boundaries", layout="wide")
st.title("🌾 Upload Farm Boundaries")
st.caption(
    "Upload a GeoJSON to see your farms on the map. Farms with existing observation "
    "history are matched to their current risk assessment; brand-new farms can get a "
    "live, preliminary Sentinel-1 assessment once you fill in a couple of details."
)

ID_CANDIDATES = ["block_id", "id", "ID", "Id", "field_id", "field_name", "name", "Name", "FID", "fid"]
RISK_COLORS = {
    "Normal": "green", "Watch": "yellow", "Moderate Risk": "orange",
    "High Risk": "red", "Critical": "darkred", "No baseline available": "grey",
}


def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


uploaded = st.file_uploader("Farm boundaries (GeoJSON)", type=["geojson", "json"])
if uploaded is None:
    st.info("Upload a GeoJSON of farm boundaries — a name/ID property per feature is all that's required.")
    st.stop()

# Reset any stale new-farm analysis if a different file is uploaded.
upload_key = f"{uploaded.name}:{uploaded.size}"
if st.session_state.get("upload_key") != upload_key:
    st.session_state["upload_key"] = upload_key
    st.session_state.pop("new_farm_report", None)
    st.session_state.pop("new_farm_obs", None)

tmp_upload_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_upload_path = tmp.name
    uploaded_gdf = gpd.read_file(tmp_upload_path)
except Exception as exc:
    st.error(f"Couldn't read that file as GeoJSON: {exc}")
    st.stop()
finally:
    if tmp_upload_path:
        Path(tmp_upload_path).unlink(missing_ok=True)

if uploaded_gdf.crs is None:
    uploaded_gdf = uploaded_gdf.set_crs(epsg=4326)
elif uploaded_gdf.crs.to_epsg() != 4326:
    uploaded_gdf = uploaded_gdf.to_crs(epsg=4326)
uploaded_gdf = uploaded_gdf[uploaded_gdf.geometry.notna() & ~uploaded_gdf.geometry.is_empty].reset_index(drop=True)

if uploaded_gdf.empty:
    st.error("No valid geometries in the uploaded file.")
    st.stop()

# ---- ID detection (only real requirement) --------------------------------
id_col = next((c for c in ID_CANDIDATES if c in uploaded_gdf.columns), None)
if id_col:
    uploaded_gdf["block_id"] = uploaded_gdf[id_col].astype(str)
else:
    uploaded_gdf["block_id"] = [f"field_{i + 1}" for i in range(len(uploaded_gdf))]
    st.info("No name/ID property found in the file — auto-generated IDs (field_1, field_2, ...) were assigned.")

if uploaded_gdf["block_id"].duplicated().any():
    dupes = sorted(uploaded_gdf.loc[uploaded_gdf["block_id"].duplicated(), "block_id"].unique())
    st.warning(f"Duplicate IDs found, which may cause mismatched results: {', '.join(dupes)}")

# ---- Immediate preview map, no analysis required --------------------------
st.subheader(f"Uploaded farms ({len(uploaded_gdf)})")
preview_center = {"lat": uploaded_gdf.geometry.centroid.y.mean(), "lon": uploaded_gdf.geometry.centroid.x.mean()}
preview_fig = px.choropleth_map(
    uploaded_gdf, geojson=uploaded_gdf.__geo_interface__, locations=uploaded_gdf.index,
    color="block_id", hover_name="block_id",
    map_style="open-street-map", center=preview_center, zoom=11, opacity=0.6,
)
st.plotly_chart(preview_fig, use_container_width=True)

# ---- Split into established vs new -------------------------------------
obs_path = Path("data/metadata/observations.csv")
known_block_ids = set()
if obs_path.exists():
    known_block_ids = set(pd.read_csv(obs_path, usecols=["block_id"])["block_id"].astype(str))

uploaded_gdf["upload_status"] = uploaded_gdf["block_id"].apply(lambda b: "Established" if b in known_block_ids else "New")
established_ids = set(uploaded_gdf.loc[uploaded_gdf["upload_status"] == "Established", "block_id"])
new_gdf = uploaded_gdf[uploaded_gdf["upload_status"] == "New"].reset_index(drop=True)

c1, c2 = st.columns(2)
c1.metric("Established farms (existing history)", len(established_ids))
c2.metric("New farms (need live data)", len(new_gdf))

report_frames = []

# ---- Established farms: match to the existing report -------------------
report_path = Path("outputs/reports/block_report.csv")
if established_ids:
    if not report_path.exists():
        st.warning("No outputs/reports/block_report.csv found yet — run `python main.py` first to get existing risk scores.")
    else:
        existing_report = pd.read_csv(report_path)
        existing_report["block_id"] = existing_report["block_id"].astype(str)
        matched = existing_report[existing_report["block_id"].isin(established_ids)].copy()
        matched["upload_status"] = "Established"
        report_frames.append(matched)

        unmatched = established_ids - set(matched["block_id"])
        if unmatched:
            st.warning(f"No existing report row for: {', '.join(sorted(unmatched))} — rerun `python main.py` to refresh block_report.csv.")

# ---- New farms: fill in details, then live Sentinel-1 fetch + scoring ----
if len(new_gdf):
    st.subheader(f"Analyze {len(new_gdf)} new farm(s)")

    # Pull sensible defaults from the existing baseline, if there's only one
    # value on record — still fully editable below.
    default_variety, default_crop_type = "", ""
    baseline_preview_path = Path("outputs/reports/healthy_baseline.csv")
    baseline_varieties, baseline_crop_types = [], []
    if baseline_preview_path.exists():
        bl_preview = pd.read_csv(baseline_preview_path)
        if "variety" in bl_preview.columns:
            baseline_varieties = sorted(bl_preview["variety"].dropna().unique().tolist())
            if len(baseline_varieties) == 1:
                default_variety = baseline_varieties[0]
        if "crop_type" in bl_preview.columns:
            baseline_crop_types = sorted(bl_preview["crop_type"].dropna().unique().tolist())
            if len(baseline_crop_types) == 1:
                default_crop_type = baseline_crop_types[0]

    def _col_or_default(gdf, col, default_val):
        if col in gdf.columns:
            return gdf[col].astype(str).replace({"nan": "", "None": ""})
        return pd.Series([default_val] * len(gdf), index=gdf.index)

    edit_df = pd.DataFrame(
        {
            "block_id": new_gdf["block_id"],
            "variety": _col_or_default(new_gdf, "variety", default_variety),
            "crop_type": _col_or_default(new_gdf, "crop_type", default_crop_type),
            "planting_date": _col_or_default(new_gdf, "planting_date", ""),
        }
    )
    edit_df["planting_date"] = pd.to_datetime(edit_df["planting_date"], errors="coerce").dt.date

    st.caption("crop stage = ratoon vs plant cane (used to pick the right comparison baseline), not species.")
    column_config = {
        "block_id": st.column_config.TextColumn("Farm ID", disabled=True),
        "planting_date": st.column_config.DateColumn("Planting date", format="YYYY-MM-DD"),
    }
    column_config["variety"] = (
        st.column_config.SelectboxColumn("Variety", options=baseline_varieties)
        if baseline_varieties else st.column_config.TextColumn("Variety")
    )
    column_config["crop_type"] = (
        st.column_config.SelectboxColumn("Crop stage", options=baseline_crop_types)
        if baseline_crop_types else st.column_config.TextColumn("Crop stage")
    )
    edited = st.data_editor(edit_df, use_container_width=True, hide_index=True, column_config=column_config, key="new_farm_meta_editor")

    with st.sidebar:
        st.header("New-farm analysis")
        d1, d2 = st.columns(2)
        start_date = d1.date_input("Start date", value=date.today() - timedelta(days=120), key="nf_start")
        end_date = d2.date_input("End date", value=date.today(), key="nf_end")
        st.caption("Free credentials: dataspace.copernicus.eu → User Settings → OAuth clients.")
        sh_client_id = st.text_input("Client ID", value=_secret("SH_CLIENT_ID"), type="password", key="nf_id")
        sh_client_secret = st.text_input("Client secret", value=_secret("SH_CLIENT_SECRET"), type="password", key="nf_secret")
        sh_base_url = _secret("SH_BASE_URL", "https://sh.dataspace.copernicus.eu")
        sh_token_url = _secret(
            "SH_TOKEN_URL", "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        )
        analyze_clicked = st.button("Fetch & analyze new farms", type="primary", use_container_width=True)

    if analyze_clicked:
        incomplete = edited[
            edited["variety"].isna() | (edited["variety"].astype(str).str.strip() == "")
            | edited["crop_type"].isna() | (edited["crop_type"].astype(str).str.strip() == "")
            | edited["planting_date"].isna()
        ]
        if len(incomplete):
            st.error(f"Fill in variety / crop stage / planting date for: {', '.join(incomplete['block_id'])}")
        elif not sh_client_id or not sh_client_secret:
            st.error("Sentinel Hub client ID/secret are required.")
        elif start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            baseline_path = Path("outputs/reports/healthy_baseline.csv")
            if not baseline_path.exists():
                st.error("No outputs/reports/healthy_baseline.csv found — run `python main.py` first to fit a baseline.")
            else:
                baseline = pd.read_csv(baseline_path)
                new_gdf_ready = new_gdf.drop(
                    columns=[c for c in ("variety", "crop_type", "planting_date") if c in new_gdf.columns]
                ).merge(edited, on="block_id", how="left")
                new_gdf_ready["planting_date"] = new_gdf_ready["planting_date"].astype(str)

                tmp_path = None
                try:
                    # Duplicate block_id -> field_id so the shared Sentinel Hub engine
                    # (built around field_id/crop_type) can read this file unmodified.
                    temp_gdf = new_gdf_ready.copy()
                    temp_gdf["field_id"] = temp_gdf["block_id"]
                    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                        tmp.write(temp_gdf.to_json().encode())
                        tmp_path = tmp.name

                    with st.status("Fetching Sentinel-1 data for new farms...", expanded=True) as status:
                        fields_gdf = load_fields(tmp_path)

                        status.update(label="Authenticating with Sentinel Hub...")
                        config = authenticate_sentinel_hub(sh_client_id, sh_client_secret, sh_base_url, sh_token_url)

                        status.update(label="Searching the Sentinel-1 catalog...")
                        acquisitions = search_sentinel1_catalog(fields_gdf, str(start_date), str(end_date), config)
                        field_acqs = match_field_acquisitions(fields_gdf, acquisitions)

                        status.update(label="Requesting field-clipped rasters...")
                        requests_info = build_process_requests(fields_gdf, field_acqs, config)
                        s1_df = extract_timeseries(requests_info, config)

                        status.update(label="Computing RVI and scoring against the existing baseline...")
                        # Recompute observed_rvi via the canonical pipeline formula
                        # (not the growth-curves engine's own RVI column) so new-farm
                        # values are directly comparable to the baseline.
                        s1_df["observed_rvi"] = calculate_rvi(s1_df["VV_dB"].to_numpy(), s1_df["VH_dB"].to_numpy(), units="db")
                        s1_df = s1_df.rename(columns={"field_id": "block_id"})

                        meta = new_gdf_ready[["block_id", "variety", "planting_date"]]
                        obs_new = s1_df.merge(meta, on="block_id")
                        obs_new["healthy"] = False

                        scored = attach_baseline(obs_new, baseline)
                        scored = score_rules(scored)
                        no_baseline = sorted(scored.loc[scored["expected_rvi"].isna(), "block_id"].unique().tolist())

                        latest = scored.sort_values("date").groupby("block_id").tail(1).copy()
                        keep = [
                            "block_id", "date", "observed_rvi", "expected_rvi", "difference",
                            "percent_deviation", "z_score", "growth_rate", "risk_level",
                        ]
                        new_report = latest[keep].copy()
                        new_report["upload_status"] = "New (preliminary)"
                        new_report.loc[new_report["block_id"].isin(no_baseline), "risk_level"] = "No baseline available"

                        status.update(label="Done.", state="complete")

                    st.session_state["new_farm_report"] = new_report
                    st.session_state["new_farm_obs"] = obs_new

                    missing_farms = set(new_gdf["block_id"]) - set(new_report["block_id"])
                    if missing_farms:
                        st.warning(f"No qualifying Sentinel-1 acquisitions in this date range for: {', '.join(sorted(missing_farms))}.")
                    if no_baseline:
                        st.warning(
                            f"No healthy baseline exists yet for: {', '.join(no_baseline)} "
                            "(new variety/crop-stage combination) — risk can't be assessed until "
                            "a healthy season is recorded for them."
                        )
                except Exception as exc:
                    st.error(f"Failed to analyze new farms: {exc}")
                finally:
                    if tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)
    elif "new_farm_report" not in st.session_state:
        st.info("Fill in the table above and set Sentinel Hub credentials in the sidebar, then click **Fetch & analyze new farms**.")

if st.session_state.get("new_farm_report") is not None:
    nfr = st.session_state["new_farm_report"]
    still_uploaded = nfr[nfr["block_id"].isin(new_gdf["block_id"])]
    if len(still_uploaded):
        report_frames.append(still_uploaded)
        for a in build_alerts(still_uploaded[still_uploaded["risk_level"] != "No baseline available"]):
            st.warning(a)

# ---- Combined risk view (only for farms that have been scored) -----------
if report_frames:
    combined_report = pd.concat(report_frames, ignore_index=True)
    combined_geom = uploaded_gdf.drop(columns=["upload_status"]).merge(combined_report, on="block_id", how="inner")

    st.subheader("Combined risk view")
    center = {"lat": combined_geom.geometry.centroid.y.mean(), "lon": combined_geom.geometry.centroid.x.mean()}
    fig = px.choropleth_map(
        combined_geom, geojson=combined_geom.__geo_interface__, locations=combined_geom.index,
        color="risk_level", hover_name="block_id", hover_data=["upload_status", "percent_deviation"],
        color_discrete_map=RISK_COLORS, map_style="open-street-map", center=center, zoom=11, opacity=0.65,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(combined_report, use_container_width=True)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download combined report CSV", combined_report.to_csv(index=False).encode(),
        "uploaded_farms_report.csv", "text/csv",
    )

    new_obs = st.session_state.get("new_farm_obs")
    if new_obs is not None:
        if obs_path.exists():
            existing_obs = pd.read_csv(obs_path)
            cols = [c for c in existing_obs.columns if c in new_obs.columns]
            merged_obs = pd.concat([existing_obs, new_obs[cols]], ignore_index=True)
        else:
            merged_obs = new_obs
        dl2.download_button(
            "Download updated observations.csv", merged_obs.to_csv(index=False).encode(),
            "observations.csv", "text/csv",
            help="Commit this to data/metadata/observations.csv and rerun main.py to fully incorporate the new farms.",
        )
