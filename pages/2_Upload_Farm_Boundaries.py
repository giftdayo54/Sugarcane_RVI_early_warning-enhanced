"""
pages/2_Upload_Farm_Boundaries.py

Third task: bring farm/block boundaries into the RVI early-warning system
by uploading a GeoJSON in the browser (as an alternative to committing
data/vectors/blocks.geojson to the repo and rerunning main.py).

Required GeoJSON feature properties: block_id, variety, crop_type,
planting_date (YYYY-MM-DD).

Behavior:
- A farm whose block_id already has rows in data/metadata/observations.csv
  is "Established" -> matched directly to its existing row in
  outputs/reports/block_report.csv (no new satellite calls; this also
  lets you re-map an existing block onto an updated/corrected boundary
  shape, since the uploaded geometry is what gets displayed).
- A farm whose block_id has never been seen is "New" -> Sentinel-1 VH/VV
  is fetched live via Sentinel Hub for exactly its date range, converted
  to RVI with the same formula the rest of the pipeline uses
  (src.rvi_calculation.calculate_rvi), and scored against the EXISTING
  healthy baseline (outputs/reports/healthy_baseline.csv) for its
  variety/crop_type. This is a preliminary, rule-based-only assessment —
  no Isolation Forest, since that model was trained on the full
  historical population and refitting it on a handful of new points
  would be meaningless. It stays "preliminary" until the farm
  accumulates its own healthy-season history and is folded into a full
  `python main.py` rerun (a ready-to-commit observations.csv is offered
  as a download for exactly that).
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
    "Add farms by uploading a GeoJSON. Farms with existing observation history are "
    "matched to their current risk assessment; brand-new farms get a live, preliminary "
    "Sentinel-1 assessment against the existing healthy baseline."
)

REQUIRED_PROPS = ["block_id", "variety", "crop_type", "planting_date"]
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
    st.info(
        "Upload a GeoJSON where each feature has `block_id`, `variety`, `crop_type`, "
        "and `planting_date` (YYYY-MM-DD) properties."
    )
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

missing = [c for c in REQUIRED_PROPS if c not in uploaded_gdf.columns]
if missing:
    st.error(f"Missing required propert{'y' if len(missing) == 1 else 'ies'}: {', '.join(missing)}")
    st.stop()

if uploaded_gdf.crs is None:
    uploaded_gdf = uploaded_gdf.set_crs(epsg=4326)
elif uploaded_gdf.crs.to_epsg() != 4326:
    uploaded_gdf = uploaded_gdf.to_crs(epsg=4326)
uploaded_gdf = uploaded_gdf[uploaded_gdf.geometry.notna() & ~uploaded_gdf.geometry.is_empty].copy()
uploaded_gdf["block_id"] = uploaded_gdf["block_id"].astype(str)

if uploaded_gdf.empty:
    st.error("No valid geometries in the uploaded file.")
    st.stop()

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

# ---- New farms: live Sentinel-1 fetch + score against existing baseline ----
if len(new_gdf):
    st.subheader(f"Analyze {len(new_gdf)} new farm(s)")
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
        if not sh_client_id or not sh_client_secret:
            st.error("Sentinel Hub client ID/secret are required.")
        elif start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            baseline_path = Path("outputs/reports/healthy_baseline.csv")
            if not baseline_path.exists():
                st.error("No outputs/reports/healthy_baseline.csv found — run `python main.py` first to fit a baseline.")
            else:
                baseline = pd.read_csv(baseline_path)
                tmp_path = None
                try:
                    # Duplicate block_id -> field_id so the shared Sentinel Hub engine
                    # (built around field_id/crop_type) can read this file unmodified.
                    temp_gdf = new_gdf.copy()
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

                        meta = new_gdf[["block_id", "variety", "planting_date"]]
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
                            "(new variety/crop_type combination) — risk can't be assessed until "
                            "a healthy season is recorded for them."
                        )
                except Exception as exc:
                    st.error(f"Failed to analyze new farms: {exc}")
                finally:
                    if tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)
    elif "new_farm_report" not in st.session_state:
        st.info("Set a date range and Sentinel Hub credentials in the sidebar, then click **Fetch & analyze new farms**.")

if st.session_state.get("new_farm_report") is not None:
    nfr = st.session_state["new_farm_report"]
    still_uploaded = nfr[nfr["block_id"].isin(new_gdf["block_id"])]
    if len(still_uploaded):
        report_frames.append(still_uploaded)
        for a in build_alerts(still_uploaded[still_uploaded["risk_level"] != "No baseline available"]):
            st.warning(a)

# ---- Combined view -------------------------------------------------------
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
