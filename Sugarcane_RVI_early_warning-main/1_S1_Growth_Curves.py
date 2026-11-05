"""
pages/1_S1_Growth_Curves.py

Second task in the dashboard: on-demand Sentinel-1 VH/VV growth curves for
any uploaded field boundaries, pulled live from Sentinel Hub (Process API) —
no local SNAP preprocessing required. This is independent of the block-level
RVI early-warning pipeline on the Home page; use it to explore raw growth
curves for a new field, sanity-check backscatter, or monitor fields that
aren't yet part of the `blocks.geojson` inventory.

Only a valid boundary is required to run this. A name/ID property
(field_id/block_id/id/name/...) is picked up automatically if present, or
auto-generated if not; crop_type is purely a display label here (it
doesn't gate the fetch) and can be added after results come back, via the
editable table below the summary.
"""
import io
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.s1_growth_curves import (
    apply_smoothing,
    authenticate_sentinel_hub,
    build_growth_curves_figure,
    build_process_requests,
    calculate_summary_metrics,
    extract_timeseries,
    load_fields,
    match_field_acquisitions,
    search_sentinel1_catalog,
)

st.set_page_config(page_title="S1 Growth Curves", layout="wide")
st.title("🛰️ Sentinel-1 Growth Curves")
st.caption(
    "Upload field boundaries and fetch VH/VV backscatter growth curves directly from "
    "Sentinel Hub. Only a valid boundary is required — labels can be added after fetching."
)


def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


with st.sidebar:
    st.header("Inputs")
    geojson_file = st.file_uploader("Field boundaries (GeoJSON)", type=["geojson", "json"])

    scope, selected_field = "All fields", None
    if geojson_file is not None:
        # Reset stale results if a different file is uploaded.
        upload_key = f"{geojson_file.name}:{geojson_file.size}"
        if st.session_state.get("gc_upload_key") != upload_key:
            st.session_state["gc_upload_key"] = upload_key
            st.session_state.pop("gc_df", None)
            st.session_state.pop("gc_summary", None)

        preview_ids, tmp_preview = [], None
        try:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                tmp.write(geojson_file.getvalue())
                tmp_preview = tmp.name
            preview_ids = sorted(load_fields(tmp_preview)["field_id"].tolist())
        except Exception as exc:
            st.error(f"Couldn't read that file: {exc}")
        finally:
            if tmp_preview:
                Path(tmp_preview).unlink(missing_ok=True)

        if preview_ids:
            st.caption(f"{len(preview_ids)} field(s) detected.")
            scope = st.radio("Scope", ["All fields", "Single field"], horizontal=True)
            if scope == "Single field":
                selected_field = st.selectbox("Field", preview_ids)

    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start date", value=date.today() - timedelta(days=180))
    end_date = c2.date_input("End date", value=date.today())

    st.markdown("---")
    st.subheader("Sentinel Hub credentials")
    st.caption("Free credentials: dataspace.copernicus.eu → User Settings → OAuth clients.")
    sh_client_id = st.text_input("Client ID", value=_secret("SH_CLIENT_ID"), type="password")
    sh_client_secret = st.text_input("Client secret", value=_secret("SH_CLIENT_SECRET"), type="password")
    with st.expander("Advanced"):
        sh_base_url = st.text_input("Base URL", value=_secret("SH_BASE_URL", "https://sh.dataspace.copernicus.eu"))
        sh_token_url = st.text_input(
            "Token URL",
            value=_secret(
                "SH_TOKEN_URL", "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            ),
        )
        max_threads = st.slider("Parallel downloads", 1, 10, 5)

    run_clicked = st.button("Run", type="primary", use_container_width=True)

if run_clicked:
    if geojson_file is None:
        st.error("Please upload a GeoJSON file of field boundaries.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    elif not sh_client_id or not sh_client_secret:
        st.error("Sentinel Hub client ID/secret are required (sidebar, or Streamlit secrets).")
    else:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                tmp.write(geojson_file.getvalue())
                tmp_path = tmp.name

            with st.status("Fetching Sentinel-1 growth curves...", expanded=True) as status:
                fields_gdf = load_fields(tmp_path)
                if scope == "Single field" and selected_field:
                    fields_gdf = fields_gdf[fields_gdf["field_id"] == selected_field].reset_index(drop=True)
                st.write(f"Analyzing {len(fields_gdf)} field(s).")

                status.update(label="Authenticating with Sentinel Hub...")
                config = authenticate_sentinel_hub(sh_client_id, sh_client_secret, sh_base_url, sh_token_url)

                status.update(label="Searching the Sentinel-1 catalog...")
                acquisitions = search_sentinel1_catalog(fields_gdf, str(start_date), str(end_date), config)
                field_acqs = match_field_acquisitions(fields_gdf, acquisitions)
                st.write(f"Found {len(acquisitions)} candidate acquisition(s) over the field extent.")

                status.update(label="Requesting field-clipped rasters from the Process API...")
                requests_info = build_process_requests(fields_gdf, field_acqs, config)
                st.write(f"Downloading {len(requests_info)} field × date raster(s)...")
                df = extract_timeseries(requests_info, config, max_threads=max_threads)

                status.update(label="Smoothing and computing summary metrics...")
                df = apply_smoothing(df)
                summary_df = calculate_summary_metrics(df, str(start_date))

                status.update(label="Done.", state="complete")

            st.session_state["gc_df"] = df
            st.session_state["gc_summary"] = summary_df
        except Exception as exc:
            st.error(f"Failed to generate growth curves: {exc}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

if st.session_state.get("gc_df") is None:
    st.info("Upload a GeoJSON of field boundaries, set a date range and credentials, then click **Run**.")
    st.stop()

df = st.session_state["gc_df"]
summary_df = st.session_state["gc_summary"]

# ---- Optional: assign/edit display labels after fetching -----------------
st.subheader("Field labels")
st.caption(
    "Optional — crop_type is just a display label here (e.g. plant/ratoon cane), not used "
    "in the fetch itself. Add or fix it now, or skip it entirely."
)
field_ids = sorted(df["field_id"].unique())
current_labels = df.drop_duplicates("field_id").set_index("field_id")["crop_type"]
label_df = pd.DataFrame({"field_id": field_ids, "crop_type": [current_labels.get(f, "unspecified") for f in field_ids]})
edited_labels = st.data_editor(
    label_df, hide_index=True, use_container_width=True, key="gc_label_editor",
    column_config={"field_id": st.column_config.TextColumn("Field ID", disabled=True), "crop_type": st.column_config.TextColumn("Label")},
)

label_map = dict(zip(edited_labels["field_id"], edited_labels["crop_type"]))
df = df.copy()
df["crop_type"] = df["field_id"].map(label_map).fillna(df["crop_type"])
summary_df = summary_df.copy()
summary_df["crop_type"] = summary_df["field_id"].map(label_map).fillna(summary_df["crop_type"])

st.subheader("Field growth summary")
st.dataframe(summary_df, use_container_width=True)

st.subheader("Growth curves")
fig = build_growth_curves_figure(df)
st.pyplot(fig, use_container_width=True)

dl1, dl2, dl3 = st.columns(3)
dl1.download_button("Download time-series CSV", df.to_csv(index=False).encode(), "s1_timeseries.csv", "text/csv")

png_buf = io.BytesIO()
fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
dl2.download_button("Download growth curves PNG", png_buf.getvalue(), "growth_curves.png", "image/png")

dl3.download_button("Download summary CSV", summary_df.to_csv(index=False).encode(), "growth_summary.csv", "text/csv")
