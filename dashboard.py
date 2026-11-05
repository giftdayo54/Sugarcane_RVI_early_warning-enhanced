from pathlib import Path
import streamlit as st, pandas as pd, geopandas as gpd, plotly.express as px
from src.visualization import growth_curve
from src.gap_reporting import load_gap_report
st.set_page_config(page_title='Sugarcane RVI Early Warning',layout='wide')
st.title('Sugarcane Anomaly Early Warning System')
report_path=Path('outputs/reports/block_report.csv'); vector_path=Path('outputs/vectors/block_status.geojson')
if not report_path.exists(): st.info('Run python main.py first.'); st.stop()
report=pd.read_csv(report_path); c1,c2,c3=st.columns(3)
c1.metric('Blocks',len(report)); c2.metric('High/Critical',report.risk_level.isin(['High Risk','Critical']).sum()); c3.metric('Mean deviation',f"{report.percent_deviation.mean():.1f}%")
if vector_path.exists():
    g=gpd.read_file(vector_path).to_crs(4326); center={'lat':g.geometry.centroid.y.mean(),'lon':g.geometry.centroid.x.mean()}
    fig=px.choropleth_map(g,geojson=g.__geo_interface__,locations=g.index,color='risk_level',hover_name='block_id',
        color_discrete_map={'Normal':'green','Watch':'yellow','Moderate Risk':'orange','High Risk':'red','Critical':'darkred'},map_style='open-street-map',center=center,zoom=11,opacity=.65)
    st.plotly_chart(fig,use_container_width=True)
st.subheader('Risk ranking'); st.dataframe(report,use_container_width=True)
# ---- Canopy gaps (from pages/4_Gap_Analysis.py or scripts/run_gap_analysis.py) -------------------------
gaps = load_gap_report('outputs')
st.subheader('Canopy gaps (Sentinel-1 + Sentinel-2)')
if gaps is None or gaps.empty:
    st.caption('No gap analysis saved yet. Run it from the Gap Analysis page or `python scripts/run_gap_analysis.py`.')
else:
    if 'synthetic' in gaps and gaps['synthetic'].astype(bool).any(): st.warning('Some rows come from SYNTHETIC demo data.')
    g1,g2,g3=st.columns(3); g1.metric('Fields with gap results',len(gaps)); g2.metric('Total gap area',f"{gaps.total_gap_ha.sum():.1f} ha")
    g3.metric('Green gaps missed by NDVI',f"{gaps.weedy_gap_ha.sum():.1f} ha")
    merged=report.merge(gaps[['block_id','date','gap_pct','bare_gap_pct','weedy_gap_pct','gap_severity']].rename(columns={'date':'gap_image_date'}),on='block_id',how='left') if 'block_id' in report else gaps
    flag=merged[(merged.risk_level.isin(['High Risk','Critical']))&(merged.gap_pct.fillna(0)>=10)] if 'risk_level' in merged and 'gap_pct' in merged else merged.iloc[0:0]
    if len(flag): st.error('Radar anomaly AND >=10% canopy gap: '+', '.join(flag.block_id.astype(str)))
    st.dataframe(merged if 'gap_pct' in merged else gaps,use_container_width=True)
csv=report.to_csv(index=False).encode(); st.download_button('Download block report',csv,'block_report.csv','text/csv')
for p,label,mime in [(Path('outputs/rasters/latest_risk.tif'),'Download QGIS GeoTIFF','image/tiff'),(vector_path,'Download GeoJSON','application/geo+json')]:
    if p.exists(): st.download_button(label,p.read_bytes(),p.name,mime)
