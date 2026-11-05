from __future__ import annotations
from pathlib import Path
import geopandas as gpd, pandas as pd

def block_report(scored, affected=None):
    latest=scored.sort_values('date').groupby('block_id').tail(1).copy()
    if affected is not None: latest=latest.merge(affected,on='block_id',how='left')
    keep=['block_id','date','observed_rvi','expected_rvi','difference','percent_deviation','z_score','growth_rate','risk_level']
    if 'affected_area_percent' in latest: keep+=['affected_area_percent']
    return latest[keep].sort_values('percent_deviation')

def export_vectors(blocks_path, report, out_base):
    g=gpd.read_file(blocks_path).merge(report,on='block_id',how='left'); p=Path(out_base); p.parent.mkdir(parents=True,exist_ok=True)
    g.to_file(p.with_suffix('.gpkg'),driver='GPKG'); g.to_file(p.with_suffix('.geojson'),driver='GeoJSON')
    # Shapefile truncates field names; export explicitly for legacy interoperability.
    g.rename(columns={'observed_rvi':'obs_rvi','expected_rvi':'exp_rvi','percent_deviation':'pct_dev','risk_level':'risk'}).to_file(p.with_suffix('.shp'))
    return g
