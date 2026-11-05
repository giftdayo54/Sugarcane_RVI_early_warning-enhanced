from __future__ import annotations
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import geometry_mask

def zonal_stats(raster_path, blocks_path, date, block_id='block_id'):
    gdf=gpd.read_file(blocks_path)
    rows=[]
    with rasterio.open(raster_path) as src:
        gdf=gdf.to_crs(src.crs); arr=src.read(1,masked=True)
        for _,r in gdf.iterrows():
            inside=geometry_mask([r.geometry],src.shape,src.transform,invert=True,all_touched=False)
            vals=np.asarray(arr[inside].compressed(),dtype=float)
            rows.append({block_id:r[block_id],'date':pd.Timestamp(date),
                         'observed_rvi':float(np.nanmean(vals)) if vals.size else np.nan,
                         'rvi_std':float(np.nanstd(vals)) if vals.size else np.nan,
                         'valid_pixels':int(vals.size)})
    return pd.DataFrame(rows)
