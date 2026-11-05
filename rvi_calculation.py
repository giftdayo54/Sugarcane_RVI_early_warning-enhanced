from __future__ import annotations
from pathlib import Path
import numpy as np, rasterio
from .preprocessing import db_to_linear

def calculate_rvi(vv, vh, units='db', clip=(0.0,4.0)):
    vv=np.asarray(vv,dtype=np.float32); vh=np.asarray(vh,dtype=np.float32)
    if units.lower()=='db': vv, vh = db_to_linear(vv), db_to_linear(vh)
    valid=np.isfinite(vv)&np.isfinite(vh)&((vv+vh)>0)
    out=np.full(vv.shape,np.nan,np.float32)
    out[valid]=4.0*vh[valid]/(vv[valid]+vh[valid])
    return np.clip(out,*clip)

def rvi_geotiff(vv_path, vh_path, out_path, units='db', nodata=-9999.0):
    """Inputs must already share grid, CRS, dimensions, orbit direction and geometry."""
    with rasterio.open(vv_path) as a, rasterio.open(vh_path) as b:
        if (a.crs,a.transform,a.shape)!=(b.crs,b.transform,b.shape):
            raise ValueError('VV and VH grids differ; align first')
        rvi=calculate_rvi(a.read(1),b.read(1),units=units)
        profile=a.profile.copy(); profile.update(dtype='float32',nodata=nodata,count=1,compress='deflate',tiled=True)
        Path(out_path).parent.mkdir(parents=True,exist_ok=True)
        with rasterio.open(out_path,'w',**profile) as dst:
            dst.write(np.where(np.isfinite(rvi),rvi,nodata).astype('float32'),1)
            dst.update_tags(index='RVI',formula='4*VH/(VV+VH)',input_units=units)
    return Path(out_path)
