from __future__ import annotations
from pathlib import Path
import numpy as np, rasterio

def anomaly_raster(observed_path, expected, expected_std, out_path, z_threshold=-2.0, nodata=-9999.0):
    """Create 0..4 risk raster: 0 normal, 1 watch, 2 moderate, 3 high, 4 critical."""
    with rasterio.open(observed_path) as src:
        obs=src.read(1).astype(float); obs[obs==src.nodata]=np.nan
        pct=100*(obs-expected)/max(float(expected),1e-6); z=(obs-expected)/max(float(expected_std),0.03)
        risk=np.select([pct<=-40,pct<=-30,pct<=-20,pct<=-10],[4,3,2,1],default=0).astype('uint8')
        risk[~np.isfinite(obs)]=255
        profile=src.profile.copy(); profile.update(dtype='uint8',nodata=255,count=1,compress='deflate',photometric='minisblack')
        Path(out_path).parent.mkdir(parents=True,exist_ok=True)
        with rasterio.open(out_path,'w',**profile) as dst:
            dst.write(risk,1); dst.write_colormap(1,{0:(0,128,0,255),1:(255,255,0,255),2:(255,165,0,255),3:(255,0,0,255),4:(128,0,0,255),255:(0,0,0,0)})
            dst.update_tags(classes='0 Normal;1 Watch;2 Moderate Risk;3 High Risk;4 Critical')
    return Path(out_path)
