from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def db_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert power backscatter in dB to linear sigma0/gamma0."""
    return np.power(10.0, x / 10.0, dtype=np.float32)

def align_to_reference(src_path, reference_path, dst_path):
    """Reproject/resample a polarization to the exact reference grid."""
    with rasterio.open(reference_path) as ref, rasterio.open(src_path) as src:
        profile=ref.profile.copy(); profile.update(dtype="float32", count=1, compress="deflate")
        out=np.full((ref.height, ref.width), np.nan, np.float32)
        reproject(rasterio.band(src,1), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref.transform, dst_crs=ref.crs, resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
        with rasterio.open(dst_path,'w',**profile) as dst: dst.write(out,1)
    return Path(dst_path)

def run_snap_gpt(safe_path, output_path, graph_xml, gpt='gpt', timeout=7200):
    """Run a configurable SNAP graph: orbit, thermal/border noise, calibration,
    optional Refined Lee speckle filter, terrain flattening/correction, GeoTIFF export.
    The XML graph is deliberately external because DEM, CRS and SNAP versions vary."""
    cmd=[gpt, str(graph_xml), f'-Pinput={safe_path}', f'-Poutput={output_path}']
    return subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout)
