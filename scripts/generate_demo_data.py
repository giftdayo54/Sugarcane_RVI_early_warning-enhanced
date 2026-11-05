"""
Generates a small synthetic demo dataset (3 blocks, 3 seasons) so the
dashboard's home page has something to show immediately after deploy,
without requiring real SNAP-preprocessed rasters. Mirrors the synthetic
smoke test already in notebooks/Sugarcane_RVI_Early_Warning.ipynb.

Run once from the project root: python scripts/generate_demo_data.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

rng = np.random.default_rng(42)

# --- Block polygons -------------------------------------------------------
# Three small adjacent blocks near 34.80E, 15.80S (same origin as the
# notebook's synthetic raster), ~1 ha rectangles for illustration.
block_defs = [
    {"block_id": "A01", "estate": "Demo Estate", "variety": "NCo376", "crop_type": "ratoon", "cx": 34.800, "cy": -15.800},
    {"block_id": "A02", "estate": "Demo Estate", "variety": "NCo376", "crop_type": "ratoon", "cx": 34.805, "cy": -15.800},
    {"block_id": "A12", "estate": "Demo Estate", "variety": "NCo376", "crop_type": "ratoon", "cx": 34.810, "cy": -15.800},
]
half = 0.0015  # ~330 m square at this latitude
rows = []
for b in block_defs:
    geom = box(b["cx"] - half, b["cy"] - half, b["cx"] + half, b["cy"] + half)
    rows.append({**{k: v for k, v in b.items() if k not in ("cx", "cy")}, "geometry": geom})

blocks = gpd.GeoDataFrame(rows, crs="EPSG:4326")
blocks_path = Path("data/vectors/blocks.geojson")
blocks_path.parent.mkdir(parents=True, exist_ok=True)
blocks.to_file(blocks_path, driver="GeoJSON")
print(f"Wrote {blocks_path} ({len(blocks)} block(s)).")

# --- Observations -----------------------------------------------------
# 3 years x 3 blocks x 18 fortnightly ages. Block A12's 2025 season is
# marked unhealthy (excluded from the baseline) and develops a real RVI
# stress dip after ~84 days, so the demo shows a genuine flagged block
# rather than an all-green dashboard.
records = []
for year in (2023, 2024, 2025):
    planting_date = pd.Timestamp(year, 1, 1)
    for b in block_defs:
        block_id = b["block_id"]
        healthy_season = not (year == 2025 and block_id == "A12")
        for age in range(0, 252, 14):
            expected = 0.25 + 0.65 * np.sin(np.pi * min(age, 250) / 300)
            observed = expected + rng.normal(0, 0.035)
            if year == 2025 and block_id == "A12" and age > 84:
                observed -= 0.24
            records.append(
                {
                    "block_id": block_id,
                    "date": (planting_date + pd.Timedelta(days=age)).date().isoformat(),
                    "planting_date": planting_date.date().isoformat(),
                    "variety": b["variety"],
                    "crop_type": b["crop_type"],
                    "healthy": healthy_season,
                    "observed_rvi": round(float(np.clip(observed, 0, 4)), 4),
                    "rainfall_mm": round(float(rng.uniform(0, 25)), 1),
                    "irrigation_mm": round(float(rng.uniform(0, 15)), 1),
                    "yield_t_ha": "",
                    "is_problem": int(not healthy_season),
                }
            )

observations = pd.DataFrame(records)
obs_path = Path("data/metadata/observations.csv")
obs_path.parent.mkdir(parents=True, exist_ok=True)
observations.to_csv(obs_path, index=False)
print(f"Wrote {obs_path} ({len(observations)} row(s)).")
