from __future__ import annotations
from pathlib import Path
import logging, random, yaml
import numpy as np

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)

def ensure_dirs(*paths):
    for p in paths: Path(p).mkdir(parents=True, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)

def get_logger(name="sugarcane_rvi"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)
