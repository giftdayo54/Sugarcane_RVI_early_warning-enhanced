"""
src/raster_calc.py

Safe raster calculator for Sentinel-2 / Sentinel-1 band maths.

Users type an expression such as ``(B08 - B04) / (B08 + B04)`` or
``(NIR - RED) / (NIR + RED)``. The expression is parsed with ``ast`` and
evaluated against a whitelist of operators/functions -- it is NEVER passed to
``eval``/``exec``, so a pasted expression cannot run arbitrary code.

Variables
---------
Sentinel-2 L2A surface reflectance (0-1): B01 B02 B03 B04 B05 B06 B07 B08 B8A
B09 B11 B12, plus aliases (BLUE, GREEN, RED, RE1, RE2, RE3, NIR, NIR2, SWIR1,
SWIR2 ...). Sentinel-1 (when a composite has been pulled): VV, VH (linear
sigma0), VV_dB, VH_dB. Preset index names (NDVI, NDRE ...) can also be used
inside expressions, e.g. ``NDVI - NDRE``.
"""
from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------
# Band catalogue
# --------------------------------------------------------------------------

# band -> (common name, centre wavelength nm, native resolution m)
S2_BAND_INFO: dict[str, tuple[str, int, int]] = {
    "B01": ("Coastal aerosol", 443, 60),
    "B02": ("Blue", 490, 10),
    "B03": ("Green", 560, 10),
    "B04": ("Red", 665, 10),
    "B05": ("Red edge 1", 705, 20),
    "B06": ("Red edge 2", 740, 20),
    "B07": ("Red edge 3", 783, 20),
    "B08": ("NIR", 842, 10),
    "B8A": ("Narrow NIR", 865, 20),
    "B09": ("Water vapour", 945, 60),
    "B11": ("SWIR 1", 1610, 20),
    "B12": ("SWIR 2", 2190, 20),
}
S2_BANDS = list(S2_BAND_INFO)

S1_LAYERS = ["VV", "VH", "VV_DB", "VH_DB"]

ALIASES: dict[str, str] = {
    "COASTAL": "B01", "BLUE": "B02", "GREEN": "B03", "RED": "B04",
    "RE1": "B05", "RE2": "B06", "RE3": "B07", "REDEDGE1": "B05",
    "REDEDGE2": "B06", "REDEDGE3": "B07", "NIR": "B08", "NIR2": "B8A",
    "NNIR": "B8A", "WV": "B09", "SWIR1": "B11", "SWIR2": "B12",
    "SWIR": "B11",
}

MAX_EXPR_LEN = 600
MAX_NODES = 250


class ExpressionError(ValueError):
    """Raised for any invalid / unsafe / unevaluable expression."""


# --------------------------------------------------------------------------
# Index presets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexPreset:
    name: str
    formula: str
    description: str
    vmin: float
    vmax: float
    cmap: str


_P = IndexPreset
PRESETS: dict[str, IndexPreset] = {
    p.name: p
    for p in [
        _P("NDVI", "(B08 - B04) / (B08 + B04)",
           "Normalised Difference Vegetation Index - greenness / canopy cover.", -0.1, 0.95, "RdYlGn"),
        _P("NDRE", "(B8A - B05) / (B8A + B05)",
           "Normalised Difference Red Edge - chlorophyll / N status; stays sensitive in dense canopy where NDVI saturates.",
           0.0, 0.6, "RdYlGn"),
        _P("GNDVI", "(B08 - B03) / (B08 + B03)",
           "Green NDVI - chlorophyll content.", 0.0, 0.9, "RdYlGn"),
        _P("NDWI", "(B03 - B08) / (B03 + B08)",
           "NDWI (McFeeters) - open water / waterlogging. Positive = water.", -0.8, 0.4, "BrBG"),
        _P("NDMI", "(B08 - B11) / (B08 + B11)",
           "Normalised Difference Moisture Index (= NDWI of Gao) - canopy water content / water stress.",
           -0.2, 0.7, "Blues"),
        _P("MNDWI", "(B03 - B11) / (B03 + B11)",
           "Modified NDWI - open water, suppresses built-up/soil noise.", -0.8, 0.4, "BrBG"),
        _P("EVI", "2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1)",
           "Enhanced Vegetation Index - less saturation in dense canopy.", 0.0, 1.0, "RdYlGn"),
        _P("SAVI", "1.5 * (B08 - B04) / (B08 + B04 + 0.5)",
           "Soil-Adjusted VI (L = 0.5) - useful for sparse canopy / early growth.", 0.0, 1.0, "RdYlGn"),
        _P("MSAVI2", "(2 * B08 + 1 - sqrt((2 * B08 + 1) ** 2 - 8 * (B08 - B04))) / 2",
           "Modified SAVI - self-adjusting soil correction.", 0.0, 0.9, "RdYlGn"),
        _P("CIRE", "(B07 / B05) - 1",
           "Chlorophyll Index Red Edge - leaf chlorophyll.", 0.0, 8.0, "YlGn"),
        _P("NBR", "(B08 - B12) / (B08 + B12)",
           "Normalised Burn Ratio - burn scars / trash-burn mapping.", -0.5, 0.8, "RdYlGn"),
        _P("BSI", "((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))",
           "Bare Soil Index - high values = exposed soil (gap candidates).", -0.4, 0.4, "YlOrBr"),
        _P("NDTI", "(B11 - B12) / (B11 + B12)",
           "Normalised Difference Tillage Index - crop residue / trash cover.", -0.1, 0.3, "YlOrBr"),
        _P("RVI", "4 * VH / (VV + VH)",
           "Sentinel-1 Radar Vegetation Index - the same formula used by this repo's early-warning pipeline.",
           0.0, 1.5, "YlGn"),
        _P("VH_VV_DB", "VH_dB - VV_dB",
           "Sentinel-1 cross-pol ratio in dB (VH/VV) - volume scattering vs surface scattering.", -14.0, -4.0, "viridis"),
    ]
}

PRESET_ORDER = list(PRESETS)


# --------------------------------------------------------------------------
# Safe evaluator
# --------------------------------------------------------------------------

def _where(c, a, b):
    return np.where(np.asarray(c).astype(bool), a, b)


FUNCTIONS: dict[str, Callable] = {
    "sqrt": np.sqrt, "abs": np.abs, "log": np.log, "ln": np.log,
    "log10": np.log10, "exp": np.exp,
    "min": np.minimum, "max": np.maximum,
    "minimum": np.minimum, "maximum": np.maximum,
    "clip": np.clip, "where": _where, "square": np.square,
}
FUNCTION_ARITY = {"sqrt": 1, "abs": 1, "log": 1, "ln": 1, "log10": 1, "exp": 1, "square": 1,
                  "min": 2, "max": 2, "minimum": 2, "maximum": 2, "clip": 3, "where": 3}
CONSTANTS = {"PI": np.pi}

_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
    ast.Pow: np.power, ast.Mod: np.mod, ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or,
}
_CMPOPS = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


def _canonical(name: str) -> str:
    """Case-insensitive variable name -> canonical layer key (B08, VH_DB, ...)."""
    up = name.upper()
    return ALIASES.get(up, up)


def _parse(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("Enter an expression, e.g. (B08 - B04) / (B08 + B04).")
    if len(expr) > MAX_EXPR_LEN:
        raise ExpressionError(f"Expression too long (max {MAX_EXPR_LEN} characters).")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Syntax error: {exc.msg} (column {exc.offset}).") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExpressionError("Expression is too complex.")
    return tree


def referenced_names(expr: str) -> set[str]:
    """Canonical variable names used by the expression (functions excluded)."""
    tree = _parse(expr)
    func_nodes = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {_canonical(n.id) for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in func_nodes}


def _suggest(name: str, pool) -> str:
    close = difflib.get_close_matches(name.upper(), [p.upper() for p in pool], n=1, cutoff=0.6)
    return f" Did you mean {close[0]}?" if close else ""


class _Evaluator:
    def __init__(self, layers: dict[str, np.ndarray]):
        self.layers = layers
        self.shape = next(iter(layers.values())).shape if layers else None

    # -- name resolution ----------------------------------------------------
    def resolve(self, raw: str, depth: int):
        key = _canonical(raw)
        if key in self.layers:
            return self.layers[key]
        if key in CONSTANTS:
            return np.float32(CONSTANTS[key])
        if key in PRESETS:  # preset index used as a variable
            if depth > 4:
                raise ExpressionError("Preset references nested too deeply.")
            return self.visit(_parse(PRESETS[key].formula).body, depth + 1)
        if key in S2_BANDS or key in S1_LAYERS:
            raise ExpressionError(f"Layer {key} is not available for this scene"
                                  + (" - pull a Sentinel-1 composite first." if key in S1_LAYERS else "."))
        pool = list(self.layers) + list(ALIASES) + list(PRESETS)
        raise ExpressionError(f"Unknown name '{raw}'." + _suggest(raw, pool))

    # -- node visitor -------------------------------------------------------
    def visit(self, node, depth: int = 0):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError("Only numeric constants are allowed.")
            return np.float32(node.value)  # float32 => huge powers overflow to inf instead of hanging
        if isinstance(node, ast.Name):
            return self.resolve(node.id, depth)
        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise ExpressionError(f"Operator {type(node.op).__name__} is not allowed.")
            left, right = self.visit(node.left, depth), self.visit(node.right, depth)
            if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                return op(self._as_bool(left), self._as_bool(right))
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            val = self.visit(node.operand, depth)
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return val
            if isinstance(node.op, ast.Invert):
                return np.logical_not(self._as_bool(val))
            raise ExpressionError("Operator not allowed.")
        if isinstance(node, ast.Compare):
            result, left = None, self.visit(node.left, depth)
            for op_node, comp in zip(node.ops, node.comparators):
                op = _CMPOPS.get(type(op_node))
                if op is None:
                    raise ExpressionError("Only < <= > >= == != comparisons are allowed.")
                right = self.visit(comp, depth)
                part = op(left, right)
                result = part if result is None else np.logical_and(result, part)
                left = right
            return result
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ExpressionError("Only simple function calls like sqrt(x) are allowed.")
            fname = node.func.id.lower()
            if fname not in FUNCTIONS:
                raise ExpressionError(f"Unknown function '{node.func.id}'. Allowed: {', '.join(sorted(FUNCTIONS))}.")
            if len(node.args) != FUNCTION_ARITY[fname]:
                raise ExpressionError(f"{fname}() takes {FUNCTION_ARITY[fname]} argument(s).")
            return FUNCTIONS[fname](*[self.visit(a, depth) for a in node.args])
        raise ExpressionError(f"'{type(node).__name__}' is not allowed in expressions.")

    @staticmethod
    def _as_bool(v):
        return np.asarray(v).astype(bool)


def build_layers(bands: dict[str, np.ndarray] | None = None, s1: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Assemble the variable namespace (canonical upper-case keys) from S2 bands and S1 VV/VH (linear)."""
    layers: dict[str, np.ndarray] = {}
    for k, v in (bands or {}).items():
        layers[k.upper()] = np.asarray(v, dtype=np.float32)
    if s1:
        vv, vh = np.asarray(s1["VV"], dtype=np.float32), np.asarray(s1["VH"], dtype=np.float32)
        with np.errstate(all="ignore"):
            layers.update({
                "VV": vv, "VH": vh,
                "VV_DB": np.where(vv > 0, 10 * np.log10(vv), np.nan).astype(np.float32),
                "VH_DB": np.where(vh > 0, 10 * np.log10(vh), np.nan).astype(np.float32),
            })
    return layers


def evaluate(expr: str, layers: dict[str, np.ndarray], valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Evaluate ``expr`` over the layers; returns float32 with NaN where invalid/undefined."""
    if not layers:
        raise ExpressionError("No raster layers are loaded.")
    tree = _parse(expr)
    ev = _Evaluator(layers)
    with np.errstate(all="ignore"):
        out = ev.visit(tree.body)
        out = np.asarray(out)
        if out.dtype == bool:
            out = out.astype(np.float32)
        out = out.astype(np.float32, copy=False)
    if out.ndim == 0:
        out = np.full(ev.shape, out, dtype=np.float32)
    if out.shape != ev.shape:
        raise ExpressionError("Result shape does not match the scene.")
    out = np.where(np.isfinite(out), out, np.nan).astype(np.float32)
    if valid_mask is not None:
        out = np.where(valid_mask, out, np.nan).astype(np.float32)
    return out


def validate_expression(expr: str, layers: dict[str, np.ndarray]) -> tuple[bool, str]:
    """Cheap check (evaluates on a 2x2 dummy grid) used to give instant feedback in the UI."""
    try:
        dummy = {k: np.full((2, 2), 0.5, dtype=np.float32) for k in layers}
        evaluate(expr, dummy)
        return True, "OK"
    except ExpressionError as exc:
        return False, str(exc)


def available_presets(layers: dict[str, np.ndarray]) -> list[str]:
    """Preset names whose bands are all present in ``layers``."""
    ok = []
    for name in PRESET_ORDER:
        try:
            if all(n in layers for n in referenced_names(PRESETS[name].formula)):
                ok.append(name)
        except ExpressionError:
            continue
    return ok


def summarize(arr: np.ndarray) -> dict[str, float]:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {k: float("nan") for k in ("min", "p05", "median", "mean", "p95", "max", "std")} | {"n_valid": 0}
    p05, med, p95 = np.percentile(v, [5, 50, 95])
    return {"min": float(v.min()), "p05": float(p05), "median": float(med), "mean": float(v.mean()),
            "p95": float(p95), "max": float(v.max()), "std": float(v.std()), "n_valid": int(v.size)}
