import numpy as np
import pytest

from src.raster_calc import (ExpressionError, PRESETS, available_presets, build_layers, evaluate, referenced_names,
                             validate_expression)


@pytest.fixture
def layers():
    rng = np.random.default_rng(0)
    bands = {b: rng.uniform(0.02, 0.5, (6, 7)).astype("float32")
             for b in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]}
    s1 = {"VV": rng.uniform(0.05, 0.2, (6, 7)).astype("float32"), "VH": rng.uniform(0.01, 0.05, (6, 7)).astype("float32")}
    return build_layers(bands, s1)


def test_ndvi_matches_numpy(layers):
    out = evaluate("(B08 - B04) / (B08 + B04)", layers)
    ref = (layers["B08"] - layers["B04"]) / (layers["B08"] + layers["B04"])
    assert np.allclose(out, ref, atol=1e-6)


def test_aliases_and_case_insensitive(layers):
    assert np.allclose(evaluate("(nir - red)/(NIR + Red)", layers), evaluate("(B08 - B04) / (B08 + B04)", layers))


def test_presets_all_evaluate_and_match_pipeline_rvi(layers):
    for name, p in PRESETS.items():
        out = evaluate(p.formula, layers)
        assert out.shape == (6, 7) and np.isfinite(out).any(), name
    from src.rvi_calculation import calculate_rvi
    assert np.allclose(evaluate("RVI", layers), calculate_rvi(layers["VV"], layers["VH"], units="linear", clip=(-99, 99)), atol=1e-5)


def test_preset_name_usable_inside_expression(layers):
    assert np.allclose(evaluate("NDVI * 2", layers), 2 * evaluate(PRESETS["NDVI"].formula, layers), atol=1e-6)


def test_divide_by_zero_becomes_nan():
    L = {"B04": np.zeros((2, 2), "float32"), "B08": np.zeros((2, 2), "float32")}
    out = evaluate("(B08 - B04) / (B08 + B04)", L)
    assert np.isnan(out).all()


def test_valid_mask_applied(layers):
    m = np.zeros((6, 7), bool); m[2, 3] = True
    out = evaluate("B08", layers, m)
    assert np.isfinite(out).sum() == 1


def test_masks_and_where(layers):
    out = evaluate("where(NDVI > 0.0, 1, 0)", layers)
    assert set(np.unique(out)).issubset({0.0, 1.0})
    both = evaluate("(B08 > 0.2) & (B04 < 0.3)", layers)
    assert set(np.unique(both)).issubset({0.0, 1.0})


def test_s1_missing_gives_helpful_error():
    L = build_layers({"B04": np.ones((2, 2)), "B08": np.ones((2, 2))})
    with pytest.raises(ExpressionError, match="Sentinel-1"):
        evaluate("VH_dB", L)


def test_unknown_name_suggests(layers):
    with pytest.raises(ExpressionError, match="Did you mean"):
        evaluate("B044 + 1", layers)


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hi')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd')",
    "B04.__class__",
    "[x for x in range(3)]",
    "lambda: 1",
    "B04 if True else B08",
    "sqrt(x=B04)",
    "eval('1')",
    "'abc'",
    "B04[0]",
    "1 if 1 else 2",
    "True",
])
def test_unsafe_or_unsupported_rejected(layers, bad):
    with pytest.raises(ExpressionError):
        evaluate(bad, layers)


def test_huge_power_does_not_hang(layers):
    out = evaluate("9 ** 9 ** 9 + B04", layers)  # float32 overflow -> inf -> NaN, returns instantly
    assert np.isnan(out).all()


def test_length_and_syntax_limits(layers):
    with pytest.raises(ExpressionError):
        evaluate("B04+" * 400 + "1", layers)
    with pytest.raises(ExpressionError, match="Syntax"):
        evaluate("(B08 - B04", layers)
    with pytest.raises(ExpressionError):
        evaluate("   ", layers)


def test_function_arity(layers):
    with pytest.raises(ExpressionError, match="argument"):
        evaluate("sqrt(B04, B08)", layers)


def test_referenced_and_available(layers):
    assert referenced_names("(NIR - RED)/(NIR + RED)") == {"B08", "B04"}
    assert "NDVI" in available_presets(layers) and "RVI" in available_presets(layers)
    no_s1 = {k: v for k, v in layers.items() if not k.startswith(("VV", "VH"))}
    assert "RVI" not in available_presets(no_s1)
    assert validate_expression("B08 / B04", layers)[0]
    assert not validate_expression("B99", layers)[0]
