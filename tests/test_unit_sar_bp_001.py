import importlib

import numpy as np


def test_unit_sar_bp_001_smoke():
    m = importlib.import_module("src.sar.bp_unit")
    unit = m.UnitSarBp001()
    out = unit.run({"seed": 42, "scenario": "SAR_SCN_01", "input_shape": (64, 64)})

    assert "image" in out and "metrics" in out
    assert out["image"].shape == (64, 64)
    assert out["image"].dtype == np.float32
    assert out["metrics"]["nan_count"] == 0


def test_unit_sar_bp_001_reproducible():
    m = importlib.import_module("src.sar.bp_unit")
    unit = m.UnitSarBp001()
    cfg = {"seed": 7, "scenario": "SAR_SCN_02", "input_shape": "64x64"}

    out_a = unit.run(cfg)
    out_b = unit.run(cfg)

    assert np.array_equal(out_a["image"], out_b["image"])
    assert out_a["metrics"] == out_b["metrics"]


def test_unit_sar_bp_001_input_shape_parsing():
    m = importlib.import_module("src.sar.bp_unit")
    unit = m.UnitSarBp001()

    assert unit._parse_input_shape(None) == (256, 256)
    assert unit._parse_input_shape([32, 48]) == (32, 48)
    assert unit._parse_input_shape((40, 50)) == (40, 50)
    assert unit._parse_input_shape("24x36") == (24, 36)


def test_unit_sar_bp_001_input_shape_invalid():
    m = importlib.import_module("src.sar.bp_unit")
    unit = m.UnitSarBp001()

    try:
        unit._parse_input_shape(123)
    except ValueError as exc:
        assert "input_shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid input_shape")
