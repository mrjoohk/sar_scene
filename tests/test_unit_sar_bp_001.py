import importlib

import numpy as np


def test_unit_sar_bp_001_smoke():
    m = importlib.import_module("sar.bp_unit")
    unit = m.UnitSarBp001()
    out = unit.run({"seed": 42, "scenario": "SAR_SCN_01", "input_shape": (64, 64)})

    assert "image" in out and "metrics" in out
    assert out["image"].shape == (64, 64)
    assert out["image"].dtype == np.float32
    assert out["metrics"]["nan_count"] == 0


def test_unit_sar_bp_001_reproducible():
    m = importlib.import_module("sar.bp_unit")
    unit = m.UnitSarBp001()
    cfg = {"seed": 7, "scenario": "SAR_SCN_02", "input_shape": "64x64"}

    out_a = unit.run(cfg)
    out_b = unit.run(cfg)

    assert np.array_equal(out_a["image"], out_b["image"])
    assert out_a["metrics"] == out_b["metrics"]
