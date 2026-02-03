import importlib

import numpy as np


def test_sar_pipeline_smoke():
    m = importlib.import_module("src.pipeline.run_sar_pipeline")
    out = m.run_sar_pipeline({
        "scenario": "SAR_SCN_01",
        "seed": 42,
        "input_shape": (60, 60),
        "hw_profile": "cpu",
    })

    assert out["image"].shape == (60, 60)
    assert out["image"].dtype == np.float32
    assert out["metrics"]["nan_count"] == 0


def test_sar_pipeline_metrics_acceptance():
    m = importlib.import_module("src.pipeline.run_sar_pipeline")
    out = m.run_sar_pipeline({
        "scenario": "SAR_SCN_01",
        "seed": 42,
        "input_shape": (60, 60),
        "hw_profile": "cpu",
    })

    assert out["metrics"]["resolution_est"] <= 0.3
    assert out["metrics"]["latency_s"] <= 2.0
