"""SAR pipeline runner (Lite)."""

from __future__ import annotations

from typing import Any, Dict

from src.sar.bp_unit import UnitSarBp001


def run_sar_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run SAR pipeline and return image + metrics."""
    scenario = cfg.get("scenario", "SAR_SCN_00")
    seed = cfg.get("seed", 0)
    input_shape = cfg.get("input_shape", (256, 256))
    hw_profile = cfg.get("hw_profile", "cpu")

    unit = UnitSarBp001()
    out = unit.run({
        "seed": seed,
        "scenario": scenario,
        "input_shape": input_shape,
        "hw_profile": hw_profile,
    })

    return {
        "image": out["image"],
        "metrics": out["metrics"],
    }
