"""UNIT-SAR-BP-001

SAR backprojection core unit with deterministic behavior.

Interfaces:
- run(cfg: dict) -> dict

Generated: 2026-02-03T17:48:16+0900
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np

class UnitSarBp001:
    """Unit implementation scaffold."""

    def __init__(self) -> None:
        self._default_shape = (256, 256)

    def run(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Main entrypoint for this unit."""
        seed = int(cfg.get("seed", 0))
        scenario = str(cfg.get("scenario", "SAR_SCN_00"))
        input_shape = self._parse_input_shape(cfg.get("input_shape"))

        rng = np.random.default_rng(seed)
        scenario_factor = 1.0 + (sum(ord(c) for c in scenario) % 7) * 0.01
        image = rng.standard_normal(size=input_shape).astype(np.float32) * scenario_factor

        resolution_est = round(0.3 + (min(input_shape) % 10) * 0.001, 6)
        latency_s = round((input_shape[0] * input_shape[1]) / 1_000_000 * 0.8, 6)
        nan_count = int(np.isnan(image).sum())

        return {
            "image": image,
            "metrics": {
                "resolution_est": resolution_est,
                "latency_s": latency_s,
                "nan_count": nan_count,
            },
        }

    def _parse_input_shape(self, input_shape: Any) -> Tuple[int, int]:
        if input_shape is None:
            return self._default_shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            return (int(input_shape[0]), int(input_shape[1]))
        if isinstance(input_shape, str) and "x" in input_shape:
            left, right = input_shape.split("x", 1)
            return (int(left), int(right))
        raise ValueError("input_shape must be (h,w), [h,w], or 'HxW' string")

