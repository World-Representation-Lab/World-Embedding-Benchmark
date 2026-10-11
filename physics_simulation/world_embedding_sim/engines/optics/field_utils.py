"""Shared optical-field statistics and scalar bounds."""
from __future__ import annotations

import numpy as np


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _summary(field: np.ndarray, extent: list[float]) -> dict[str, float]:
    arr = np.asarray(field, dtype=float)
    peak = float(np.max(arr)) if arr.size else 0.0
    power = float(np.sum(arr))
    if power <= 1.0e-12:
        return {
            "peak": peak,
            "power": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "spot_radius": 0.0,
        }
    x0, x1, y0, y1 = extent
    ny, nx = arr.shape
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xg, yg = np.meshgrid(xs, ys)
    cx = float(np.sum(arr * xg) / power)
    cy = float(np.sum(arr * yg) / power)
    radius = float(np.sqrt(np.sum(arr * ((xg - cx) ** 2 + (yg - cy) ** 2)) / power))
    return {
        "peak": peak,
        "power": power,
        "centroid_x": cx,
        "centroid_y": cy,
        "spot_radius": radius,
    }
