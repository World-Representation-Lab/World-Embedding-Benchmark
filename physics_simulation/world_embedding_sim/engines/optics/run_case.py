#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

from field_utils import (
    _clamp,
)


WAVE_FAMILIES = {
    "singleSlitFraunhofer",
    "doubleSlitInterference",
    "circularApertureAiry",
    "knifeEdgeFresnel",
    "diffractionGrating",
    "thinFilmInterference",
    "gaussianBeamPropagation",
    "waveplatePolarization",
}
MEEP_FAMILIES = {
    "meepCylinderScattering",
    "meepWaveguideMode",
    "meepPhotonicCrystalDefect",
    "meepPhotonicCrystalWaveguide",
    "meepGratingDiffraction",
    "meepPlasmonicDimer",
    "meepDipoleRadiation",
    "meepEvanescentTIR",
}
MITSUBA_BACKENDS = {"mitsuba2", "mitsuba3"}
SOLVER_BACKENDS = MITSUBA_BACKENDS | {"hcipy", "meep", "surrogate"}
GEOMETRIC_BACKEND_DEFAULTS = ["mitsuba3", "mitsuba2", "surrogate"]
WAVE_BACKEND_DEFAULTS = ["hcipy", "surrogate"]
MEEP_BACKEND_DEFAULTS = ["meep"]


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_pvd(path: Path, datasets: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, rel_file in datasets:
        lines.append(
            f'    <DataSet timestep="{float(t):.9g}" group="" part="0" file="{rel_file}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    _write_text(path, "\n".join(lines))


def _write_vti(
    path: Path, *, arrays: dict[str, np.ndarray], extent: list[float]
) -> None:
    names = list(arrays.keys())
    if not names:
        raise ValueError("VTI export requires at least one array")
    first = np.asarray(arrays[names[0]], dtype=float)
    if first.ndim != 2:
        raise ValueError("VTI arrays must be 2D")
    ny, nx = first.shape
    x0, x1, y0, y1 = [float(v) for v in extent]
    dx = (x1 - x0) / float(max(1, nx - 1))
    dy = (y1 - y0) / float(max(1, ny - 1))
    whole_extent = f"0 {nx - 1} 0 {ny - 1} 0 0"
    array_xml: list[str] = []
    for name, arr in arrays.items():
        vals = np.asarray(arr, dtype=float)
        if vals.shape != (ny, nx):
            continue
        payload = " ".join(f"{float(v):.9g}" for v in vals.reshape(-1))
        array_xml.append(
            f'        <DataArray type="Float32" Name="{name}" format="ascii">{payload}</DataArray>'
        )
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{whole_extent}" Origin="{x0:.9g} {y0:.9g} 0" Spacing="{dx:.9g} {dy:.9g} 1">\n'
        f'    <Piece Extent="{whole_extent}">\n'
        '      <PointData Scalars="irradiance">\n'
        + "\n".join(array_xml)
        + "\n      </PointData>\n"
        "      <CellData></CellData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    _write_text(path, xml)


def _write_vtp_lines(
    path: Path,
    *,
    points: np.ndarray,
    lines: list[list[int]],
    cell_data: Optional[dict[str, np.ndarray]] = None,
) -> None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = np.zeros((0, 3), dtype=float)
    safe_lines = [
        list(map(int, line))
        for line in lines
        if isinstance(line, list) and len(line) >= 2
    ]
    connectivity = " ".join(str(v) for line in safe_lines for v in line)
    offsets = " ".join(
        str(sum(len(line) for line in safe_lines[: i + 1]))
        for i in range(len(safe_lines))
    )
    pts_flat = " ".join(f"{float(v):.9g}" for v in pts.reshape(-1))

    cell_arrays: list[str] = []
    for name, values in (cell_data or {}).items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != len(safe_lines):
            continue
        payload = " ".join(f"{float(v):.9g}" for v in arr)
        cell_arrays.append(
            f'      <DataArray type="Float32" Name="{name}" format="ascii">{payload}</DataArray>'
        )
    cell_xml = "\n".join(cell_arrays)

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{int(pts.shape[0])}" NumberOfVerts="0" NumberOfLines="{len(safe_lines)}" NumberOfStrips="0" NumberOfPolys="0">\n'
        "      <PointData></PointData>\n"
        f"      <CellData>\n{cell_xml}\n      </CellData>\n"
        "      <Points>\n"
        f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts_flat}</DataArray>\n'
        "      </Points>\n"
        "      <Lines>\n"
        f'        <DataArray type="Int32" Name="connectivity" format="ascii">{connectivity}</DataArray>\n'
        f'        <DataArray type="Int32" Name="offsets" format="ascii">{offsets}</DataArray>\n'
        "      </Lines>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )
    _write_text(path, xml)


def _load_case(case_dir: Path) -> dict[str, Any]:
    case = _read_json(case_dir / "solver_input.json")
    if case:
        return case
    return _read_json(case_dir / "case_meta.json")


def _mitsuba_major_version() -> int | None:
    if importlib.util.find_spec("mitsuba") is None:
        return None
    try:
        import mitsuba as mi
    except Exception:
        return None
    version = getattr(mi, "__version__", None)
    if isinstance(version, str) and version:
        major = version.split(".", 1)[0]
        if major.isdigit():
            return int(major)
    # Mitsuba 2's Python module in this project has no __version__.
    return 2


def _has_mitsuba2() -> bool:
    return _mitsuba_major_version() == 2


def _has_mitsuba3() -> bool:
    return _mitsuba_major_version() == 3


def _has_hcipy() -> bool:
    return importlib.util.find_spec("hcipy") is not None


def _has_meep() -> bool:
    return importlib.util.find_spec("meep") is not None


def _case_backend_preferences(case: dict[str, Any]) -> list[str]:
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    family = str(case.get("family") or case.get("case_prefix") or "unknown")
    if family in MEEP_FAMILIES:
        defaults = MEEP_BACKEND_DEFAULTS
    elif family in WAVE_FAMILIES:
        defaults = WAVE_BACKEND_DEFAULTS
    else:
        defaults = GEOMETRIC_BACKEND_DEFAULTS
    raw = solver.get("backend_preference")
    prefs = [str(x).strip().lower() for x in raw] if isinstance(raw, list) else []
    prefs = [p for p in prefs if p in SOLVER_BACKENDS]
    if family not in WAVE_FAMILIES and prefs == ["mitsuba2"]:
        return list(defaults)
    return prefs or list(defaults)


def _select_backend(requested: str, strict_mitsuba: bool, case: dict[str, Any]) -> str:
    requested = (requested or "auto").strip().lower()
    if requested not in {"auto", *SOLVER_BACKENDS}:
        raise ValueError(f"Unknown backend: {requested}")
    if requested == "surrogate":
        return "surrogate"
    if requested == "hcipy":
        if _has_hcipy():
            return "hcipy"
        raise RuntimeError(
            "HCIPy is not available. Install HCIPy or rerun with --backend surrogate."
        )
    if requested == "meep":
        if _has_meep():
            return "meep"
        raise RuntimeError(
            "Meep Python bindings are not available. Set optics.python.meep in runtime.json to an interpreter with Meep installed; see docs/environments.md."
        )
    if requested == "mitsuba3":
        if _has_mitsuba3():
            return "mitsuba3"
        raise RuntimeError(
            "Mitsuba 3 Python bindings are not available. Install Mitsuba 3 or rerun with --backend surrogate --no-strict-mitsuba."
        )
    if requested == "mitsuba2":
        if _has_mitsuba2():
            return "mitsuba2"
        raise RuntimeError(
            "Mitsuba 2 Python bindings are not available in this Python environment. The bundled geometric-optics examples use Mitsuba 3; see docs/environments.md."
        )
    if requested == "auto":
        for pref in _case_backend_preferences(case):
            if pref == "hcipy" and _has_hcipy():
                return "hcipy"
            if pref == "meep" and _has_meep():
                return "meep"
            if pref == "mitsuba3" and _has_mitsuba3():
                return "mitsuba3"
            if pref == "mitsuba2" and _has_mitsuba2():
                return "mitsuba2"
            if pref == "surrogate":
                return "surrogate"
        family = str(case.get("family") or case.get("case_prefix") or "unknown")
        if family in MEEP_FAMILIES:
            if _has_meep():
                return "meep"
            return "surrogate"
        if family in WAVE_FAMILIES:
            if _has_hcipy():
                return "hcipy"
            return "surrogate"
        if _has_mitsuba3():
            return "mitsuba3"
        if _has_mitsuba2():
            return "mitsuba2"
        return "surrogate"
    if strict_mitsuba and any(
        pref in MITSUBA_BACKENDS for pref in _case_backend_preferences(case)
    ):
        raise RuntimeError(
            "No requested Mitsuba backend is available. Install Mitsuba 3, source Mitsuba 2, or rerun with --backend surrogate --no-strict-mitsuba."
        )
    return "surrogate"


def _scan_values(case: dict[str, Any]) -> np.ndarray:
    scan = case.get("scan") if isinstance(case.get("scan"), dict) else {}
    count = max(2, int(scan.get("count", 24)))
    start = float(scan.get("start", 0.0))
    stop = float(scan.get("stop", 1.0))
    return np.linspace(start, stop, count)


def _field_grid(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    vis = (
        case.get("visualization") if isinstance(case.get("visualization"), dict) else {}
    )
    res = vis.get("field_resolution", [160, 96])
    if not isinstance(res, list) or len(res) != 2:
        res = [160, 96]
    nx = max(32, int(res[0]))
    ny = max(24, int(res[1]))
    extent = vis.get("field_extent", [-1.25, 1.25, -0.6, 0.6])
    if not isinstance(extent, list) or len(extent) != 4:
        extent = [-1.25, 1.25, -0.6, 0.6]
    x0, x1, y0, y1 = [float(v) for v in extent]
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xg, yg = np.meshgrid(xs, ys)
    return xg, yg, [x0, x1, y0, y1]


def _gaussian2d(
    xg: np.ndarray,
    yg: np.ndarray,
    *,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    amp: float = 1.0,
) -> np.ndarray:
    sx = max(1.0e-6, float(sigma_x))
    sy = max(1.0e-6, float(sigma_y))
    return float(amp) * np.exp(
        -0.5 * (((xg - float(x0)) / sx) ** 2 + ((yg - float(y0)) / sy) ** 2)
    )


def _field_summary(field: np.ndarray, extent: list[float]) -> dict[str, float]:
    arr = np.asarray(field, dtype=float)
    peak = float(np.max(arr)) if arr.size else 0.0
    total = float(np.sum(arr))
    if total <= 1.0e-12:
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
    cx = float(np.sum(arr * xg) / total)
    cy = float(np.sum(arr * yg) / total)
    r2 = (xg - cx) ** 2 + (yg - cy) ** 2
    spot_radius = float(np.sqrt(max(0.0, np.sum(arr * r2) / total)))
    return {
        "peak": peak,
        "power": total,
        "centroid_x": cx,
        "centroid_y": cy,
        "spot_radius": spot_radius,
    }


def _axis_coords(
    extent: list[float], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x0, x1, y0, y1 = [float(v) for v in extent]
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    return xs, ys


def _central_profile(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(0, dtype=float)
    return arr[arr.shape[0] // 2]


def _half_max_width(profile: np.ndarray, coords: np.ndarray) -> float:
    prof = np.asarray(profile, dtype=float).reshape(-1)
    xs = np.asarray(coords, dtype=float).reshape(-1)
    if prof.size < 3 or xs.size != prof.size:
        return 0.0
    peak = float(np.max(prof))
    if peak <= 0.0:
        return 0.0
    support = np.where(prof >= 0.5 * peak)[0]
    if support.size < 2:
        return 0.0
    return float(abs(xs[support[-1]] - xs[support[0]]))


def _peak_indices(profile: np.ndarray, *, threshold_rel: float = 0.1) -> np.ndarray:
    prof = np.asarray(profile, dtype=float).reshape(-1)
    if prof.size < 3:
        return np.zeros(0, dtype=int)
    peak = float(np.max(prof))
    if peak <= 0.0:
        return np.zeros(0, dtype=int)
    mask = (
        (prof[1:-1] >= prof[:-2])
        & (prof[1:-1] >= prof[2:])
        & (prof[1:-1] >= threshold_rel * peak)
    )
    return np.where(mask)[0] + 1


def _mean_spacing_from_indices(indices: np.ndarray, coords: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=int).reshape(-1)
    xs = np.asarray(coords, dtype=float).reshape(-1)
    if idx.size < 2 or xs.size == 0:
        return 0.0
    idx = idx[(idx >= 0) & (idx < xs.size)]
    if idx.size < 2:
        return 0.0
    return float(np.mean(np.diff(xs[idx])))


def _fringe_visibility_from_profile(profile: np.ndarray) -> float:
    prof = np.asarray(profile, dtype=float).reshape(-1)
    peaks = _peak_indices(prof, threshold_rel=0.15)
    if peaks.size < 2:
        return 0.0
    peak_vals = prof[peaks]
    trough_samples: list[float] = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        lo = min(a, b)
        hi = max(a, b)
        if hi - lo >= 2:
            trough_samples.append(float(np.min(prof[lo : hi + 1])))
    if not trough_samples:
        return 0.0
    i_max = float(np.mean(peak_vals))
    i_min = float(np.mean(trough_samples))
    denom = i_max + i_min
    return float((i_max - i_min) / denom) if denom > 1.0e-12 else 0.0


def _radial_profile(
    field: np.ndarray, extent: list[float], bins: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    xs, ys = _axis_coords(extent, arr.shape)
    xg, yg = np.meshgrid(xs, ys)
    rr = np.sqrt(xg * xg + yg * yg).reshape(-1)
    vals = arr.reshape(-1)
    r_max = float(np.max(rr))
    if r_max <= 0.0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    edges = np.linspace(0.0, r_max, max(8, int(bins)) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.zeros_like(centers)
    for i in range(centers.size):
        mask = (rr >= edges[i]) & (rr < edges[i + 1])
        if np.any(mask):
            prof[i] = float(np.mean(vals[mask]))
    return centers, prof


def _airy_ring_radius_from_field(field: np.ndarray, extent: list[float]) -> float:
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    xs, _ = _axis_coords(extent, arr.shape)
    profile = _central_profile(arr)
    if profile.size < 8:
        return 0.0
    center = profile.size // 2
    peaks = _peak_indices(profile, threshold_rel=0.002)
    side = [idx for idx in peaks if abs(int(idx) - center) >= 6]
    if not side:
        return 0.0
    best = min(side, key=lambda idx: abs(int(idx) - center))
    return float(abs(xs[int(best)] - xs[center]))


def _transition_width_from_profile(
    profile: np.ndarray, coords: np.ndarray, *, lo: float = 0.1, hi: float = 0.9
) -> float:
    prof = np.asarray(profile, dtype=float).reshape(-1)
    xs = np.asarray(coords, dtype=float).reshape(-1)
    if prof.size < 2 or xs.size != prof.size:
        return 0.0
    p_min = float(np.min(prof))
    p_max = float(np.max(prof))
    denom = p_max - p_min
    if denom <= 1.0e-12:
        return 0.0
    norm = (prof - p_min) / denom
    lo_idx = np.where(norm >= float(lo))[0]
    hi_idx = np.where(norm >= float(hi))[0]
    if lo_idx.size == 0 or hi_idx.size == 0:
        return 0.0
    return float(abs(xs[hi_idx[0]] - xs[lo_idx[0]]))


def _edge_overshoot_from_profile(profile: np.ndarray) -> float:
    prof = np.asarray(profile, dtype=float).reshape(-1)
    if prof.size < 8:
        return 0.0
    bright_side = float(np.mean(prof[-max(4, prof.size // 8) :]))
    if bright_side <= 1.0e-12:
        return 0.0
    return float(max(0.0, float(np.max(prof)) - bright_side) / bright_side)


def _fresnel_unpolarized(n1: float, n2: float, theta_i_deg: float) -> float:
    theta_i = math.radians(float(theta_i_deg))
    sin_t = (float(n1) / float(n2)) * math.sin(theta_i)
    if abs(sin_t) >= 1.0:
        return 1.0
    theta_t = math.asin(sin_t)
    ci = math.cos(theta_i)
    ct = math.cos(theta_t)
    rs = ((n1 * ci) - (n2 * ct)) / ((n1 * ci) + (n2 * ct))
    rp = ((n2 * ci) - (n1 * ct)) / ((n2 * ci) + (n1 * ct))
    return float(0.5 * (rs * rs + rp * rp))


def _fresnel_p(n1: float, n2: float, theta_i_deg: float) -> float:
    theta_i = math.radians(float(theta_i_deg))
    sin_t = (float(n1) / float(n2)) * math.sin(theta_i)
    if abs(sin_t) >= 1.0:
        return 1.0
    theta_t = math.asin(sin_t)
    ci = math.cos(theta_i)
    ct = math.cos(theta_t)
    rp = ((n2 * ci) - (n1 * ct)) / ((n2 * ci) + (n1 * ct))
    return float(rp * rp)


def _linear_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.size == 0 or yp.size != yt.size:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if ss_tot <= 1.0e-12:
        return 1.0 if ss_res <= 1.0e-12 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _bundle_values(center: float, half_width: float, count: int) -> list[float]:
    if count <= 1:
        return [center]
    return [
        float(v) for v in np.linspace(center - half_width, center + half_width, count)
    ]


def _normalize2(x: float, y: float) -> tuple[float, float]:
    norm = math.hypot(float(x), float(y))
    if norm <= 1.0e-12:
        return 0.0, 0.0
    return float(x) / norm, float(y) / norm


def _cross2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def _reflect2(
    direction: tuple[float, float], normal: tuple[float, float]
) -> tuple[float, float]:
    dx, dy = _normalize2(direction[0], direction[1])
    nx, ny = _normalize2(normal[0], normal[1])
    dot = dx * nx + dy * ny
    return _normalize2(dx - 2.0 * dot * nx, dy - 2.0 * dot * ny)


def _refract2(
    direction: tuple[float, float],
    normal: tuple[float, float],
    n1: float,
    n2: float,
) -> tuple[float, float] | None:
    dx, dy = _normalize2(direction[0], direction[1])
    nx, ny = _normalize2(normal[0], normal[1])
    cos_i = -(dx * nx + dy * ny)
    if cos_i < 0.0:
        nx, ny = -nx, -ny
        cos_i = -(dx * nx + dy * ny)
    eta = float(n1) / max(1.0e-12, float(n2))
    k = 1.0 - eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if k < 0.0:
        return None
    tx = eta * dx + (eta * cos_i - math.sqrt(k)) * nx
    ty = eta * dy + (eta * cos_i - math.sqrt(k)) * ny
    return _normalize2(tx, ty)


def _ray_circle_intersections(
    origin: tuple[float, float],
    direction: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> list[float]:
    dx, dy = _normalize2(direction[0], direction[1])
    ox = float(origin[0]) - float(center[0])
    oy = float(origin[1]) - float(center[1])
    rr = max(1.0e-12, float(radius) * float(radius))
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - rr
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return []
    root = math.sqrt(max(0.0, disc))
    vals = [0.5 * (-b - root), 0.5 * (-b + root)]
    return [float(t) for t in sorted(vals) if t > 1.0e-6]


def _ray_segment_intersection(
    origin: tuple[float, float],
    direction: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    t_min: float = 1.0e-6,
) -> tuple[float, float, tuple[float, float]] | None:
    d = _normalize2(direction[0], direction[1])
    s = (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    denom = _cross2(d, s)
    if abs(denom) <= 1.0e-9:
        return None
    qmp = (float(a[0]) - float(origin[0]), float(a[1]) - float(origin[1]))
    t = _cross2(qmp, s) / denom
    u = _cross2(qmp, d) / denom
    if t <= float(t_min) or u < -1.0e-6 or u > 1.0 + 1.0e-6:
        return None
    px = float(origin[0]) + d[0] * t
    py = float(origin[1]) + d[1] * t
    return float(t), float(u), (px, py)


def _line_point_at_x(
    start: tuple[float, float], direction: tuple[float, float], x_target: float
) -> tuple[float, float] | None:
    dx, dy = _normalize2(direction[0], direction[1])
    if abs(dx) <= 1.0e-9:
        return None
    scale = (float(x_target) - float(start[0])) / dx
    if scale <= 1.0e-6:
        return None
    return float(x_target), float(start[1]) + scale * dy


def _point_field(
    xg: np.ndarray,
    yg: np.ndarray,
    points: list[tuple[float, float]],
    *,
    sigma_x: float,
    sigma_y: float,
    weights: list[float] | None = None,
) -> np.ndarray:
    sx = max(1.0e-6, float(sigma_x))
    sy = max(1.0e-6, float(sigma_y))
    out = np.zeros_like(xg, dtype=float)
    if not points:
        return out
    raw_weights = (
        weights
        if isinstance(weights, list) and len(weights) == len(points)
        else [1.0] * len(points)
    )
    for (px, py), weight in zip(points, raw_weights):
        out += float(weight) * np.exp(
            -0.5 * (((xg - float(px)) / sx) ** 2 + ((yg - float(py)) / sy) ** 2)
        )
    return out


def _polyline_glow_field(
    xg: np.ndarray,
    yg: np.ndarray,
    polylines: list[list[list[float]]],
    *,
    sigma: float,
) -> np.ndarray:
    sig = max(1.0e-6, float(sigma))
    inv = 0.5 / (sig * sig)
    out = np.zeros_like(xg, dtype=float)
    for poly in polylines:
        if not isinstance(poly, list) or len(poly) < 2:
            continue
        for p0, p1 in zip(poly[:-1], poly[1:]):
            x0 = float(p0[0])
            y0 = float(p0[1])
            x1 = float(p1[0])
            y1 = float(p1[1])
            vx = x1 - x0
            vy = y1 - y0
            seg_len2 = max(1.0e-12, vx * vx + vy * vy)
            tt = ((xg - x0) * vx + (yg - y0) * vy) / seg_len2
            tt = np.clip(tt, 0.0, 1.0)
            px = x0 + tt * vx
            py = y0 + tt * vy
            dist2 = (xg - px) ** 2 + (yg - py) ** 2
            out += np.exp(-dist2 * inv)
    return out


def _ray_irradiance_field(
    xg: np.ndarray,
    yg: np.ndarray,
    polylines: list[list[list[float]]],
    detector_points: list[tuple[float, float]],
    *,
    beam_sigma: float,
    detector_sigma_y: float,
    beam_weight: float = 0.10,
    detector_weights: list[float] | None = None,
) -> np.ndarray:
    dx = abs(float(xg[0, 1] - xg[0, 0])) if xg.shape[1] > 1 else 1.0
    dy = abs(float(yg[1, 0] - yg[0, 0])) if yg.shape[0] > 1 else 1.0
    sigma_x = max(1.8 * dx, 0.010)
    sigma_y = max(float(detector_sigma_y), 1.8 * dy)
    screen = _point_field(
        xg,
        yg,
        detector_points,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        weights=detector_weights,
    )
    beam = _polyline_glow_field(
        xg, yg, polylines, sigma=max(float(beam_sigma), 1.5 * max(dx, dy))
    )
    return screen + float(beam_weight) * beam


def _offset_strip_polygon(
    points: list[tuple[float, float]],
    offset: tuple[float, float],
    *,
    z: float = 0.0,
) -> list[list[float]]:
    front = [[float(px), float(py), z] for px, py in points]
    back = [
        [float(px) + float(offset[0]), float(py) + float(offset[1]), z]
        for px, py in reversed(points)
    ]
    return front + back


def _segment_strip_polygon(
    p0: tuple[float, float],
    p1: tuple[float, float],
    thickness: float,
    *,
    z: float = 0.012,
) -> list[list[float]]:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    length = math.hypot(dx, dy)
    if length <= 1.0e-12:
        return [
            [float(p0[0]), float(p0[1]), z],
            [float(p0[0]), float(p0[1]), z],
            [float(p0[0]), float(p0[1]), z],
        ]
    nx = -dy / length
    ny = dx / length
    half = 0.5 * float(thickness)
    return [
        [float(p0[0]) + nx * half, float(p0[1]) + ny * half, z],
        [float(p1[0]) + nx * half, float(p1[1]) + ny * half, z],
        [float(p1[0]) - nx * half, float(p1[1]) - ny * half, z],
        [float(p0[0]) - nx * half, float(p0[1]) - ny * half, z],
    ]


def _axis_strip_polygon(
    center_x: float,
    center_y: float,
    angle_deg: float,
    length: float,
    thickness: float,
    *,
    z: float = 0.018,
) -> list[list[float]]:
    theta = math.radians(float(angle_deg))
    ux = math.cos(theta)
    uy = math.sin(theta)
    half = 0.5 * float(length)
    p0 = (float(center_x) - ux * half, float(center_y) - uy * half)
    p1 = (float(center_x) + ux * half, float(center_y) + uy * half)
    return _segment_strip_polygon(p0, p1, float(thickness), z=z)


def _rect_polygon2d(
    center_x: float,
    center_y: float,
    half_w: float,
    half_h: float,
    *,
    angle_deg: float = 0.0,
    z: float = 0.012,
) -> list[list[float]]:
    theta = math.radians(float(angle_deg))
    ct = math.cos(theta)
    st = math.sin(theta)
    out: list[list[float]] = []
    for xx, yy in [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]:
        out.append(
            [
                float(center_x) + xx * ct - yy * st,
                float(center_y) + xx * st + yy * ct,
                z,
            ]
        )
    return out


def _annulus_polygon2d(
    center_x: float,
    center_y: float,
    inner_r: float,
    outer_r: float,
    *,
    n: int = 72,
    z: float = 0.0,
) -> list[list[float]]:
    outer: list[list[float]] = []
    inner: list[list[float]] = []
    count = max(16, int(n))
    for i in range(count):
        t = 2.0 * math.pi * float(i) / float(count)
        ct = math.cos(t)
        st = math.sin(t)
        outer.append(
            [
                float(center_x) + float(outer_r) * ct,
                float(center_y) + float(outer_r) * st,
                z,
            ]
        )
    for i in range(count - 1, -1, -1):
        t = 2.0 * math.pi * float(i) / float(count)
        ct = math.cos(t)
        st = math.sin(t)
        inner.append(
            [
                float(center_x) + float(inner_r) * ct,
                float(center_y) + float(inner_r) * st,
                z,
            ]
        )
    return outer + inner


def _component_spec(
    style: str, polygons: list[list[list[float]]], **overrides: Any
) -> dict[str, Any]:
    return {"style": str(style), "polygons": polygons, **overrides}


def _wave_field_resolution(case: dict[str, Any]) -> tuple[int, int]:
    vis = (
        case.get("visualization") if isinstance(case.get("visualization"), dict) else {}
    )
    res = vis.get("field_resolution", [192, 128])
    if not isinstance(res, list) or len(res) != 2:
        res = [192, 128]
    return max(48, int(res[0])), max(48, int(res[1]))


def _wave_solver_grid_pixels(case: dict[str, Any]) -> int:
    vis = (
        case.get("visualization") if isinstance(case.get("visualization"), dict) else {}
    )
    px = vis.get("solver_grid_pixels", 256)
    return max(96, int(px))


def _wave_extent(case: dict[str, Any]) -> list[float]:
    vis = (
        case.get("visualization") if isinstance(case.get("visualization"), dict) else {}
    )
    extent = vis.get("field_extent", [-0.01, 0.01, -0.006, 0.006])
    if not isinstance(extent, list) or len(extent) != 4:
        extent = [-0.01, 0.01, -0.006, 0.006]
    return [float(v) for v in extent]


def _hcipy_output_grid(case: dict[str, Any]) -> Any:
    from hcipy import make_uniform_grid

    nx, ny = _wave_field_resolution(case)
    x0, x1, y0, y1 = _wave_extent(case)
    return make_uniform_grid(
        (nx, ny), extent=(x1 - x0, y1 - y0), center=(0.5 * (x0 + x1), 0.5 * (y0 + y1))
    )


def _hcipy_pupil_grid(case: dict[str, Any], diameter: float) -> Any:
    from hcipy import make_pupil_grid

    px = _wave_solver_grid_pixels(case)
    return make_pupil_grid((px, px), diameter=float(diameter))


def _hcipy_wavefront(
    mask: np.ndarray,
    *,
    grid: Any,
    wavelength_nm: float,
    phase_slope_x: float = 0.0,
    phase_slope_y: float = 0.0,
) -> Any:
    from hcipy import Field, Wavefront

    lam = max(1.0e-12, float(wavelength_nm) * 1.0e-9)
    phase = (
        2.0
        * math.pi
        * (
            float(phase_slope_x) * np.asarray(grid.x, dtype=float)
            + float(phase_slope_y) * np.asarray(grid.y, dtype=float)
        )
        / lam
    )
    electric_field = Field(np.asarray(mask, dtype=complex) * np.exp(1j * phase), grid)
    wf = Wavefront(electric_field, wavelength=lam)
    try:
        wf.total_power = 1.0
    except Exception:
        pass
    return wf


def _hcipy_result_arrays(out_wavefront: Any) -> dict[str, np.ndarray]:
    power = np.asarray(out_wavefront.power.shaped, dtype=float)
    efield = np.asarray(out_wavefront.electric_field.shaped, dtype=complex)
    amplitude = np.abs(efield)
    phase = np.angle(efield)
    return {
        "irradiance": power,
        "amplitude": amplitude,
        "phase_rad": phase,
    }


def _single_slit_mask(
    grid: Any, *, slit_width: float, slit_height: float
) -> np.ndarray:
    return (
        (np.abs(np.asarray(grid.x, dtype=float)) <= 0.5 * float(slit_width))
        & (np.abs(np.asarray(grid.y, dtype=float)) <= 0.5 * float(slit_height))
    ).astype(float)


def _double_slit_mask(
    grid: Any,
    *,
    slit_width: float,
    slit_height: float,
    slit_separation: float,
    phase_delay_rad: float,
) -> np.ndarray:
    x = np.asarray(grid.x, dtype=float)
    y = np.asarray(grid.y, dtype=float)
    lower = (
        (np.abs(x + 0.5 * float(slit_separation)) <= 0.5 * float(slit_width))
        & (np.abs(y) <= 0.5 * float(slit_height))
    ).astype(complex)
    upper = (
        (np.abs(x - 0.5 * float(slit_separation)) <= 0.5 * float(slit_width))
        & (np.abs(y) <= 0.5 * float(slit_height))
    ).astype(complex)
    return np.asarray(
        lower + upper * np.exp(1j * float(phase_delay_rad)), dtype=complex
    )


def _circular_mask(grid: Any, *, diameter: float) -> np.ndarray:
    x = np.asarray(grid.x, dtype=float)
    y = np.asarray(grid.y, dtype=float)
    return ((x * x + y * y) <= (0.5 * float(diameter)) ** 2).astype(float)


def _knife_edge_mask(
    grid: Any, *, beam_width: float, beam_height: float, edge_offset: float
) -> np.ndarray:
    x = np.asarray(grid.x, dtype=float)
    y = np.asarray(grid.y, dtype=float)
    beam = (np.abs(x) <= 0.5 * float(beam_width)) & (
        np.abs(y) <= 0.5 * float(beam_height)
    )
    unblocked = x <= float(edge_offset)
    return (beam & unblocked).astype(float)


def _pack_polylines(
    polylines: list[list[list[float]]], *, wavelength_nm: float, strength: float
) -> tuple[np.ndarray, list[list[int]], dict[str, np.ndarray]]:
    points: list[list[float]] = []
    lines: list[list[int]] = []
    wavelengths: list[float] = []
    strengths: list[float] = []
    for poly in polylines:
        ids: list[int] = []
        for pt in poly:
            ids.append(len(points))
            points.append([float(pt[0]), float(pt[1]), float(pt[2])])
        if len(ids) >= 2:
            lines.append(ids)
            wavelengths.append(float(wavelength_nm))
            strengths.append(float(strength))
    return (
        np.asarray(points, dtype=float),
        lines,
        {
            "wavelength_nm": np.asarray(wavelengths, dtype=float),
            "strength": np.asarray(strengths, dtype=float),
        },
    )


def _merge_polyline_sets(
    chunks: list[tuple[np.ndarray, list[list[int]], dict[str, np.ndarray]]]
) -> tuple[np.ndarray, list[list[int]], dict[str, np.ndarray]]:
    points: list[list[float]] = []
    lines: list[list[int]] = []
    cell_data: dict[str, list[float]] = {}
    offset = 0
    for pts, lns, cdata in chunks:
        for key, arr in cdata.items():
            cell_data.setdefault(key, [])
            cell_data[key].extend(
                float(v) for v in np.asarray(arr, dtype=float).reshape(-1)
            )
        for pt in np.asarray(pts, dtype=float):
            points.append([float(pt[0]), float(pt[1]), float(pt[2])])
        for line in lns:
            lines.append([offset + int(i) for i in line])
        offset += int(np.asarray(pts).shape[0])
    return (
        np.asarray(points, dtype=float),
        lines,
        {k: np.asarray(v, dtype=float) for k, v in cell_data.items()},
    )


def _simulate_plane_mirror(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    mirror = (case.get("optics") or {}).get("mirror") or {}
    source = ((case.get("illumination") or {}).get("source")) or {}
    receiver = case.get("receiver") if isinstance(case.get("receiver"), dict) else {}
    reflectance = float(mirror.get("reflectance", 0.92))
    source_distance = float(source.get("source_distance", 1.0))
    screen_distance = float(receiver.get("screen_distance", 1.2))
    divergence = float(source.get("divergence_deg", 0.4))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    for value in values:
        theta = math.radians(float(value))
        sigma = 0.04 + 0.004 * divergence + 0.002 * abs(float(value))
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        for ang in _bundle_values(float(value), max(0.5, divergence), bundle_count):
            th = math.radians(ang)
            endpoint = (2.0 * screen_distance * math.tan(th), screen_distance)
            detector_points.append(endpoint)
            polylines.append(
                [
                    [-source_distance, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [endpoint[0], endpoint[1], 0.0],
                ]
            )
        irr = reflectance * _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.012, 0.75 * sigma),
            detector_sigma_y=max(0.60 * sigma, 1.0e-3),
            beam_weight=0.13,
        )
        arrays = {
            "irradiance": irr,
            "reflection_efficiency": np.full_like(irr, reflectance),
            "optical_path": np.full_like(irr, source_distance + screen_distance),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=reflectance
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        out_dir = _normalize2(2.0 * screen_distance * math.tan(theta), screen_distance)
        normal = _normalize2(1.0 - out_dir[0], -out_dir[1])
        tangent = _normalize2(-normal[1], normal[0])
        seg_half = 0.11 * min(abs(extent[1] - extent[0]), abs(extent[3] - extent[2]))
        mirror_poly = _segment_strip_polygon(
            (-seg_half * tangent[0], -seg_half * tangent[1]),
            (seg_half * tangent[0], seg_half * tangent[1]),
            0.010 * max(abs(extent[1] - extent[0]), abs(extent[3] - extent[2])),
        )
        component_specs = [
            _component_spec(
                "metal", [mirror_poly], representation="Wireframe", opacity=0.72
            )
        ]
        frames.append(
            {
                "value": float(value),
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    centroids = [s["centroid_x"] for s in summaries]
    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "spot_shift_span": float(max(centroids) - min(centroids)) if centroids else 0.0,
        "reflection_efficiency_mean": reflectance,
        "reflection_law_error_deg": 0.0,
    }
    return frames, metrics


def _simulate_dielectric_block(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    block = (case.get("optics") or {}).get("block") or {}
    source = ((case.get("illumination") or {}).get("source")) or {}
    receiver = case.get("receiver") if isinstance(case.get("receiver"), dict) else {}
    n1 = 1.0
    n2 = float(block.get("int_ior", 1.5))
    thickness = float(block.get("thickness", 0.32))
    source_distance = float(source.get("source_distance", 1.1))
    screen_distance = float(receiver.get("screen_distance", 0.9))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    trans_vals: list[float] = []
    snell_errs: list[float] = []
    for value in values:
        theta_i = math.radians(float(value))
        theta_t = math.asin(
            max(-0.999999, min(0.999999, (n1 / n2) * math.sin(theta_i)))
        )
        exit_x = thickness * math.tan(theta_t)
        reflectance = _fresnel_unpolarized(n1, n2, float(value))
        trans = 1.0 - reflectance
        sigma = 0.035 + 0.0025 * abs(float(value))
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        for ang in _bundle_values(float(value), 0.6, bundle_count):
            th_i = math.radians(ang)
            th_t = math.asin(max(-0.999999, min(0.999999, (n1 / n2) * math.sin(th_i))))
            x_exit_ray = thickness * math.tan(th_t)
            endpoint = (
                x_exit_ray + screen_distance * math.tan(th_i),
                thickness + screen_distance,
            )
            detector_points.append(endpoint)
            polylines.append(
                [
                    [-source_distance, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [x_exit_ray, thickness, 0.0],
                    [endpoint[0], endpoint[1], 0.0],
                ]
            )
        irr = trans * _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.012, 0.70 * sigma),
            detector_sigma_y=max(0.65 * sigma, 1.0e-3),
            beam_weight=0.12,
        )
        arrays = {
            "irradiance": irr,
            "transmission_efficiency": np.full_like(irr, trans),
            "optical_path": np.full_like(
                irr, source_distance + n2 * thickness + screen_distance
            ),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=trans
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        trans_vals.append(trans)
        snell_errs.append(
            abs(
                math.degrees(math.asin((n1 / n2) * math.sin(theta_i)))
                - math.degrees(theta_t)
            )
        )
        block_center_x = 0.5 * exit_x
        block_half_w = max(0.18, abs(exit_x) + 0.16)
        block_poly = _rect_polygon2d(
            block_center_x, 0.5 * thickness, block_half_w, 0.5 * thickness
        )
        component_specs = [
            _component_spec(
                "glass", [block_poly], representation="Wireframe", opacity=0.62
            )
        ]
        frames.append(
            {
                "value": float(value),
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "spot_shift_span": float(
            max((s["centroid_x"] for s in summaries), default=0.0)
            - min((s["centroid_x"] for s in summaries), default=0.0)
        ),
        "transmission_efficiency_mean": float(np.mean(trans_vals))
        if trans_vals
        else 0.0,
        "snell_residual_deg": float(max(snell_errs)) if snell_errs else 0.0,
    }
    return frames, metrics


def _simulate_concave_mirror(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    mirror = (case.get("optics") or {}).get("mirror") or {}
    f = max(0.08, float(mirror.get("focal_length", 0.42)))
    do = max(float(mirror.get("object_distance", 1.2)), f + 0.05)
    aperture = max(0.01, float(mirror.get("aperture_radius", 0.09)))
    reflectance = _clamp(float(mirror.get("reflectance", 0.94)), 0.1, 0.999)
    di = 1.0 / max(1.0e-6, (1.0 / f) - (1.0 / do))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 11)))
    object_pt = (-do, 0.0)
    focus_pt = (-di, 0.0)
    xc = -0.5 * (do + di)
    a = -xc
    b = math.sqrt(max(1.0e-6, do * di))
    usable_aperture = min(aperture, 0.92 * b)
    mirror_arc: list[tuple[float, float]] = []
    for yy in np.linspace(-usable_aperture, usable_aperture, 48):
        root = math.sqrt(max(0.0, 1.0 - (float(yy) * float(yy)) / max(1.0e-12, b * b)))
        mirror_arc.append((xc + a * root, float(yy)))
    mirror_shell = _offset_strip_polygon(
        mirror_arc, (max(0.006, 0.08 * usable_aperture), 0.0)
    )
    component_specs = [
        _component_spec(
            "metal", [mirror_shell], representation="Wireframe", opacity=0.68
        )
    ]

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    focus_sigmas: list[float] = []
    for value in values:
        sensor_distance = float(value)
        detector_x = -sensor_distance
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        for yy in _bundle_values(0.0, usable_aperture, bundle_count):
            root = math.sqrt(max(0.0, 1.0 - (yy * yy) / max(1.0e-12, b * b)))
            hit = (xc + a * root, float(yy))
            out_dir = (focus_pt[0] - hit[0], focus_pt[1] - hit[1])
            det_pt = _line_point_at_x(hit, out_dir, detector_x)
            if det_pt is None:
                continue
            detector_points.append(det_pt)
            polylines.append(
                [
                    [object_pt[0], object_pt[1], 0.0],
                    [hit[0], hit[1], 0.0],
                    [det_pt[0], det_pt[1], 0.0],
                ]
            )
        endpoint_spread = (
            float(np.std([pt[1] for pt in detector_points])) if detector_points else 0.0
        )
        sigma = max(0.012, 0.009 + 1.00 * endpoint_spread)
        base_field = _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.010, 0.12 * usable_aperture),
            detector_sigma_y=sigma,
            beam_weight=0.07,
        )
        focus_amp = reflectance / max(1.0e-5, sigma * sigma)
        base_peak = float(np.max(base_field)) if base_field.size else 0.0
        irr = base_field * (focus_amp / max(base_peak, 1.0e-12))
        amp = float(np.max(irr)) if irr.size else 0.0
        arrays = {
            "irradiance": irr,
            "defocus_radius": np.full_like(irr, sigma),
            "focus_gain_field": np.full_like(irr, amp),
            "reflection_efficiency": np.full_like(irr, reflectance),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=reflectance
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        focus_sigmas.append(sigma)
        frames.append(
            {
                "value": sensor_distance,
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    peaks = [s["peak"] for s in summaries]
    if focus_sigmas and len(values):
        min_sigma = float(min(focus_sigmas))
        candidates = [
            i
            for i, sigma in enumerate(focus_sigmas)
            if float(sigma) <= min_sigma * (1.0 + 1.0e-6)
        ]
        best_idx = min(candidates or [0], key=lambda i: abs(float(values[i]) - di))
    else:
        best_idx = int(np.argmax(peaks)) if peaks else 0
    metrics = {
        "peak_irradiance_max": max(peaks, default=0.0),
        "peak_irradiance_mean": float(np.mean(peaks)) if peaks else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "focus_distance_best": float(values[best_idx]) if len(values) else di,
        "focus_gain": float(max(peaks) / max(min(peaks), 1.0e-6)) if peaks else 0.0,
        "min_spot_radius": float(min(focus_sigmas)) if focus_sigmas else 0.0,
        "reflection_efficiency_mean": reflectance,
    }
    return frames, metrics


def _simulate_tir(case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    waveguide = (case.get("optics") or {}).get("waveguide") or {}
    n_core = float(waveguide.get("n_core", 1.55))
    n_clad = float(waveguide.get("n_cladding", 1.0))
    screen_offset = float(waveguide.get("screen_offset", 0.25))
    critical = math.degrees(math.asin(max(0.0, min(0.999999, n_clad / n_core))))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    refl_vals: list[float] = []
    trans_vals: list[float] = []
    for value in values:
        theta = float(value)
        if theta < critical:
            theta_t = math.degrees(
                math.asin(
                    max(
                        -0.999999,
                        min(
                            0.999999, (n_core / n_clad) * math.sin(math.radians(theta))
                        ),
                    )
                )
            )
            x0 = screen_offset + 0.22 * math.tan(math.radians(theta_t))
            y0 = -0.36
            trans = 1.0 - _fresnel_unpolarized(n_core, n_clad, theta)
            refl = 1.0 - trans
        else:
            theta_t = None
            x0 = (
                screen_offset
                + 0.08
                + 0.18 * math.tan(math.radians(theta - critical + 2.0))
            )
            y0 = 0.18
            refl = min(0.999, 0.96 + 0.02 * math.tanh(0.15 * (theta - critical)))
            trans = 0.0
        sigma = 0.035 + 0.0015 * abs(theta - critical)
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        for ang in _bundle_values(theta, 0.8, bundle_count):
            if ang < critical:
                theta_out = math.degrees(
                    math.asin(
                        max(
                            -0.999999,
                            min(
                                0.999999,
                                (n_core / n_clad) * math.sin(math.radians(ang)),
                            ),
                        )
                    )
                )
                endpoint = (
                    screen_offset + 0.22 * math.tan(math.radians(theta_out)),
                    -0.36,
                )
                detector_points.append(endpoint)
                polylines.append(
                    [
                        [0.0, 0.0, 0.0],
                        [screen_offset, 0.0, 0.0],
                        [endpoint[0], endpoint[1], 0.0],
                    ]
                )
            else:
                endpoint = (
                    screen_offset
                    + 0.08
                    + 0.18 * math.tan(math.radians(ang - critical + 2.0)),
                    0.18,
                )
                detector_points.append(endpoint)
                polylines.append(
                    [
                        [0.0, 0.0, 0.0],
                        [screen_offset, 0.0, 0.0],
                        [endpoint[0], endpoint[1], 0.0],
                    ]
                )
        irr = max(refl, trans) * _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.012, 0.65 * sigma),
            detector_sigma_y=sigma,
            beam_weight=0.12,
        )
        arrays = {
            "irradiance": irr,
            "reflection_efficiency": np.full_like(irr, refl),
            "transmission_efficiency": np.full_like(irr, trans),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=max(refl, trans)
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        refl_vals.append(refl)
        trans_vals.append(trans)
        guide_half = 0.045
        guide_poly = _rect_polygon2d(
            0.5 * (screen_offset + max(0.15, x0)),
            0.0,
            0.5 * abs(max(0.15, x0) - 0.0) + 0.08,
            guide_half,
        )
        interface_poly = _segment_strip_polygon(
            (screen_offset, -0.22), (screen_offset, 0.22), 0.010
        )
        component_specs = [
            _component_spec(
                "glass", [guide_poly], representation="Wireframe", opacity=0.42
            ),
            _component_spec(
                "interface", [interface_poly], representation="Wireframe", opacity=0.72
            ),
        ]
        frames.append(
            {
                "value": theta,
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    transition_idx = (
        int(np.argmax(np.asarray(refl_vals) > np.asarray(trans_vals)))
        if refl_vals
        else 0
    )
    critical_est = (
        float(values[min(transition_idx, len(values) - 1)]) if len(values) else critical
    )
    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "reflection_efficiency_mean": float(np.mean(refl_vals)) if refl_vals else 0.0,
        "transmission_efficiency_mean": float(np.mean(trans_vals))
        if trans_vals
        else 0.0,
        "critical_angle_est_deg": critical_est,
        "critical_angle_error_deg": abs(critical_est - critical),
    }
    return frames, metrics


def _simulate_brewster(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    interface = (case.get("optics") or {}).get("interface") or {}
    n1 = float(interface.get("n1", 1.0))
    n2 = float(interface.get("n2", 1.5))
    brewster = math.degrees(math.atan2(n2, n1))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    rp_vals: list[float] = []
    for value in values:
        rp = _fresnel_p(n1, n2, float(value))
        x0 = 0.28 * math.tan(math.radians(float(value)))
        sigma = 0.04 + 0.002 * abs(float(value) - brewster)
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        for ang in _bundle_values(float(value), 0.6, bundle_count):
            endpoint = (0.28 * math.tan(math.radians(ang)), 0.42)
            detector_points.append(endpoint)
            polylines.append(
                [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [endpoint[0], endpoint[1], 0.0]]
            )
        irr = max(rp, 1.0e-4) * _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.012, 0.65 * sigma),
            detector_sigma_y=sigma,
            beam_weight=0.13,
        )
        arrays = {
            "irradiance": irr,
            "reflectance_p": np.full_like(irr, rp),
            "polarization_degree": np.full_like(irr, 1.0),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=max(rp, 1.0e-4)
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        rp_vals.append(rp)
        out_dir = _normalize2(x0, 0.42)
        normal = _normalize2(1.0 - out_dir[0], -out_dir[1])
        tangent = _normalize2(-normal[1], normal[0])
        seg_half = 0.10 * min(abs(extent[1] - extent[0]), abs(extent[3] - extent[2]))
        interface_poly = _segment_strip_polygon(
            (-seg_half * tangent[0], -seg_half * tangent[1]),
            (seg_half * tangent[0], seg_half * tangent[1]),
            0.010 * max(abs(extent[1] - extent[0]), abs(extent[3] - extent[2])),
        )
        component_specs = [
            _component_spec(
                "interface", [interface_poly], representation="Wireframe", opacity=0.72
            )
        ]
        frames.append(
            {
                "value": float(value),
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    min_idx = int(np.argmin(rp_vals)) if rp_vals else 0
    brewster_est = float(values[min_idx]) if len(values) else brewster
    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "reflection_efficiency_mean": float(np.mean(rp_vals)) if rp_vals else 0.0,
        "brewster_angle_est_deg": brewster_est,
        "brewster_angle_error_deg": abs(brewster_est - brewster),
    }
    return frames, metrics


def _simulate_prism(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    prism = (case.get("optics") or {}).get("prism") or {}
    apex = math.radians(float(prism.get("apex_angle_deg", 48.0)))
    iors = (
        prism.get("ior_rgb")
        if isinstance(prism.get("ior_rgb"), dict)
        else {"r": 1.515, "g": 1.522, "b": 1.531}
    )
    n_r = float(iors.get("r", 1.515))
    n_g = float(iors.get("g", 1.522))
    n_b = float(iors.get("b", 1.531))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    spans: list[float] = []
    for value in values:
        trim = math.radians(float(value))
        base_shift = 0.38 * math.tan(0.55 * trim)
        color_scale = 3.6
        xr = base_shift + color_scale * (n_r - n_g) * apex
        xg0 = base_shift
        xb = base_shift + color_scale * (n_b - n_g) * apex
        sigma = 0.025 + 0.0015 * abs(float(value))
        field_r = _gaussian2d(
            xg, yg, x0=xr, y0=0.12, sigma_x=sigma, sigma_y=sigma, amp=1.0
        )
        field_g = _gaussian2d(
            xg, yg, x0=xg0, y0=0.0, sigma_x=sigma, sigma_y=sigma, amp=1.0
        )
        field_b = _gaussian2d(
            xg, yg, x0=xb, y0=-0.12, sigma_x=sigma, sigma_y=sigma, amp=1.0
        )
        irr = 0.30 * field_r + 0.59 * field_g + 0.11 * field_b
        arrays = {
            "irradiance": irr,
            "irradiance_r": field_r,
            "irradiance_g": field_g,
            "irradiance_b": field_b,
            "dispersion_index": np.full_like(irr, abs(xr - xb)),
        }
        chunks = []
        for wl, x_out, y_out, strength in [
            (650.0, xr, 0.12, 1.0),
            (550.0, xg0, 0.0, 1.0),
            (450.0, xb, -0.12, 1.0),
        ]:
            polylines = []
            for offset in _bundle_values(0.0, 0.03, bundle_count):
                polylines.append(
                    [
                        [-0.7, offset, 0.0],
                        [0.0, offset, 0.0],
                        [0.15, 0.0, 0.0],
                        [x_out, y_out + 0.20 * offset, 0.0],
                    ]
                )
            chunks.append(
                _pack_polylines(polylines, wavelength_nm=wl, strength=strength)
            )
        pts, lines, cdata = _merge_polyline_sets(chunks)
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        spans.append(abs(xr - xb))
        frames.append(
            {
                "value": float(value),
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
            }
        )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "dispersion_span": float(max(spans)) if spans else 0.0,
        "spot_shift_span": float(
            max((s["centroid_x"] for s in summaries), default=0.0)
            - min((s["centroid_x"] for s in summaries), default=0.0)
        ),
    }
    return frames, metrics


def _simulate_beam_splitter(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    splitter = (case.get("optics") or {}).get("beam_splitter") or {}
    base_reflectance = _clamp(float(splitter.get("reflectance", 0.5)), 0.1, 0.9)
    base_transmittance = _clamp(float(splitter.get("transmittance", 0.5)), 0.1, 0.9)
    nominal_angle = float(splitter.get("angle_deg", 45.0))
    total_nominal = max(1.0e-6, base_reflectance + base_transmittance)
    base_reflectance /= total_nominal
    base_transmittance /= total_nominal
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    refl_vals: list[float] = []
    trans_vals: list[float] = []
    balance_errors: list[float] = []
    separations: list[float] = []
    centroids: list[float] = []
    for value in values:
        trim = float(value)
        reflectance = _clamp(base_reflectance + 0.0035 * trim, 0.18, 0.82)
        transmittance = _clamp(base_transmittance - 0.0035 * trim, 0.18, 0.82)
        total = max(1.0e-6, reflectance + transmittance)
        reflectance /= total
        transmittance /= total

        sigma_ref = 0.038 + 0.0015 * abs(trim)
        sigma_trans = 0.038 + 0.0012 * abs(trim)
        polylines_ref: list[list[list[float]]] = []
        polylines_trans: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        detector_weights: list[float] = []
        ref_endpoints: list[tuple[float, float]] = []
        trans_endpoints: list[tuple[float, float]] = []
        for offset in _bundle_values(0.0, 0.08, bundle_count):
            ref_endpoint = (
                -0.48 + 0.14 * math.tan(math.radians(0.35 * trim)),
                0.34 + 0.0035 * trim + 0.40 * offset,
            )
            trans_endpoint = (
                0.52 + 0.12 * math.tan(math.radians(0.22 * trim)),
                -0.28 + 0.0025 * trim + 0.25 * offset,
            )
            ref_endpoints.append(ref_endpoint)
            trans_endpoints.append(trans_endpoint)
            detector_points.extend([ref_endpoint, trans_endpoint])
            detector_weights.extend([reflectance, transmittance])
            polylines_ref.append(
                [
                    [-0.65, offset, 0.0],
                    [0.0, offset, 0.0],
                    [ref_endpoint[0], ref_endpoint[1], 0.0],
                ]
            )
            polylines_trans.append(
                [
                    [-0.65, offset, 0.0],
                    [0.0, offset, 0.0],
                    [trans_endpoint[0], trans_endpoint[1], 0.0],
                ]
            )
        polylines = polylines_ref + polylines_trans
        irr = _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.012, 0.5 * min(sigma_ref, sigma_trans)),
            detector_sigma_y=max(0.85 * max(sigma_ref, sigma_trans), 1.0e-3),
            beam_weight=0.10,
            detector_weights=detector_weights,
        )
        arrays = {
            "irradiance": irr,
            "reflection_efficiency": np.full_like(irr, reflectance),
            "transmission_efficiency": np.full_like(irr, transmittance),
            "split_balance_field": np.full_like(irr, abs(reflectance - transmittance)),
        }
        pts, lines, cdata = _merge_polyline_sets(
            [
                _pack_polylines(
                    polylines_ref, wavelength_nm=550.0, strength=reflectance
                ),
                _pack_polylines(
                    polylines_trans, wavelength_nm=550.0, strength=transmittance
                ),
            ]
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        refl_vals.append(reflectance)
        trans_vals.append(transmittance)
        balance_errors.append(abs(reflectance - transmittance))
        ref_center = (
            np.mean(np.asarray(ref_endpoints, dtype=float), axis=0)
            if ref_endpoints
            else np.asarray([0.0, 0.0])
        )
        trans_center = (
            np.mean(np.asarray(trans_endpoints, dtype=float), axis=0)
            if trans_endpoints
            else np.asarray([0.0, 0.0])
        )
        separations.append(float(np.linalg.norm(ref_center - trans_center)))
        centroids.append(summary["centroid_x"])
        plate_half = 0.12 * min(abs(extent[1] - extent[0]), abs(extent[3] - extent[2]))
        angle = math.radians(nominal_angle)
        tangent = (math.cos(angle), math.sin(angle))
        plate_poly = _segment_strip_polygon(
            (-plate_half * tangent[0], -plate_half * tangent[1]),
            (plate_half * tangent[0], plate_half * tangent[1]),
            0.018 * min(abs(extent[1] - extent[0]), abs(extent[3] - extent[2])),
        )
        component_specs = [
            _component_spec(
                "glass", [plate_poly], representation="Wireframe", opacity=0.68
            )
        ]
        frames.append(
            {
                "value": trim,
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "spot_shift_span": float(max(centroids) - min(centroids)) if centroids else 0.0,
        "reflection_efficiency_mean": float(np.mean(refl_vals)) if refl_vals else 0.0,
        "transmission_efficiency_mean": float(np.mean(trans_vals))
        if trans_vals
        else 0.0,
        "split_balance_error": float(np.mean(balance_errors))
        if balance_errors
        else 0.0,
        "branch_separation": float(np.mean(separations)) if separations else 0.0,
    }
    return frames, metrics


def _simulate_thin_lens(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    lens = (case.get("optics") or {}).get("lens") or {}
    f = float(lens.get("focal_length", 0.35))
    do = max(float(lens.get("object_distance", 0.9)), f + 0.05)
    aperture = float(lens.get("aperture_radius", 0.05))
    di = 1.0 / max(1.0e-6, (1.0 / f) - (1.0 / do))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    focus_sigmas: list[float] = []
    for value in values:
        sensor_distance = float(value)
        sigma = 0.012 + 0.18 * abs(sensor_distance - di) / max(di, 1.0e-6) * max(
            0.4, aperture / 0.05
        )
        amp = 1.0 / max(1.0e-5, sigma * sigma)
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        object_x = -do
        for h in _bundle_values(0.0, aperture, bundle_count):
            target_y = h * (1.0 - sensor_distance / max(di, 1.0e-6))
            detector_points.append((sensor_distance, target_y))
            polylines.append(
                [[object_x, 0.0, 0.0], [0.0, h, 0.0], [sensor_distance, target_y, 0.0]]
            )
        base_field = _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.006, 0.08 * max(aperture, 1.0e-6)),
            detector_sigma_y=sigma,
            beam_weight=0.08,
        )
        base_peak = float(np.max(base_field)) if base_field.size else 0.0
        irr = base_field * (amp / max(base_peak, 1.0e-12))
        arrays = {
            "irradiance": irr,
            "defocus_radius": np.full_like(irr, sigma),
            "focus_gain_field": np.full_like(irr, amp),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=1.0
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        focus_sigmas.append(sigma)
        lens_poly = _segment_strip_polygon(
            (0.0, -aperture), (0.0, aperture), max(0.018, 0.18 * aperture)
        )
        component_specs = [
            _component_spec(
                "glass", [lens_poly], representation="Wireframe", opacity=0.68
            )
        ]
        frames.append(
            {
                "value": sensor_distance,
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    peaks = [s["peak"] for s in summaries]
    if focus_sigmas and len(values):
        min_sigma = float(min(focus_sigmas))
        candidates = [
            i
            for i, sigma in enumerate(focus_sigmas)
            if float(sigma) <= min_sigma * (1.0 + 1.0e-6)
        ]
        best_idx = min(candidates or [0], key=lambda i: abs(float(values[i]) - di))
    else:
        best_idx = int(np.argmax(peaks)) if peaks else 0
    metrics = {
        "peak_irradiance_max": max(peaks, default=0.0),
        "peak_irradiance_mean": float(np.mean(peaks)) if peaks else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "focus_distance_best": float(values[best_idx]) if len(values) else di,
        "focus_gain": float(max(peaks) / max(min(peaks), 1.0e-6)) if peaks else 0.0,
        "min_spot_radius": float(min(focus_sigmas)) if focus_sigmas else 0.0,
    }
    return frames, metrics


def _simulate_sphere_caustic(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    sphere = (case.get("optics") or {}).get("sphere") or {}
    radius = float(sphere.get("radius", 0.2))
    n = float(sphere.get("int_ior", 1.52))
    f = (n * radius) / max(1.0e-6, 2.0 * (n - 1.0))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))
    illum = (case.get("illumination") or {}).get("source") or {}
    source_distance = max(float(illum.get("source_distance", 0.8)), radius + 0.1)
    beam_half_width = max(0.65 * radius, float(illum.get("beam_half_width", radius)))
    usable_half_width = min(0.96 * radius, beam_half_width)
    sphere_shell = _annulus_polygon2d(0.0, 0.0, 0.97 * radius, radius, n=96)
    component_specs = [
        _component_spec(
            "glass", [sphere_shell], representation="Wireframe", opacity=0.66
        )
    ]

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    gains: list[float] = []
    for value in values:
        screen_distance = float(value)
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        for yy in _bundle_values(0.0, usable_half_width, bundle_count):
            origin = (-source_distance, float(yy))
            incoming = (1.0, 0.0)
            hits_in = _ray_circle_intersections(origin, incoming, (0.0, 0.0), radius)
            if not hits_in:
                continue
            p_in = (origin[0] + hits_in[0], origin[1])
            normal_in = _normalize2(p_in[0], p_in[1])
            inside_dir = _refract2(incoming, normal_in, 1.0, n)
            if inside_dir is None:
                continue
            start_inside = (
                p_in[0] + 1.0e-5 * inside_dir[0],
                p_in[1] + 1.0e-5 * inside_dir[1],
            )
            hits_out = _ray_circle_intersections(
                start_inside, inside_dir, (0.0, 0.0), radius
            )
            if not hits_out:
                continue
            p_out = (
                start_inside[0] + hits_out[0] * inside_dir[0],
                start_inside[1] + hits_out[0] * inside_dir[1],
            )
            normal_out = _normalize2(p_out[0], p_out[1])
            out_dir = _refract2(inside_dir, normal_out, n, 1.0)
            if out_dir is None:
                out_dir = _reflect2(inside_dir, normal_out)
            det_pt = _line_point_at_x(p_out, out_dir, screen_distance)
            if det_pt is None:
                continue
            if (
                det_pt[0] < extent[0]
                or det_pt[0] > extent[1]
                or det_pt[1] < extent[2]
                or det_pt[1] > extent[3]
            ):
                continue
            detector_points.append(det_pt)
            polylines.append(
                [
                    [origin[0], origin[1], 0.0],
                    [p_in[0], p_in[1], 0.0],
                    [p_out[0], p_out[1], 0.0],
                    [det_pt[0], det_pt[1], 0.0],
                ]
            )
        y_hits = [pt[1] for pt in detector_points]
        sigma = max(0.012, 0.008 + 0.25 * (float(np.std(y_hits)) if y_hits else 0.0))
        irr = _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.010, 0.08 * radius),
            detector_sigma_y=sigma,
            beam_weight=0.10,
        )
        amp = float(np.max(irr)) if irr.size else 0.0
        arrays = {
            "irradiance": irr,
            "caustic_gain_field": np.full_like(irr, amp),
            "optical_path": np.full_like(irr, radius + screen_distance),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=amp
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        gains.append(amp)
        frames.append(
            {
                "value": screen_distance,
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    best_idx = int(np.argmax(gains)) if gains else 0
    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "focus_distance_best": float(values[best_idx]) if len(values) else f,
        "caustic_gain": float(max(gains) / max(np.mean(gains), 1.0e-6))
        if gains
        else 0.0,
    }
    return frames, metrics


def _simulate_corner_cube(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    cube = (case.get("optics") or {}).get("corner_cube") or {}
    aperture = max(0.01, float(cube.get("aperture_radius", 0.1)))
    reflectance = _clamp(float(cube.get("reflectance", 0.92)), 0.1, 0.999)
    face_error = max(0.0, float(cube.get("face_error_deg", 0.12)))
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))
    receiver = case.get("receiver") if isinstance(case.get("receiver"), dict) else {}
    detector_x = -max(float(receiver.get("screen_distance", 0.6)), 0.2)
    tip = (aperture, 0.0)
    leg = aperture * math.sqrt(2.0)
    upper_angle = math.radians(135.0 - 0.5 * face_error)
    lower_angle = math.radians(-135.0 + 0.5 * face_error)
    upper_left = (
        tip[0] + leg * math.cos(upper_angle),
        tip[1] + leg * math.sin(upper_angle),
    )
    lower_left = (
        tip[0] + leg * math.cos(lower_angle),
        tip[1] + leg * math.sin(lower_angle),
    )
    component_specs = [
        _component_spec(
            "metal",
            [
                [
                    [upper_left[0], upper_left[1], 0.0],
                    [tip[0], tip[1], 0.0],
                    [lower_left[0], lower_left[1], 0.0],
                ]
            ],
            representation="Wireframe",
            opacity=0.68,
        )
    ]

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    return_vals: list[float] = []
    retro_errs: list[float] = []
    offsets: list[float] = []
    centroids: list[float] = []
    for value in values:
        theta = float(value)
        incoming_dir = _normalize2(
            math.cos(math.radians(theta)), math.sin(math.radians(theta))
        )
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        frame_retro_errs: list[float] = []
        frame_returns: list[float] = []
        x_start = min(float(extent[0]) + 0.02, detector_x - 0.12)
        x_entry = min(upper_left[0], lower_left[0])
        for yy in _bundle_values(0.0, 0.8 * aperture, bundle_count):
            origin_y = float(yy) - (x_entry - x_start) * incoming_dir[1] / max(
                abs(incoming_dir[0]), 1.0e-9
            )
            origin = (x_start, origin_y)
            first_seg = (upper_left, tip)
            second_seg = (lower_left, tip)
            first_hit = _ray_segment_intersection(
                origin, incoming_dir, first_seg[0], first_seg[1]
            )
            if first_hit is None:
                first_seg = (lower_left, tip)
                second_seg = (upper_left, tip)
                first_hit = _ray_segment_intersection(
                    origin, incoming_dir, first_seg[0], first_seg[1]
                )
            if first_hit is None:
                continue
            _, _, p1 = first_hit
            tangent1 = (
                first_seg[1][0] - first_seg[0][0],
                first_seg[1][1] - first_seg[0][1],
            )
            normal1 = _normalize2(-tangent1[1], tangent1[0])
            dir1 = _reflect2(incoming_dir, normal1)
            second_hit = _ray_segment_intersection(
                (p1[0] + 1.0e-5 * dir1[0], p1[1] + 1.0e-5 * dir1[1]),
                dir1,
                second_seg[0],
                second_seg[1],
            )
            if second_hit is None:
                continue
            _, _, p2 = second_hit
            tangent2 = (
                second_seg[1][0] - second_seg[0][0],
                second_seg[1][1] - second_seg[0][1],
            )
            normal2 = _normalize2(-tangent2[1], tangent2[0])
            dir2 = _reflect2(dir1, normal2)
            det_pt = _line_point_at_x(p2, dir2, detector_x)
            if det_pt is None:
                continue
            detector_points.append(det_pt)
            polylines.append(
                [
                    [origin[0], origin[1], 0.0],
                    [p1[0], p1[1], 0.0],
                    [p2[0], p2[1], 0.0],
                    [det_pt[0], det_pt[1], 0.0],
                ]
            )
            wanted = (-incoming_dir[0], -incoming_dir[1])
            cos_err = _clamp(dir2[0] * wanted[0] + dir2[1] * wanted[1], -1.0, 1.0)
            frame_retro_errs.append(math.degrees(math.acos(cos_err)))
            frame_returns.append(reflectance * reflectance)
        retro_error = float(np.mean(frame_retro_errs)) if frame_retro_errs else 90.0
        return_eff = float(np.mean(frame_returns)) if frame_returns else 0.0
        spread_y = (
            float(np.std([pt[1] for pt in detector_points])) if detector_points else 1.0
        )
        offset = spread_y
        sigma = max(0.010, 0.008 + 0.20 * spread_y)
        irr = return_eff * _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=max(0.010, 0.08 * aperture),
            detector_sigma_y=sigma,
            beam_weight=0.07,
        )
        arrays = {
            "irradiance": irr,
            "return_efficiency": np.full_like(irr, return_eff),
            "retroreflection_error_deg": np.full_like(irr, retro_error),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=return_eff
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        return_vals.append(return_eff)
        retro_errs.append(retro_error)
        offsets.append(abs(offset))
        centroids.append(summary["centroid_x"])
        frames.append(
            {
                "value": theta,
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "spot_shift_span": float(max(centroids) - min(centroids)) if centroids else 0.0,
        "retroreflection_error_deg": float(max(retro_errs)) if retro_errs else 0.0,
        "return_efficiency_mean": float(np.mean(return_vals)) if return_vals else 0.0,
        "return_spot_offset": float(max(offsets)) if offsets else 0.0,
    }
    return frames, metrics


def _empty_polydata() -> tuple[np.ndarray, list[list[int]], dict[str, np.ndarray]]:
    return np.zeros((0, 3), dtype=float), [], {}


def _simulate_single_slit(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    slit = optics.get("slit") if isinstance(optics.get("slit"), dict) else {}
    wavelength_nm = float(slit.get("wavelength_nm", 532.0))
    slit_width = float(slit.get("slit_width", 80.0e-6))
    slit_height = float(slit.get("slit_height", 600.0e-6))
    pupil_diameter = float(slit.get("pupil_diameter", 1.0e-3))
    screen_distance = float(
        (case.get("receiver") or {}).get(
            "screen_distance", slit.get("screen_distance", 1.2)
        )
    )
    extent = _wave_extent(case)

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    widths: list[float] = []
    centroids: list[float] = []

    if backend == "hcipy":
        from hcipy import FraunhoferPropagator

        pupil_grid = _hcipy_pupil_grid(case, pupil_diameter)
        output_grid = _hcipy_output_grid(case)
        propagator = FraunhoferPropagator(
            pupil_grid, output_grid, focal_length=screen_distance
        )
        points, lines, cell_data = _empty_polydata()

        for value in values:
            theta = math.radians(float(value))
            mask = _single_slit_mask(
                pupil_grid, slit_width=slit_width, slit_height=slit_height
            )
            wf = _hcipy_wavefront(
                mask,
                grid=pupil_grid,
                wavelength_nm=wavelength_nm,
                phase_slope_x=math.sin(theta),
            )
            out = propagator(wf)
            arrays = _hcipy_result_arrays(out)
            arrays["scan_value"] = np.full_like(arrays["irradiance"], float(value))
            summary = _field_summary(arrays["irradiance"], extent)
            xs, _ = _axis_coords(extent, arrays["irradiance"].shape)
            widths.append(_half_max_width(_central_profile(arrays["irradiance"]), xs))
            centroids.append(summary["centroid_x"])
            summaries.append(summary)
            frames.append(
                {
                    "value": float(value),
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )
    else:
        xg, yg, _ = _field_grid(case)
        points, lines, cell_data = _empty_polydata()
        lam = wavelength_nm * 1.0e-9
        x_scale = max(1.0e-6, lam * screen_distance / max(slit_width, 1.0e-9))
        y_scale = max(1.0e-6, 0.45 * (extent[3] - extent[2]))
        for value in values:
            theta = math.radians(float(value))
            x0 = screen_distance * math.tan(theta)
            irr = (np.sinc((xg - x0) / x_scale) ** 2) * np.exp(
                -0.5 * (yg / y_scale) ** 2
            )
            arrays = {
                "irradiance": irr,
                "amplitude": np.sqrt(np.maximum(irr, 0.0)),
                "phase_rad": np.full_like(irr, theta),
            }
            summary = _field_summary(irr, extent)
            xs, _ = _axis_coords(extent, irr.shape)
            widths.append(_half_max_width(_central_profile(irr), xs))
            centroids.append(summary["centroid_x"])
            summaries.append(summary)
            frames.append(
                {
                    "value": float(value),
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "spot_shift_span": float(max(centroids) - min(centroids)) if centroids else 0.0,
        "central_lobe_width": float(np.mean(widths)) if widths else 0.0,
    }
    return frames, metrics


def _simulate_double_slit(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    slit = (
        optics.get("double_slit") if isinstance(optics.get("double_slit"), dict) else {}
    )
    wavelength_nm = float(slit.get("wavelength_nm", 532.0))
    slit_width = float(slit.get("slit_width", 55.0e-6))
    slit_height = float(slit.get("slit_height", 550.0e-6))
    slit_separation = float(slit.get("slit_separation", 240.0e-6))
    pupil_diameter = float(slit.get("pupil_diameter", 1.0e-3))
    screen_distance = float(
        (case.get("receiver") or {}).get(
            "screen_distance", slit.get("screen_distance", 1.0)
        )
    )
    extent = _wave_extent(case)

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    visibilities: list[float] = []
    spacings: list[float] = []
    peak_positions: list[float] = []

    if backend == "hcipy":
        from hcipy import Field, FraunhoferPropagator, Wavefront

        pupil_grid = _hcipy_pupil_grid(case, pupil_diameter)
        output_grid = _hcipy_output_grid(case)
        propagator = FraunhoferPropagator(
            pupil_grid, output_grid, focal_length=screen_distance
        )
        points, lines, cell_data = _empty_polydata()

        for value in values:
            phase_delay = math.radians(float(value))
            mask = _double_slit_mask(
                pupil_grid,
                slit_width=slit_width,
                slit_height=slit_height,
                slit_separation=slit_separation,
                phase_delay_rad=phase_delay,
            )
            wf = Wavefront(
                Field(mask, pupil_grid), wavelength=max(1.0e-12, wavelength_nm * 1.0e-9)
            )
            try:
                wf.total_power = 1.0
            except Exception:
                pass
            out = propagator(wf)
            arrays = _hcipy_result_arrays(out)
            arrays["scan_value"] = np.full_like(arrays["irradiance"], float(value))
            summary = _field_summary(arrays["irradiance"], extent)
            profile = _central_profile(arrays["irradiance"])
            xs, _ = _axis_coords(extent, arrays["irradiance"].shape)
            visibilities.append(_fringe_visibility_from_profile(profile))
            spacings.append(
                abs(
                    _mean_spacing_from_indices(
                        _peak_indices(profile, threshold_rel=0.15), xs
                    )
                )
            )
            peak_positions.append(
                float(xs[int(np.argmax(profile))]) if profile.size else 0.0
            )
            summaries.append(summary)
            frames.append(
                {
                    "value": float(value),
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )
    else:
        xg, yg, _ = _field_grid(case)
        points, lines, cell_data = _empty_polydata()
        lam = wavelength_nm * 1.0e-9
        beta = math.pi * slit_width / max(lam * screen_distance, 1.0e-12)
        gamma = math.pi * slit_separation / max(lam * screen_distance, 1.0e-12)
        y_scale = max(1.0e-6, 0.45 * (extent[3] - extent[2]))
        for value in values:
            phase_delay = math.radians(float(value))
            envelope = np.sinc((beta / math.pi) * xg) ** 2
            fringes = np.cos(gamma * xg + 0.5 * phase_delay) ** 2
            irr = envelope * fringes * np.exp(-0.5 * (yg / y_scale) ** 2)
            arrays = {
                "irradiance": irr,
                "amplitude": np.sqrt(np.maximum(irr, 0.0)),
                "phase_rad": np.full_like(irr, phase_delay),
            }
            summary = _field_summary(irr, extent)
            profile = _central_profile(irr)
            xs, _ = _axis_coords(extent, irr.shape)
            visibilities.append(_fringe_visibility_from_profile(profile))
            spacings.append(
                abs(
                    _mean_spacing_from_indices(
                        _peak_indices(profile, threshold_rel=0.15), xs
                    )
                )
            )
            peak_positions.append(
                float(xs[int(np.argmax(profile))]) if profile.size else 0.0
            )
            summaries.append(summary)
            frames.append(
                {
                    "value": float(value),
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "fringe_visibility": float(np.mean(visibilities)) if visibilities else 0.0,
        "fringe_spacing": float(np.mean([v for v in spacings if v > 0.0]))
        if any(v > 0.0 for v in spacings)
        else 0.0,
        "fringe_shift_span": float(max(peak_positions) - min(peak_positions))
        if peak_positions
        else 0.0,
    }
    return frames, metrics


def _simulate_circular_airy(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    aperture = (
        optics.get("aperture") if isinstance(optics.get("aperture"), dict) else {}
    )
    wavelength_nm = float(aperture.get("wavelength_nm", 532.0))
    base_diameter = float(aperture.get("diameter", 0.4e-3))
    pupil_diameter = float(aperture.get("pupil_diameter", 1.0e-3))
    focal_length = float(
        (case.get("receiver") or {}).get(
            "screen_distance", aperture.get("focal_length", 1.0)
        )
    )
    extent = _wave_extent(case)

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    ring_radii: list[float] = []
    widths: list[float] = []

    if backend == "hcipy":
        from hcipy import FraunhoferPropagator

        pupil_grid = _hcipy_pupil_grid(case, pupil_diameter)
        output_grid = _hcipy_output_grid(case)
        propagator = FraunhoferPropagator(
            pupil_grid, output_grid, focal_length=focal_length
        )
        points, lines, cell_data = _empty_polydata()

        for value in values:
            scale = float(value)
            mask = _circular_mask(pupil_grid, diameter=base_diameter * scale)
            wf = _hcipy_wavefront(mask, grid=pupil_grid, wavelength_nm=wavelength_nm)
            out = propagator(wf)
            arrays = _hcipy_result_arrays(out)
            arrays["scan_value"] = np.full_like(arrays["irradiance"], scale)
            summary = _field_summary(arrays["irradiance"], extent)
            xs, _ = _axis_coords(extent, arrays["irradiance"].shape)
            widths.append(_half_max_width(_central_profile(arrays["irradiance"]), xs))
            ring_radii.append(
                _airy_ring_radius_from_field(arrays["irradiance"], extent)
            )
            summaries.append(summary)
            frames.append(
                {
                    "value": scale,
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )
    else:
        xg, yg, _ = _field_grid(case)
        points, lines, cell_data = _empty_polydata()
        lam = wavelength_nm * 1.0e-9
        for value in values:
            scale = float(value)
            radius = np.sqrt(xg * xg + yg * yg)
            airy_scale = max(
                1.0e-6, 1.22 * lam * focal_length / max(base_diameter * scale, 1.0e-9)
            )
            irr = (np.sinc(radius / airy_scale) ** 2) * (
                1.0 + 0.12 * np.cos(8.0 * math.pi * radius / airy_scale)
            )
            irr = np.maximum(irr, 0.0)
            arrays = {
                "irradiance": irr,
                "amplitude": np.sqrt(np.maximum(irr, 0.0)),
                "phase_rad": np.zeros_like(irr),
            }
            summary = _field_summary(irr, extent)
            xs, _ = _axis_coords(extent, irr.shape)
            widths.append(_half_max_width(_central_profile(irr), xs))
            ring_radii.append(_airy_ring_radius_from_field(irr, extent))
            summaries.append(summary)
            frames.append(
                {
                    "value": scale,
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "central_lobe_width": float(np.mean(widths)) if widths else 0.0,
        "airy_ring_radius": float(np.mean([v for v in ring_radii if v > 0.0]))
        if any(v > 0.0 for v in ring_radii)
        else 0.0,
    }
    return frames, metrics


def _simulate_knife_edge(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    edge = optics.get("edge") if isinstance(optics.get("edge"), dict) else {}
    wavelength_nm = float(edge.get("wavelength_nm", 532.0))
    beam_width = float(edge.get("beam_width", 1.8e-3))
    beam_height = float(edge.get("beam_height", 1.2e-3))
    propagation_distance = float(edge.get("propagation_distance", 0.35))
    extent = _wave_extent(case)

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    transition_widths: list[float] = []
    overshoots: list[float] = []

    if backend == "hcipy":
        from hcipy import FresnelPropagator

        plane_grid = _hcipy_output_grid(case)
        propagator = FresnelPropagator(plane_grid, propagation_distance)
        points, lines, cell_data = _empty_polydata()

        for value in values:
            edge_offset = float(value)
            mask = _knife_edge_mask(
                plane_grid,
                beam_width=beam_width,
                beam_height=beam_height,
                edge_offset=edge_offset,
            )
            wf = _hcipy_wavefront(mask, grid=plane_grid, wavelength_nm=wavelength_nm)
            out = propagator(wf)
            arrays = _hcipy_result_arrays(out)
            arrays["scan_value"] = np.full_like(arrays["irradiance"], edge_offset)
            summary = _field_summary(arrays["irradiance"], extent)
            profile = _central_profile(arrays["irradiance"])
            xs, _ = _axis_coords(extent, arrays["irradiance"].shape)
            transition_widths.append(_transition_width_from_profile(profile, xs))
            overshoots.append(_edge_overshoot_from_profile(profile))
            summaries.append(summary)
            frames.append(
                {
                    "value": edge_offset,
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )
    else:
        xg, yg, _ = _field_grid(case)
        points, lines, cell_data = _empty_polydata()
        sigma = max(1.0e-6, 0.08 * (extent[1] - extent[0]))
        beam = np.exp(-0.5 * (yg / max(1.0e-6, 0.22 * (extent[3] - extent[2]))) ** 2)
        for value in values:
            edge_offset = float(value)
            transition = 0.5 * (1.0 + np.tanh((xg - edge_offset) / sigma))
            oscillation = 1.0 + 0.15 * np.sin(
                14.0 * math.pi * (xg - edge_offset) / max(sigma, 1.0e-6)
            ) * np.exp(-np.abs(xg - edge_offset) / max(2.5 * sigma, 1.0e-6))
            irr = np.clip(transition * oscillation * beam, 0.0, None)
            arrays = {
                "irradiance": irr,
                "amplitude": np.sqrt(np.maximum(irr, 0.0)),
                "phase_rad": np.zeros_like(irr),
            }
            summary = _field_summary(irr, extent)
            profile = _central_profile(irr)
            xs, _ = _axis_coords(extent, irr.shape)
            transition_widths.append(_transition_width_from_profile(profile, xs))
            overshoots.append(_edge_overshoot_from_profile(profile))
            summaries.append(summary)
            frames.append(
                {
                    "value": edge_offset,
                    "arrays": arrays,
                    "points": points,
                    "lines": lines,
                    "cell_data": cell_data,
                    "summary": summary,
                }
            )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "edge_transition_width": float(
            np.mean([v for v in transition_widths if v > 0.0])
        )
        if any(v > 0.0 for v in transition_widths)
        else 0.0,
        "edge_overshoot": float(np.mean(overshoots)) if overshoots else 0.0,
    }
    return frames, metrics


def _simulate_diffraction_grating(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _ = backend
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    grating = optics.get("grating") if isinstance(optics.get("grating"), dict) else {}
    slit_width = float(grating.get("slit_width", 32.0e-6))
    slit_pitch = float(grating.get("slit_pitch", 160.0e-6))
    slit_count = max(2, int(grating.get("slit_count", 9)))
    screen_distance = float((case.get("receiver") or {}).get("screen_distance", 1.1))
    xg, yg, extent = _field_grid(case)
    points, lines, cell_data = _empty_polydata()
    y_scale = max(1.0e-6, 0.45 * (extent[3] - extent[2]))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    spacings: list[float] = []
    order_counts: list[float] = []
    for wavelength_nm in values:
        lam = max(1.0e-12, float(wavelength_nm) * 1.0e-9)
        denom_scale = max(lam * screen_distance, 1.0e-12)
        envelope = np.sinc((slit_width * xg) / denom_scale) ** 2
        alpha = math.pi * slit_pitch * xg / denom_scale
        den = np.sin(alpha)
        num = np.sin(slit_count * alpha)
        grating_factor = np.ones_like(alpha, dtype=float)
        mask = np.abs(den) > 1.0e-7
        grating_factor[mask] = (num[mask] / (float(slit_count) * den[mask])) ** 2
        irr = np.maximum(
            0.0, envelope * grating_factor * np.exp(-0.5 * (yg / y_scale) ** 2)
        )
        arrays = {
            "irradiance": irr,
            "amplitude": np.sqrt(np.maximum(irr, 0.0)),
            "phase_rad": np.zeros_like(irr),
            "wavelength_nm": np.full_like(irr, float(wavelength_nm)),
        }
        summary = _field_summary(irr, extent)
        profile = _central_profile(irr)
        xs, _ = _axis_coords(extent, irr.shape)
        peaks = _peak_indices(profile, threshold_rel=0.12)
        spacing = abs(_mean_spacing_from_indices(peaks, xs))
        if spacing > 0.0:
            spacings.append(spacing)
        order_counts.append(float(peaks.size))
        summaries.append(summary)
        frames.append(
            {
                "value": float(wavelength_nm),
                "arrays": arrays,
                "points": points,
                "lines": lines,
                "cell_data": cell_data,
                "summary": summary,
            }
        )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "grating_order_spacing": float(np.mean(spacings)) if spacings else 0.0,
        "grating_order_count": float(np.mean(order_counts)) if order_counts else 0.0,
        "spectral_spread": float(max(spacings) - min(spacings))
        if len(spacings) >= 2
        else 0.0,
    }
    return frames, metrics


def _thin_film_reflectance(
    n0: float,
    n1: float,
    n2: float,
    angle_deg: float,
    thickness_nm: float,
    wavelength_nm: np.ndarray,
) -> np.ndarray:
    theta0 = math.radians(float(angle_deg))
    sin1 = np.clip(
        (float(n0) / max(float(n1), 1.0e-12)) * math.sin(theta0), -0.999, 0.999
    )
    cos0 = max(1.0e-6, math.cos(theta0))
    cos1 = np.sqrt(np.maximum(1.0e-12, 1.0 - sin1 * sin1))
    sin2 = np.clip((float(n1) / max(float(n2), 1.0e-12)) * sin1, -0.999, 0.999)
    cos2 = np.sqrt(np.maximum(1.0e-12, 1.0 - sin2 * sin2))
    r01 = ((n0 * cos0) - (n1 * cos1)) / ((n0 * cos0) + (n1 * cos1))
    r12 = ((n1 * cos1) - (n2 * cos2)) / ((n1 * cos1) + (n2 * cos2))
    lam_nm = np.maximum(np.asarray(wavelength_nm, dtype=float), 1.0e-6)
    delta = 4.0 * math.pi * n1 * float(thickness_nm) * cos1 / lam_nm
    phase = np.exp(1j * delta)
    refl = np.abs((r01 + r12 * phase) / (1.0 + r01 * r12 * phase)) ** 2
    return np.clip(np.asarray(refl, dtype=float), 0.0, 1.0)


def _simulate_thin_film(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _ = backend
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    film = optics.get("film") if isinstance(optics.get("film"), dict) else {}
    n0 = float(film.get("n_incident", 1.0))
    n1 = float(film.get("n_film", 1.46))
    n2 = float(film.get("n_substrate", 1.52))
    angle_deg = float(film.get("angle_deg", 0.0))
    center_nm = float(film.get("wavelength_center_nm", 550.0))
    span_nm = float(film.get("wavelength_span_nm", 220.0))
    xg, yg, extent = _field_grid(case)
    points, lines, cell_data = _empty_polydata()
    x_norm = (xg - extent[0]) / max(extent[1] - extent[0], 1.0e-12)
    wavelength_axis = center_nm - 0.5 * span_nm + span_nm * x_norm
    y_texture = 1.0 + 0.035 * np.cos(
        2.0 * math.pi * (yg - extent[2]) / max(extent[3] - extent[2], 1.0e-12)
    )

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    contrasts: list[float] = []
    fringe_periods: list[float] = []
    reflectances: list[float] = []
    for thickness_nm in values:
        refl = _thin_film_reflectance(
            n0, n1, n2, angle_deg, float(thickness_nm), wavelength_axis
        )
        irr = np.clip(refl * y_texture, 0.0, None)
        arrays = {
            "irradiance": irr,
            "amplitude": np.sqrt(np.maximum(irr, 0.0)),
            "phase_rad": np.full_like(
                irr, 4.0 * math.pi * n1 * float(thickness_nm) / max(center_nm, 1.0e-6)
            ),
            "reflection_efficiency": refl,
        }
        summary = _field_summary(irr, extent)
        profile = _central_profile(irr)
        p_max = float(np.max(profile)) if profile.size else 0.0
        p_min = float(np.min(profile)) if profile.size else 0.0
        contrasts.append(float((p_max - p_min) / max(p_max + p_min, 1.0e-12)))
        reflectances.append(float(np.mean(refl)))
        cos1 = math.sqrt(
            max(
                1.0e-12,
                1.0
                - ((n0 / max(n1, 1.0e-12)) * math.sin(math.radians(angle_deg))) ** 2,
            )
        )
        fringe_periods.append(
            float(
                center_nm
                * center_nm
                / max(2.0 * n1 * max(float(thickness_nm), 1.0e-6) * cos1, 1.0e-6)
            )
        )
        summaries.append(summary)
        frames.append(
            {
                "value": float(thickness_nm),
                "arrays": arrays,
                "points": points,
                "lines": lines,
                "cell_data": cell_data,
                "summary": summary,
            }
        )

    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "thin_film_reflectance_mean": float(np.mean(reflectances))
        if reflectances
        else 0.0,
        "thin_film_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
        "thin_film_fringe_period_nm": float(np.mean(fringe_periods))
        if fringe_periods
        else 0.0,
    }
    return frames, metrics


def _simulate_gaussian_beam(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _ = backend
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    beam = (
        optics.get("gaussian_beam")
        if isinstance(optics.get("gaussian_beam"), dict)
        else {}
    )
    waist = max(1.0e-8, float(beam.get("waist_radius", 0.35e-3)))
    wavelength_nm = float(beam.get("wavelength_nm", 532.0))
    lam = max(1.0e-12, wavelength_nm * 1.0e-9)
    waist_pos = float(beam.get("waist_position", 0.0))
    power = float(beam.get("power", 1.0))
    rayleigh = math.pi * waist * waist / lam
    divergence = math.degrees(math.atan(lam / (math.pi * waist)))
    xg, yg, extent = _field_grid(case)
    points, lines, cell_data = _empty_polydata()
    rr2 = xg * xg + yg * yg

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    waists: list[float] = []
    for z in values:
        z_rel = float(z) - waist_pos
        w_z = waist * math.sqrt(1.0 + (z_rel / max(rayleigh, 1.0e-12)) ** 2)
        amp = power * (waist / max(w_z, 1.0e-12)) ** 2
        irr = amp * np.exp(-2.0 * rr2 / max(w_z * w_z, 1.0e-18))
        gouy = math.atan2(z_rel, max(rayleigh, 1.0e-12))
        arrays = {
            "irradiance": irr,
            "amplitude": np.sqrt(np.maximum(irr, 0.0)),
            "phase_rad": np.full_like(irr, gouy),
            "beam_radius": np.full_like(irr, w_z),
        }
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        waists.append(w_z)
        frames.append(
            {
                "value": float(z),
                "arrays": arrays,
                "points": points,
                "lines": lines,
                "cell_data": cell_data,
                "summary": summary,
            }
        )

    peaks = [s["peak"] for s in summaries]
    metrics = {
        "peak_irradiance_max": max(peaks, default=0.0),
        "peak_irradiance_mean": float(np.mean(peaks)) if peaks else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "beam_waist_min": float(min(waists)) if waists else 0.0,
        "rayleigh_range": float(rayleigh),
        "divergence_angle_deg": float(divergence),
        "focus_gain": float(max(peaks) / max(min(peaks), 1.0e-12))
        if len(peaks) >= 2
        else 0.0,
    }
    return frames, metrics


def _waveplate_transmission(
    theta_deg: float,
    retardance_deg: float,
    input_deg: float,
    analyzer_deg: float,
    extinction_ratio: float,
) -> float:
    theta = math.radians(float(theta_deg))
    delta = math.radians(float(retardance_deg))
    alpha = math.radians(float(input_deg))
    beta = math.radians(float(analyzer_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.asarray([[c, -s], [s, c]], dtype=complex)
    inv = np.asarray([[c, s], [-s, c]], dtype=complex)
    plate = (
        rot @ np.asarray([[1.0, 0.0], [0.0, np.exp(1j * delta)]], dtype=complex) @ inv
    )
    ein = np.asarray([math.cos(alpha), math.sin(alpha)], dtype=complex)
    analyzer = np.asarray([math.cos(beta), math.sin(beta)], dtype=complex)
    ideal = float(abs(np.vdot(analyzer, plate @ ein)) ** 2)
    floor = 1.0 / max(1.0, float(extinction_ratio))
    return float(np.clip(floor + (1.0 - floor) * ideal, 0.0, 1.0))


def _simulate_waveplate(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _ = backend
    values = _scan_values(case)
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    waveplate = (
        optics.get("waveplate") if isinstance(optics.get("waveplate"), dict) else {}
    )
    retardance = float(waveplate.get("retardance_deg", 90.0))
    input_pol = float(waveplate.get("input_polarization_deg", 0.0))
    analyzer = float(waveplate.get("analyzer_angle_deg", 45.0))
    extinction = float(waveplate.get("extinction_ratio", 160.0))
    xg, yg, extent = _field_grid(case)
    points, lines, cell_data = _empty_polydata()
    sigma = 0.13 * max(extent[1] - extent[0], extent[3] - extent[2])

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    intensities: list[float] = []
    for theta in values:
        transmission = _waveplate_transmission(
            float(theta), retardance, input_pol, analyzer, extinction
        )
        irr = _gaussian2d(
            xg, yg, x0=0.0, y0=0.0, sigma_x=sigma, sigma_y=sigma, amp=transmission
        )
        arrays = {
            "irradiance": irr,
            "transmission_efficiency": np.full_like(irr, transmission),
            "retardance_deg": np.full_like(irr, retardance),
            "polarization_degree": np.full_like(irr, 1.0 - 1.0 / max(extinction, 1.0)),
        }
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        intensities.append(transmission)
        frames.append(
            {
                "value": float(theta),
                "arrays": arrays,
                "points": points,
                "lines": lines,
                "cell_data": cell_data,
                "summary": summary,
            }
        )

    i_max = max(intensities) if intensities else 0.0
    i_min = min(intensities) if intensities else 0.0
    denom = i_max + i_min
    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "transmission_efficiency_mean": float(np.mean(intensities))
        if intensities
        else 0.0,
        "waveplate_modulation_depth": float((i_max - i_min) / denom)
        if denom > 1.0e-12
        else 0.0,
        "polarization_contrast": float(i_max / max(i_min, 1.0e-6))
        if intensities
        else 0.0,
        "retardance_error_deg": float(
            min(abs(retardance - 90.0), abs(retardance - 180.0))
        ),
    }
    return frames, metrics


def _simulate_polarizer(
    case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = _scan_values(case)
    xg, yg, extent = _field_grid(case)
    polarizers = (case.get("optics") or {}).get("polarizers") or {}
    ext_ratio = max(1.0, float(polarizers.get("extinction_ratio", 120.0)))
    theta0 = float(polarizers.get("initial_polarization_deg", 0.0))
    floor = 1.0 / ext_ratio
    bundle_count = int(((case.get("visualization") or {}).get("ray_bundle_count", 9)))

    frames: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []
    intensities: list[float] = []
    for value in values:
        theta = math.radians(float(value) - theta0)
        transmission = floor + (1.0 - floor) * (math.cos(theta) ** 2)
        sigma = 0.05
        polylines: list[list[list[float]]] = []
        detector_points: list[tuple[float, float]] = []
        for h in _bundle_values(0.0, 0.08, bundle_count):
            endpoint = (0.48, h)
            detector_points.append(endpoint)
            polylines.append(
                [
                    [-0.45, h, 0.0],
                    [-0.12, h, 0.0],
                    [0.12, h, 0.0],
                    [endpoint[0], endpoint[1], 0.0],
                ]
            )
        base_field = _ray_irradiance_field(
            xg,
            yg,
            polylines,
            detector_points,
            beam_sigma=0.012,
            detector_sigma_y=sigma,
            beam_weight=0.10,
        )
        base_peak = float(np.max(base_field)) if base_field.size else 0.0
        irr = transmission * base_field / max(base_peak, 1.0e-12)
        arrays = {
            "irradiance": irr,
            "transmission_efficiency": np.full_like(irr, transmission),
            "polarization_degree": np.full_like(irr, 1.0 - floor),
        }
        pts, lines, cdata = _pack_polylines(
            polylines, wavelength_nm=550.0, strength=transmission
        )
        summary = _field_summary(irr, extent)
        summaries.append(summary)
        intensities.append(transmission)
        plate_h = 0.13
        plate_w = 0.012
        axis_len = plate_h * 1.55
        axis_thick = max(0.003, plate_w * 0.55)
        gauge_cx = 0.63
        gauge_cy = -0.23
        gauge_half_h = 0.105
        gauge_half_w = 0.012
        fill_half_h = max(0.004, gauge_half_h * float(np.clip(transmission, 0.0, 1.0)))
        fill_cy = gauge_cy - gauge_half_h + fill_half_h
        component_specs = [
            _component_spec(
                "polarizer",
                [
                    _rect_polygon2d(-0.12, 0.0, plate_w, plate_h),
                    _rect_polygon2d(0.12, 0.0, plate_w, plate_h),
                ],
                representation="Wireframe",
                opacity=0.66,
            ),
            _component_spec(
                "polarizer_axis",
                [_axis_strip_polygon(-0.12, 0.0, theta0, axis_len, axis_thick)],
                representation="Surface With Edges",
                opacity=0.72,
            ),
            _component_spec(
                "analyzer_axis",
                [_axis_strip_polygon(0.12, 0.0, float(value), axis_len, axis_thick)],
                representation="Surface With Edges",
                opacity=0.82,
            ),
            _component_spec(
                "gauge_frame",
                [_rect_polygon2d(gauge_cx, gauge_cy, gauge_half_w, gauge_half_h)],
                representation="Wireframe",
                opacity=0.70,
            ),
            _component_spec(
                "intensity_gauge",
                [_rect_polygon2d(gauge_cx, fill_cy, gauge_half_w * 0.72, fill_half_h)],
                representation="Surface With Edges",
                opacity=0.62,
            ),
        ]
        frames.append(
            {
                "value": float(value),
                "arrays": arrays,
                "points": pts,
                "lines": lines,
                "cell_data": cdata,
                "summary": summary,
                "component_specs": component_specs,
            }
        )

    theta_vals = np.radians(values - theta0)
    target = floor + (1.0 - floor) * (np.cos(theta_vals) ** 2)
    metrics = {
        "peak_irradiance_max": max((s["peak"] for s in summaries), default=0.0),
        "peak_irradiance_mean": float(np.mean([s["peak"] for s in summaries]))
        if summaries
        else 0.0,
        "integrated_power_mean": float(np.mean([s["power"] for s in summaries]))
        if summaries
        else 0.0,
        "transmission_efficiency_mean": float(np.mean(intensities))
        if intensities
        else 0.0,
        "malus_r2": _linear_r2(
            np.asarray(intensities, dtype=float), np.asarray(target, dtype=float)
        ),
        "extinction_ratio": float(max(intensities) / max(min(intensities), 1.0e-6))
        if intensities
        else 0.0,
    }
    return frames, metrics


_SIMULATORS: dict[str, Any] = {
    "planeMirrorReflection": _simulate_plane_mirror,
    "concaveMirrorFocus": _simulate_concave_mirror,
    "dielectricBlockRefraction": _simulate_dielectric_block,
    "totalInternalReflection": _simulate_tir,
    "fresnelBrewster": _simulate_brewster,
    "beamSplitterPlate": _simulate_beam_splitter,
    "prismDispersion": _simulate_prism,
    "thinLensFocus": _simulate_thin_lens,
    "glassSphereCaustic": _simulate_sphere_caustic,
    "cornerCubeRetroreflector": _simulate_corner_cube,
    "polarizerMalus": _simulate_polarizer,
}

_WAVE_SIMULATORS: dict[str, Any] = {
    "singleSlitFraunhofer": _simulate_single_slit,
    "doubleSlitInterference": _simulate_double_slit,
    "circularApertureAiry": _simulate_circular_airy,
    "knifeEdgeFresnel": _simulate_knife_edge,
    "diffractionGrating": _simulate_diffraction_grating,
    "thinFilmInterference": _simulate_thin_film,
    "gaussianBeamPropagation": _simulate_gaussian_beam,
    "waveplatePolarization": _simulate_waveplate,
}


def _save_texture_png(field: np.ndarray, out_png: Path) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    arr = np.asarray(field, dtype=float)
    if arr.size == 0:
        return False
    norm = arr - float(np.min(arr))
    denom = float(np.max(norm))
    if denom > 1.0e-12:
        norm = norm / denom
    img = np.clip(255.0 * norm, 0.0, 255.0).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(out_png)
    return True


def _render_mitsuba_reference_frames(
    case_dir: Path,
    case: dict[str, Any],
    frames: list[dict[str, Any]],
    strict_mitsuba: bool,
    backend: str,
) -> list[dict[str, Any]]:
    if not frames:
        return []
    try:
        import mitsuba as mi
    except Exception as exc:
        if strict_mitsuba:
            raise RuntimeError(
                f"{backend} import failed during reference rendering"
            ) from exc
        return [{"status": "skipped", "reason": f"mitsuba_import_failed: {exc}"}]

    expected_major = 3 if backend == "mitsuba3" else 2
    actual_major = _mitsuba_major_version()
    if actual_major != expected_major:
        msg = (
            f"requested {backend}, but imported Mitsuba major version is {actual_major}"
        )
        if strict_mitsuba:
            raise RuntimeError(msg)
        return [{"status": "skipped", "reason": msg}]

    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    variants = (
        solver.get("variant_preferences")
        if isinstance(solver.get("variant_preferences"), list)
        else ["scalar_rgb"]
    )
    variants = list(dict.fromkeys(str(v) for v in variants))
    if (
        any("polarized" in v for v in variants)
        and "scalar_spectral_polarized" not in variants
    ):
        variants.insert(0, "scalar_spectral_polarized")
    if backend == "mitsuba3" and "scalar_rgb_polarized" in variants:
        variants = [v for v in variants if v != "scalar_rgb_polarized"]
        variants.insert(0, "scalar_spectral_polarized")
    for fallback in ("scalar_rgb", "scalar_spectral"):
        if fallback not in variants:
            variants.append(fallback)
    variant_ok = None
    for cand in variants:
        available = set(mi.variants()) if hasattr(mi, "variants") else set()
        if available and str(cand) not in available:
            continue
        try:
            mi.set_variant(str(cand))
            variant_ok = str(cand)
            break
        except Exception:
            continue
    if variant_ok is None:
        if strict_mitsuba:
            raise RuntimeError(f"No usable {backend} variant from {variants}")
        return [{"status": "skipped", "reason": f"no_variant_from_{variants}"}]

    try:
        if backend == "mitsuba3":
            T = mi.ScalarTransform4f
            load_dict = mi.load_dict
            look_at = lambda **kw: T().look_at(**kw)
            scale = lambda values: T().scale(values)
        else:
            from mitsuba.core import ScalarTransform4f as T
            from mitsuba.core.xml import load_dict

            look_at = T.look_at
            scale = T.scale
    except Exception as exc:
        if strict_mitsuba:
            raise RuntimeError(f"{backend} scene API import failed") from exc
        return [{"status": "skipped", "reason": f"mitsuba_core_failed: {exc}"}]

    ref_count = int(solver.get("reference_frames", 3))
    if ref_count <= 0:
        return []
    ref_count = max(1, ref_count)
    ref_indices = sorted(
        set(int(round(v)) for v in np.linspace(0, len(frames) - 1, ref_count))
    )
    manifest: list[dict[str, Any]] = []
    textures_dir = case_dir / "scene_assets" / "reference_textures"
    ref_dir = case_dir / "results" / "reference"

    for idx in ref_indices:
        frame = frames[idx]
        texture_png = textures_dir / f"frame_{idx:04d}.png"
        out_exr = ref_dir / f"frame_{idx:04d}.exr"
        field = np.asarray(frame["arrays"]["irradiance"], dtype=float)
        if not _save_texture_png(field, texture_png):
            if strict_mitsuba:
                raise RuntimeError(
                    "Pillow is required to generate Mitsuba reference textures."
                )
            manifest.append(
                {"frame": idx, "status": "skipped", "reason": "texture_png_failed"}
            )
            continue
        try:
            scene = load_dict(
                {
                    "type": "scene",
                    "integrator": {"type": "path"},
                    "sensor": {
                        "type": "perspective",
                        "fov": 28.0,
                        "to_world": look_at(
                            origin=[0, 0, 2.6], target=[0, 0, 0], up=[0, 1, 0]
                        ),
                        "sampler": {
                            "type": "independent",
                            "sample_count": max(
                                4, int(solver.get("samples_per_pixel", 64))
                            ),
                        },
                        "film": {
                            "type": "hdrfilm",
                            "width": int(field.shape[1]),
                            "height": int(field.shape[0]),
                        },
                    },
                    "light": {
                        "type": "constant",
                        "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
                    },
                    "screen": {
                        "type": "rectangle",
                        "to_world": scale([1.6, 1.0, 1.0]),
                        "bsdf": {
                            "type": "diffuse",
                            "reflectance": {
                                "type": "bitmap",
                                "filename": str(texture_png),
                            },
                        },
                    },
                }
            )
            if backend == "mitsuba3":
                img = mi.render(
                    scene, spp=max(4, int(solver.get("samples_per_pixel", 64)))
                )
                mi.Bitmap(img).write(str(out_exr))
            else:
                sensor = scene.sensors()[0]
                sensor.film().set_destination_file(str(out_exr))
                scene.integrator().render(scene, sensor)
                sensor.film().develop()
            manifest.append(
                {
                    "frame": idx,
                    "status": "ok",
                    "variant": variant_ok,
                    "output": str(out_exr),
                }
            )
        except Exception as exc:
            if strict_mitsuba:
                raise
            manifest.append(
                {
                    "frame": idx,
                    "status": "failed",
                    "reason": str(exc),
                    "variant": variant_ok,
                }
            )
    return manifest


def _simulate_case(
    case: dict[str, Any], backend: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family = str(case.get("family") or case.get("case_prefix") or "unknown")
    if family in MEEP_FAMILIES:
        if backend != "meep":
            raise KeyError(
                f"Meep/FDTD family {family} requires --backend meep or auto in a Meep-enabled Python."
            )
        import run_meep_case

        return run_meep_case.simulate_case(case, backend="meep")
    if family in _WAVE_SIMULATORS:
        if backend in MITSUBA_BACKENDS:
            raise KeyError(
                f"Wave-optics family {family} is not supported by Mitsuba transport backends; use --backend hcipy or auto."
            )
        return _WAVE_SIMULATORS[family](case, backend)
    if backend == "hcipy":
        raise KeyError(
            f"HCIPy backend is only supported for wave-optics families, got geometric family: {family}"
        )
    if family not in _SIMULATORS:
        raise KeyError(f"Unsupported optics family: {family}")
    return _SIMULATORS[family](case)


def _write_outputs(
    case_dir: Path,
    case: dict[str, Any],
    frames: list[dict[str, Any]],
    metrics: dict[str, Any],
    backend: str,
    strict_mitsuba: bool,
) -> None:
    results_dir = case_dir / "results"
    field_dir = results_dir / "sensor_fields"
    ray_dir = results_dir / "ray_paths"
    field_dir.mkdir(parents=True, exist_ok=True)
    ray_dir.mkdir(parents=True, exist_ok=True)

    field_datasets: list[tuple[float, str]] = []
    ray_datasets: list[tuple[float, str]] = []
    series_frames: list[dict[str, Any]] = []
    extent = (case.get("visualization") or {}).get(
        "field_extent", [-1.25, 1.25, -0.6, 0.6]
    )
    if not isinstance(extent, list) or len(extent) != 4:
        extent = [-1.25, 1.25, -0.6, 0.6]

    for idx, frame in enumerate(frames):
        field_name = f"frame_{idx:04d}.vti"
        ray_name = f"frame_{idx:04d}.vtp"
        _write_vti(field_dir / field_name, arrays=frame["arrays"], extent=extent)
        _write_vtp_lines(
            ray_dir / ray_name,
            points=frame["points"],
            lines=frame["lines"],
            cell_data=frame["cell_data"],
        )
        field_datasets.append((float(idx), f"sensor_fields/{field_name}"))
        ray_datasets.append((float(idx), f"ray_paths/{ray_name}"))
        series_frames.append(
            {
                "frame": idx,
                "scan_value": float(frame["value"]),
                "summary": frame["summary"],
                "field_file": field_name,
                "ray_file": ray_name,
                **(
                    {"component_specs": frame["component_specs"]}
                    if isinstance(frame.get("component_specs"), list)
                    else {}
                ),
            }
        )

    _write_pvd(results_dir / "sensor_fields.pvd", field_datasets)
    _write_pvd(results_dir / "ray_paths.pvd", ray_datasets)
    _write_json(
        results_dir / "scan_series.json",
        {
            "parameter": (case.get("scan") or {}).get("parameter", "scan_value"),
            "units": (case.get("scan") or {}).get("units", ""),
            "frames": series_frames,
        },
    )

    reference_manifest = []
    if backend in MITSUBA_BACKENDS:
        reference_manifest = _render_mitsuba_reference_frames(
            case_dir, case, frames, strict_mitsuba, backend
        )
    _write_json(
        results_dir / "reference_manifest.json",
        {"backend": backend, "frames": reference_manifest},
    )

    metrics = dict(metrics)
    metrics["backend"] = backend
    metrics["n_frames"] = len(frames)
    _write_json(case_dir / "postProcessing" / "metrics.json", metrics)


def _should_rerun(case_dir: Path, force_rerun: bool) -> bool:
    if force_rerun:
        return True
    log = case_dir / "log.solve"
    return not (log.exists() and "END" in log.read_text(errors="ignore"))


def run_case(
    case_dir: Path, *, backend: str, strict_mitsuba: bool, force_rerun: bool
) -> None:
    case_dir = case_dir.expanduser().resolve()
    if not _should_rerun(case_dir, force_rerun):
        print(f"[optics_bench] skip existing END log: {case_dir.name}")
        return
    case = _load_case(case_dir)
    selected_backend = _select_backend(backend, strict_mitsuba, case)
    frames, metrics = _simulate_case(case, selected_backend)
    _write_outputs(case_dir, case, frames, metrics, selected_backend, strict_mitsuba)
    _write_text(
        case_dir / "log.solve",
        "\n".join(
            [
                f"backend={selected_backend}",
                f"n_frames={len(frames)}",
                "status=ok",
                "END",
                "",
            ]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a generated optics benchmark case"
    )
    parser.add_argument("case_dir", type=Path)
    parser.add_argument(
        "--backend",
        choices=["auto", "mitsuba3", "mitsuba2", "hcipy", "meep", "surrogate"],
        default="auto",
    )
    parser.add_argument(
        "--strict-mitsuba", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    force_rerun = _bool_env("FORCE_RERUN_CASES", default=False)
    case_dir = args.case_dir.expanduser().resolve()
    try:
        run_case(
            case_dir,
            backend=args.backend,
            strict_mitsuba=bool(args.strict_mitsuba),
            force_rerun=force_rerun,
        )
        print(f"[optics_bench] run_case ok case={case_dir.name}")
    except Exception as exc:
        _write_text(
            case_dir / "log.solve",
            "\n".join(
                [
                    f"status=fail",
                    f"error={exc}",
                    "FAIL",
                    "",
                ]
            ),
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
