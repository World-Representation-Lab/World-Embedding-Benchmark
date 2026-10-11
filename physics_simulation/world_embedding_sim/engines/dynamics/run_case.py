#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from trajectory import (
    _quat_to_rotmat_np,
    _mech_cfg,
    _env_cfg,
    _case_dim,
    _family,
    _gravity,
    _polyline_points,
    _polyline_arclength,
    _polyline_progress,
    _convex_hull_xy,
    _compute_speed_series,
    _first_sustain_time,
    _find_object,
    _scalar_series_value,
    _rotate_xy,
    _rect_polygon,
    _polygon_vertical_envelope,
    _vehicle_local_sections,
    _transform_body_points_3d,
    _vehicle_polygon_xz,
    _vehicle_polygon_xy,
    _poly_axes,
    _signed_gap_poly_poly,
    _projected_shape,
    _signed_gap_objects,
    _object_rotation_matrix,
    _obb_penetration,
    _metrics_from_trajectory,
)


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_vtp_points(
    path: Path,
    *,
    points: np.ndarray,
    point_data: dict[str, np.ndarray],
) -> None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = np.zeros((0, 3), dtype=float)
    n = int(pts.shape[0])
    pts_flat = " ".join(f"{float(v):.9g}" for v in pts.reshape(-1))
    connectivity = " ".join(str(i) for i in range(n))
    offsets = " ".join(str(i + 1) for i in range(n))

    data_arrays: list[str] = []
    for name, values in point_data.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n:
            continue
        payload = " ".join(f"{float(v):.9g}" for v in arr)
        data_arrays.append(
            f'      <DataArray type="Float32" Name="{name}" format="ascii">{payload}</DataArray>'
        )
    point_data_xml = (
        "\n".join(data_arrays)
        if data_arrays
        else '      <DataArray type="Float32" Name="speed_mag" format="ascii"></DataArray>'
    )

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n'
        "      <PointData>\n"
        f"{point_data_xml}\n"
        "      </PointData>\n"
        "      <Points>\n"
        f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts_flat}</DataArray>\n'
        "      </Points>\n"
        "      <Verts>\n"
        f'        <DataArray type="Int32" Name="connectivity" format="ascii">{connectivity}</DataArray>\n'
        f'        <DataArray type="Int32" Name="offsets" format="ascii">{offsets}</DataArray>\n'
        "      </Verts>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )
    _write_text(path, xml)


def _write_vtp_polys(
    path: Path,
    *,
    points: np.ndarray,
    polys: list[list[int]],
    point_data: Optional[dict[str, np.ndarray]] = None,
    cell_data: Optional[dict[str, np.ndarray]] = None,
) -> None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = np.zeros((0, 3), dtype=float)
    n_pts = int(pts.shape[0])
    safe_polys = [
        list(map(int, poly))
        for poly in polys
        if isinstance(poly, list) and len(poly) >= 3
    ]
    n_polys = int(len(safe_polys))
    pts_flat = " ".join(f"{float(v):.9g}" for v in pts.reshape(-1))
    connectivity = " ".join(str(v) for poly in safe_polys for v in poly)
    offsets = " ".join(
        str(sum(len(poly) for poly in safe_polys[: i + 1])) for i in range(n_polys)
    )

    point_arrays: list[str] = []
    for name, values in (point_data or {}).items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n_pts:
            continue
        payload = " ".join(f"{float(v):.9g}" for v in arr)
        point_arrays.append(
            f'      <DataArray type="Float32" Name="{name}" format="ascii">{payload}</DataArray>'
        )
    point_data_xml = "\n".join(point_arrays)

    cell_arrays: list[str] = []
    for name, values in (cell_data or {}).items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n_polys:
            continue
        payload = " ".join(f"{float(v):.9g}" for v in arr)
        cell_arrays.append(
            f'      <DataArray type="Float32" Name="{name}" format="ascii">{payload}</DataArray>'
        )
    cell_data_xml = "\n".join(cell_arrays)

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{n_polys}">\n'
        f"      <PointData>\n{point_data_xml}\n      </PointData>\n"
        f"      <CellData>\n{cell_data_xml}\n      </CellData>\n"
        "      <Points>\n"
        f'        <DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts_flat}</DataArray>\n'
        "      </Points>\n"
        "      <Polys>\n"
        f'        <DataArray type="Int32" Name="connectivity" format="ascii">{connectivity}</DataArray>\n'
        f'        <DataArray type="Int32" Name="offsets" format="ascii">{offsets}</DataArray>\n'
        "      </Polys>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )
    _write_text(path, xml)


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


def _interp_series(
    times_src: np.ndarray, values_src: np.ndarray, times_query: np.ndarray
) -> np.ndarray:
    ts = np.asarray(times_src, dtype=float).reshape(-1)
    vs = np.asarray(values_src, dtype=float)
    tq = np.asarray(times_query, dtype=float).reshape(-1)
    if ts.size == 0 or tq.size == 0:
        if vs.ndim <= 1:
            return np.zeros((tq.size,), dtype=float)
        return np.zeros((tq.size, vs.shape[1]), dtype=float)
    if ts.size == 1:
        if vs.ndim <= 1:
            return np.full((tq.size,), float(vs.reshape(-1)[0]), dtype=float)
        row = vs.reshape(1, -1)[0]
        return np.repeat(row.reshape(1, -1), tq.size, axis=0)
    if vs.ndim == 1:
        return np.interp(tq, ts, vs, left=float(vs[0]), right=float(vs[-1]))
    out = np.zeros((tq.size, vs.shape[1]), dtype=float)
    for j in range(vs.shape[1]):
        out[:, j] = np.interp(
            tq, ts, vs[:, j], left=float(vs[0, j]), right=float(vs[-1, j])
        )
    return out


def _load_case(case_dir: Path) -> dict[str, Any]:
    case = _read_json(case_dir / "solver_input.json")
    if case:
        return case
    return _read_json(case_dir / "case_meta.json")


def _solver_cfg(case: dict[str, Any]) -> dict[str, Any]:
    return case.get("solver") if isinstance(case.get("solver"), dict) else {}


def _case_name(case: dict[str, Any], case_dir: Path) -> str:
    return str(case.get("case_name") or case_dir.name)


def _duration(case: dict[str, Any]) -> float:
    return float(_solver_cfg(case).get("duration", 5.0))


def _step_size(case: dict[str, Any]) -> float:
    return float(_solver_cfg(case).get("step_size", 0.005))


def _sample_count(case: dict[str, Any]) -> int:
    return max(40, int(_solver_cfg(case).get("sample_count", 160)))


def _has_pychrono() -> bool:
    return importlib.util.find_spec("pychrono") is not None


def _select_backend(requested: str, strict_chrono: bool) -> str:
    requested = (requested or "auto").strip().lower()
    if requested not in {"auto", "chrono", "surrogate"}:
        raise ValueError(f"Unknown backend: {requested}")
    if requested == "surrogate":
        return "surrogate"
    if _has_pychrono():
        return "chrono"
    if strict_chrono or requested == "chrono":
        raise RuntimeError(
            "PyChrono is not available. Install `pychrono` or rerun with --backend surrogate --no-strict-chrono."
        )
    return "surrogate"


def _times(duration: float, dt: float) -> np.ndarray:
    n = max(2, int(math.ceil(duration / max(dt, 1.0e-6))) + 1)
    return np.linspace(0.0, duration, n)


def _rgb(r: float, g: float, b: float) -> list[float]:
    return [float(r), float(g), float(b)]


def _circle_track(
    *,
    name: str,
    radius: float,
    mass: float,
    color: list[float],
    positions: np.ndarray,
    scalars: Optional[dict[str, list[float]]] = None,
    extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    out = {
        "name": name,
        "kind": "circle",
        "size": [float(radius)],
        "mass": float(mass),
        "color": color,
        "positions": positions.tolist(),
        "scalars": scalars or {},
    }
    if extras:
        out.update(extras)
    return out


def _box_track(
    *,
    name: str,
    size: list[float],
    mass: float,
    color: list[float],
    positions: np.ndarray,
    sizes: Optional[np.ndarray] = None,
    scalars: Optional[dict[str, list[float]]] = None,
    render_shape: Optional[str] = None,
    extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    out = {
        "name": name,
        "kind": "box",
        "size": [float(size[0]), float(size[1]), float(size[2])],
        "mass": float(mass),
        "color": color,
        "positions": positions.tolist(),
        "scalars": scalars or {},
    }
    if sizes is not None:
        arr = np.asarray(sizes, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == len(positions) and arr.shape[1] >= 3:
            out["sizes"] = arr[:, :3].tolist()
    if render_shape:
        out["render_shape"] = str(render_shape)
    if extras:
        out.update(extras)
    return out


def _segment_track(
    *,
    name: str,
    color: list[float],
    points_a: np.ndarray,
    points_b: np.ndarray,
    width: float = 2.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "segment",
        "color": color,
        "width": float(width),
        "points_a": points_a.tolist(),
        "points_b": points_b.tolist(),
    }


def _capsule_track(
    *,
    name: str,
    color: list[float],
    points_a: np.ndarray,
    points_b: np.ndarray,
    radius: float,
    mass: float = 0.0,
    scalars: Optional[dict[str, list[float]]] = None,
    extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    center = 0.5 * (
        np.asarray(points_a, dtype=float) + np.asarray(points_b, dtype=float)
    )
    out = {
        "name": name,
        "kind": "capsule",
        "size": [float(radius)],
        "mass": float(mass),
        "color": color,
        "positions": center.tolist(),
        "points_a": np.asarray(points_a, dtype=float).tolist(),
        "points_b": np.asarray(points_b, dtype=float).tolist(),
        "scalars": scalars or {},
    }
    if extras:
        out.update(extras)
    return out


def _clip_speed(v: np.ndarray, vmax: float = 60.0) -> np.ndarray:
    mag = np.linalg.norm(v)
    if mag <= vmax or mag < 1.0e-9:
        return v
    return v * (vmax / mag)


def _j2_stress_from_eq_strain(
    eq_strain: float,
    *,
    young_mpa: float = 210000.0,
    yield_mpa: float = 280.0,
    hardening_mpa: float = 1400.0,
) -> tuple[float, float]:
    eps = max(0.0, float(eq_strain))
    trial = young_mpa * eps
    if trial <= yield_mpa:
        return float(trial), 0.0
    dep = (trial - yield_mpa) / max(young_mpa + hardening_mpa, 1.0e-9)
    sigma = yield_mpa + hardening_mpa * dep
    return float(sigma), float(dep)


def _polyline_point_at(points: np.ndarray, s: float) -> np.ndarray:
    cum, total = _polyline_arclength(points)
    if total <= 1.0e-9 or points.shape[0] < 2:
        return points[0].copy()
    ss = max(0.0, min(float(s), total))
    idx = int(np.searchsorted(cum, ss, side="right") - 1)
    idx = max(0, min(idx, points.shape[0] - 2))
    seg_len = max(float(cum[idx + 1] - cum[idx]), 1.0e-9)
    alpha = (ss - float(cum[idx])) / seg_len
    return (1.0 - alpha) * points[idx] + alpha * points[idx + 1]


def _maze_wall_boxes(
    guide_path: np.ndarray,
    *,
    corridor_width: float,
    wall_thickness: float,
    wall_height: float,
) -> list[dict[str, list[float]]]:
    boxes: list[dict[str, list[float]]] = []
    offset = 0.5 * float(corridor_width) + 0.5 * float(wall_thickness)
    for i in range(guide_path.shape[0] - 1):
        a = guide_path[i]
        b = guide_path[i + 1]
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        if abs(dx) >= abs(dy):
            length = abs(dx) + float(wall_thickness)
            midx = 0.5 * (float(a[0]) + float(b[0]))
            yref = float(a[1])
            boxes.append(
                {
                    "size": [length, float(wall_thickness), float(wall_height)],
                    "pos": [midx, yref + offset, 0.5 * float(wall_height)],
                }
            )
            boxes.append(
                {
                    "size": [length, float(wall_thickness), float(wall_height)],
                    "pos": [midx, yref - offset, 0.5 * float(wall_height)],
                }
            )
        else:
            length = abs(dy) + float(wall_thickness)
            midy = 0.5 * (float(a[1]) + float(b[1]))
            xref = float(a[0])
            boxes.append(
                {
                    "size": [float(wall_thickness), length, float(wall_height)],
                    "pos": [xref + offset, midy, 0.5 * float(wall_height)],
                }
            )
            boxes.append(
                {
                    "size": [float(wall_thickness), length, float(wall_height)],
                    "pos": [xref - offset, midy, 0.5 * float(wall_height)],
                }
            )

    if guide_path.shape[0] >= 2:
        start = guide_path[0]
        second = guide_path[1]
        end = guide_path[-1]
        prev = guide_path[-2]
        if abs(second[0] - start[0]) >= abs(second[1] - start[1]):
            boxes.append(
                {
                    "size": [
                        float(wall_thickness),
                        float(corridor_width) + 2.0 * float(wall_thickness),
                        float(wall_height),
                    ],
                    "pos": [
                        float(start[0]) - 0.5 * float(wall_thickness),
                        float(start[1]),
                        0.5 * float(wall_height),
                    ],
                }
            )
        else:
            boxes.append(
                {
                    "size": [
                        float(corridor_width) + 2.0 * float(wall_thickness),
                        float(wall_thickness),
                        float(wall_height),
                    ],
                    "pos": [
                        float(start[0]),
                        float(start[1]) - 0.5 * float(wall_thickness),
                        0.5 * float(wall_height),
                    ],
                }
            )
        if abs(end[0] - prev[0]) >= abs(end[1] - prev[1]):
            boxes.append(
                {
                    "size": [
                        float(wall_thickness),
                        float(corridor_width) + 2.0 * float(wall_thickness),
                        float(wall_height),
                    ],
                    "pos": [
                        float(end[0]) + 0.5 * float(wall_thickness),
                        float(end[1]),
                        0.5 * float(wall_height),
                    ],
                }
            )
        else:
            boxes.append(
                {
                    "size": [
                        float(corridor_width) + 2.0 * float(wall_thickness),
                        float(wall_thickness),
                        float(wall_height),
                    ],
                    "pos": [
                        float(end[0]),
                        float(end[1]) + 0.5 * float(wall_thickness),
                        0.5 * float(wall_height),
                    ],
                }
            )
    return boxes


def _maze_schedule(raw: Any, duration: float) -> list[dict[str, Any]]:
    items = (
        raw
        if isinstance(raw, list) and raw
        else [{"time": 0.0, "gravity_xy": [0.0, 0.0]}]
    )
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        gravity_xy = item.get(
            "gravity_xy", [float(item.get("gx", 0.0)), float(item.get("gy", 0.0))]
        )
        if not isinstance(gravity_xy, list) or len(gravity_xy) < 2:
            gravity_xy = [0.0, 0.0]
        rows.append(
            {
                "time": float(item.get("time", 0.0)),
                "gravity_xy": [float(gravity_xy[0]), float(gravity_xy[1])],
            }
        )
    rows.sort(key=lambda x: x["time"])
    if not rows:
        rows = [{"time": 0.0, "gravity_xy": [0.0, 0.0]}]
    first_g = rows[0]["gravity_xy"]
    if rows[0]["time"] <= 1.0e-9 and (
        abs(float(first_g[0])) > 1.0e-9 or abs(float(first_g[1])) > 1.0e-9
    ):
        hold = min(0.5, max(0.25, 0.12 * float(duration)))
        rows[0] = {"time": hold, "gravity_xy": [float(first_g[0]), float(first_g[1])]}
        rows.insert(0, {"time": 0.0, "gravity_xy": [0.0, 0.0]})
        rows.sort(key=lambda x: x["time"])
    if rows[-1]["time"] < float(duration):
        rows.append(
            {"time": float(duration), "gravity_xy": list(rows[-1]["gravity_xy"])}
        )
    return rows


def _maze_gravity_xy(schedule: list[dict[str, Any]], t: float) -> np.ndarray:
    if not schedule:
        return np.asarray([0.0, 0.0], dtype=float)
    if len(schedule) == 1:
        row = schedule[0]
        return np.asarray(
            [float(row["gravity_xy"][0]), float(row["gravity_xy"][1])], dtype=float
        )
    tt = float(t)
    if tt <= float(schedule[0]["time"]):
        row = schedule[0]
        return np.asarray(
            [float(row["gravity_xy"][0]), float(row["gravity_xy"][1])], dtype=float
        )
    for i in range(len(schedule) - 1):
        a = schedule[i]
        b = schedule[i + 1]
        ta = float(a["time"])
        tb = float(b["time"])
        if tt > tb + 1.0e-12:
            continue
        span = max(tb - ta, 1.0e-9)
        alpha = max(0.0, min(1.0, (tt - ta) / span))
        # Smoothstep transition suppresses unrealistic impulse-like tilt jumps.
        s = alpha * alpha * (3.0 - 2.0 * alpha)
        ga = np.asarray(
            [float(a["gravity_xy"][0]), float(a["gravity_xy"][1])], dtype=float
        )
        gb = np.asarray(
            [float(b["gravity_xy"][0]), float(b["gravity_xy"][1])], dtype=float
        )
        return (1.0 - s) * ga + s * gb
    row = schedule[-1]
    return np.asarray(
        [float(row["gravity_xy"][0]), float(row["gravity_xy"][1])], dtype=float
    )


def _circle_intersections_2d(
    c0: np.ndarray, r0: float, c1: np.ndarray, r1: float
) -> list[np.ndarray]:
    d = float(np.linalg.norm(c1 - c0))
    if d <= 1.0e-9 or d > (r0 + r1) or d < abs(r0 - r1):
        return []
    a = (r0 * r0 - r1 * r1 + d * d) / max(2.0 * d, 1.0e-9)
    h_sq = max(r0 * r0 - a * a, 0.0)
    h = math.sqrt(h_sq)
    mid = c0 + a * (c1 - c0) / d
    perp = np.asarray([-(c1 - c0)[1], (c1 - c0)[0]], dtype=float) / d
    return [mid + h * perp, mid - h * perp]


def _fourbar_joint_points(
    mech: dict[str, Any], theta_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(mech.get("anchor_a", [0.0, 0.0, 0.42]), dtype=float)
    d = np.asarray(mech.get("anchor_d", [0.9, 0.0, 0.42]), dtype=float)
    crank = float(mech.get("crank_length", 0.28))
    coupler = float(mech.get("coupler_length", 0.78))
    rocker = float(mech.get("rocker_length", 0.62))
    b = a + np.asarray(
        [crank * math.cos(theta_rad), crank * math.sin(theta_rad), 0.0], dtype=float
    )
    options = _circle_intersections_2d(b[:2], coupler, d[:2], rocker)
    if not options:
        fallback = d[:2] + np.asarray([-rocker, 0.0], dtype=float)
        c_xy = fallback
    else:
        branch = str(mech.get("assembly_branch", "open")).strip().lower()
        c_xy = (
            max(options, key=lambda p: p[1])
            if branch == "open"
            else min(options, key=lambda p: p[1])
        )
    c = np.asarray([float(c_xy[0]), float(c_xy[1]), float(a[2])], dtype=float)
    return a, b, c, d


def _scene_oblique(title: str, ground_value: Optional[float] = 0.0) -> dict[str, Any]:
    return {
        "view_plane": "oblique_xy",
        "ground_value": ground_value,
        "camera_yaw_deg": 40.0,
        "camera_pitch_deg": 22.0,
        "title": title,
    }


_SEDAN_FALLBACK = {
    "profile": "sedan",
    "chassis_size": [4.889501, 1.84912, 1.465576],
    "wheel_offsets_xy": [
        [1.388, 0.7979],
        [1.388, -0.7979],
        [-1.388, 0.7979],
        [-1.388, -0.7979],
    ],
    "wheel_size_xy": [0.67736, 0.205],
    "chassis_mass": 1550.0,
    "chassis_inertia": [222.8, 944.1, 1053.5],
    "corner_mass": 29.843,
    "outline_xy": [
        [-2.4448, -0.9246],
        [-1.4861, -0.9246],
        [2.4448, -0.5592],
        [2.4448, 0.5592],
        [-1.4861, 0.9246],
        [-2.4448, 0.9246],
    ],
}

_BMW_E90_FALLBACK = {
    "profile": "bmw_e90",
    "chassis_size": [4.521385, 2.01413, 1.250201],
    "wheel_offsets_xy": [
        [0.0, 0.750062],
        [0.0, -0.750062],
        [-2.75717, 0.7493],
        [-2.75717, -0.7493],
    ],
    "wheel_size_xy": [0.63645, 0.24],
    "chassis_mass": 1530.0,
    "chassis_inertia": [620.0, 2280.0, 2550.0],
    "corner_mass": 68.085738777,
    "outline_xy": [
        [-4.4772 / 2.0, -2.0112 / 2.0],
        [-1.74, -2.0112 / 2.0],
        [4.51728 / 2.0, -0.70],
        [4.51728 / 2.0, 0.70],
        [-1.74, 2.0112 / 2.0],
        [-4.4772 / 2.0, 2.0112 / 2.0],
    ],
    "outline_xz": [
        [-4.4772 / 2.0, -1.250201 / 2.0],
        [-1.55, -1.250201 / 2.0],
        [1.75, -0.18],
        [4.51728 / 2.0, 0.02],
        [4.51728 / 2.0, 1.250201 / 2.0],
        [-1.9, 1.250201 / 2.0],
        [-4.4772 / 2.0, 0.24],
    ],
}

_VEHICLE_FALLBACKS = {
    "sedan": _SEDAN_FALLBACK,
    "bmw_e90": _BMW_E90_FALLBACK,
    "bmw": _BMW_E90_FALLBACK,
}


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _x_envelope_outline(
    points: list[tuple[float, float]], bins: int = 64
) -> list[list[float]]:
    if not points:
        return []
    uniq = sorted({(float(x), float(y)) for x, y in points})
    if len(uniq) <= 8:
        return [[float(x), float(y)] for x, y in uniq]
    xs = [p[0] for p in uniq]
    x_min = float(min(xs))
    x_max = float(max(xs))
    if abs(x_max - x_min) <= 1.0e-9:
        return _convex_hull_xy(uniq)
    n_bins = max(12, min(int(bins), len(uniq)))
    span = x_max - x_min
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for x, y in uniq:
        alpha = (x - x_min) / span
        idx = min(n_bins - 1, max(0, int(alpha * (n_bins - 1))))
        buckets[idx].append((x, y))
    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for bucket in buckets:
        if not bucket:
            continue
        bx = float(sum(p[0] for p in bucket) / len(bucket))
        by_min = float(min(p[1] for p in bucket))
        by_max = float(max(p[1] for p in bucket))
        lower.append((bx, by_min))
        upper.append((bx, by_max))
    outline = lower + list(reversed(upper))
    cleaned: list[tuple[float, float]] = []
    for p in outline:
        if (
            not cleaned
            or abs(cleaned[-1][0] - p[0]) > 1.0e-9
            or abs(cleaned[-1][1] - p[1]) > 1.0e-9
        ):
            cleaned.append(p)
    if len(cleaned) < 3:
        return _convex_hull_xy(uniq)
    return [[float(x), float(y)] for x, y in cleaned]


def _obj_bounds_outline(path: Path) -> tuple[list[float], list[list[float]]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    xy_points: list[tuple[float, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        xy_points.append((x, y))
    if not xs:
        size = list(_SEDAN_FALLBACK["chassis_size"])
        outline = [list(p) for p in _SEDAN_FALLBACK["outline_xy"]]
        return size, outline
    size = [
        float(max(xs) - min(xs)),
        float(max(ys) - min(ys)),
        float(max(zs) - min(zs)),
    ]
    outline = _x_envelope_outline(xy_points)
    cx = 0.5 * (float(max(xs)) + float(min(xs)))
    cy = 0.5 * (float(max(ys)) + float(min(ys)))
    outline = [[float(p[0] - cx), float(p[1] - cy)] for p in outline]
    return size, outline


def _obj_contact_outline_xy(path: Path) -> list[list[float]]:
    xy_points: list[tuple[float, float]] = []
    xs: list[float] = []
    ys: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        x = float(parts[1])
        y = float(parts[2])
        xs.append(x)
        ys.append(y)
        xy_points.append((x, y))
    if not xs:
        return [list(p) for p in _SEDAN_FALLBACK["outline_xy"]]
    outline = _convex_hull_xy(xy_points)
    cx = 0.5 * (float(max(xs)) + float(min(xs)))
    cy = 0.5 * (float(max(ys)) + float(min(ys)))
    return [[float(p[0] - cx), float(p[1] - cy)] for p in outline]


def _obj_side_outline(path: Path) -> list[list[float]]:
    xs: list[float] = []
    zs: list[float] = []
    xz_points: list[tuple[float, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        x = float(parts[1])
        z = float(parts[3])
        xs.append(x)
        zs.append(z)
        xz_points.append((x, z))
    if not xs:
        return [
            list(p)
            for p in _BMW_E90_FALLBACK.get(
                "outline_xz", [[-2.0, -0.6], [2.0, -0.6], [2.0, 0.6], [-2.0, 0.6]]
            )
        ]
    outline = _x_envelope_outline(xz_points)
    cx = 0.5 * (float(max(xs)) + float(min(xs)))
    cz = 0.5 * (float(max(zs)) + float(min(zs)))
    return [[float(p[0] - cx), float(p[1] - cz)] for p in outline]


def _obj_contact_outline_xz(path: Path) -> list[list[float]]:
    xz_points: list[tuple[float, float]] = []
    xs: list[float] = []
    zs: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        x = float(parts[1])
        z = float(parts[3])
        xs.append(x)
        zs.append(z)
        xz_points.append((x, z))
    if not xs:
        return [
            list(p)
            for p in _BMW_E90_FALLBACK.get(
                "outline_xz", [[-2.0, -0.6], [2.0, -0.6], [2.0, 0.6], [-2.0, 0.6]]
            )
        ]
    outline = _convex_hull_xy(xz_points)
    cx = 0.5 * (float(max(xs)) + float(min(xs)))
    cz = 0.5 * (float(max(zs)) + float(min(zs)))
    return [[float(p[0] - cx), float(p[1] - cz)] for p in outline]


def _extract_number(text: str, key: str, default: float) -> float:
    pattern = rf'"{re.escape(key)}"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else float(default)


def _vehicle_profile_key(profile: str) -> str:
    key = str(profile or "sedan").strip().lower()
    if key in {"bmw", "bmw_e90", "bmw-e90"}:
        return "bmw_e90"
    return "sedan"


def _vehicle_mesh_paths(profile: str) -> dict[str, str]:
    key = _vehicle_profile_key(profile)
    if key == "bmw_e90":
        return {
            "chassis_vis": "bmw/chassis/bmw_e90_chassis.obj",
            "chassis_col": "bmw/chassis/bmw_e90_chassis_col.obj",
            "tire_front": "bmw/tire/bmw_e90_front_Tire.obj",
            "tire_rear": "bmw/tire/bmw_e90_rear_Tire.obj",
        }
    return {
        "chassis_vis": "sedan/sedan_chassis_vis.obj",
        "chassis_col": "sedan/sedan_chassis_col.obj",
        "tire_front": "sedan/sedan_tire.obj",
        "tire_rear": "sedan/sedan_tire.obj",
    }


def _load_vehicle_specs(
    profile: str = "sedan", data_root: Optional[Path] = None
) -> dict[str, Any]:
    key = _vehicle_profile_key(profile)
    spec = json.loads(json.dumps(_VEHICLE_FALLBACKS[key]))
    chassis_inertia = np.asarray(spec["chassis_inertia"], dtype=float)
    corner_mass = float(spec["corner_mass"])

    if data_root is not None:
        if key == "sedan":
            chassis_json = _read_json(
                data_root / "sedan" / "chassis" / "Sedan_Chassis.json"
            )
            wheel_json = _read_json(data_root / "sedan" / "wheel" / "Sedan_Wheel.json")
            vehicle_json = _read_json(
                data_root / "sedan" / "vehicle" / "Sedan_Vehicle.json"
            )
            suspension_json = _read_json(
                data_root / "sedan" / "suspension" / "Sedan_DoubleWishbone.json"
            )
            tire_text = (
                (data_root / "sedan" / "tire" / "Sedan_TMeasyTire.json").read_text(
                    errors="ignore"
                )
                if (data_root / "sedan" / "tire" / "Sedan_TMeasyTire.json").exists()
                else ""
            )

            chassis_comp = (
                ((chassis_json.get("Components") or [{}])[0])
                if isinstance(chassis_json.get("Components"), list)
                else {}
            )
            spec["chassis_mass"] = float(chassis_comp.get("Mass", spec["chassis_mass"]))
            spec["chassis_inertia"] = [
                float(x)
                for x in chassis_comp.get("Moments of Inertia", spec["chassis_inertia"])
            ]
            chassis_inertia = np.asarray(spec["chassis_inertia"], dtype=float)

            wheel_mass = float(wheel_json.get("Mass", 13.2))
            spindle = (
                suspension_json.get("Spindle")
                if isinstance(suspension_json.get("Spindle"), dict)
                else {}
            )
            upright = (
                suspension_json.get("Upright")
                if isinstance(suspension_json.get("Upright"), dict)
                else {}
            )
            upper = (
                suspension_json.get("Upper Control Arm")
                if isinstance(suspension_json.get("Upper Control Arm"), dict)
                else {}
            )
            lower = (
                suspension_json.get("Lower Control Arm")
                if isinstance(suspension_json.get("Lower Control Arm"), dict)
                else {}
            )
            tire_mass = _extract_number(tire_text, "Mass [kg]", 11.5)
            corner_mass = (
                wheel_mass
                + tire_mass
                + float(spindle.get("Mass", 0.0))
                + float(upright.get("Mass", 0.0))
                + float(upper.get("Mass", 0.0))
                + float(lower.get("Mass", 0.0))
            )
            spec["corner_mass"] = float(corner_mass)

            axle_locs: list[float] = []
            for axle in vehicle_json.get("Axles", []):
                if not isinstance(axle, dict):
                    continue
                loc = axle.get("Suspension Location")
                if isinstance(loc, list) and len(loc) >= 1:
                    axle_locs.append(float(loc[0]))
            if len(axle_locs) >= 2:
                y_half = float(spindle.get("COM", [0.0, 0.7979, 0.0])[1])
                spec["wheel_offsets_xy"] = [
                    [float(axle_locs[0]), y_half],
                    [float(axle_locs[0]), -y_half],
                    [float(axle_locs[1]), y_half],
                    [float(axle_locs[1]), -y_half],
                ]

        mesh_paths = _vehicle_mesh_paths(key)
        vis_mesh = data_root / mesh_paths["chassis_vis"]
        front_tire_mesh = data_root / mesh_paths["tire_front"]
        rear_tire_mesh = data_root / mesh_paths["tire_rear"]
        if vis_mesh.exists():
            spec["chassis_size"], spec["outline_xy"] = _obj_bounds_outline(vis_mesh)
            spec["outline_xz"] = _obj_side_outline(vis_mesh)
            spec["contact_outline_xy"] = _obj_contact_outline_xy(vis_mesh)
            spec["contact_outline_xz"] = _obj_contact_outline_xz(vis_mesh)
        if front_tire_mesh.exists() and rear_tire_mesh.exists():
            front_size, _ = _obj_bounds_outline(front_tire_mesh)
            rear_size, _ = _obj_bounds_outline(rear_tire_mesh)
            avg_len = 0.5 * (
                max(front_size[0], front_size[2]) + max(rear_size[0], rear_size[2])
            )
            avg_width = 0.5 * (front_size[1] + rear_size[1])
            spec["wheel_size_xy"] = [float(avg_len), float(avg_width)]

    inertia = chassis_inertia.copy()
    axle_x = sorted({round(float(item[0]), 6) for item in spec["wheel_offsets_xy"]})
    if axle_x:
        x_center = 0.5 * (axle_x[0] + axle_x[-1])
        spec["wheel_offsets_xy"] = [
            [float(ox - x_center), float(oy)] for ox, oy in spec["wheel_offsets_xy"]
        ]
    for ox, oy in spec["wheel_offsets_xy"]:
        inertia[0] += corner_mass * (float(oy) ** 2)
        inertia[1] += corner_mass * (float(ox) ** 2)
        inertia[2] += corner_mass * (float(ox) ** 2 + float(oy) ** 2)
    spec["mass"] = float(spec["chassis_mass"] + 4.0 * corner_mass)
    spec["inertia"] = [float(x) for x in inertia.tolist()]
    return spec


def _vehicle_track(
    *,
    name: str,
    spec: dict[str, Any],
    color: list[float],
    positions: np.ndarray,
    yaw_deg: list[float],
    visual_offset: Optional[list[float]] = None,
    scalars_extra: Optional[dict[str, list[float]]] = None,
    color_metric: Optional[str] = None,
) -> dict[str, Any]:
    offset = np.asarray(
        visual_offset
        if isinstance(visual_offset, list) and len(visual_offset) >= 3
        else [0.0, 0.0, 0.0],
        dtype=float,
    )
    scalars_payload: dict[str, list[float]] = {"yaw_deg": [float(x) for x in yaw_deg]}
    if isinstance(scalars_extra, dict):
        for key, series in scalars_extra.items():
            if isinstance(series, list):
                scalars_payload[str(key)] = [float(v) for v in series]
    out = _box_track(
        name=name,
        size=list(spec["chassis_size"]),
        mass=float(spec["mass"]),
        color=color,
        positions=positions,
        scalars=scalars_payload,
        render_shape="vehicle_mesh",
        extras={
            "outline_xy": [
                [float(p[0] + offset[0]), float(p[1] + offset[1])]
                for p in spec["outline_xy"]
            ],
            "outline_xz": [
                [float(p[0] + offset[0]), float(p[1] + offset[2])]
                for p in spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"])
            ],
            "contact_outline_xy": [
                [float(p[0] + offset[0]), float(p[1] + offset[1])]
                for p in spec.get("contact_outline_xy", spec["outline_xy"])
            ],
            "contact_outline_xz": [
                [float(p[0] + offset[0]), float(p[1] + offset[2])]
                for p in spec.get(
                    "contact_outline_xz",
                    spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"]),
                )
            ],
            "wheel_offsets_xy": [
                [float(p[0] + offset[0]), float(p[1] + offset[1])]
                for p in spec["wheel_offsets_xy"]
            ],
            "wheel_size_xy": [
                float(spec["wheel_size_xy"][0]),
                float(spec["wheel_size_xy"][1]),
            ],
            "vehicle_profile": str(spec.get("profile", "sedan")),
        },
    )
    if color_metric:
        out["color_metric"] = str(color_metric)
    return out


def _vehicle_profile_from_case(case: dict[str, Any]) -> str:
    mech = _mech_cfg(case)
    return _vehicle_profile_key(str(mech.get("asset_profile", "sedan")))


def _vehicle_solver_variant(case: dict[str, Any]) -> str:
    mech = _mech_cfg(case)
    raw = (
        str(mech.get("vehicle_model") or mech.get("solver_variant") or "official_mesh")
        .strip()
        .lower()
    )
    if raw in {"bmw_e90_full_vehicle", "full_vehicle", "full_vehicle_official"}:
        return "full_vehicle"
    return "official_mesh"


def _vehicle_data_root(veh: Any, profile: str) -> Path:
    key = _vehicle_profile_key(profile)
    if key == "bmw_e90":
        return (
            Path(veh.GetVehicleDataFile("bmw/chassis/bmw_e90_chassis.obj"))
            .resolve()
            .parents[2]
        )
    return Path(veh.GetVehicleDataFile("sedan/Sedan.json")).resolve().parent.parent


def _vehicle_object_template(
    name: str,
    spec: dict[str, Any],
    color: list[float],
    profile: str,
    visual_offset: Optional[list[float]] = None,
    wheel_width_scale: float = 1.0,
) -> dict[str, Any]:
    offset = np.asarray(
        visual_offset
        if isinstance(visual_offset, list) and len(visual_offset) >= 3
        else [0.0, 0.0, 0.0],
        dtype=float,
    )
    return {
        "name": name,
        "kind": "box",
        "size": list(spec["chassis_size"]),
        "mass": float(spec["mass"]),
        "color": color,
        "positions": [],
        "scalars": {"yaw_deg": []},
        "render_shape": "vehicle_mesh",
        "outline_xy": [
            [float(p[0] + offset[0]), float(p[1] + offset[1])]
            for p in spec["outline_xy"]
        ],
        "outline_xz": [
            [float(p[0] + offset[0]), float(p[1] + offset[2])]
            for p in spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"])
        ],
        "contact_outline_xy": [
            [float(p[0] + offset[0]), float(p[1] + offset[1])]
            for p in spec.get("contact_outline_xy", spec["outline_xy"])
        ],
        "contact_outline_xz": [
            [float(p[0] + offset[0]), float(p[1] + offset[2])]
            for p in spec.get(
                "contact_outline_xz",
                spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"]),
            )
        ],
        "wheel_offsets_xy": [
            [float(p[0] + offset[0]), float(p[1] + offset[1])]
            for p in spec["wheel_offsets_xy"]
        ],
        "wheel_size_xy": [
            float(spec["wheel_size_xy"][0]),
            float(spec["wheel_size_xy"][1]),
        ],
        "wheel_width_scale": max(0.8, min(2.6, float(wheel_width_scale))),
        "vehicle_profile": profile,
    }


def _chrono_vehicle_visual_offset(vehicle: Any) -> list[float]:
    body = vehicle.GetChassisBody()
    ref = vehicle.GetPos()
    dx = float(body.GetPos().x - ref.x)
    dy = float(body.GetPos().y - ref.y)
    dz = float(body.GetPos().z - ref.z)
    yaw = math.radians(_chrono_yaw_deg(vehicle))
    c = math.cos(yaw)
    s = math.sin(yaw)
    # Convert the chassis-body COM offset from world space back into the
    # vehicle reference frame before attaching local render silhouettes.
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    # The BMW chassis COM offset should be predominantly longitudinal.
    # Suppress lateral drift from frame-convention differences so both
    # vehicles share the same silhouette scale/orientation in renders.
    if abs(local_y) > 1.0e-3:
        local_y = 0.0
    return [float(local_x), float(local_y), float(dz)]


def _chrono_vehicle_pose(vehicle: Any) -> tuple[list[float], float]:
    body = vehicle.GetChassisBody() if hasattr(vehicle, "GetChassisBody") else vehicle
    pos = body.GetPos()
    return [float(pos.x), float(pos.y), float(pos.z)], _chrono_yaw_deg(body)


def _chrono_comp(vec: Any, name: str) -> float:
    value = getattr(vec, name, 0.0)
    if callable(value):
        value = value()
    try:
        return float(value)
    except Exception:
        return 0.0


def _chrono_vec_np(vec: Any) -> np.ndarray:
    return np.asarray(
        [_chrono_comp(vec, "x"), _chrono_comp(vec, "y"), _chrono_comp(vec, "z")],
        dtype=float,
    )


def _chrono_spin_rate_about_local_axis(
    body: Any, chrono: Any, local_axis: list[float]
) -> float:
    p0 = _chrono_point_local_to_world(body, chrono, [0.0, 0.0, 0.0])
    p1 = _chrono_point_local_to_world(body, chrono, local_axis)
    axis = np.asarray(p1 - p0, dtype=float)
    nrm = float(np.linalg.norm(axis))
    if nrm <= 1.0e-9:
        return 0.0
    axis /= nrm
    omega = _chrono_vec_np(body.GetAngVelParent())
    return float(np.dot(omega, axis))


def _driver_sequence_entries(
    veh: Any, sequence: list[dict[str, Any]], duration: float
) -> Any:
    rows: list[dict[str, float]] = []
    raw_items = (
        sequence
        if isinstance(sequence, list) and sequence
        else [{"time": 0.0, "steering": 0.0, "throttle": 0.0, "braking": 0.0}]
    )
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "time": float(item.get("time", 0.0)),
                "steering": float(item.get("steering", 0.0)),
                "throttle": float(item.get("throttle", 0.0)),
                "braking": float(item.get("braking", 0.0)),
                "clutch": float(item.get("clutch", 0.0)),
            }
        )
    if not rows:
        rows.append(
            {
                "time": 0.0,
                "steering": 0.0,
                "throttle": 0.0,
                "braking": 0.0,
                "clutch": 0.0,
            }
        )
    if float(rows[-1]["time"]) < float(duration):
        rows.append({**rows[-1], "time": float(duration)})
    return veh.vector_Entry(
        [
            veh.DataDriverEntry(
                r["time"], r["steering"], r["throttle"], r["braking"], r["clutch"]
            )
            for r in rows
        ]
    )


def _simulate_surrogate_pendulum(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_pendulum import (
        _simulate_surrogate_pendulum as implementation,
    )

    return implementation(case)


def _simulate_surrogate_double_pendulum(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_double_pendulum import (
        _simulate_surrogate_double_pendulum as implementation,
    )

    return implementation(case)


def _simulate_surrogate_triple_pendulum(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_triple_pendulum import (
        _simulate_surrogate_triple_pendulum as implementation,
    )

    return implementation(case)


def _simulate_surrogate_rigid_collision(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_rigid_collision import (
        _simulate_surrogate_rigid_collision as implementation,
    )

    return implementation(case)


def _simulate_surrogate_stack(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_stack import _simulate_surrogate_stack as implementation

    return implementation(case)


def _simulate_surrogate_mass_spring_damper(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_mass_spring_damper import (
        _simulate_surrogate_mass_spring_damper as implementation,
    )

    return implementation(case)


def _simulate_surrogate_rolling(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_rolling import _simulate_surrogate_rolling as implementation

    return implementation(case)


def _simulate_surrogate_vehicle_collision(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_vehicle_collision import (
        _simulate_surrogate_vehicle_collision as implementation,
    )

    return implementation(case)


def _simulate_surrogate_vehicle_lane_change(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_vehicle_lane_change import (
        _simulate_surrogate_vehicle_lane_change as implementation,
    )

    return implementation(case)


def _simulate_surrogate_block_tower(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_block_tower import (
        _simulate_surrogate_block_tower as implementation,
    )

    return implementation(case)


def _simulate_surrogate_ramp_jump(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_ramp_jump import (
        _simulate_surrogate_ramp_jump as implementation,
    )

    return implementation(case)


def _simulate_surrogate_sphere_drop(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_sphere_drop import (
        _simulate_surrogate_sphere_drop as implementation,
    )

    return implementation(case)


def _simulate_surrogate_gyroscope(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_gyroscope import (
        _simulate_surrogate_gyroscope as implementation,
    )

    return implementation(case)


def _simulate_surrogate_unbalanced_rotor(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_unbalanced_rotor import (
        _simulate_surrogate_unbalanced_rotor as implementation,
    )

    return implementation(case)


def _simulate_surrogate_articulated_chain(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_articulated_chain import (
        _simulate_surrogate_articulated_chain as implementation,
    )

    return implementation(case)


def _simulate_surrogate_ball_maze(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_ball_maze import (
        _simulate_surrogate_ball_maze as implementation,
    )

    return implementation(case)


def _simulate_surrogate_slider_crank(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_slider_crank import (
        _simulate_surrogate_slider_crank as implementation,
    )

    return implementation(case)


def _simulate_surrogate_four_bar(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_four_bar import (
        _simulate_surrogate_four_bar as implementation,
    )

    return implementation(case)


def _simulate_surrogate_vehicle_obstacle_crossing(
    case: dict[str, Any]
) -> dict[str, Any]:
    from families.surrogate_vehicle_obstacle_crossing import (
        _simulate_surrogate_vehicle_obstacle_crossing as implementation,
    )

    return implementation(case)


def _simulate_surrogate_granular_collapse(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_granular_collapse import (
        _simulate_surrogate_granular_collapse as implementation,
    )

    return implementation(case)


def _simulate_surrogate_ball_chute_pile(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_ball_chute_pile import (
        _simulate_surrogate_ball_chute_pile as implementation,
    )

    return implementation(case)


def _integrate_angular_speed(
    times: np.ndarray, omega_series: np.ndarray, theta0: float
) -> np.ndarray:
    theta = np.zeros_like(omega_series, dtype=float)
    if len(theta) == 0:
        return theta
    theta[0] = float(theta0)
    if len(theta) == 1:
        return theta
    dt = np.diff(times)
    avg = 0.5 * (omega_series[:-1] + omega_series[1:])
    theta[1:] = float(theta0) + np.cumsum(avg * dt)
    return theta


def _gear_motor_profile(
    mech: dict[str, Any], times: np.ndarray, default_speed: float
) -> np.ndarray:
    base = np.full((len(times),), float(default_speed), dtype=float)
    profile = (
        mech.get("motor_profile") if isinstance(mech.get("motor_profile"), dict) else {}
    )
    profile_type = str(profile.get("type", "constant")).strip().lower()
    points_raw = profile.get("points")
    if (
        profile_type in {"", "constant"}
        or not isinstance(points_raw, list)
        or len(points_raw) < 2
    ):
        return base

    points: list[tuple[float, float]] = []
    for item in points_raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except Exception:
            continue
    if len(points) < 2:
        return base

    points.sort(key=lambda x: x[0])
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    xs[0] = min(xs[0], float(times[0]))
    xs[-1] = max(xs[-1], float(times[-1]))

    if profile_type in {"step", "piecewise_step"}:
        out = np.zeros_like(base)
        cursor = 0
        for i, t in enumerate(times):
            while cursor + 1 < len(xs) and float(t) >= float(xs[cursor + 1]):
                cursor += 1
            out[i] = ys[cursor]
        return out

    return np.interp(times, xs, ys, left=ys[0], right=ys[-1]).astype(float)


def _simulate_surrogate_gear_pair(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_gear_pair import (
        _simulate_surrogate_gear_pair as implementation,
    )

    return implementation(case)


def _add_csv_export(
    data: dict[str, Any], file_name: str, header: list[str], rows: list[list[float]]
) -> None:
    exports = data.setdefault("exports", [])
    if not isinstance(exports, list):
        exports = []
        data["exports"] = exports
    exports.append(
        {"file_name": str(file_name), "header": [str(h) for h in header], "rows": rows}
    )


def _cantilever_shape(s: np.ndarray) -> np.ndarray:
    ss = np.clip(s, 0.0, 1.0)
    return ss * ss * (3.0 - 2.0 * ss)


def _simulate_surrogate_cantilever_beam_transient(
    case: dict[str, Any]
) -> dict[str, Any]:
    from families.surrogate_cantilever_beam_transient import (
        _simulate_surrogate_cantilever_beam_transient as implementation,
    )

    return implementation(case)


def _simulate_surrogate_modal_block(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_modal_block import (
        _simulate_surrogate_modal_block as implementation,
    )

    return implementation(case)


def _simulate_surrogate_harmonic_frf(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_harmonic_frf import (
        _simulate_surrogate_harmonic_frf as implementation,
    )

    return implementation(case)


def _simulate_surrogate_stress_wave(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_stress_wave import (
        _simulate_surrogate_stress_wave as implementation,
    )

    return implementation(case)


def _simulate_surrogate_drop_weight_impact(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_drop_weight_impact import (
        _simulate_surrogate_drop_weight_impact as implementation,
    )

    return implementation(case)


def _simulate_surrogate_j2_plastic_beam(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_j2_plastic_beam import (
        _simulate_surrogate_j2_plastic_beam as implementation,
    )

    return implementation(case)


def _simulate_surrogate_lemaitre_damage(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_lemaitre_damage import (
        _simulate_surrogate_lemaitre_damage as implementation,
    )

    return implementation(case)


def _simulate_surrogate_rubber_block(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_rubber_block import (
        _simulate_surrogate_rubber_block as implementation,
    )

    return implementation(case)


def _simulate_surrogate_flexible_crank_slider(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_flexible_crank_slider import (
        _simulate_surrogate_flexible_crank_slider as implementation,
    )

    return implementation(case)


def _simulate_surrogate_reissner_shell_modal(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_reissner_shell_modal import (
        _simulate_surrogate_reissner_shell_modal as implementation,
    )

    return implementation(case)


def _simulate_surrogate_structural_demo(case: dict[str, Any]) -> dict[str, Any]:
    from families.surrogate_structural_demo import (
        _simulate_surrogate_structural_demo as implementation,
    )

    return implementation(case)


def _simulate_surrogate(case: dict[str, Any]) -> dict[str, Any]:
    family = _family(case)
    if family in {
        "cantileverBeamTransient",
        "harmonicResponseFRF",
        "stressWavePropagation",
        "dropWeightImpactSMC",
        "j2PlasticBeam",
        "lemaitreDamageBar",
        "rubberBlockCompression",
        "flexibleCrankSlider",
        "reissnerShellModal",
    }:
        return _simulate_surrogate_structural_demo(case)
    if family == "pendulum":
        return _simulate_surrogate_pendulum(case)
    if family == "doublePendulum":
        return _simulate_surrogate_double_pendulum(case)
    if family == "triplePendulum":
        return _simulate_surrogate_triple_pendulum(case)
    if family == "rigidCollision":
        return _simulate_surrogate_rigid_collision(case)
    if family == "stackStability":
        return _simulate_surrogate_stack(case)
    if family == "massSpringDamper":
        return _simulate_surrogate_mass_spring_damper(case)
    if family == "rollingVsSliding":
        return _simulate_surrogate_rolling(case)
    if family == "rollingSlidingTransition":
        return _simulate_surrogate_rolling(case)
    if family == "vehicleCollision3D":
        return _simulate_surrogate_vehicle_collision(case)
    if family == "vehicleLaneChange3D":
        return _simulate_surrogate_vehicle_lane_change(case)
    if family == "vehicleObstacleCrossing3D":
        return _simulate_surrogate_vehicle_obstacle_crossing(case)
    if family == "blockTower3D":
        return _simulate_surrogate_block_tower(case)
    if family == "rampJump3D":
        return _simulate_surrogate_ramp_jump(case)
    if family == "sphereDrop3D":
        return _simulate_surrogate_sphere_drop(case)
    if family == "gyroscopeTop3D":
        return _simulate_surrogate_gyroscope(case)
    if family == "gyroscopeRotor3D":
        return _simulate_surrogate_unbalanced_rotor(case)
    if family == "unbalancedRotor3D":
        return _simulate_surrogate_unbalanced_rotor(case)
    if family == "articulatedChain":
        return _simulate_surrogate_articulated_chain(case)
    if family == "rollingBallMaze3D":
        return _simulate_surrogate_ball_maze(case)
    if family == "sliderCrank3D":
        raise ValueError("Family sliderCrank3D has been removed.")
    if family == "fourBarLinkage3D":
        raise ValueError("Family fourBarLinkage3D has been removed.")
    if family == "granularCollapse3D":
        return _simulate_surrogate_granular_collapse(case)
    if family == "gearPair3D":
        return _simulate_surrogate_gear_pair(case)
    if family == "ballChutePile3D":
        return _simulate_surrogate_ball_chute_pile(case)
    raise ValueError(f"Unsupported surrogate family: {family}")


def _vehicle_world_outline_xy(
    local_outline_xy: np.ndarray, pos_xy: np.ndarray, yaw_deg: float
) -> np.ndarray:
    if local_outline_xy.ndim != 2 or local_outline_xy.shape[0] < 3:
        return np.zeros((0, 2), dtype=float)
    return _rotate_xy(local_outline_xy, yaw_deg) + np.asarray(pos_xy, dtype=float)


def _scene_project(
    point: np.ndarray, view_plane: str, scene: Optional[dict[str, Any]] = None
) -> tuple[float, float]:
    if view_plane == "xz":
        return float(point[0]), float(point[2])
    if view_plane == "oblique_xy":
        yaw = math.radians(float((scene or {}).get("camera_yaw_deg", 40.0)))
        pitch = math.radians(float((scene or {}).get("camera_pitch_deg", 22.0)))
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        xr = x * math.cos(yaw) - y * math.sin(yaw)
        yr = x * math.sin(yaw) + y * math.cos(yaw)
        return xr, yr * math.sin(pitch) + z * math.cos(pitch)
    return float(point[0]), float(point[1])


def _scene_bounds(
    objects: list[dict[str, Any]],
    view_plane: str,
    scene: Optional[dict[str, Any]] = None,
) -> dict[str, list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    z_min_clip: Optional[float] = None
    if isinstance(scene, dict) and isinstance(
        scene.get("bounds_min_world_z"), (int, float)
    ):
        z_min_clip = float(scene["bounds_min_world_z"])
    for obj in objects:
        visible_series: Optional[list[float]] = None
        scalars = obj.get("scalars")
        if isinstance(scalars, dict) and isinstance(scalars.get("visible"), list):
            visible_series = scalars.get("visible")
        if obj.get("kind") in {"segment", "capsule"}:
            pa = np.asarray(obj.get("points_a", []), dtype=float)
            pb = np.asarray(obj.get("points_b", []), dtype=float)
            if len(pa):
                for p in pa:
                    if (
                        z_min_clip is not None
                        and len(p) >= 3
                        and float(p[2]) < z_min_clip
                    ):
                        continue
                    xp, yp = _scene_project(p, view_plane, scene)
                    xs.append(xp)
                    ys.append(yp)
            if len(pb):
                for p in pb:
                    if (
                        z_min_clip is not None
                        and len(p) >= 3
                        and float(p[2]) < z_min_clip
                    ):
                        continue
                    xp, yp = _scene_project(p, view_plane, scene)
                    xs.append(xp)
                    ys.append(yp)
            continue
        pos = np.asarray(obj.get("positions", []), dtype=float)
        if len(pos):
            for idx, p in enumerate(pos):
                if visible_series is not None and idx < len(visible_series):
                    try:
                        if float(visible_series[idx]) <= 0.5:
                            continue
                    except Exception:
                        pass
                if z_min_clip is not None and len(p) >= 3 and float(p[2]) < z_min_clip:
                    continue
                xp, yp = _scene_project(p, view_plane, scene)
                xs.append(xp)
                ys.append(yp)
    if not xs:
        return {"xlim": [-1.0, 1.0], "ylim": [-1.0, 1.0]}
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = max(1.0, xmax - xmin)
    dy = max(1.0, ymax - ymin)
    margin_frac = 0.16
    if isinstance(scene, dict) and isinstance(
        scene.get("bounds_margin_fraction"), (int, float)
    ):
        margin_frac = max(0.02, min(0.35, float(scene["bounds_margin_fraction"])))
    mx = margin_frac * dx
    my = margin_frac * dy
    return {"xlim": [xmin - mx, xmax + mx], "ylim": [ymin - my, ymax + my]}


def _chrono_vec(chrono: Any, arr: Union[list[float], np.ndarray]) -> Any:
    return chrono.ChVector3d(float(arr[0]), float(arr[1]), float(arr[2]))


def _chrono_quat_z(chrono: Any, angle_rad: float) -> Any:
    if hasattr(chrono, "QuatFromAngleZ"):
        return chrono.QuatFromAngleZ(float(angle_rad))
    if hasattr(chrono, "Q_from_AngZ"):
        return chrono.Q_from_AngZ(float(angle_rad))
    half = 0.5 * float(angle_rad)
    return chrono.ChQuaterniond(math.cos(half), 0.0, 0.0, math.sin(half))


def _chrono_yaw_deg(body: Any) -> float:
    rot = body.GetRot()
    e0 = float(rot.e0)
    e1 = float(rot.e1)
    e2 = float(rot.e2)
    e3 = float(rot.e3)
    siny_cosp = 2.0 * (e0 * e3 + e1 * e2)
    cosy_cosp = 1.0 - 2.0 * (e2 * e2 + e3 * e3)
    return float(math.degrees(math.atan2(siny_cosp, cosy_cosp)))


def _chrono_quaternion(body: Any) -> list[float]:
    rot = body.GetRot()
    return [float(rot.e0), float(rot.e1), float(rot.e2), float(rot.e3)]


def _chrono_point_local_to_world(
    body: Any, chrono: Any, point: Union[list[float], np.ndarray]
) -> np.ndarray:
    local = _chrono_vec(chrono, point)
    if hasattr(body, "TransformPointLocalToParent"):
        out = body.TransformPointLocalToParent(local)
    elif hasattr(body, "Point_Body2World"):
        out = body.Point_Body2World(local)
    else:
        pos = body.GetPos()
        return np.asarray(
            [float(pos.x + point[0]), float(pos.y + point[1]), float(pos.z + point[2])],
            dtype=float,
        )
    return np.asarray([float(out.x), float(out.y), float(out.z)], dtype=float)


def _chrono_joint_frame_y(chrono: Any, pos: Union[list[float], np.ndarray]) -> Any:
    orient = getattr(chrono, "Q_ROTATE_Z_TO_Y", None)
    if orient is None:
        orient = chrono.QuatFromAngleX(-0.5 * math.pi)
    return chrono.ChFramed(_chrono_vec(chrono, pos), orient)


def _chrono_material(
    chrono: Any, *, friction: float, restitution: float, rolling_friction: float = 0.0
) -> Any:
    mat = chrono.ChContactMaterialNSC()
    if hasattr(mat, "SetFriction"):
        mat.SetFriction(float(friction))
    if hasattr(mat, "SetRestitution"):
        mat.SetRestitution(float(restitution))
    if hasattr(mat, "SetRollingFriction"):
        mat.SetRollingFriction(float(rolling_friction))
    return mat


def _chrono_density_for_box(mass: float, size: list[float]) -> float:
    vol = max(float(size[0]) * float(size[1]) * float(size[2]), 1.0e-6)
    return float(mass) / vol


def _chrono_density_for_sphere(mass: float, radius: float) -> float:
    vol = max((4.0 / 3.0) * math.pi * radius**3, 1.0e-6)
    return float(mass) / vol


def _chrono_density_for_cylinder(mass: float, radius: float, length: float) -> float:
    vol = max(math.pi * radius * radius * max(float(length), 1.0e-6), 1.0e-6)
    return float(mass) / vol


def _chrono_material_smc(
    chrono: Any, *, friction: float, restitution: float, rolling_friction: float = 0.0
) -> Any:
    mat = chrono.ChContactMaterialSMC()
    if hasattr(mat, "SetFriction"):
        mat.SetFriction(float(friction))
    if hasattr(mat, "SetRestitution"):
        mat.SetRestitution(float(restitution))
    if hasattr(mat, "SetRollingFriction"):
        mat.SetRollingFriction(float(rolling_friction))
    if hasattr(mat, "SetYoungModulus"):
        mat.SetYoungModulus(2.0e7)
    return mat


def _chrono_body_box_between(
    system: Any,
    chrono: Any,
    *,
    point_a: Union[list[float], np.ndarray],
    point_b: Union[list[float], np.ndarray],
    width: float,
    thickness: float,
    mass: float,
    material: Any,
    collide: bool = False,
) -> Any:
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    delta = b - a
    length = max(float(np.linalg.norm(delta)), 1.0e-4)
    body = chrono.ChBodyEasyBox(
        length,
        max(float(width), 1.0e-4),
        max(float(thickness), 1.0e-4),
        _chrono_density_for_box(
            mass, [length, max(float(width), 1.0e-4), max(float(thickness), 1.0e-4)]
        ),
        True,
        collide,
        material,
    )
    body.SetPos(_chrono_vec(chrono, 0.5 * (a + b)))
    body.SetRot(_chrono_quat_z(chrono, math.atan2(float(delta[1]), float(delta[0]))))
    system.Add(body)
    return body


def _chrono_add_ground(system: Any, chrono: Any, case: dict[str, Any]) -> None:
    env = _env_cfg(case)
    ground = env.get("ground") if isinstance(env.get("ground"), dict) else {}
    if not ground or not bool(ground.get("enabled", True)):
        return
    size = ground.get("size", [40.0, 10.0])
    if not isinstance(size, list) or len(size) < 2:
        size = [40.0, 10.0]
    thickness = float(ground.get("thickness", 0.25))
    mat = _chrono_material(
        chrono,
        friction=float(ground.get("friction", 0.4)),
        restitution=float(ground.get("restitution", 0.05)),
    )
    body = chrono.ChBodyEasyBox(
        float(size[0]), float(size[1]), thickness, 1000.0, True, True, mat
    )
    body.SetPos(_chrono_vec(chrono, [0.0, 0.0, -0.5 * thickness]))
    body.SetFixed(True)
    system.Add(body)


def _chrono_num_contacts(system: Any) -> int:
    try:
        if hasattr(system, "GetNumContacts"):
            value = system.GetNumContacts()
            return int(value() if callable(value) else value)
        if hasattr(system, "GetContactContainer"):
            cc = system.GetContactContainer()
            if cc is not None:
                for name in ("GetNumContacts",):
                    if hasattr(cc, name):
                        value = getattr(cc, name)()
                        return int(value() if callable(value) else value)
    except Exception:
        pass
    return 0


def _chrono_sampling_loop(
    *,
    case: dict[str, Any],
    system: Any,
    trackers: list[dict[str, Any]],
    scene: dict[str, Any],
    step_hook: Optional[Callable[[int, float], None]] = None,
) -> dict[str, Any]:
    total_steps = max(
        2, int(math.ceil(_duration(case) / max(_step_size(case), 1.0e-6)))
    )
    stride = max(1, total_steps // max(2, _sample_count(case) - 1))

    times: list[float] = []
    objects: list[dict[str, Any]] = []
    for tr in trackers:
        base = {
            "name": tr["name"],
            "kind": tr["kind"],
            "color": tr["color"],
        }
        if isinstance(tr.get("static_fields"), dict):
            base.update(tr["static_fields"])
        if tr["kind"] == "segment":
            base["width"] = float(tr.get("width", 2.0))
            base["points_a"] = []
            base["points_b"] = []
        elif tr["kind"] == "capsule":
            base["size"] = tr["size"]
            base["mass"] = float(tr.get("mass", 0.0))
            base["positions"] = []
            base["points_a"] = []
            base["points_b"] = []
            base["scalars"] = {name: [] for name in tr.get("scalars", {})}
        else:
            base["size"] = tr["size"]
            base["mass"] = float(tr.get("mass", 0.0))
            base["positions"] = []
            base["scalars"] = {name: [] for name in tr.get("scalars", {})}
            if bool(tr.get("track_quaternion", False)):
                base["quaternions"] = []
            if tr.get("render_shape"):
                base["render_shape"] = str(tr["render_shape"])
        objects.append(base)

    for step in range(total_steps + 1):
        if step_hook is not None:
            step_hook(step, float(system.GetChTime()))
        if step == 0 or step == total_steps or step % stride == 0:
            times.append(float(system.GetChTime()))
            for out, tr in zip(objects, trackers):
                sample = tr["sample"]()
                if tr["kind"] == "segment":
                    out["points_a"].append(sample["point_a"])
                    out["points_b"].append(sample["point_b"])
                elif tr["kind"] == "capsule":
                    out["positions"].append(sample["position"])
                    out["points_a"].append(sample["point_a"])
                    out["points_b"].append(sample["point_b"])
                    for key in tr.get("scalars", {}):
                        out["scalars"][key].append(
                            float(sample["scalars"].get(key, 0.0))
                        )
                else:
                    out["positions"].append(sample["position"])
                    if "quaternions" in out:
                        out["quaternions"].append(sample["quaternion"])
                    for key in tr.get("scalars", {}):
                        out["scalars"][key].append(
                            float(sample["scalars"].get(key, 0.0))
                        )
        if step == total_steps:
            break
        system.DoStepDynamics(_step_size(case))

    scene = dict(scene)
    scene.update(_scene_bounds(objects, scene.get("view_plane", "xz"), scene))
    return {"times": times, "objects": objects, "scene": scene}


def _simulate_chrono_pendulum(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_pendulum import _simulate_chrono_pendulum as implementation

    return implementation(case, chrono)


def _simulate_chrono_double_pendulum(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_double_pendulum import (
        _simulate_chrono_double_pendulum as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_triple_pendulum(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_triple_pendulum import (
        _simulate_chrono_triple_pendulum as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_boxes(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_boxes import _simulate_chrono_boxes as implementation

    return implementation(case, chrono)


def _simulate_chrono_rolling(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_rolling import _simulate_chrono_rolling as implementation

    return implementation(case, chrono)


def _chrono_init_bmw_vehicle_scene(
    case: dict[str, Any], chrono: Any
) -> tuple[Any, Any, Any, dict[str, Any], float, float, Any, Any]:
    import pychrono.vehicle as veh

    profile = _vehicle_profile_from_case(case)
    if profile != "bmw_e90":
        raise RuntimeError(
            f"BMW full-vehicle scene currently supports only bmw_e90, got: {profile}"
        )

    data_root = _vehicle_data_root(veh, profile)
    spec = _load_vehicle_specs(profile, data_root)
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))
    solver_type = (
        str(_solver_cfg(case).get("solver_type", "BARZILAIBORWEIN")).strip().upper()
    )
    solver_attr = getattr(chrono.ChSolver, f"Type_{solver_type}", None)
    if solver_attr is not None:
        system.SetSolverType(solver_attr)
    solver = system.GetSolver() if hasattr(system, "GetSolver") else None
    if solver is not None and hasattr(solver, "AsIterative"):
        iterative = solver.AsIterative()
        if iterative is not None and hasattr(iterative, "SetMaxIterations"):
            iterative.SetMaxIterations(
                int(_solver_cfg(case).get("max_iterations", 150))
            )
    if hasattr(system, "SetMaxPenetrationRecoverySpeed"):
        system.SetMaxPenetrationRecoverySpeed(
            float(_solver_cfg(case).get("max_penetration_recovery_speed", 4.0))
        )

    terrain = veh.RigidTerrain(system)
    env = _env_cfg(case)
    ground = env.get("ground") if isinstance(env.get("ground"), dict) else {}
    terrain_mat = chrono.ChContactMaterialNSC()
    if hasattr(terrain_mat, "SetFriction"):
        terrain_mat.SetFriction(float(ground.get("friction", 0.9)))
    if hasattr(terrain_mat, "SetRestitution"):
        terrain_mat.SetRestitution(float(ground.get("restitution", 0.01)))
    ground_size = (
        ground.get("size", [120.0, 48.0])
        if isinstance(ground.get("size"), list)
        else [120.0, 48.0]
    )
    terrain.AddPatch(
        terrain_mat, chrono.CSYSNORM, float(ground_size[0]), float(ground_size[1])
    )

    mech = _mech_cfg(case)
    step = _step_size(case)
    duration = _duration(case)
    tire_model = str(mech.get("tire_model", "TMSIMPLE")).strip().upper()
    tire_enum = getattr(
        veh, f"TireModelType_{tire_model}", getattr(veh, "TireModelType_TMSIMPLE")
    )
    collision_enum = getattr(
        veh,
        f"CollisionType_{str(mech.get('collision_type', 'MESH')).strip().upper()}",
        getattr(veh, "CollisionType_MESH"),
    )
    return veh, system, terrain, spec, step, duration, tire_enum, collision_enum


def _chrono_add_bmw_vehicle(
    veh: Any,
    chrono: Any,
    system: Any,
    step: float,
    tire_enum: Any,
    collision_enum: Any,
    cfg: dict[str, Any],
) -> Any:
    viz_mesh = getattr(chrono, "VisualizationType_MESH", 2)
    viz_none = getattr(chrono, "VisualizationType_NONE", 0)
    vehicle = veh.BMW_E90(system)
    vehicle.SetInitPosition(
        chrono.ChCoordsysd(
            _chrono_vec(chrono, cfg.get("position", [0.0, 0.0, 0.6])),
            _chrono_quat_z(
                chrono, math.radians(float(cfg.get("initial_yaw_deg", 0.0)))
            ),
        )
    )
    forward = np.asarray(
        [
            math.cos(math.radians(float(cfg.get("initial_yaw_deg", 0.0)))),
            math.sin(math.radians(float(cfg.get("initial_yaw_deg", 0.0)))),
            0.0,
        ],
        dtype=float,
    )
    init_vel = np.asarray(cfg.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    vehicle.SetInitFwdVel(float(np.dot(init_vel, forward)))
    vehicle.SetChassisCollisionType(collision_enum)
    vehicle.SetTireType(tire_enum)
    vehicle.SetTireStepSize(step)
    vehicle.Initialize()
    vehicle.SetChassisVisualizationType(viz_mesh)
    vehicle.SetSuspensionVisualizationType(viz_none)
    vehicle.SetSteeringVisualizationType(viz_none)
    vehicle.SetWheelVisualizationType(viz_none)
    vehicle.SetTireVisualizationType(viz_none)
    return vehicle


def _simulate_chrono_vehicle_collision_full(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_vehicle_collision_full import (
        _simulate_chrono_vehicle_collision_full as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_vehicle_lane_change(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_vehicle_lane_change import (
        _simulate_chrono_vehicle_lane_change as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_ramp_jump(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_ramp_jump import _simulate_chrono_ramp_jump as implementation

    return implementation(case, chrono)


def _simulate_chrono_sphere_drop(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_sphere_drop import (
        _simulate_chrono_sphere_drop as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_gyroscope(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_gyroscope import _simulate_chrono_gyroscope as implementation

    return implementation(case, chrono)


def _simulate_chrono_ball_maze(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_ball_maze import _simulate_chrono_ball_maze as implementation

    return implementation(case, chrono)


def _simulate_chrono_slider_crank(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_slider_crank import (
        _simulate_chrono_slider_crank as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_four_bar(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_four_bar import _simulate_chrono_four_bar as implementation

    return implementation(case, chrono)


def _simulate_chrono_vehicle_obstacle_crossing(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_vehicle_obstacle_crossing import (
        _simulate_chrono_vehicle_obstacle_crossing as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_granular_collapse(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_granular_collapse import (
        _simulate_chrono_granular_collapse as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_ball_chute_pile(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_ball_chute_pile import (
        _simulate_chrono_ball_chute_pile as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono_gear_pair(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    from families.chrono_gear_pair import _simulate_chrono_gear_pair as implementation

    return implementation(case, chrono)


def _simulate_chrono_vehicle_collision(
    case: dict[str, Any], chrono: Any
) -> dict[str, Any]:
    from families.chrono_vehicle_collision import (
        _simulate_chrono_vehicle_collision as implementation,
    )

    return implementation(case, chrono)


def _simulate_chrono(case: dict[str, Any]) -> dict[str, Any]:
    import pychrono as chrono

    family = _family(case)
    if family in {
        "cantileverBeamTransient",
        "harmonicResponseFRF",
        "stressWavePropagation",
        "dropWeightImpactSMC",
        "j2PlasticBeam",
        "lemaitreDamageBar",
        "rubberBlockCompression",
        "flexibleCrankSlider",
        "reissnerShellModal",
    }:
        # Structural-demo families are synthesized in benchmark space to keep
        # compatibility across Chrono builds without optional FEA backends.
        return _simulate_surrogate_structural_demo(case)
    if family == "pendulum":
        scene = str(case.get("scene", "")).strip().lower()
        mechanism_type = str(_mech_cfg(case).get("type", "")).strip().lower()
        if scene == "triplependulum" or "triple" in mechanism_type:
            return _simulate_chrono_triple_pendulum(case, chrono)
        if scene == "doublependulum" or "double" in mechanism_type:
            return _simulate_chrono_double_pendulum(case, chrono)
        return _simulate_chrono_pendulum(case, chrono)
    if family == "doublePendulum":
        return _simulate_chrono_double_pendulum(case, chrono)
    if family == "triplePendulum":
        return _simulate_chrono_triple_pendulum(case, chrono)
    if family == "rigidCollision":
        # Chrono NSC contact in this cluster build is strongly dissipative for
        # these short box impacts, so use the analytic rigid-body impulse path
        # to preserve visible post-impact separation and restitution.
        return _simulate_surrogate_rigid_collision(case)
    if family in {"stackStability", "blockTower3D"}:
        return _simulate_chrono_boxes(case, chrono)
    if family == "massSpringDamper":
        return _simulate_surrogate_mass_spring_damper(case)
    if family == "vehicleCollision3D":
        raise ValueError("Family vehicleCollision3D is disabled.")
    if family == "vehicleLaneChange3D":
        return _simulate_chrono_vehicle_lane_change(case, chrono)
    if family == "vehicleObstacleCrossing3D":
        return _simulate_chrono_vehicle_obstacle_crossing(case, chrono)
    if family == "rampJump3D":
        return _simulate_chrono_ramp_jump(case, chrono)
    if family == "sphereDrop3D":
        return _simulate_chrono_sphere_drop(case, chrono)
    if family == "gyroscopeTop3D":
        return _simulate_chrono_gyroscope(case, chrono)
    if family == "gyroscopeRotor3D":
        return _simulate_surrogate_unbalanced_rotor(case)
    if family == "unbalancedRotor3D":
        return _simulate_surrogate_unbalanced_rotor(case)
    if family == "articulatedChain":
        return _simulate_surrogate_articulated_chain(case)
    if family == "rollingBallMaze3D":
        return _simulate_chrono_ball_maze(case, chrono)
    if family == "sliderCrank3D":
        raise ValueError("Family sliderCrank3D has been removed.")
    if family == "fourBarLinkage3D":
        raise ValueError("Family fourBarLinkage3D has been removed.")
    if family == "granularCollapse3D":
        return _simulate_chrono_granular_collapse(case, chrono)
    if family == "gearPair3D":
        return _simulate_chrono_gear_pair(case, chrono)
    if family == "ballChutePile3D":
        return _simulate_chrono_ball_chute_pile(case, chrono)
    if family == "rollingVsSliding":
        return _simulate_chrono_rolling(case, chrono)
    if family == "rollingSlidingTransition":
        return _simulate_surrogate_rolling(case)
    raise ValueError(f"Unsupported Chrono family: {family}")


def _apply_runtime_overrides(
    case: dict[str, Any],
    *,
    ball_count: Optional[int],
    random_seed: Optional[int],
    ball_radius: Optional[float],
    chute_angle_deg: Optional[float],
) -> dict[str, Any]:
    if not isinstance(case, dict):
        return case
    family = _family(case)
    if family != "ballChutePile3D":
        return case
    mech = case.get("mechanism") if isinstance(case.get("mechanism"), dict) else {}
    case["mechanism"] = mech
    if ball_count is not None:
        mech["ball_count"] = max(1, int(ball_count))
    if random_seed is not None:
        mech["random_seed"] = int(random_seed)
        case["seed"] = int(random_seed)
    if ball_radius is not None:
        mech["ball_radius"] = max(0.002, float(ball_radius))
    if chute_angle_deg is not None:
        mech["chute_angle_deg"] = float(chute_angle_deg)
    return case


def _solve(
    case_dir: Path,
    backend: str,
    strict_chrono: bool,
    *,
    ball_count: Optional[int],
    random_seed: Optional[int],
    ball_radius: Optional[float],
    chute_angle_deg: Optional[float],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    case = _load_case(case_dir)
    case = _apply_runtime_overrides(
        case,
        ball_count=ball_count,
        random_seed=random_seed,
        ball_radius=ball_radius,
        chute_angle_deg=chute_angle_deg,
    )
    actual_backend = _select_backend(backend, strict_chrono)
    data = (
        _simulate_chrono(case)
        if actual_backend == "chrono"
        else _simulate_surrogate(case)
    )
    data["backend"] = actual_backend
    data["family"] = _family(case)
    data["dim"] = _case_dim(case)
    data["case_name"] = _case_name(case, case_dir)
    data["scene"] = dict(data.get("scene", {}))
    if bool(data["scene"].get("auto_bounds", True)) or not (
        isinstance(data["scene"].get("xlim"), list)
        and isinstance(data["scene"].get("ylim"), list)
    ):
        data["scene"].update(
            _scene_bounds(
                list(data.get("objects", [])),
                data["scene"].get("view_plane", "xz"),
                data["scene"],
            )
        )
    metrics = _metrics_from_trajectory(case, data, actual_backend)
    return case, data, actual_backend


def _write_ball_chute_exports(
    case_dir: Path, case: dict[str, Any], data: dict[str, Any]
) -> None:
    times = np.asarray(data.get("times", []), dtype=float).reshape(-1)
    if times.size < 2:
        return
    mech = _mech_cfg(case)
    family = _family(case)
    if family != "ballChutePile3D":
        return
    dt_out = max(1.0e-4, float(mech.get("export_interval_s", 0.005)))
    t_end = float(times[-1])
    n_out = max(1, int(math.floor(t_end / dt_out + 1.0e-9)) + 1)
    query_times = np.asarray([i * dt_out for i in range(n_out)], dtype=float)
    if query_times[-1] < t_end - 1.0e-9:
        query_times = np.append(query_times, t_end)
    # Use zero-order hold for VTP export to avoid linear interpolation
    # re-introducing visual penetration between already-corrected states.
    query_indices = np.searchsorted(times, query_times, side="right") - 1
    query_indices = np.clip(query_indices, 0, times.size - 1).astype(int)

    objects = data.get("objects") if isinstance(data.get("objects"), list) else []
    balls = [
        obj
        for obj in objects
        if isinstance(obj, dict) and str(obj.get("name", "")).startswith("ball_")
    ]
    if not balls:
        return

    post = case_dir / "postProcessing"
    vtp_dir = post / "ball_stream_vtp"
    vtp_dir.mkdir(parents=True, exist_ok=True)
    field_vtp_dir = post / "velocity_field_vtp"
    field_vtp_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[float, str]] = []
    field_datasets: list[tuple[float, str]] = []
    csv_rows: list[list[float]] = []
    contact_pairs_series = (
        (data.get("global_series") or {}).get("contact_pairs")
        if isinstance(data.get("global_series"), dict)
        else None
    )
    contact_pairs_interp = (
        _interp_series(
            times, np.asarray(contact_pairs_series, dtype=float), query_times
        )
        if isinstance(contact_pairs_series, list)
        and len(contact_pairs_series) == len(times)
        else None
    )
    gseries = (
        data.get("global_series") if isinstance(data.get("global_series"), dict) else {}
    )
    stage_names = [
        "stage_chute_count",
        "stage_flight_count",
        "stage_box_count",
        "stage_rest_count",
    ]
    stage_interp: dict[str, np.ndarray] = {}
    for key in stage_names:
        raw = gseries.get(key)
        if isinstance(raw, list) and len(raw) == len(times):
            stage_interp[key] = _interp_series(
                times, np.asarray(raw, dtype=float), query_times
            )

    probe_cfg = (
        data.get("field_probes") if isinstance(data.get("field_probes"), dict) else {}
    )
    probe_points = (
        np.asarray(probe_cfg.get("points", []), dtype=float)
        if isinstance(probe_cfg.get("points"), list)
        else np.zeros((0, 3), dtype=float)
    )
    if probe_points.ndim != 2 or probe_points.shape[1] < 3:
        probe_points = np.zeros((0, 3), dtype=float)

    for frame_idx, tq in enumerate(query_times):
        tidx = int(query_indices[frame_idx])
        points: list[list[float]] = []
        speed_vals: list[float] = []
        omega_vals: list[float] = []
        radius_vals: list[float] = []
        contact_vals: list[float] = []
        inflight_vals: list[float] = []
        total_ke = 0.0
        contacting = 0

        for obj in balls:
            pos_src = np.asarray(obj.get("positions", []), dtype=float)
            if (
                pos_src.ndim != 2
                or pos_src.shape[0] != times.size
                or pos_src.shape[1] < 3
            ):
                continue
            scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
            visible_src = (
                np.asarray(scalars.get("visible", []), dtype=float)
                if isinstance(scalars.get("visible"), list)
                else np.ones((times.size,), dtype=float)
            )
            visible = float(visible_src[tidx]) if visible_src.size > tidx else 1.0
            if visible <= 0.5:
                continue

            pos = pos_src[tidx, :3]
            speed_src = (
                np.asarray(scalars.get("speed_mag", []), dtype=float)
                if isinstance(scalars.get("speed_mag"), list)
                else np.zeros((times.size,), dtype=float)
            )
            omega_src = (
                np.asarray(scalars.get("omega_mag", []), dtype=float)
                if isinstance(scalars.get("omega_mag"), list)
                else np.zeros((times.size,), dtype=float)
            )
            radius_src = (
                np.asarray(scalars.get("radius", []), dtype=float)
                if isinstance(scalars.get("radius"), list)
                else np.full(
                    (times.size,), float((obj.get("size") or [0.05])[0]), dtype=float
                )
            )
            contact_src = (
                np.asarray(scalars.get("contact_force_mag", []), dtype=float)
                if isinstance(scalars.get("contact_force_mag"), list)
                else np.zeros((times.size,), dtype=float)
            )
            inflight_src = (
                np.asarray(scalars.get("in_flight", []), dtype=float)
                if isinstance(scalars.get("in_flight"), list)
                else np.zeros((times.size,), dtype=float)
            )

            speed = float(speed_src[tidx]) if speed_src.size > tidx else 0.0
            omega = float(omega_src[tidx]) if omega_src.size > tidx else 0.0
            radius = (
                float(radius_src[tidx])
                if radius_src.size > tidx
                else float((obj.get("size") or [0.05])[0])
            )
            contact = float(contact_src[tidx]) if contact_src.size > tidx else 0.0
            inflight = float(inflight_src[tidx]) if inflight_src.size > tidx else 0.0

            mass = float(obj.get("mass", 0.0) or 0.0)
            inertia = 0.4 * mass * radius * radius
            total_ke += 0.5 * mass * speed * speed + 0.5 * inertia * omega * omega
            if contact > 1.0e-6:
                contacting += 1

            points.append([float(pos[0]), float(pos[1]), float(pos[2])])
            speed_vals.append(speed)
            omega_vals.append(omega)
            radius_vals.append(radius)
            contact_vals.append(contact)
            inflight_vals.append(inflight)

        pts = (
            np.asarray(points, dtype=float) if points else np.zeros((0, 3), dtype=float)
        )
        vtp_name = f"frame_{frame_idx:05d}.vtp"
        _write_vtp_points(
            vtp_dir / vtp_name,
            points=pts,
            point_data={
                "velocity_magnitude": np.asarray(speed_vals, dtype=float),
                "speed_mag": np.asarray(speed_vals, dtype=float),
                "omega_mag": np.asarray(omega_vals, dtype=float),
                "radius": np.asarray(radius_vals, dtype=float),
                "contact_force_mag": np.asarray(contact_vals, dtype=float),
                "in_flight": np.asarray(inflight_vals, dtype=float),
            },
        )
        datasets.append((float(tq), f"ball_stream_vtp/{vtp_name}"))
        contact_pairs = (
            int(round(float(contact_pairs_interp[frame_idx])))
            if isinstance(contact_pairs_interp, np.ndarray)
            and frame_idx < contact_pairs_interp.size
            else int(max(0, contacting // 2))
        )
        stage_vals = [
            float(
                stage_interp.get("stage_chute_count", np.zeros((0,), dtype=float))[
                    frame_idx
                ]
            )
            if frame_idx < len(stage_interp.get("stage_chute_count", []))
            else 0.0,
            float(
                stage_interp.get("stage_flight_count", np.zeros((0,), dtype=float))[
                    frame_idx
                ]
            )
            if frame_idx < len(stage_interp.get("stage_flight_count", []))
            else 0.0,
            float(
                stage_interp.get("stage_box_count", np.zeros((0,), dtype=float))[
                    frame_idx
                ]
            )
            if frame_idx < len(stage_interp.get("stage_box_count", []))
            else 0.0,
            float(
                stage_interp.get("stage_rest_count", np.zeros((0,), dtype=float))[
                    frame_idx
                ]
            )
            if frame_idx < len(stage_interp.get("stage_rest_count", []))
            else 0.0,
        ]
        csv_rows.append([float(tq), float(total_ke), float(contact_pairs)] + stage_vals)

        if probe_points.size:
            if points:
                src_points = np.asarray(points, dtype=float)
                speed_arr = np.asarray(speed_vals, dtype=float)
                contact_arr = np.asarray(contact_vals, dtype=float)
                radii_arr = np.asarray(radius_vals, dtype=float)
                diff = probe_points[:, None, :3] - src_points[None, :, :3]
                d2 = np.sum(diff * diff, axis=2)
                sigma = (
                    max(float(np.mean(radii_arr) * 1.8), 0.08)
                    if radii_arr.size
                    else 0.08
                )
                weights = np.exp(-d2 / max(2.0 * sigma * sigma, 1.0e-9))
                speed_field = np.sum(weights * speed_arr[None, :], axis=1) / np.maximum(
                    np.sum(weights, axis=1), 1.0e-9
                )
                contact_field = np.sum(
                    weights * (contact_arr > 1.0e-6)[None, :], axis=1
                )
                occupancy = np.max(
                    (
                        np.sqrt(d2) <= (2.5 * np.maximum(radii_arr[None, :], 1.0e-6))
                    ).astype(float),
                    axis=1,
                )
            else:
                speed_field = np.zeros((probe_points.shape[0],), dtype=float)
                contact_field = np.zeros((probe_points.shape[0],), dtype=float)
                occupancy = np.zeros((probe_points.shape[0],), dtype=float)
            field_name = f"frame_{frame_idx:05d}.vtp"
            _write_vtp_points(
                field_vtp_dir / field_name,
                points=probe_points[:, :3],
                point_data={
                    "velocity_magnitude": np.asarray(speed_field, dtype=float),
                    "speed_mag": np.asarray(speed_field, dtype=float),
                    "contact_density": np.asarray(contact_field, dtype=float),
                    "occupancy": np.asarray(occupancy, dtype=float),
                },
            )
            field_datasets.append((float(tq), f"velocity_field_vtp/{field_name}"))

    _write_pvd(post / "ball_stream.pvd", datasets)
    if field_datasets:
        _write_pvd(post / "velocity_field.pvd", field_datasets)
    _write_csv(
        post / "ball_stream_frame_stats.csv",
        [
            "time_s",
            "total_kinetic_energy_j",
            "contact_pairs",
            "stage_chute_count",
            "stage_flight_count",
            "stage_box_count",
            "stage_rest_count",
        ],
        csv_rows,
    )
    _write_csv(
        post / "energy_log.csv",
        [
            "time_s",
            "total_kinetic_energy_j",
            "contact_pairs",
            "stage_chute_count",
            "stage_flight_count",
            "stage_box_count",
            "stage_rest_count",
        ],
        csv_rows,
    )

    static_scene_dir = post / "static_scene_vtp"
    static_scene_dir.mkdir(parents=True, exist_ok=True)
    static_manifest: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict) or str(obj.get("kind", "")) != "box":
            continue
        positions = obj.get("positions")
        size = np.asarray(obj.get("size", []), dtype=float)
        quats = obj.get("quaternions")
        if not isinstance(positions, list) or not positions or size.size < 3:
            continue
        center = np.asarray(positions[0], dtype=float)
        quat = (
            np.asarray(quats[0], dtype=float)
            if isinstance(quats, list) and quats
            else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        )
        sx, sy, sz = [float(v) for v in size[:3]]
        hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
        local_pts = np.asarray(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [-hx, hy, -hz],
                [hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [-hx, hy, hz],
                [hx, hy, hz],
            ],
            dtype=float,
        )
        world_pts = local_pts @ _quat_to_rotmat_np(quat).T + center.reshape(1, 3)
        polys = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 3, 7, 5],
        ]
        cell_count = len(polys)
        opacity = float(obj.get("opacity", 1.0) or 1.0)
        color = obj.get("color", [0.7, 0.7, 0.7])
        color_rgb = np.asarray(
            color[:3]
            if isinstance(color, list) and len(color) >= 3
            else [0.7, 0.7, 0.7],
            dtype=float,
        )
        file_name = f"{str(obj.get('name', 'static_box')).strip()}.vtp"
        _write_vtp_polys(
            static_scene_dir / file_name,
            points=world_pts,
            polys=polys,
            cell_data={
                "recommended_opacity": np.full((cell_count,), opacity, dtype=float),
                "color_r": np.full((cell_count,), float(color_rgb[0]), dtype=float),
                "color_g": np.full((cell_count,), float(color_rgb[1]), dtype=float),
                "color_b": np.full((cell_count,), float(color_rgb[2]), dtype=float),
            },
        )
        static_manifest.append(
            {
                "name": str(obj.get("name", file_name[:-4])),
                "file": f"static_scene_vtp/{file_name}",
                "kind": "box",
                "opacity": opacity,
                "color": [
                    float(color_rgb[0]),
                    float(color_rgb[1]),
                    float(color_rgb[2]),
                ],
            }
        )
    _write_json(post / "static_scene_manifest.json", {"objects": static_manifest})


def _write_case_exports(
    case_dir: Path, case: dict[str, Any], data: dict[str, Any]
) -> None:
    exports = data.get("exports")
    if not isinstance(exports, list):
        exports = []
    for item in exports:
        if isinstance(item, dict):
            file_name = str(item.get("file_name", "")).strip()
            header = item.get("header")
            rows = item.get("rows")
            if (
                not file_name
                or not isinstance(header, list)
                or not isinstance(rows, list)
            ):
                continue
            cleaned_rows: list[list[float]] = []
            for row in rows:
                if isinstance(row, list):
                    cleaned_rows.append([float(v) for v in row])
            if cleaned_rows:
                _write_csv(
                    case_dir / "postProcessing" / file_name,
                    [str(h) for h in header],
                    cleaned_rows,
                )
    static_files = data.get("static_files")
    if isinstance(static_files, list):
        for item in static_files:
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file_name", "")).strip()
            content = item.get("content")
            if file_name and isinstance(content, str):
                _write_text(case_dir / file_name, content)
    _write_ball_chute_exports(case_dir, case, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dynamics benchmark case")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument(
        "--backend", choices=["auto", "chrono", "surrogate"], default="auto"
    )
    parser.add_argument(
        "--strict-chrono",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require PyChrono availability; fail fast if unavailable.",
    )
    parser.add_argument(
        "--ball-count",
        type=int,
        default=None,
        help="Override ball count for ballChutePile3D.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override random seed for ballChutePile3D.",
    )
    parser.add_argument(
        "--ball-radius",
        type=float,
        default=None,
        help="Override mean ball radius [m] for ballChutePile3D.",
    )
    parser.add_argument(
        "--chute-angle-deg",
        type=float,
        default=None,
        help="Override chute angle [deg] for ballChutePile3D.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    force_rerun = _bool_env("FORCE_RERUN_CASES", False)
    log_path = case_dir / "log.solve"
    if (
        log_path.exists()
        and "END" in log_path.read_text(errors="ignore")
        and not force_rerun
    ):
        print(f"[dynamics_bench] skip existing END: {case_dir.name}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"BEGIN backend={args.backend} strict_chrono={int(args.strict_chrono)}\n"
    )

    try:
        case, data, actual_backend = _solve(
            case_dir,
            args.backend,
            bool(args.strict_chrono),
            ball_count=args.ball_count,
            random_seed=args.random_seed,
            ball_radius=args.ball_radius,
            chute_angle_deg=args.chute_angle_deg,
        )
        _write_json(case_dir / "results" / "trajectory.json", data)
        _write_json(
            case_dir / "postProcessing" / "metrics.json",
            _metrics_from_trajectory(case, data, actual_backend),
        )
        _write_case_exports(case_dir, case, data)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"END backend={actual_backend}\n")
        print(f"[dynamics_bench] solved {case_dir.name} backend={actual_backend}")
    except Exception:
        tb = traceback.format_exc()
        with log_path.open("a", encoding="utf-8") as f:
            f.write("FAIL\n")
            f.write(tb)
        raise


if __name__ == "__main__":
    main()
