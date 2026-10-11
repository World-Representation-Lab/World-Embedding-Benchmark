"""Trajectory geometry, contact measurements, and physical diagnostics."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _quat_to_rotmat_np(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.size != 4:
        return np.eye(3, dtype=float)
    w, x, y, z = [float(v) for v in q]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _mech_cfg(case: dict[str, Any]) -> dict[str, Any]:
    return case.get("mechanism") if isinstance(case.get("mechanism"), dict) else {}


def _env_cfg(case: dict[str, Any]) -> dict[str, Any]:
    return case.get("environment") if isinstance(case.get("environment"), dict) else {}


def _case_dim(case: dict[str, Any]) -> str:
    dim = str(case.get("dim", "")).strip().upper()
    return dim if dim in {"2D", "3D"} else "2D"


def _family(case: dict[str, Any]) -> str:
    return str(
        case.get("family") or case.get("case_prefix") or case.get("scene") or "unknown"
    )


def _gravity(case: dict[str, Any]) -> np.ndarray:
    env = _env_cfg(case)
    raw = env.get("gravity", [0.0, 0.0, -9.81])
    if not isinstance(raw, list) or len(raw) != 3:
        raw = [0.0, 0.0, -9.81]
    return np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=float)


def _polyline_points(
    raw: Any, *, default: Optional[list[list[float]]] = None
) -> np.ndarray:
    points = (
        raw if isinstance(raw, list) and raw else (default or [[0.0, 0.0], [1.0, 0.0]])
    )
    out: list[list[float]] = []
    for item in points:
        if not isinstance(item, list) or len(item) < 2:
            continue
        out.append([float(item[0]), float(item[1])])
    if len(out) < 2:
        out = [[0.0, 0.0], [1.0, 0.0]]
    return np.asarray(out, dtype=float)


def _polyline_arclength(points: np.ndarray) -> tuple[np.ndarray, float]:
    if points.shape[0] < 2:
        return np.asarray([0.0], dtype=float), 0.0
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum, float(cum[-1])


def _polyline_progress(point_xy: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 2:
        return 0.0, float(np.linalg.norm(point_xy - points[0]))
    cum, total = _polyline_arclength(points)
    best_progress = 0.0
    best_dist = float("inf")
    p = np.asarray(point_xy[:2], dtype=float)
    for i in range(points.shape[0] - 1):
        a = points[i]
        b = points[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1.0e-12:
            continue
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        dist = float(np.linalg.norm(p - proj))
        if dist < best_dist:
            best_dist = dist
            best_progress = float(cum[i] + t * np.linalg.norm(ab))
    return float(best_progress), float(best_dist if np.isfinite(best_dist) else total)


def _convex_hull_xy(points: list[tuple[float, float]]) -> list[list[float]]:
    uniq = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in points))
    if len(uniq) <= 2:
        return [[float(x), float(y)] for x, y in uniq]

    def cross(
        o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in uniq:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(uniq):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) > 48:
        step = max(1, len(hull) // 48)
        hull = hull[::step]
    return [[float(x), float(y)] for x, y in hull]


def _compute_speed_series(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return np.zeros((len(times), 3), dtype=float)
    dt = np.diff(times).reshape(-1, 1)
    dp = np.diff(positions, axis=0)
    vel = np.zeros_like(positions)
    vel[1:] = dp / np.maximum(dt, 1.0e-6)
    vel[0] = vel[1]
    return vel


def _first_sustain_time(
    times: np.ndarray, values: np.ndarray, threshold: float
) -> float:
    for i in range(len(times)):
        if np.all(values[i:] <= threshold):
            return float(times[i])
    return float(times[-1])


def _find_object(objects: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for obj in objects:
        if str(obj.get("name")) == name:
            return obj
    return {}


def _scalar_series_value(values: Any, idx: int, default: float = 0.0) -> float:
    if isinstance(values, list) and idx < len(values):
        try:
            return float(values[idx])
        except Exception:
            return float(default)
    return float(default)


def _rotate_xy(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    ang = math.radians(float(yaw_deg))
    c = math.cos(ang)
    s = math.sin(ang)
    rot = np.asarray([[c, -s], [s, c]], dtype=float)
    return np.asarray(points, dtype=float) @ rot.T


def _rect_polygon(
    center_xy: np.ndarray, size_xy: np.ndarray, yaw_deg: float = 0.0
) -> np.ndarray:
    hx = 0.5 * float(size_xy[0])
    hy = 0.5 * float(size_xy[1])
    base = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=float)
    return _rotate_xy(base, yaw_deg) + np.asarray(center_xy, dtype=float)


def _polygon_vertical_envelope(
    points: np.ndarray, xq: float
) -> tuple[float, float] | None:
    if points.ndim != 2 or points.shape[0] < 3:
        return None
    hits: list[float] = []
    n = int(points.shape[0])
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        x0 = float(p0[0])
        x1 = float(p1[0])
        if (xq < min(x0, x1) - 1.0e-9) or (xq > max(x0, x1) + 1.0e-9):
            continue
        if abs(x1 - x0) <= 1.0e-9:
            hits.extend([float(p0[1]), float(p1[1])])
            continue
        t = (xq - x0) / (x1 - x0)
        if -1.0e-9 <= t <= 1.0 + 1.0e-9:
            hits.append(float(p0[1] + t * (float(p1[1]) - float(p0[1]))))
    if len(hits) < 2:
        return None
    return float(min(hits)), float(max(hits))


def _vehicle_local_sections(
    obj: dict[str, Any], n_sections: int = 18
) -> list[np.ndarray]:
    outline_xy = np.asarray(obj.get("outline_xy", []), dtype=float)
    outline_xz = np.asarray(obj.get("outline_xz", []), dtype=float)
    if (
        outline_xy.ndim != 2
        or outline_xy.shape[0] < 6
        or outline_xz.ndim != 2
        or outline_xz.shape[0] < 6
    ):
        return []
    x_min = max(float(np.min(outline_xy[:, 0])), float(np.min(outline_xz[:, 0])))
    x_max = min(float(np.max(outline_xy[:, 0])), float(np.max(outline_xz[:, 0])))
    if x_max - x_min <= 1.0e-6:
        return []
    sections: list[np.ndarray] = []
    for xq in np.linspace(x_min, x_max, max(10, int(n_sections))):
        env_xy = _polygon_vertical_envelope(outline_xy, float(xq))
        env_xz = _polygon_vertical_envelope(outline_xz, float(xq))
        if env_xy is None or env_xz is None:
            continue
        y_min, y_max = env_xy
        z_min, z_max = env_xz
        if (y_max - y_min) <= 1.0e-6 or (z_max - z_min) <= 1.0e-6:
            continue
        y_half = max(abs(y_min), abs(y_max))
        z_span = z_max - z_min
        roof_half = max(
            0.18 * y_half,
            0.58 * y_half - 0.10 * y_half * min(1.0, z_span / max(0.6, z_span)),
        )
        belt_z = z_min + 0.52 * z_span
        shoulder = 0.94 * y_half
        section = np.asarray(
            [
                [xq, -y_half, z_min],
                [xq, -shoulder, belt_z],
                [xq, -roof_half, z_max],
                [xq, roof_half, z_max],
                [xq, shoulder, belt_z],
                [xq, y_half, z_min],
            ],
            dtype=float,
        )
        sections.append(section)
    return sections


def _transform_body_points_3d(
    points_xyz: np.ndarray, x: float, y: float, z: float, yaw_deg: float
) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        return np.zeros((0, 3), dtype=float)
    xy = _rotate_xy(points_xyz[:, :2], yaw_deg) + np.asarray(
        [float(x), float(y)], dtype=float
    )
    zz = points_xyz[:, 2] + float(z)
    return np.column_stack([xy[:, 0], xy[:, 1], zz])


def _vehicle_polygon_xz(
    obj: dict[str, Any], pos: np.ndarray, idx: int
) -> np.ndarray | None:
    sections = _vehicle_local_sections(obj)
    if not sections:
        return None
    scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
    yaw = _scalar_series_value(scalars.get("yaw_deg"), idx, 0.0)
    world_points: list[tuple[float, float]] = []
    for sec in sections:
        world = _transform_body_points_3d(
            sec, float(pos[0]), float(pos[1]), float(pos[2]), yaw
        )
        for p in world:
            world_points.append((float(p[0]), float(p[2])))
    hull = _convex_hull_xy(world_points)
    if len(hull) >= 3:
        return np.asarray(hull, dtype=float)
    return None


def _vehicle_polygon_xy(obj: dict[str, Any], pos: np.ndarray, idx: int) -> np.ndarray:
    outline = obj.get("outline_xy")
    if isinstance(outline, list):
        pts = np.asarray(
            [
                [float(p[0]), float(p[1])]
                for p in outline
                if isinstance(p, list) and len(p) >= 2
            ],
            dtype=float,
        )
        if len(pts) >= 3:
            scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
            yaw = _scalar_series_value(scalars.get("yaw_deg"), idx, 0.0)
            return _rotate_xy(pts, yaw) + np.asarray(
                [float(pos[0]), float(pos[1])], dtype=float
            )
    size = np.asarray(obj.get("size", [1.0, 1.0, 1.0]), dtype=float)
    return _rect_polygon(
        np.asarray([float(pos[0]), float(pos[1])], dtype=float),
        np.asarray([size[0], size[1]], dtype=float),
        0.0,
    )


def _poly_axes(poly: np.ndarray) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    if poly.ndim != 2 or poly.shape[0] < 2:
        return axes
    for i in range(poly.shape[0]):
        p0 = poly[i]
        p1 = poly[(i + 1) % poly.shape[0]]
        edge = p1 - p0
        n = np.asarray([-edge[1], edge[0]], dtype=float)
        mag = float(np.linalg.norm(n))
        if mag > 1.0e-9:
            axes.append(n / mag)
    return axes


def _signed_gap_poly_poly(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    gaps: list[float] = []
    for axis in _poly_axes(poly_a) + _poly_axes(poly_b):
        a_proj = poly_a @ axis
        b_proj = poly_b @ axis
        gaps.append(
            max(
                float(np.min(a_proj) - np.max(b_proj)),
                float(np.min(b_proj) - np.max(a_proj)),
            )
        )
    return max(gaps) if gaps else 0.0


def _projected_shape(obj: dict[str, Any], idx: int, plane: str) -> tuple[str, Any]:
    def _size_at_index() -> np.ndarray:
        size_series = obj.get("sizes")
        if isinstance(size_series, list) and idx < len(size_series):
            item = size_series[idx]
            if isinstance(item, list) and len(item) >= 3:
                return np.asarray(
                    [float(item[0]), float(item[1]), float(item[2])], dtype=float
                )
        size = obj.get("size", [1.0, 1.0, 1.0])
        if isinstance(size, list) and len(size) >= 3:
            return np.asarray(
                [float(size[0]), float(size[1]), float(size[2])], dtype=float
            )
        return np.asarray([1.0, 1.0, 1.0], dtype=float)

    positions = obj.get("positions", [])
    if not isinstance(positions, list) or idx >= len(positions):
        return "none", None
    pos = np.asarray(positions[idx], dtype=float)
    kind = str(obj.get("kind", ""))
    if kind == "circle":
        if plane == "xz":
            return "circle", (
                np.asarray([float(pos[0]), float(pos[2])], dtype=float),
                float((obj.get("size") or [0.1])[0]),
            )
        return "circle", (
            np.asarray([float(pos[0]), float(pos[1])], dtype=float),
            float((obj.get("size") or [0.1])[0]),
        )
    if kind == "box":
        size = _size_at_index()
        if plane == "xz":
            if obj.get("render_shape") == "vehicle_mesh":
                poly = _vehicle_polygon_xz(obj, pos, idx)
                if poly is not None:
                    return "poly", poly
                outline_xz = obj.get("contact_outline_xz", obj.get("outline_xz"))
                if isinstance(outline_xz, list):
                    pts = np.asarray(
                        [
                            [float(p[0]), float(p[1])]
                            for p in outline_xz
                            if isinstance(p, list) and len(p) >= 2
                        ],
                        dtype=float,
                    )
                    if len(pts) >= 3:
                        return "poly", pts + np.asarray(
                            [float(pos[0]), float(pos[2])], dtype=float
                        )
            return "poly", _rect_polygon(
                np.asarray([float(pos[0]), float(pos[2])], dtype=float),
                np.asarray([size[0], size[2]], dtype=float),
                0.0,
            )
        if obj.get("render_shape") == "vehicle_mesh":
            outline = obj.get("contact_outline_xy", obj.get("outline_xy"))
            if isinstance(outline, list):
                pts = np.asarray(
                    [
                        [float(p[0]), float(p[1])]
                        for p in outline
                        if isinstance(p, list) and len(p) >= 2
                    ],
                    dtype=float,
                )
                if len(pts) >= 3:
                    scalars = (
                        obj.get("scalars")
                        if isinstance(obj.get("scalars"), dict)
                        else {}
                    )
                    yaw = _scalar_series_value(scalars.get("yaw_deg"), idx, 0.0)
                    return "poly", _rotate_xy(pts, yaw) + np.asarray(
                        [float(pos[0]), float(pos[1])], dtype=float
                    )
            return "poly", _vehicle_polygon_xy(obj, pos, idx)
        scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
        yaw = _scalar_series_value(scalars.get("yaw_deg"), idx, 0.0)
        return "poly", _rect_polygon(
            np.asarray([float(pos[0]), float(pos[1])], dtype=float),
            np.asarray([size[0], size[1]], dtype=float),
            yaw,
        )
    return "none", None


def _signed_gap_objects(
    obj_a: dict[str, Any], obj_b: dict[str, Any], idx: int, plane: str
) -> float | None:
    kind_a, shape_a = _projected_shape(obj_a, idx, plane)
    kind_b, shape_b = _projected_shape(obj_b, idx, plane)
    if kind_a == "none" or kind_b == "none":
        return None
    if kind_a == "circle" and kind_b == "circle":
        ca, ra = shape_a
        cb, rb = shape_b
        return float(np.linalg.norm(ca - cb) - (ra + rb))
    if kind_a == "poly" and kind_b == "poly":
        return float(
            _signed_gap_poly_poly(
                np.asarray(shape_a, dtype=float), np.asarray(shape_b, dtype=float)
            )
        )
    return None


def _object_rotation_matrix(obj: dict[str, Any], idx: int) -> np.ndarray:
    quaternions = obj.get("quaternions")
    if isinstance(quaternions, list) and idx < len(quaternions):
        quat = quaternions[idx]
        if isinstance(quat, list) and len(quat) >= 4:
            return _quat_to_rotmat_np(np.asarray(quat[:4], dtype=float))
    scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
    yaw = math.radians(_scalar_series_value(scalars.get("yaw_deg"), idx, 0.0))
    return np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _obb_penetration(obj_a: dict[str, Any], obj_b: dict[str, Any], idx: int) -> float:
    positions_a = obj_a.get("positions")
    positions_b = obj_b.get("positions")
    if not isinstance(positions_a, list) or not isinstance(positions_b, list):
        return 0.0
    if idx >= len(positions_a) or idx >= len(positions_b):
        return 0.0
    center_a = np.asarray(positions_a[idx], dtype=float)
    center_b = np.asarray(positions_b[idx], dtype=float)
    half_a = 0.5 * np.asarray(obj_a.get("size", [1.0, 1.0, 1.0]), dtype=float)[:3]
    half_b = 0.5 * np.asarray(obj_b.get("size", [1.0, 1.0, 1.0]), dtype=float)[:3]
    rot_a = _object_rotation_matrix(obj_a, idx)
    rot_b = _object_rotation_matrix(obj_b, idx)
    rel_rot = rot_a.T @ rot_b
    abs_rot = np.abs(rel_rot) + 1.0e-10
    translation = rot_a.T @ (center_b - center_a)
    overlaps: list[float] = []

    for i in range(3):
        radius_a = float(half_a[i])
        radius_b = float(np.dot(half_b, abs_rot[i, :]))
        overlap = radius_a + radius_b - abs(float(translation[i]))
        if overlap <= 0.0:
            return 0.0
        overlaps.append(overlap)

    for j in range(3):
        radius_a = float(np.dot(half_a, abs_rot[:, j]))
        radius_b = float(half_b[j])
        distance = abs(float(np.dot(translation, rel_rot[:, j])))
        overlap = radius_a + radius_b - distance
        if overlap <= 0.0:
            return 0.0
        overlaps.append(overlap)

    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            axis_norm = math.sqrt(max(0.0, 1.0 - float(rel_rot[i, j]) ** 2))
            if axis_norm < 1.0e-7:
                continue
            radius_a = float(half_a[i1] * abs_rot[i2, j] + half_a[i2] * abs_rot[i1, j])
            radius_b = float(half_b[j1] * abs_rot[i, j2] + half_b[j2] * abs_rot[i, j1])
            distance = abs(
                float(
                    translation[i2] * rel_rot[i1, j] - translation[i1] * rel_rot[i2, j]
                )
            )
            overlap = radius_a + radius_b - distance
            if overlap <= 0.0:
                return 0.0
            overlaps.append(overlap / axis_norm)

    return float(min(overlaps)) if overlaps else 0.0


def _metrics_from_trajectory(
    case: dict[str, Any], data: dict[str, Any], backend: str
) -> dict[str, Any]:
    times = np.asarray(data.get("times", []), dtype=float)
    objects = list(data.get("objects", []))
    g = abs(float(_gravity(case)[2]))

    speeds: list[np.ndarray] = []
    energy = np.zeros(len(times), dtype=float)
    max_height = 0.0
    max_speed = 0.0
    max_ke = 0.0
    for obj in objects:
        if obj.get("kind") == "segment":
            continue
        positions = np.asarray(obj.get("positions", []), dtype=float)
        if len(positions) != len(times) or len(times) == 0:
            continue
        mass = float(obj.get("mass", 0.0))
        vel = _compute_speed_series(times, positions)
        sp = np.linalg.norm(vel, axis=1)
        speeds.append(sp)
        max_speed = max(max_speed, float(np.max(sp)))
        max_height = max(max_height, float(np.max(positions[:, 2])))
        ke = 0.5 * mass * sp * sp
        height = positions[:, 2] - float(np.min(positions[:, 2]))
        pe = mass * g * np.maximum(0.0, height)
        energy += ke + pe
        max_ke = max(max_ke, float(np.max(ke)))
    energy_ref = max(abs(float(energy[0])) if len(energy) else 0.0, 1.0e-6)
    energy_drift = (
        100.0 * (float(np.max(energy) - np.min(energy)) / energy_ref)
        if len(energy)
        else 0.0
    )
    stacked_speed = (
        np.max(np.vstack(speeds), axis=0)
        if speeds
        else np.zeros(len(times), dtype=float)
    )
    settling_time = (
        _first_sustain_time(times, stacked_speed, threshold=0.05) if len(times) else 0.0
    )
    late_start = max(0, int(0.90 * len(stacked_speed)))
    late_max_speed = (
        float(np.max(stacked_speed[late_start:]))
        if late_start < len(stacked_speed)
        else 0.0
    )

    metrics: dict[str, Any] = {
        "backend": backend,
        "family": _family(case),
        "dim": _case_dim(case),
        "duration_sec": float(times[-1]) if len(times) else 0.0,
        "n_steps": int(len(times)),
        "max_speed": float(max_speed),
        "max_height": float(max_height),
        "max_kinetic_energy": float(max_ke),
        "energy_drift_pct": float(energy_drift),
        "settling_time": float(settling_time),
        "late_max_speed": float(late_max_speed),
        "contact_events": 0,
        "max_angle_deg": 0.0,
        "final_angle_deg": 0.0,
        "max_relative_angle_deg": 0.0,
        "angle_sign_changes": 0,
        "swing_span_deg": 0.0,
        "impact_speed": 0.0,
        "post_impact_speed": 0.0,
        "restitution_est": 0.0,
        "min_center_distance": 0.0,
        "min_projected_gap": 0.0,
        "min_signed_gap": 0.0,
        "max_penetration_depth": 0.0,
        "contact_gap_at_peak_impact": 0.0,
        "relative_speed_drop": 0.0,
        "relative_speed_drop_ratio": 0.0,
        "stack_toppled": False,
        "top_lateral_disp": 0.0,
        "top_height_drop": 0.0,
        "roll_capture_time": 0.0,
        "slip_distance": 0.0,
        "roll_distance": 0.0,
        "initial_slip_ratio": 0.0,
        "final_slip_ratio": 0.0,
        "lane_offset_final": 0.0,
        "bounce_count": 0,
        "max_rebound_height": 0.0,
        "precession_turns": 0.0,
        "orbit_radius_mean": 0.0,
        "tilt_span_deg": 0.0,
        "path_progress": 0.0,
        "path_completion_ratio": 0.0,
        "path_clearance": 0.0,
        "slider_stroke": 0.0,
        "rocker_span_deg": 0.0,
        "coupler_span": 0.0,
        "vehicle_pitch_span_deg": 0.0,
        "obstacle_clearance": 0.0,
        "granular_spread_gain": 0.0,
        "granular_com_drop": 0.0,
        "gear_ratio_est": 0.0,
        "gear_ratio_error": 0.0,
        "plate_center_displacement": 0.0,
    }
    for obj in objects:
        scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
        for name, raw in scalars.items():
            if not isinstance(raw, list) or not raw:
                continue
            vals = np.asarray(raw, dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                metrics[f"scalar_{name}_min"] = min(
                    float(metrics.get(f"scalar_{name}_min", np.inf)),
                    float(np.min(vals)),
                )
                metrics[f"scalar_{name}_max"] = max(
                    float(metrics.get(f"scalar_{name}_max", -np.inf)),
                    float(np.max(vals)),
                )

    family = _family(case)
    mech = _mech_cfg(case)
    if family == "massSpringDamper":
        gseries = (
            data.get("global_series")
            if isinstance(data.get("global_series"), dict)
            else {}
        )
        disp = (
            np.asarray(gseries.get("displacement_norm_m", []), dtype=float)
            if isinstance(gseries.get("displacement_norm_m"), list)
            else np.zeros((0,), dtype=float)
        )
        if len(disp):
            metrics["response_amplitude_m"] = float(np.max(disp) - np.min(disp))
            metrics["max_angle_deg"] = 0.0
            metrics["lane_offset_final"] = float(disp[-1])
        masses = [
            obj for obj in objects if str(obj.get("name", "")).startswith("mass_")
        ]
        if masses:
            metrics["contact_events"] = 1
        return metrics

    if family == "articulatedChain":
        link_angles: list[np.ndarray] = []
        for obj in objects:
            if str(obj.get("name", "")).startswith("link_"):
                pa = np.asarray(obj.get("points_a", []), dtype=float)
                pb = np.asarray(obj.get("points_b", []), dtype=float)
                if len(pa) == len(times) and len(pb) == len(times):
                    vec = pb - pa
                    link_angles.append(np.degrees(np.arctan2(vec[:, 0], -vec[:, 2])))
        if link_angles:
            tip_ang = link_angles[-1]
            metrics["max_angle_deg"] = float(np.max(np.abs(tip_ang)))
            metrics["final_angle_deg"] = float(tip_ang[-1])
            metrics["swing_span_deg"] = float(np.max(tip_ang) - np.min(tip_ang))
            if len(link_angles) > 1:
                rel = link_angles[-1] - link_angles[-2]
                metrics["max_relative_angle_deg"] = float(np.max(np.abs(rel)))
            sign = np.sign(tip_ang)
            sign[np.abs(tip_ang) < 1.0] = 0.0
            for idx in range(1, len(sign)):
                if sign[idx] == 0.0:
                    sign[idx] = sign[idx - 1]
            metrics["angle_sign_changes"] = (
                int(np.sum(np.abs(np.diff(sign)) > 1.5)) if len(sign) > 1 else 0
            )
        return metrics

    if family in {"unbalancedRotor3D", "gyroscopeRotor3D"}:
        rotor_disk = _find_object(objects, "rotor_disk")
        if rotor_disk:
            scalars = (
                rotor_disk.get("scalars")
                if isinstance(rotor_disk.get("scalars"), dict)
                else {}
            )
            disp = (
                np.asarray(scalars.get("lateral_displacement_m", []), dtype=float)
                if isinstance(scalars.get("lateral_displacement_m"), list)
                else np.zeros((0,), dtype=float)
            )
            if len(disp):
                metrics["response_amplitude_m"] = float(np.max(disp))
                metrics["orbit_radius_mean"] = float(np.mean(disp))
                metrics["max_angle_deg"] = 0.0
                metrics["precession_turns"] = 0.0
            return metrics

    if family == "pendulum":
        pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
        bob_names = [
            name
            for name in ("bob", "bob_1", "bob_2", "bob_3")
            if _find_object(objects, name)
        ]
        link_angles: list[np.ndarray] = []
        previous = np.repeat(pivot[None, :], len(times), axis=0)
        for name in bob_names:
            pos = np.asarray(
                _find_object(objects, name).get("positions", []), dtype=float
            )
            if len(pos) != len(times):
                continue
            link_angles.append(
                np.degrees(
                    np.arctan2(
                        pos[:, 0] - previous[:, 0], -(pos[:, 2] - previous[:, 2])
                    )
                )
            )
            previous = pos
        if link_angles:
            ang = link_angles[-1]
            metrics["max_angle_deg"] = float(np.max(np.abs(ang)))
            metrics["final_angle_deg"] = float(ang[-1])
            if len(link_angles) > 1:
                metrics["max_relative_angle_deg"] = float(
                    np.max(np.abs(link_angles[-1] - link_angles[-2]))
                )
            sign = np.sign(ang)
            sign[np.abs(ang) < 1.0] = 0.0
            for idx in range(1, len(sign)):
                if sign[idx] == 0.0:
                    sign[idx] = sign[idx - 1]
            metrics["angle_sign_changes"] = (
                int(np.sum(np.abs(np.diff(sign)) > 1.5)) if len(sign) > 1 else 0
            )
            metrics["swing_span_deg"] = float(np.max(ang) - np.min(ang))
        return metrics

    if family == "doublePendulum":
        p1 = np.asarray(
            _find_object(objects, "bob_1").get("positions", []), dtype=float
        )
        p2 = np.asarray(
            _find_object(objects, "bob_2").get("positions", []), dtype=float
        )
        pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
        if len(p1) and len(p2):
            ang1 = np.degrees(np.arctan2(p1[:, 0] - pivot[0], -(p1[:, 2] - pivot[2])))
            ang2 = np.degrees(np.arctan2(p2[:, 0] - p1[:, 0], -(p2[:, 2] - p1[:, 2])))
            metrics["max_angle_deg"] = float(np.max(np.abs(ang2)))
            metrics["final_angle_deg"] = float(ang2[-1])
            metrics["max_relative_angle_deg"] = float(np.max(np.abs(ang2 - ang1)))
            sign = np.sign(ang2)
            sign[np.abs(ang2) < 1.0] = 0.0
            for idx in range(1, len(sign)):
                if sign[idx] == 0.0:
                    sign[idx] = sign[idx - 1]
            metrics["angle_sign_changes"] = (
                int(np.sum(np.abs(np.diff(sign)) > 1.5)) if len(sign) > 1 else 0
            )
            metrics["swing_span_deg"] = float(np.max(ang2) - np.min(ang2))
        return metrics

    if family == "triplePendulum":
        p1 = np.asarray(
            _find_object(objects, "bob_1").get("positions", []), dtype=float
        )
        p2 = np.asarray(
            _find_object(objects, "bob_2").get("positions", []), dtype=float
        )
        p3 = np.asarray(
            _find_object(objects, "bob_3").get("positions", []), dtype=float
        )
        pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
        if len(p1) and len(p2) and len(p3):
            ang2 = np.degrees(np.arctan2(p2[:, 0] - p1[:, 0], -(p2[:, 2] - p1[:, 2])))
            ang3 = np.degrees(np.arctan2(p3[:, 0] - p2[:, 0], -(p3[:, 2] - p2[:, 2])))
            metrics["max_angle_deg"] = float(np.max(np.abs(ang3)))
            metrics["final_angle_deg"] = float(ang3[-1])
            metrics["max_relative_angle_deg"] = float(np.max(np.abs(ang3 - ang2)))
            sign = np.sign(ang3)
            sign[np.abs(ang3) < 1.0] = 0.0
            for idx in range(1, len(sign)):
                if sign[idx] == 0.0:
                    sign[idx] = sign[idx - 1]
            metrics["angle_sign_changes"] = (
                int(np.sum(np.abs(np.diff(sign)) > 1.5)) if len(sign) > 1 else 0
            )
            metrics["swing_span_deg"] = float(np.max(ang3) - np.min(ang3))
        return metrics

    if family in {"rigidCollision", "vehicleCollision3D"}:
        name_a = "body_a" if family == "rigidCollision" else "vehicle_a"
        name_b = "body_b" if family == "rigidCollision" else "vehicle_b"
        obj_a = _find_object(objects, name_a)
        obj_b = _find_object(objects, name_b)
        pa = np.asarray(obj_a.get("positions", []), dtype=float)
        pb = np.asarray(obj_b.get("positions", []), dtype=float)
        if len(pa) and len(pb):
            cfg_a = (
                mech.get("body_a")
                if family == "rigidCollision" and isinstance(mech.get("body_a"), dict)
                else {}
            )
            cfg_b = (
                mech.get("body_b")
                if family == "rigidCollision" and isinstance(mech.get("body_b"), dict)
                else {}
            )
            size_a = np.asarray(
                cfg_a.get("size", obj_a.get("size", [1.0, 1.0, 1.0])), dtype=float
            )
            size_b = np.asarray(
                cfg_b.get("size", obj_b.get("size", [1.0, 1.0, 1.0])), dtype=float
            )
            va = _compute_speed_series(times, pa)
            vb = _compute_speed_series(times, pb)
            rel = pa - pb
            sep = np.linalg.norm(rel, axis=1)
            rel_speed = np.linalg.norm(va - vb, axis=1)
            rel_normal_speed = np.abs(va[:, 0] - vb[:, 0])
            idx = int(np.argmin(sep))
            size_a = np.pad(
                size_a[:3], (0, max(0, 3 - len(size_a[:3]))), constant_values=1.0
            )
            size_b = np.pad(
                size_b[:3], (0, max(0, 3 - len(size_b[:3]))), constant_values=1.0
            )
            scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
            plane = str(scene.get("view_plane", "xy")).strip().lower()
            axis_idx = (0, 2) if plane == "xz" else (0, 1)
            half_extent_sum = 0.5 * size_a + 0.5 * size_b
            projected_gap = (
                np.abs(rel[:, [axis_idx[0], axis_idx[1]]])
                - half_extent_sum[[axis_idx[0], axis_idx[1]]]
            )
            projected_gap = np.maximum(projected_gap, 0.0)
            projected_sep = np.linalg.norm(projected_gap, axis=1)
            overlap = np.all(projected_gap <= 0.0, axis=1)
            threshold = float(
                np.linalg.norm(half_extent_sum[[axis_idx[0], axis_idx[1]]]) * 0.2
            )
            metrics["min_center_distance"] = float(np.min(sep))
            metrics["min_projected_gap"] = (
                float(np.min(projected_sep)) if len(projected_sep) else 0.0
            )
            metrics["impact_speed"] = (
                float(np.max(rel_normal_speed[: idx + 1])) if idx >= 0 else 0.0
            )
            post_start = max(idx + 3, int(0.80 * len(rel_normal_speed)))
            post_slice = rel_normal_speed[min(post_start, len(rel_normal_speed) - 1) :]
            metrics["post_impact_speed"] = (
                float(np.median(post_slice)) if len(post_slice) else 0.0
            )
            metrics["restitution_est"] = float(
                metrics["post_impact_speed"] / max(metrics["impact_speed"], 1.0e-6)
            )
            metrics["relative_speed_drop"] = float(
                max(0.0, metrics["impact_speed"] - metrics["post_impact_speed"])
            )
            metrics["relative_speed_drop_ratio"] = float(
                metrics["relative_speed_drop"] / max(metrics["impact_speed"], 1.0e-6)
            )
            signed_gaps = [
                gap
                for i in range(min(len(pa), len(pb)))
                if (gap := _signed_gap_objects(obj_a, obj_b, i, plane)) is not None
            ]
            if signed_gaps:
                metrics["min_signed_gap"] = float(min(signed_gaps))
                metrics["max_penetration_depth"] = float(max(0.0, -min(signed_gaps)))
                peak_idx = idx if len(signed_gaps) else 0
                peak_idx = max(0, min(peak_idx, len(signed_gaps) - 1))
                metrics["contact_gap_at_peak_impact"] = float(signed_gaps[peak_idx])
            metrics["contact_events"] = int(
                bool(np.any(overlap))
                or metrics["min_projected_gap"] <= threshold
                or metrics["min_signed_gap"] <= 0.0
            )
        return metrics

    if family in {"stackStability", "blockTower3D"}:
        level_count = max(2, int(mech.get("levels", 4)))
        layers = max(1, int(mech.get("layers", 1))) if family == "blockTower3D" else 1
        size = np.asarray(
            mech.get(
                "block_size" if family == "blockTower3D" else "box_size",
                [0.35, 0.35, 0.28],
            ),
            dtype=float,
        )
        if family == "blockTower3D":
            top_objects = [
                _find_object(objects, f"tower_block_{level_count - 1}_{layer}")
                for layer in range(layers)
            ]
        else:
            top_objects = [_find_object(objects, f"box_{level_count - 1}")]
        lateral_peaks: list[float] = []
        drop_peaks: list[float] = []
        for top_obj in top_objects:
            top = np.asarray(top_obj.get("positions", []), dtype=float)
            if not len(top):
                continue
            lateral_peaks.append(
                float(np.max(np.linalg.norm(top[:, :2] - top[0, :2], axis=1)))
            )
            drop_peaks.append(max(0.0, float(top[0, 2] - np.min(top[:, 2]))))
        if lateral_peaks:
            metrics["top_lateral_disp"] = float(max(lateral_peaks))
            metrics["top_height_drop"] = float(max(drop_peaks))
            metrics["stack_toppled"] = bool(
                metrics["top_lateral_disp"] > 0.6 * size[0]
                or metrics["top_height_drop"] > 0.4 * size[2]
            )
        impactor_obj = _find_object(objects, "impactor")
        impactor = np.asarray(impactor_obj.get("positions", []), dtype=float)
        impactor_size = np.asarray(
            impactor_obj.get("size", [0.2, 0.2, 0.2]), dtype=float
        )
        if len(impactor):
            contacted = False
            for obj in objects:
                name = str(obj.get("name", ""))
                if name == "impactor" or not (
                    name.startswith("box_") or name.startswith("tower_block_")
                ):
                    continue
                positions = np.asarray(obj.get("positions", []), dtype=float)
                if len(positions) != len(impactor):
                    continue
                obj_size = np.asarray(obj.get("size", size.tolist()), dtype=float)
                half_sum = 0.5 * (impactor_size[:3] + obj_size[:3])
                delta = np.abs(impactor[:, :3] - positions[:, :3])
                if bool(np.any(np.all(delta <= half_sum + 0.006, axis=1))):
                    contacted = True
                    break
            metrics["contact_events"] = int(contacted)
        box_names = [
            str(obj.get("name", ""))
            for obj in objects
            if obj.get("kind") == "box" and str(obj.get("name", "")) != "impactor"
        ]
        max_pen = 0.0
        for i, name_i in enumerate(box_names):
            obj_i = _find_object(objects, name_i)
            for name_j in box_names[i + 1 :]:
                obj_j = _find_object(objects, name_j)
                n = min(
                    len(obj_i.get("positions", [])), len(obj_j.get("positions", []))
                )
                for idx in range(n):
                    max_pen = max(max_pen, _obb_penetration(obj_i, obj_j, idx))
        metrics["max_penetration_depth"] = float(max_pen)
        return metrics

    if family == "dropWeightImpactSMC":
        max_contact_force = 0.0
        exports = data.get("exports") if isinstance(data.get("exports"), list) else []
        for export in exports:
            if (
                not isinstance(export, dict)
                or str(export.get("file_name", "")) != "contact_force.csv"
            ):
                continue
            header = [str(h) for h in (export.get("header") or [])]
            rows = export.get("rows") if isinstance(export.get("rows"), list) else []
            try:
                force_idx = header.index("contact_force_n")
            except ValueError:
                force_idx = 1
            for row in rows:
                if not isinstance(row, list) or len(row) <= force_idx:
                    continue
                try:
                    max_contact_force = max(max_contact_force, float(row[force_idx]))
                except Exception:
                    continue
        metrics["max_contact_force_n"] = float(max_contact_force)
        impactor_obj = _find_object(objects, "impactor")
        impactor = np.asarray(impactor_obj.get("positions", []), dtype=float)
        radius = float(
            (impactor_obj.get("size") or [mech.get("sphere_radius_m", 0.02)])[0]
        )
        plate_objects = [
            obj for obj in objects if str(obj.get("name", "")).startswith("plate_elem_")
        ]
        if len(impactor) and plate_objects:
            center_obj = min(
                plate_objects,
                key=lambda obj: float(
                    np.linalg.norm(
                        np.asarray(
                            obj.get("positions", [[0.0, 0.0, 0.0]])[0], dtype=float
                        )[:2]
                    )
                ),
            )
            center = np.asarray(center_obj.get("positions", []), dtype=float)
            if len(center) == len(impactor):
                gap = impactor[:, 2] - radius - center[:, 2]
                velocity = _compute_speed_series(times, impactor)
                vertical_velocity = velocity[:, 2]
                impact_idx = int(np.argmin(gap))
                metrics["contact_events"] = int(bool(np.any(gap <= 0.0)))
                metrics["min_signed_gap"] = float(np.min(gap))
                metrics["max_penetration_depth"] = float(max(0.0, -np.min(gap)))
                metrics["impact_speed"] = float(
                    max(0.0, -np.min(vertical_velocity[: impact_idx + 1]))
                )
                center_z0 = float(center[0, 2])
                metrics["plate_center_displacement"] = float(
                    np.max(np.abs(center[:, 2] - center_z0))
                )
        if max_contact_force > 1.0:
            metrics["contact_events"] = 1
        return metrics

    if family == "vehicleLaneChange3D":
        vehicle = np.asarray(
            _find_object(objects, "vehicle").get("positions", []), dtype=float
        )
        if len(vehicle):
            metrics["lane_offset_final"] = float(vehicle[-1, 1] - vehicle[0, 1])
            metrics["lane_offset_peak_abs"] = float(
                np.max(np.abs(vehicle[:, 1] - vehicle[0, 1]))
            )
        return metrics

    if family == "vehicleObstacleCrossing3D":
        vehicle_obj = _find_object(objects, "vehicle")
        vehicle = np.asarray(vehicle_obj.get("positions", []), dtype=float)
        if len(vehicle):
            metrics["lane_offset_final"] = float(vehicle[-1, 0] - vehicle[0, 0])
            metrics["max_height"] = float(np.max(vehicle[:, 2]))
            metrics["obstacle_clearance"] = float(vehicle[-1, 0] - vehicle[0, 0])
            if len(vehicle) > 2:
                vx = np.gradient(vehicle[:, 0], times, edge_order=1)
                vz = np.gradient(vehicle[:, 2], times, edge_order=1)
                pitch = np.degrees(np.arctan2(vz, np.maximum(np.abs(vx), 1.0e-6)))
                metrics["vehicle_pitch_span_deg"] = float(np.max(pitch) - np.min(pitch))
            obs_names = [
                str(obj.get("name", ""))
                for obj in objects
                if str(obj.get("name", "")).startswith("obstacle_")
            ]
            signed_gaps: list[float] = []
            for obs_name in obs_names:
                obs_obj = _find_object(objects, obs_name)
                if str(obs_obj.get("render_shape", "")).strip().lower() == "speed_bump":
                    continue
                n = min(
                    len(vehicle_obj.get("positions", [])),
                    len(obs_obj.get("positions", [])),
                )
                for idx in range(n):
                    gap = _signed_gap_objects(vehicle_obj, obs_obj, idx, "xz")
                    if gap is not None:
                        signed_gaps.append(float(gap))
            if signed_gaps:
                metrics["min_signed_gap"] = float(min(signed_gaps))
                metrics["max_penetration_depth"] = float(max(0.0, -min(signed_gaps)))
        return metrics

    if family == "rampJump3D":
        vehicle = np.asarray(
            _find_object(objects, "vehicle").get("positions", []), dtype=float
        )
        if len(vehicle):
            metrics["lane_offset_final"] = float(vehicle[-1, 0] - vehicle[0, 0])
        return metrics

    if family == "sphereDrop3D":
        ball = _find_object(objects, "ball")
        pos = np.asarray(ball.get("positions", []), dtype=float)
        radius = float(
            (ball.get("size") or [mech.get("sphere", {}).get("radius", 0.2)])[0]
        )
        if len(pos):
            z = pos[:, 2]
            contact = z <= (radius + 0.03 * max(radius, 1.0))
            sep = z - radius
            metrics["contact_events"] = (
                int(np.sum((~contact[:-1]) & contact[1:]))
                if len(contact) > 1
                else int(bool(contact[0]))
            )
            first_contact = int(np.argmax(contact)) if np.any(contact) else -1
            if first_contact >= 0 and first_contact < len(z) - 2:
                peaks: list[float] = []
                for i in range(first_contact + 1, len(z) - 1):
                    if z[i - 1] <= z[i] >= z[i + 1]:
                        peaks.append(float(z[i] - radius))
                metrics["bounce_count"] = int(len(peaks))
                metrics["max_rebound_height"] = float(max(peaks) if peaks else 0.0)
            vel = _compute_speed_series(times, pos)
            vz = vel[:, 2] if len(vel) else np.zeros((0,), dtype=float)
            if first_contact > 0 and len(vz):
                metrics["impact_speed"] = float(abs(vz[max(0, first_contact - 1)]))
                post_idx = min(len(vz) - 1, first_contact + 2)
                metrics["post_impact_speed"] = float(abs(vz[post_idx]))
                metrics["restitution_est"] = float(
                    metrics["post_impact_speed"] / max(metrics["impact_speed"], 1.0e-6)
                )
            metrics["min_signed_gap"] = float(np.min(sep))
            metrics["max_penetration_depth"] = float(max(0.0, -np.min(sep)))
        return metrics

    if family in {"gyroscopeTop3D", "gyroscopeRotor3D"}:
        rotor = np.asarray(
            _find_object(objects, "rotor").get("positions", []), dtype=float
        )
        pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
        shaft_length = max(float(mech.get("shaft_length", 0.65)), 1.0e-6)
        if len(rotor):
            rel = rotor - pivot
            rel_xy = np.linalg.norm(rel[:, :2], axis=1)
            az = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
            tilt = np.degrees(np.arctan2(rel_xy, np.maximum(rel[:, 2], 1.0e-9)))
            metrics["orbit_radius_mean"] = float(np.mean(rel_xy))
            metrics["precession_turns"] = float(abs(az[-1] - az[0]) / (2.0 * math.pi))
            metrics["tilt_span_deg"] = float(np.max(tilt) - np.min(tilt))
            metrics["max_angle_deg"] = float(np.max(tilt))
            metrics["swing_span_deg"] = float(2.0 * np.max(rel_xy) / shaft_length)
        return metrics

    if family == "rollingBallMaze3D":
        ball = np.asarray(
            _find_object(objects, "ball").get("positions", []), dtype=float
        )
        guide_path = _polyline_points(
            mech.get("guide_path"),
            default=[[-8.0, -6.0], [-8.0, 2.0], [-3.4, 2.0], [-3.4, -1.0], [1.6, -1.0]],
        )
        _, total_len = _polyline_arclength(guide_path)
        if len(ball):
            progress_samples = [_polyline_progress(p[:2], guide_path) for p in ball]
            progress = max((item[0] for item in progress_samples), default=0.0)
            clearance = min((item[1] for item in progress_samples), default=0.0)
            metrics["path_progress"] = float(progress)
            metrics["path_completion_ratio"] = float(progress / max(total_len, 1.0e-6))
            metrics["path_clearance"] = float(clearance)
            metrics["lane_offset_final"] = float(
                np.linalg.norm(ball[-1, :2] - ball[0, :2])
            )
        return metrics

    if family == "sliderCrank3D":
        slider = np.asarray(
            _find_object(objects, "slider").get("positions", []), dtype=float
        )
        if len(slider):
            metrics["slider_stroke"] = float(
                np.max(slider[:, 0]) - np.min(slider[:, 0])
            )
            metrics["lane_offset_final"] = float(slider[-1, 0] - slider[0, 0])
        return metrics

    if family == "fourBarLinkage3D":
        coupler = np.asarray(
            _find_object(objects, "coupler_marker").get("positions", []), dtype=float
        )
        if not len(coupler):
            coupler_obj = _find_object(objects, "coupler")
            pa_c = np.asarray(coupler_obj.get("points_a", []), dtype=float)
            pb_c = np.asarray(coupler_obj.get("points_b", []), dtype=float)
            if len(pa_c) and len(pb_c):
                coupler = 0.5 * (pa_c + pb_c)
        rocker = _find_object(objects, "rocker")
        pa = np.asarray(rocker.get("points_a", []), dtype=float)
        pb = np.asarray(rocker.get("points_b", []), dtype=float)
        if len(coupler):
            metrics["coupler_span"] = float(
                np.max(coupler[:, 1]) - np.min(coupler[:, 1])
            )
        if len(pa) and len(pb):
            vec = pb - pa
            ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
            metrics["rocker_span_deg"] = float(np.max(ang) - np.min(ang))
            metrics["max_angle_deg"] = float(np.max(np.abs(ang)))
        return metrics

    if family == "granularCollapse3D":
        grains = [
            np.asarray(obj.get("positions", []), dtype=float)
            for obj in objects
            if str(obj.get("name", "")).startswith("grain_")
        ]
        valid = [g for g in grains if len(g)]
        if valid:
            initial = np.vstack([g[0] for g in valid])
            final = np.vstack([g[-1] for g in valid])
            metrics["granular_spread_gain"] = float(
                (np.max(final[:, 0]) - np.min(final[:, 0]))
                - (np.max(initial[:, 0]) - np.min(initial[:, 0]))
            )
            metrics["granular_com_drop"] = float(
                np.mean(initial[:, 2]) - np.mean(final[:, 2])
            )
            metrics["contact_events"] = int(metrics["granular_spread_gain"] > 0.1)
        return metrics

    if family == "gearPair3D":
        gear_a = _find_object(objects, "gear_a")
        gear_b = _find_object(objects, "gear_b")
        scalars_a = (
            gear_a.get("scalars") if isinstance(gear_a.get("scalars"), dict) else {}
        )
        scalars_b = (
            gear_b.get("scalars") if isinstance(gear_b.get("scalars"), dict) else {}
        )
        yaw_a = np.asarray(
            scalars_a.get("yaw_deg", scalars_a.get("spin_deg", [])), dtype=float
        )
        yaw_b = np.asarray(
            scalars_b.get("yaw_deg", scalars_b.get("spin_deg", [])), dtype=float
        )
        omega_a_series = np.asarray(scalars_a.get("omega_spin_rad_s", []), dtype=float)
        omega_b_series = np.asarray(scalars_b.get("omega_spin_rad_s", []), dtype=float)
        radius_a = float(mech.get("radius_a", 0.32))
        radius_b = float(mech.get("radius_b", 0.58))
        teeth_a = max(1, int(mech.get("teeth_a", max(12, int(round(36.0 * radius_a))))))
        teeth_b = max(
            1,
            int(
                mech.get(
                    "teeth_b",
                    max(12, int(round((radius_b / max(radius_a, 1.0e-6)) * teeth_a))),
                )
            ),
        )
        expect = teeth_a / max(float(teeth_b), 1.0e-6)
        if len(yaw_a) >= 2 and len(yaw_b) >= 2:
            yaw_a_u = np.unwrap(np.radians(yaw_a))
            yaw_b_u = np.unwrap(np.radians(yaw_b))
            da = abs(float(np.degrees(yaw_a_u[-1] - yaw_a_u[0])))
            db = abs(float(np.degrees(yaw_b_u[-1] - yaw_b_u[0])))
            metrics["max_angle_deg"] = float(max(da, db))
            if len(omega_a_series) != len(times):
                omega_a_series = np.gradient(yaw_a_u, times, edge_order=1)
            if len(omega_b_series) != len(times):
                omega_b_series = np.gradient(yaw_b_u, times, edge_order=1)
            warmup = max(1, len(times) // 10)
            omega_a_abs = float(np.median(np.abs(omega_a_series[warmup:])))
            omega_b_abs = float(np.median(np.abs(omega_b_series[warmup:])))
            est = omega_b_abs / max(omega_a_abs, 1.0e-6)
            metrics["gear_ratio_est"] = float(est)
            metrics["gear_ratio_error"] = float(abs(est - expect))
            metrics["max_speed"] = float(
                max(
                    np.max(np.abs(omega_a_series) * radius_a),
                    np.max(np.abs(omega_b_series) * radius_b),
                )
            )
        return metrics

    if family == "ballChutePile3D":
        balls = [obj for obj in objects if str(obj.get("name", "")).startswith("ball_")]
        visible_counts: Optional[np.ndarray] = None
        impact_count = 0
        for obj in balls:
            scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
            vis = (
                np.asarray(scalars.get("visible", []), dtype=float)
                if isinstance(scalars.get("visible"), list)
                else np.zeros((0,), dtype=float)
            )
            cfm = (
                np.asarray(scalars.get("contact_force_mag", []), dtype=float)
                if isinstance(scalars.get("contact_force_mag"), list)
                else np.zeros((0,), dtype=float)
            )
            if len(vis):
                mask = (vis > 0.5).astype(float)
                if visible_counts is None:
                    visible_counts = np.zeros_like(mask)
                n = min(len(visible_counts), len(mask))
                visible_counts[:n] += mask[:n]
            if len(cfm) > 1:
                active = cfm > 1.0e-6
                impact_count += int(np.sum((~active[:-1]) & active[1:]))
        metrics["contact_events"] = int(
            max(metrics.get("contact_events", 0), impact_count)
        )
        metrics["active_ball_peak"] = (
            int(np.max(visible_counts))
            if isinstance(visible_counts, np.ndarray) and len(visible_counts)
            else 0
        )
        gseries = (
            data.get("global_series")
            if isinstance(data.get("global_series"), dict)
            else {}
        )
        cps = (
            np.asarray(gseries.get("contact_pairs", []), dtype=float)
            if isinstance(gseries.get("contact_pairs"), list)
            else np.zeros((0,), dtype=float)
        )
        if len(cps):
            metrics["contact_pairs_peak"] = float(np.max(cps))
        return metrics

    if family in {"rollingVsSliding", "rollingSlidingTransition"}:
        roller = _find_object(objects, "roller")
        pos = np.asarray(roller.get("positions", []), dtype=float)
        omega = np.asarray(
            (roller.get("scalars") or {}).get("omega_y", []), dtype=float
        )
        radius = float(mech.get("radius", 0.25))
        if len(pos):
            vel = _compute_speed_series(times, pos)
            slope = math.radians(float(mech.get("slope_angle_deg", 0.0)))
            tangent = np.asarray([math.cos(slope), 0.0, -math.sin(slope)], dtype=float)
            vx = vel @ tangent
            if len(omega) != len(times):
                omega = np.zeros(len(times), dtype=float)
            slip = np.abs(vx - omega * radius)
            denom = np.maximum(np.maximum(np.abs(vx), np.abs(omega * radius)), 1.0e-6)
            slip_ratio = slip / denom
            capture_time = _first_sustain_time(times, slip_ratio, threshold=0.05)
            idx = min(len(times) - 1, int(np.searchsorted(times, capture_time)))
            metrics["roll_capture_time"] = float(capture_time)
            metrics["slip_distance"] = float(pos[idx, 0] - pos[0, 0])
            metrics["roll_distance"] = float(pos[-1, 0] - pos[0, 0])
            metrics["initial_slip_ratio"] = float(slip_ratio[0])
            metrics["final_slip_ratio"] = float(slip_ratio[-1])
        return metrics

    return metrics
