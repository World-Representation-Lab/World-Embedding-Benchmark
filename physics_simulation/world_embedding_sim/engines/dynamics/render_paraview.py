#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import vtk  # type: ignore


BG_RGB = [0x72 / 255.0, 0x6F / 255.0, 0x69 / 255.0]
TEXT_RGB = [0.96, 0.96, 0.94]
PENDULUM_FAMILIES = {"pendulum", "doublePendulum", "triplePendulum"}
_BOX_FACE_POLYS = [
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
]
_BOX_LOCAL_CORNERS = [
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _vec3(values: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        return float(values[0]), float(values[1]), float(values[2])
    return default


def _series_vec3(values: Any, idx: int, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if isinstance(values, list) and values:
        j = min(max(idx, 0), len(values) - 1)
        return _vec3(values[j], default)
    return default


def _series_scalar(values: Any, idx: int, default: float = 0.0) -> float:
    if isinstance(values, list) and values:
        j = min(max(idx, 0), len(values) - 1)
        try:
            return float(values[j])
        except Exception:
            return float(default)
    try:
        return float(values)
    except Exception:
        return float(default)


def _series_scalar_values(values: Any, idx: int, count: int, default: float = 0.0) -> list[float]:
    if count <= 0:
        return []
    if isinstance(values, list) and values:
        j = min(max(idx, 0), len(values) - 1)
        frame = values[j]
        if isinstance(frame, (list, tuple)):
            out: list[float] = []
            for v in frame[:count]:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float(default))
            if len(out) < count:
                out.extend([out[-1] if out else float(default)] * (count - len(out)))
            return out
        try:
            val = float(frame)
        except Exception:
            val = float(default)
        return [val for _ in range(count)]
    try:
        val = float(values)
    except Exception:
        val = float(default)
    return [val for _ in range(count)]


def _nice_label(metric: str) -> str:
    mapping = {
        "speed_mag": "Speed (m/s)",
        "mass_speed_m_s": "Block Speed (m/s)",
        "velocity_magnitude": "Velocity Magnitude (m/s)",
        "slip_ratio": "Slip Ratio (-)",
        "slip_speed_m_s": "Slip Speed (m/s)",
        "signed_slip_m_s": "Signed Slip Speed (m/s)",
        "friction_power_w": "Frictional Power (W)",
        "surface_point_speed_m_s": "Speed m/s",
        "surface_speed_m_s": "Surface Speed (m/s)",
        "u_mag": "Displacement Magnitude (m)",
        "displacement_m": "Displacement (m)",
        "lateral_displacement_m": "Lateral Displacement (m)",
        "bearing_force_n": "Bearing Force (N)",
        "critical_ratio": "Spin / Critical Speed (-)",
        "von_mises_mpa": "Von Mises (MPa)",
        "sigma11_mpa": "Stress (MPa)",
        "PEEQ": "PEEQ",
        "D": "Damage",
        "Uz": "Uz (m)",
        "visual_scalar": "Field Value",
        "curve_m": "Curve (m)",
        "omega_spin_rad_s": "Angular Speed (rad/s)",
        "omega_spin_abs_rad_s": "Angular Speed (rad/s)",
    }
    return mapping.get(metric, metric)


def _quat_to_rotmat(quat: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = quat
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1.0e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _rot_y(angle_deg: float) -> list[list[float]]:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(angle_deg: float) -> list[list[float]]:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _apply_rot(rot: list[list[float]], p: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2],
        rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2],
        rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2],
    )


def _box_corners(pos: tuple[float, float, float], size: tuple[float, float, float], rot: list[list[float]]) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for lx, ly, lz in _BOX_LOCAL_CORNERS:
        local = (lx * size[0], ly * size[1], lz * size[2])
        wx, wy, wz = _apply_rot(rot, local)
        out.append((pos[0] + wx, pos[1] + wy, pos[2] + wz))
    return out


def _round_key(p: tuple[float, float, float], ndigits: int = 6) -> tuple[float, float, float]:
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits), round(float(p[2]), ndigits))


def _rotate_xz(x: float, z: float, angle_rad: float) -> tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (c * x - s * z, s * x + c * z)


def _involute_point(base_radius: float, t: float) -> tuple[float, float]:
    return (
        base_radius * (math.cos(t) + t * math.sin(t)),
        base_radius * (math.sin(t) - t * math.cos(t)),
    )


def _gear_outline_points(
    outer_radius: float,
    teeth: int,
    hole_radius: float,
    pressure_angle_deg: float = 20.0,
    samples_flank: int = 18,
    samples_arc: int = 16,
) -> list[tuple[float, float]]:
    teeth = max(int(teeth), 6)
    pressure_angle = math.radians(float(pressure_angle_deg))
    module = 2.0 * outer_radius / max(float(teeth + 2), 1.0)
    pitch_radius = max(hole_radius * 1.5, outer_radius - module)
    addendum = outer_radius - pitch_radius
    dedendum = 1.25 * module
    root_radius = max(hole_radius * 1.22, pitch_radius - dedendum)
    base_radius = min(pitch_radius, max(root_radius, pitch_radius * math.cos(pressure_angle)))
    inv_alpha = math.tan(pressure_angle) - pressure_angle
    half_tooth_angle = math.pi / (2.0 * teeth)

    r_start = max(base_radius, root_radius)
    t_start = math.sqrt(max((r_start / max(base_radius, 1.0e-9)) ** 2 - 1.0, 0.0))
    t_outer = math.sqrt(max((outer_radius / max(base_radius, 1.0e-9)) ** 2 - 1.0, 0.0))

    teeth_data: list[dict[str, Any]] = []
    for k in range(teeth):
        center_ang = 2.0 * math.pi * float(k) / float(teeth)
        left_rot = center_ang - half_tooth_angle - inv_alpha
        right_rot = center_ang + half_tooth_angle + inv_alpha

        left_flank: list[tuple[float, float]] = []
        right_flank: list[tuple[float, float]] = []
        for i in range(samples_flank):
            u = 0.0 if samples_flank <= 1 else float(i) / float(samples_flank - 1)
            t = t_start + u * (t_outer - t_start)
            px, pz = _involute_point(base_radius, t)
            left_flank.append(_rotate_xz(px, pz, left_rot))
            right_flank.append(_rotate_xz(px, -pz, right_rot))

        left_outer_ang = math.atan2(left_flank[-1][1], left_flank[-1][0])
        right_outer_ang = math.atan2(right_flank[-1][1], right_flank[-1][0])
        top_arc: list[tuple[float, float]] = []
        span = right_outer_ang - left_outer_ang
        if span <= 0.0:
            span += 2.0 * math.pi
        for i in range(1, samples_arc):
            a = left_outer_ang + span * float(i) / float(samples_arc)
            top_arc.append((outer_radius * math.cos(a), outer_radius * math.sin(a)))

        teeth_data.append(
            {
                "left_flank": left_flank,
                "right_flank": right_flank,
                "left_root_ang": math.atan2(left_flank[0][1], left_flank[0][0]),
                "right_root_ang": math.atan2(right_flank[0][1], right_flank[0][0]),
                "top_arc": top_arc,
            }
        )

    outline: list[tuple[float, float]] = []
    for i, tooth in enumerate(teeth_data):
        prev = teeth_data[i - 1]
        a0 = prev["right_root_ang"]
        a1 = tooth["left_root_ang"]
        span = a1 - a0
        if span <= 0.0:
            span += 2.0 * math.pi
        if not outline:
            outline.append((root_radius * math.cos(a0), root_radius * math.sin(a0)))
        for j in range(1, samples_arc + 1):
            a = a0 + span * float(j) / float(samples_arc + 1)
            outline.append((root_radius * math.cos(a), root_radius * math.sin(a)))
        outline.extend(tooth["left_flank"])
        outline.extend(tooth["top_arc"])
        outline.extend(reversed(tooth["right_flank"]))
    return outline


def _build_gear_polydata(
    outer_radius: float,
    teeth: int,
    hole_radius: float,
    pressure_angle_deg: float = 20.0,
    keyway: bool = False,
) -> vtk.vtkPolyData:
    outline = _gear_outline_points(outer_radius, teeth, hole_radius, pressure_angle_deg=pressure_angle_deg)
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    outer_loop = vtk.vtkPolyLine()
    outer_loop.GetPointIds().SetNumberOfIds(len(outline) + 1)
    for i, (x, z) in enumerate(outline):
        pts.InsertNextPoint(float(x), 0.0, float(z))
        outer_loop.GetPointIds().SetId(i, i)
    outer_loop.GetPointIds().SetId(len(outline), 0)
    lines.InsertNextCell(outer_loop)

    inner_samples = max(48, 2 * int(teeth) * 4)
    inner_offset = len(outline)
    inner_loop = vtk.vtkPolyLine()
    inner_outline: list[tuple[float, float]] = []
    if keyway:
        slot_half_width = min(hole_radius * 0.52, max(0.016, hole_radius * 0.46))
        join_x = math.sqrt(max(hole_radius * hole_radius - slot_half_width * slot_half_width, 1.0e-12))
        slot_depth = max(0.022, hole_radius * 0.78)
        slot_x = join_x + slot_depth
        join_angle = math.atan2(slot_half_width, join_x)
        arc_samples = max(inner_samples, 72)
        for i in range(arc_samples):
            u = 0.0 if arc_samples <= 1 else float(i) / float(arc_samples - 1)
            a = join_angle + u * (-(2.0 * math.pi - join_angle) - join_angle)
            inner_outline.append((hole_radius * math.cos(a), hole_radius * math.sin(a)))
        inner_outline.append((slot_x, -slot_half_width))
        inner_outline.append((slot_x, slot_half_width))
    else:
        for i in range(inner_samples):
            a = 2.0 * math.pi * float(inner_samples - i) / float(inner_samples)
            inner_outline.append((hole_radius * math.cos(a), hole_radius * math.sin(a)))
    inner_loop.GetPointIds().SetNumberOfIds(len(inner_outline) + 1)
    for i, (x, z) in enumerate(inner_outline):
        pts.InsertNextPoint(float(x), 0.0, float(z))
        inner_loop.GetPointIds().SetId(i, inner_offset + i)
    inner_loop.GetPointIds().SetId(len(inner_outline), inner_offset)
    lines.InsertNextCell(inner_loop)

    contour = vtk.vtkPolyData()
    contour.SetPoints(pts)
    contour.SetLines(lines)
    tri = vtk.vtkContourTriangulator()
    tri.SetInputData(contour)
    tri.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def _gear_keyway_enabled(name: str) -> bool:
    return str(name).startswith("gear_")


def _build_ctf(vmin: float, vmax: float) -> vtk.vtkColorTransferFunction:
    if not math.isfinite(vmin):
        vmin = 0.0
    if not math.isfinite(vmax):
        vmax = 1.0
    if vmax <= vmin + 1.0e-12:
        vmax = vmin + 1.0
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToRGB()
    pts = [
        (0.00, (0.18995, 0.07176, 0.23217)),
        (0.25, (0.25107, 0.25237, 0.63374)),
        (0.50, (0.27628, 0.59818, 0.93614)),
        (0.75, (0.97323, 0.81121, 0.15129)),
        (1.00, (0.47960, 0.01583, 0.01055)),
    ]
    span = vmax - vmin
    for t, rgb in pts:
        x = vmin + t * span
        ctf.AddRGBPoint(float(x), float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return ctf


def _augment_speed_scalars(objects: list[dict[str, Any]], times: list[float], family: str) -> None:
    if len(times) < 2:
        return
    n = len(times)
    dts = [max(1.0e-9, float(times[min(i + 1, n - 1)]) - float(times[max(i - 1, 0)])) for i in range(n)]
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        scalars = obj.setdefault("scalars", {})
        if not isinstance(scalars, dict):
            scalars = {}
            obj["scalars"] = scalars
        if isinstance(scalars.get("speed_mag"), list) and len(scalars["speed_mag"]) == n:
            continue
        centers: list[tuple[float, float, float]] = []
        if isinstance(obj.get("positions"), list) and len(obj["positions"]) == n:
            centers = [_vec3(p) for p in obj["positions"]]
        elif isinstance(obj.get("points_a"), list) and isinstance(obj.get("points_b"), list) and len(obj["points_a"]) == n and len(obj["points_b"]) == n:
            for i in range(n):
                pa = _vec3(obj["points_a"][i])
                pb = _vec3(obj["points_b"][i])
                centers.append((0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])))
        if len(centers) != n:
            continue
        speed = []
        for i in range(n):
            if i == 0:
                p0, p1 = centers[0], centers[min(1, n - 1)]
            elif i == n - 1:
                p0, p1 = centers[max(0, n - 2)], centers[n - 1]
            else:
                p0, p1 = centers[i - 1], centers[i + 1]
            dt = dts[i]
            vx = (p1[0] - p0[0]) / dt
            vy = (p1[1] - p0[1]) / dt
            vz = (p1[2] - p0[2]) / dt
            speed.append(float(math.sqrt(vx * vx + vy * vy + vz * vz)))
        scalars["speed_mag"] = speed
        kind = str(obj.get("kind", "")).strip().lower()
        if str(obj.get("color_mode", "")).strip().lower() == "fixed":
            obj.pop("default_metric", None)
            continue
        if family in PENDULUM_FAMILIES and kind in {"capsule", "segment"}:
            obj.pop("default_metric", None)
            continue
        if not obj.get("color_metric") and kind in {"circle", "box", "capsule", "segment"}:
            obj.setdefault("default_metric", "speed_mag")


def _choose_metric(objects: list[dict[str, Any]]) -> tuple[Optional[str], tuple[float, float]]:
    preferred: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        metric = obj.get("color_metric") or obj.get("default_metric")
        if isinstance(metric, str) and metric:
            preferred.append(metric)
    fallback = ["speed_mag", "velocity_magnitude", "u_mag", "displacement_m", "von_mises_mpa", "sigma11_mpa", "PEEQ", "D", "Uz", "visual_scalar", "omega_spin_abs_rad_s", "omega_spin_rad_s"]
    seen: list[str] = []
    for name in preferred + fallback:
        if name not in seen:
            seen.append(name)
    for metric in seen:
        vals: list[float] = []
        for obj in objects:
            scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
            raw = scalars.get(metric)
            if isinstance(raw, list):
                for v in raw:
                    if isinstance(v, (list, tuple)):
                        candidates = v
                    else:
                        candidates = [v]
                    for item in candidates:
                        try:
                            fv = float(item)
                        except Exception:
                            continue
                        if math.isfinite(fv):
                            vals.append(fv)
        if vals:
            vmin = min(vals)
            vmax = max(vals)
            if abs(vmin) <= max(1.0e-9, 0.01 * max(abs(vmax), 1.0)):
                vmin = 0.0
            if vmax <= vmin + 1.0e-12:
                vmax = vmin + 1.0
            return metric, (float(vmin), float(vmax))
    return None, (0.0, 1.0)


def _effective_target_seconds(family: str, duration: float, requested: float) -> float:
    if family == "rollingSlidingTransition":
        return min(float(requested), max(4.2, min(6.5, 0.95 * float(duration))))
    if family == "dropWeightImpactSMC":
        return max(6.0, min(8.0, 1.55 * float(duration)))
    if family == "lemaitreDamageBar":
        return min(float(requested), 6.0)
    if family == "sphereDrop3D":
        return max(3.0, min(4.0, 0.65 * float(duration)))
    if family == "flexibleCrankSlider":
        return min(float(requested), max(4.0, 2.5 * float(duration)))
    if family == "stressWavePropagation":
        return min(float(requested), 4.0)
    if family == "stackStability":
        return min(float(requested), 3.0)
    if family == "blockTower3D":
        return min(float(requested), 8.0)
    return float(requested)


def _selected_indices(n: int, fps: int, target_seconds: float) -> list[int]:
    if n <= 1:
        return [0]
    target_frames = min(n, max(2, int(round(float(fps) * float(target_seconds)))))
    if target_frames >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (target_frames - 1))) for i in range(target_frames)]


def _rigid_collision_contact_index(objects: list[dict[str, Any]]) -> Optional[int]:
    body_a = next((o for o in objects if isinstance(o, dict) and o.get("name") == "body_a"), None)
    body_b = next((o for o in objects if isinstance(o, dict) and o.get("name") == "body_b"), None)
    if not body_a or not body_b:
        return None
    pa = body_a.get("positions") if isinstance(body_a.get("positions"), list) else []
    pb = body_b.get("positions") if isinstance(body_b.get("positions"), list) else []
    n = min(len(pa), len(pb))
    if n <= 0:
        return None
    best_i = 0
    best_d2 = float("inf")
    for i in range(n):
        ax, ay, az = _vec3(pa[i])
        bx, by, bz = _vec3(pb[i])
        d2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz)
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _selected_indices_for_family(
    n: int,
    fps: int,
    target_seconds: float,
    family: str,
    times: list[float],
    objects: list[dict[str, Any]],
) -> list[int]:
    if family != "rigidCollision":
        return _selected_indices(n, fps, target_seconds)
    contact_idx = _rigid_collision_contact_index(objects)
    if contact_idx is None or not times:
        return _selected_indices(n, fps, min(float(target_seconds), 3.0))
    t_contact = float(times[min(contact_idx, len(times) - 1)])
    t0 = max(float(times[0]), t_contact - 0.16)
    t1 = min(float(times[-1]), t_contact + 0.20)
    start = 0
    while start < n - 1 and float(times[start]) < t0:
        start += 1
    end = start
    while end < n - 1 and float(times[end]) < t1:
        end += 1
    window = list(range(max(0, start), min(n, end + 1)))
    if len(window) <= 1:
        return _selected_indices(n, fps, min(float(target_seconds), 3.0))
    target_frames = min(len(window), max(2, int(round(float(fps) * min(float(target_seconds), 3.0)))))
    if target_frames >= len(window):
        return window
    return [window[int(round(i * (len(window) - 1) / (target_frames - 1)))] for i in range(target_frames)]


class ScalarSmoother:
    def __init__(self, objects: list[dict[str, Any]], metric_name: Optional[str]) -> None:
        self.metric_name = metric_name
        self.groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.cache: dict[tuple[str, int], dict[str, list[float]]] = {}
        if not metric_name:
            return
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if not bool(obj.get("scalar_smoothing")):
                continue
            if str(obj.get("color_metric") or obj.get("default_metric") or "") != metric_name:
                continue
            group = str(obj.get("scalar_smooth_group") or "")
            if group:
                self.groups[group].append(obj)

    def values_for(self, obj: dict[str, Any], idx: int) -> Optional[list[float]]:
        if not self.metric_name or not bool(obj.get("scalar_smoothing")):
            return None
        group = str(obj.get("scalar_smooth_group") or "")
        if not group or group not in self.groups:
            return None
        key = (group, idx)
        if key not in self.cache:
            self.cache[key] = self._build_group_frame(group, idx)
        return self.cache[key].get(str(obj.get("name", "")))

    def _build_group_frame(self, group: str, idx: int) -> dict[str, list[float]]:
        accum: dict[tuple[float, float, float], list[float]] = defaultdict(list)
        objs = self.groups[group]
        for obj in objs:
            metric = self.metric_name or str(obj.get("color_metric") or obj.get("default_metric") or "")
            scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
            val = _series_scalar(scalars.get(metric), idx, 0.0)
            kind = str(obj.get("kind", "")).strip().lower()
            if kind in {"segment", "capsule"}:
                pa = _series_vec3(obj.get("points_a"), idx)
                pb = _series_vec3(obj.get("points_b"), idx)
                accum[_round_key(pa)].append(val)
                accum[_round_key(pb)].append(val)
            elif kind == "box":
                pos = _series_vec3(obj.get("positions"), idx)
                size_src = obj.get("size") if isinstance(obj.get("size"), list) else [1.0, 1.0, 1.0]
                size = _vec3(size_src, (1.0, 1.0, 1.0))
                if isinstance(obj.get("quaternions"), list) and obj.get("quaternions"):
                    q = obj["quaternions"][min(max(idx, 0), len(obj["quaternions"]) - 1)]
                    rot = _quat_to_rotmat((float(q[0]), float(q[1]), float(q[2]), float(q[3]))) if isinstance(q, (list, tuple)) and len(q) >= 4 else _rot_z(0.0)
                else:
                    rot = _rot_z(_series_scalar((obj.get("scalars") or {}).get("yaw_deg"), idx, 0.0))
                for corner in _box_corners(pos, size, rot):
                    accum[_round_key(corner)].append(val)
        averaged = {k: (sum(v) / max(len(v), 1)) for k, v in accum.items()}
        out: dict[str, list[float]] = {}
        for obj in objs:
            name = str(obj.get("name", ""))
            kind = str(obj.get("kind", "")).strip().lower()
            if kind in {"segment", "capsule"}:
                pa = _series_vec3(obj.get("points_a"), idx)
                pb = _series_vec3(obj.get("points_b"), idx)
                out[name] = [averaged.get(_round_key(pa), 0.0), averaged.get(_round_key(pb), 0.0)]
            elif kind == "box":
                pos = _series_vec3(obj.get("positions"), idx)
                size_src = obj.get("size") if isinstance(obj.get("size"), list) else [1.0, 1.0, 1.0]
                size = _vec3(size_src, (1.0, 1.0, 1.0))
                if isinstance(obj.get("quaternions"), list) and obj.get("quaternions"):
                    q = obj["quaternions"][min(max(idx, 0), len(obj["quaternions"]) - 1)]
                    rot = _quat_to_rotmat((float(q[0]), float(q[1]), float(q[2]), float(q[3]))) if isinstance(q, (list, tuple)) and len(q) >= 4 else _rot_z(0.0)
                else:
                    rot = _rot_z(_series_scalar((obj.get("scalars") or {}).get("yaw_deg"), idx, 0.0))
                out[name] = [averaged.get(_round_key(c), 0.0) for c in _box_corners(pos, size, rot)]
        return out


@dataclass
class RenderConfig:
    family: str
    metric_name: Optional[str]
    ctf: Optional[vtk.vtkColorTransferFunction]
    metric_range: tuple[float, float]
    segment_radius: float
    smoother: ScalarSmoother


class TrajectoryActor:
    def __init__(self, obj: dict[str, Any], renderer: vtk.vtkRenderer, config: RenderConfig) -> None:
        self.obj = obj
        self.renderer = renderer
        self.config = config
        self.kind = str(obj.get("kind", "")).strip().lower()
        self.name = str(obj.get("name", ""))
        self.color = tuple(float(v) for v in (obj.get("color") or [0.7, 0.7, 0.7])[:3])
        self.opacity = float(obj.get("opacity", 1.0) or 1.0)
        self.scalars = obj.get("scalars") if isinstance(obj.get("scalars"), dict) else {}
        self.positions = obj.get("positions") if isinstance(obj.get("positions"), list) else []
        self.sizes_series = obj.get("sizes") if isinstance(obj.get("sizes"), list) else []
        self.points_a = obj.get("points_a") if isinstance(obj.get("points_a"), list) else []
        self.points_b = obj.get("points_b") if isinstance(obj.get("points_b"), list) else []
        self.size = obj.get("size") if isinstance(obj.get("size"), list) else []
        self.quaternions = obj.get("quaternions") if isinstance(obj.get("quaternions"), list) else None
        static_fields = obj.get("static_fields") if isinstance(obj.get("static_fields"), dict) else {}
        self.render_shape = str(obj.get("render_shape") or static_fields.get("render_shape") or "").strip().lower()
        self.metric_for_obj = str(obj.get("color_metric") or obj.get("default_metric") or "")
        self._main_actor: Optional[vtk.vtkActor] = None
        self._edge_actor: Optional[vtk.vtkActor] = None
        self._hole_actor: Optional[vtk.vtkActor] = None
        self._gear_marker_actor: Optional[vtk.vtkActor] = None
        self._box_points: Optional[vtk.vtkPoints] = None
        self._line_points: Optional[vtk.vtkPoints] = None
        self._circle_unit_points: list[tuple[float, float, float]] = []
        self._point_scalars: Optional[vtk.vtkFloatArray] = None
        self._mapper: Optional[vtk.vtkPolyDataMapper] = None
        self._build_pipeline(static_fields)

    def _add_prop(self, prop) -> None:
        self.renderer.AddViewProp(prop)

    def _build_poly_mapper(self, poly: vtk.vtkPolyData) -> vtk.vtkPolyDataMapper:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        if self.config.ctf is not None:
            mapper.SetLookupTable(self.config.ctf)
            mapper.SetScalarRange(float(self.config.metric_range[0]), float(self.config.metric_range[1]))
        return mapper

    def _build_pipeline(self, static_fields: dict[str, Any]) -> None:
        if self.kind == "box":
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(8)
            polys = vtk.vtkCellArray()
            for face in _BOX_FACE_POLYS:
                quad = vtk.vtkQuad()
                for j, pid in enumerate(face):
                    quad.GetPointIds().SetId(j, int(pid))
                polys.InsertNextCell(quad)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetPolys(polys)
            scal = vtk.vtkFloatArray()
            scal.SetName("metric")
            scal.SetNumberOfValues(8)
            poly.GetPointData().SetScalars(scal)
            mapper = self._build_poly_mapper(poly)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self._add_prop(actor)
            self._main_actor = actor
            self._box_points = pts
            self._point_scalars = scal
            self._mapper = mapper
            if self.opacity < 0.999 or self.config.family in {"stackStability", "blockTower3D", "ballChutePile3D"}:
                edges = vtk.vtkExtractEdges()
                edges.SetInputData(poly)
                edge_mapper = vtk.vtkPolyDataMapper()
                edge_mapper.SetInputConnection(edges.GetOutputPort())
                edge_mapper.ScalarVisibilityOff()
                edge_actor = vtk.vtkActor()
                edge_actor.SetMapper(edge_mapper)
                if self.config.family in {"stackStability", "blockTower3D", "ballChutePile3D"}:
                    edge_actor.GetProperty().SetColor(0.08, 0.08, 0.08)
                    edge_actor.GetProperty().SetLineWidth(0.8)
                else:
                    edge_actor.GetProperty().SetColor(0.82, 0.86, 0.90)
                    edge_actor.GetProperty().SetLineWidth(1.0)
                edge_actor.GetProperty().LightingOff()
                edge_actor.GetProperty().SetAmbient(1.0)
                edge_actor.GetProperty().SetDiffuse(0.0)
                self._add_prop(edge_actor)
                self._edge_actor = edge_actor
            return
        if self.kind in {"segment", "capsule"}:
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(2)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(line)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetLines(lines)
            scal = vtk.vtkFloatArray()
            scal.SetName("metric")
            scal.SetNumberOfValues(2)
            poly.GetPointData().SetScalars(scal)
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(poly)
            tube.SetNumberOfSides(20)
            tube.SetCapping(True)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
            if self.config.ctf is not None:
                mapper.SetLookupTable(self.config.ctf)
                mapper.SetScalarRange(float(self.config.metric_range[0]), float(self.config.metric_range[1]))
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self._add_prop(actor)
            self._main_actor = actor
            self._line_points = pts
            self._point_scalars = scal
            self._mapper = mapper
            self._tube = tube
            return
        if self.kind == "circle" and self.render_shape == "gear":
            teeth = int(static_fields.get("teeth", obj.get("teeth", 24)) if (obj := self.obj) else 24)
            outer_radius = float(self.size[0]) if len(self.size) >= 1 else 0.2
            hole_radius = float(static_fields.get("inner_hole_radius_abs", self.obj.get("inner_hole_radius_abs", 0.18 * outer_radius)))
            pressure_angle = float(static_fields.get("pressure_angle_deg", self.obj.get("pressure_angle_deg", 20.0)))
            face = _build_gear_polydata(
                outer_radius,
                teeth,
                hole_radius,
                pressure_angle_deg=pressure_angle,
                keyway=_gear_keyway_enabled(self.name),
            )
            extrude = vtk.vtkLinearExtrusionFilter()
            extrude.SetInputData(face)
            extrude.SetExtrusionTypeToVectorExtrusion()
            extrude.SetVector(0.0, 1.0, 0.0)
            extrude.SetScaleFactor(1.0)
            extrude.CappingOn()
            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(extrude.GetOutputPort())
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(clean.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self._add_prop(actor)
            self._main_actor = actor
            marker_pts = vtk.vtkPoints()
            marker_pts.InsertNextPoint(float(hole_radius * 1.18), 1.035, 0.0)
            marker_pts.InsertNextPoint(float(outer_radius * 0.90), 1.035, 0.0)
            marker_line = vtk.vtkLine()
            marker_line.GetPointIds().SetId(0, 0)
            marker_line.GetPointIds().SetId(1, 1)
            marker_lines = vtk.vtkCellArray()
            marker_lines.InsertNextCell(marker_line)
            marker_poly = vtk.vtkPolyData()
            marker_poly.SetPoints(marker_pts)
            marker_poly.SetLines(marker_lines)
            marker_tube = vtk.vtkTubeFilter()
            marker_tube.SetInputData(marker_poly)
            marker_tube.SetNumberOfSides(10)
            marker_tube.SetRadius(max(0.004, 0.018 * outer_radius))
            marker_tube.SetCapping(True)
            marker_mapper = vtk.vtkPolyDataMapper()
            marker_mapper.SetInputConnection(marker_tube.GetOutputPort())
            marker_mapper.ScalarVisibilityOff()
            marker_actor = vtk.vtkActor()
            marker_actor.SetMapper(marker_mapper)
            marker_actor.GetProperty().SetColor(0.035, 0.035, 0.035)
            marker_actor.GetProperty().SetAmbient(0.35)
            marker_actor.GetProperty().SetDiffuse(0.65)
            marker_actor.GetProperty().SetSpecular(0.02)
            self._add_prop(marker_actor)
            self._gear_marker_actor = marker_actor
            return
        if self.kind == "circle":
            source = vtk.vtkSphereSource()
            source.SetThetaResolution(28)
            source.SetPhiResolution(28)
            source.SetRadius(0.5)
            source.Update()
            poly = vtk.vtkPolyData()
            poly.DeepCopy(source.GetOutput())
            scal = vtk.vtkFloatArray()
            scal.SetName("metric")
            scal.SetNumberOfValues(poly.GetNumberOfPoints())
            poly.GetPointData().SetScalars(scal)
            self._circle_unit_points = [poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())]
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            if self.config.ctf is not None:
                mapper.SetLookupTable(self.config.ctf)
                mapper.SetScalarRange(float(self.config.metric_range[0]), float(self.config.metric_range[1]))
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self._add_prop(actor)
            self._main_actor = actor
            self._point_scalars = scal
            self._mapper = mapper
            return

    def _scalar_values(self, idx: int, count: int) -> Optional[list[float]]:
        metric = self.config.metric_name
        if not metric or not self.metric_for_obj or self.metric_for_obj != metric:
            return None
        smoothed = self.config.smoother.values_for(self.obj, idx)
        if isinstance(smoothed, list) and len(smoothed) == count:
            return [float(v) for v in smoothed]
        return _series_scalar_values(self.scalars.get(metric), idx, count, 0.0)

    def _circle_scalar_values(self, idx: int) -> Optional[list[float]]:
        if self._point_scalars is None:
            return None
        count = self._point_scalars.GetNumberOfValues()
        vals = self._scalar_values(idx, count)
        profile = str(self.obj.get("surface_scalar_profile", "")).strip().lower()
        if vals is not None and profile == "rotor_eccentric" and self._circle_unit_points:
            base = _series_scalar(self.scalars.get(self.config.metric_name), idx, 0.0) if self.config.metric_name else 0.0
            profiled: list[float] = []
            for _, y, z in self._circle_unit_points:
                phase = math.atan2(float(z), float(y))
                weight = 0.18 + 0.82 * (0.5 + 0.5 * math.cos(phase))
                profiled.append(float(base) * float(weight))
            return profiled
        if vals is not None and profile in {"rolling_surface_speed", "rolling_surface_point_speed"} and self._circle_unit_points:
            speed = _series_scalar(self.scalars.get("speed_m_s"), idx, 0.0)
            omega = _series_scalar(self.scalars.get("omega_y"), idx, 0.0)
            radius = float(self.size[0]) if len(self.size) >= 1 else 0.1
            slope = math.radians(float(self.obj.get("surface_slope_angle_deg", 0.0)))
            tx = math.cos(slope)
            tz = -math.sin(slope)
            profiled = []
            for x, _, z in self._circle_unit_points:
                # vtkSphereSource radius is 0.5 and the actor scales by 2R.
                rx = 2.0 * radius * float(x)
                rz = 2.0 * radius * float(z)
                vx = speed * tx + omega * rz
                vz = speed * tz - omega * rx
                profiled.append(float(math.sqrt(vx * vx + vz * vz)))
            return profiled
        return vals

    def _update_mapper_scalars(self, vals: Optional[list[float]]) -> None:
        if self._mapper is None or self._point_scalars is None:
            return
        if vals is None:
            self._mapper.ScalarVisibilityOff()
            return
        self._mapper.ScalarVisibilityOn()
        for i, v in enumerate(vals):
            self._point_scalars.SetValue(i, float(v))
        self._point_scalars.Modified()

    def _apply_prop_style(self, prop: vtk.vtkProperty, rgb: tuple[float, float, float]) -> None:
        prop.SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        prop.SetOpacity(float(self.opacity))
        prop.SetAmbient(0.18)
        prop.SetDiffuse(0.82)
        prop.SetSpecular(0.10 if self.kind == "circle" else 0.04)
        prop.SetSpecularPower(20.0)

    def _metric_rgb(self, idx: int) -> tuple[float, float, float]:
        if self.config.metric_name and self.metric_for_obj == self.config.metric_name and self.config.ctf is not None:
            val = _series_scalar(self.scalars.get(self.config.metric_name), idx, 0.0)
            rgb = self.config.ctf.GetColor(float(val))
            return float(rgb[0]), float(rgb[1]), float(rgb[2])
        return self.color

    def _visible_at(self, idx: int) -> bool:
        raw = self.scalars.get("visible") if isinstance(self.scalars, dict) else None
        if isinstance(raw, list) and raw:
            return _series_scalar(raw, idx, 1.0) > 0.5
        return True

    def _set_visible(self, visible: bool) -> None:
        state = 1 if visible else 0
        for actor in (self._main_actor, self._edge_actor, self._hole_actor, self._gear_marker_actor):
            if actor is not None:
                actor.SetVisibility(state)

    def update(self, idx: int) -> None:
        if self._main_actor is None:
            return
        visible = self._visible_at(idx)
        self._set_visible(visible)
        if not visible:
            return
        if self.kind == "box" and self._box_points is not None:
            pos = _series_vec3(self.positions, idx)
            size = _series_vec3(self.sizes_series, idx, _vec3(self.size, (1.0, 1.0, 1.0)))
            if isinstance(self.quaternions, list) and self.quaternions:
                q = self.quaternions[min(max(idx, 0), len(self.quaternions) - 1)]
                rot = _quat_to_rotmat((float(q[0]), float(q[1]), float(q[2]), float(q[3]))) if isinstance(q, (list, tuple)) and len(q) >= 4 else _rot_z(0.0)
            else:
                rot = _rot_z(_series_scalar(self.scalars.get("yaw_deg"), idx, 0.0))
            corners = _box_corners(pos, size, rot)
            for i, p in enumerate(corners):
                self._box_points.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
            self._box_points.Modified()
            self._update_mapper_scalars(self._scalar_values(idx, 8))
            self._apply_prop_style(self._main_actor.GetProperty(), self.color)
            if self._edge_actor is not None:
                if self.config.family in {"stackStability", "blockTower3D", "ballChutePile3D"}:
                    self._edge_actor.GetProperty().SetOpacity(1.0)
                else:
                    self._edge_actor.GetProperty().SetOpacity(min(1.0, max(0.45, self.opacity + 0.35)))
            return
        if self.kind in {"segment", "capsule"} and self._line_points is not None:
            pa = _series_vec3(self.points_a, idx)
            pb = _series_vec3(self.points_b, idx)
            self._line_points.SetPoint(0, float(pa[0]), float(pa[1]), float(pa[2]))
            self._line_points.SetPoint(1, float(pb[0]), float(pb[1]), float(pb[2]))
            self._line_points.Modified()
            radius = float(self.size[0]) if self.kind == "capsule" and len(self.size) >= 1 else max(self.config.segment_radius, 0.6 * float(self.obj.get("width", 1.0)) * self.config.segment_radius)
            self._tube.SetRadius(float(radius))
            self._tube.Modified()
            self._update_mapper_scalars(self._scalar_values(idx, 2))
            self._apply_prop_style(self._main_actor.GetProperty(), self.color)
            return
        if self.kind == "circle" and self.render_shape == "gear":
            pos = _series_vec3(self.positions, idx)
            thickness = float(self.obj.get("thickness", self.size[1] if len(self.size) >= 2 else 0.05))
            spin = _series_scalar(self.scalars.get("spin_deg"), idx, _series_scalar(self.scalars.get("yaw_deg"), idx, 0.0))
            spin *= float(self.obj.get("render_spin_sign", 1.0))
            self._main_actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
            self._main_actor.SetOrientation(0.0, float(spin), 0.0)
            self._main_actor.SetScale(1.0, float(thickness), 1.0)
            self._apply_prop_style(self._main_actor.GetProperty(), self._metric_rgb(idx))
            if self._gear_marker_actor is not None:
                self._gear_marker_actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
                self._gear_marker_actor.SetOrientation(0.0, float(spin), 0.0)
                self._gear_marker_actor.SetScale(1.0, float(thickness), 1.0)
            return
        if self.kind == "circle":
            pos = _series_vec3(self.positions, idx)
            radius = float(self.size[0]) if len(self.size) >= 1 else 0.1
            tf = vtk.vtkTransform()
            tf.Translate(float(pos[0]), float(pos[1]), float(pos[2]))
            tf.Scale(2.0 * radius, 2.0 * radius, 2.0 * radius)
            self._main_actor.SetUserTransform(tf)
            self._update_mapper_scalars(self._circle_scalar_values(idx))
            self._apply_prop_style(self._main_actor.GetProperty(), self._metric_rgb(idx))


def _collect_bounds(
    objects: list[dict[str, Any]],
    family: str,
    frame_indices: Optional[list[int]] = None,
) -> tuple[list[float], list[float]]:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    dynamic_pts: list[tuple[float, float, float]] = []
    static_mins = [float("inf"), float("inf"), float("inf")]
    static_maxs = [float("-inf"), float("-inf"), float("-inf")]

    def add_point(p: tuple[float, float, float], pad: float = 0.0) -> None:
        for i in range(3):
            mins[i] = min(mins[i], float(p[i]) - pad)
            maxs[i] = max(maxs[i], float(p[i]) + pad)

    def add_static_point(p: tuple[float, float, float], pad: float = 0.0) -> None:
        for i in range(3):
            static_mins[i] = min(static_mins[i], float(p[i]) - pad)
            static_maxs[i] = max(static_maxs[i], float(p[i]) + pad)

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", ""))
        kind = str(obj.get("kind", "")).strip().lower()
        size = obj.get("size") if isinstance(obj.get("size"), list) else []
        is_ball = family == "ballChutePile3D" and name.startswith("ball_")
        if kind in {"circle", "box"}:
            pts = obj.get("positions") if isinstance(obj.get("positions"), list) else []
            if family == "rigidCollision" and frame_indices:
                pts = [pts[i] for i in frame_indices if 0 <= i < len(pts)]
            if family in {"stackStability", "blockTower3D"} and name == "impactor":
                # Frame the target assembly, not the impactor's long post-impact flight.
                pts = pts[: max(2, int(math.ceil(0.22 * len(pts))))]
            for i, p in enumerate(pts):
                if is_ball:
                    vis = ((obj.get("scalars") or {}).get("visible") or [])
                    if isinstance(vis, list) and i < len(vis) and float(vis[i]) <= 0.5:
                        continue
                pos = _vec3(p)
                if kind == "circle":
                    pad = float(size[0]) if len(size) >= 1 else 0.05
                else:
                    sx, sy, sz = _vec3(size, (0.1, 0.1, 0.1))
                    pad = 0.5 * max(sx, sy, sz)
                if is_ball:
                    dynamic_pts.append(pos)
                else:
                    add_point(pos, pad)
                    add_static_point(pos, pad)
        elif kind in {"segment", "capsule"}:
            pa = obj.get("points_a") if isinstance(obj.get("points_a"), list) else []
            pb = obj.get("points_b") if isinstance(obj.get("points_b"), list) else []
            pad = float(size[0]) if kind == "capsule" and len(size) >= 1 else 0.03
            for p in pa:
                add_point(_vec3(p), pad)
                add_static_point(_vec3(p), pad)
            for p in pb:
                add_point(_vec3(p), pad)
                add_static_point(_vec3(p), pad)
    if family == "ballChutePile3D" and dynamic_pts:
        if math.isfinite(static_mins[0]):
            span = [max(0.2, static_maxs[i] - static_mins[i]) for i in range(3)]
            culled = [
                p for p in dynamic_pts
                if (static_mins[0] - 0.35 * span[0]) <= p[0] <= (static_maxs[0] + 0.35 * span[0])
                and (static_mins[1] - 0.45 * span[1]) <= p[1] <= (static_maxs[1] + 0.45 * span[1])
                and (static_mins[2] - 0.20 * span[2]) <= p[2] <= (static_maxs[2] + 0.90 * span[2])
            ]
            if culled:
                dynamic_pts = culled
        xs = sorted(p[0] for p in dynamic_pts)
        ys = sorted(p[1] for p in dynamic_pts)
        zs = sorted(p[2] for p in dynamic_pts)
        def pct(vals: list[float], q: float) -> float:
            if not vals:
                return 0.0
            k = min(max(int(round((len(vals) - 1) * q)), 0), len(vals) - 1)
            return float(vals[k])
        qmins = [pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02)]
        qmaxs = [pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98)]
        pads = [0.06, 0.06, 0.06]
        for i in range(3):
            mins[i] = min(mins[i], qmins[i] - pads[i])
            maxs[i] = max(maxs[i], qmaxs[i] + pads[i])
    if not math.isfinite(mins[0]):
        mins = [-1.0, -1.0, -1.0]
        maxs = [1.0, 1.0, 1.0]
    return mins, maxs


def _camera_parallel_scale(scene: dict[str, Any], default_scale: float) -> float:
    try:
        override = float(scene.get("camera_parallel_scale", default_scale))
    except (TypeError, ValueError):
        override = default_scale
    return max(0.05, float(override))


def _camera_shift_fraction(scene: dict[str, Any], reserve_scalar_bar: bool) -> float:
    if "scalar_bar_shift_fraction" in scene:
        try:
            return float(scene.get("scalar_bar_shift_fraction", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.18 if reserve_scalar_bar else 0.0


def _configure_camera(camera: vtk.vtkCamera, scene: dict[str, Any], mins: list[float], maxs: list[float]) -> None:
    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    cz = 0.5 * (mins[2] + maxs[2])
    ext = [maxs[i] - mins[i] for i in range(3)]
    zoom = max(0.25, float(scene.get("camera_zoom", 1.07)))
    radius = max(max(ext), 1.0e-3) * 2.2 / zoom
    view = str(scene.get("view_plane", "oblique_xy")).strip().lower()
    reserve_scalar_bar = bool(scene.get("reserve_scalar_bar", False))
    horizontal_shift = _camera_shift_fraction(scene, reserve_scalar_bar) * ext[0]
    if view == "xz":
        camera.SetPosition(cx + horizontal_shift, cy - radius, cz)
        camera.SetFocalPoint(cx + horizontal_shift, cy, cz)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.ParallelProjectionOn()
        scale_pad = 1.06 if reserve_scalar_bar else 1.0
        default_scale = scale_pad * 0.54 * max(ext[0], ext[2], 0.12) / zoom
        camera.SetParallelScale(_camera_parallel_scale(scene, default_scale))
        return
    if view == "xy":
        camera.SetPosition(cx + horizontal_shift, cy, cz + radius)
        camera.SetFocalPoint(cx + horizontal_shift, cy, cz)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.ParallelProjectionOn()
        scale_pad = 1.06 if reserve_scalar_bar else 1.0
        default_scale = scale_pad * 0.54 * max(ext[0], ext[1], 0.12) / zoom
        camera.SetParallelScale(_camera_parallel_scale(scene, default_scale))
        return
    yaw_deg = float(scene.get("camera_yaw_deg", 40.0))
    pitch_deg = float(scene.get("camera_pitch_deg", 22.0))
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    px = cx + radius * math.cos(pitch) * math.cos(yaw)
    py = cy + radius * math.cos(pitch) * math.sin(yaw)
    pz = cz + radius * math.sin(pitch)
    camera.SetPosition(px, py, pz)
    camera.SetFocalPoint(cx, cy, cz)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.ParallelProjectionOff()
    camera.SetViewAngle(29.5 if reserve_scalar_bar else 30.5)


def _scene_vector(scene: dict[str, Any], key: str, default: list[float]) -> list[float]:
    value = scene.get(key)
    if isinstance(value, list) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return default
    return default


def _follow_position(objects: list[Any], scene: dict[str, Any], idx: int) -> Optional[list[float]]:
    follow_name = str(scene.get("follow_object", "")).strip()
    if not follow_name:
        return None
    for obj in objects:
        if not isinstance(obj, dict) or str(obj.get("name", "")) != follow_name:
            continue
        positions = obj.get("positions")
        if not isinstance(positions, list) or not positions:
            return None
        use_idx = min(max(int(idx), 0), len(positions) - 1)
        pos = positions[use_idx]
        if isinstance(pos, list) and len(pos) >= 3:
            try:
                offset = _scene_vector(scene, "follow_offset", [0.0, 0.0, 0.0])
                return [float(pos[0]) + offset[0], float(pos[1]) + offset[1], float(pos[2]) + offset[2]]
            except (TypeError, ValueError):
                return None
    return None


def _update_follow_camera(
    camera: vtk.vtkCamera,
    scene: dict[str, Any],
    mins: list[float],
    maxs: list[float],
    objects: list[Any],
    idx: int,
) -> None:
    focus = _follow_position(objects, scene, idx)
    if focus is None:
        return
    ext = [maxs[i] - mins[i] for i in range(3)]
    zoom = max(0.25, float(scene.get("camera_zoom", 1.07)))
    radius = max(max(ext), 1.0e-3) * 2.2 / zoom
    view = str(scene.get("view_plane", "oblique_xy")).strip().lower()
    reserve_scalar_bar = bool(scene.get("reserve_scalar_bar", False))
    horizontal_shift = _camera_shift_fraction(scene, reserve_scalar_bar) * ext[0]
    if view == "xz":
        cy = 0.5 * (mins[1] + maxs[1])
        camera.SetPosition(focus[0] + horizontal_shift, cy - radius, focus[2])
        camera.SetFocalPoint(focus[0] + horizontal_shift, cy, focus[2])
        camera.SetViewUp(0.0, 0.0, 1.0)
        return
    if view == "xy":
        cz = 0.5 * (mins[2] + maxs[2])
        camera.SetPosition(focus[0] + horizontal_shift, focus[1], cz + radius)
        camera.SetFocalPoint(focus[0] + horizontal_shift, focus[1], cz)
        camera.SetViewUp(0.0, 1.0, 0.0)
        return


def _apply_scene_bounds(scene: dict[str, Any], family: str, mins: list[float], maxs: list[float]) -> tuple[list[float], list[float]]:
    """Honor solver-specified fixed view bounds when the case provides them."""
    if family in {"rigidCollision", "massSpringDamper"}:
        return mins, maxs
    if bool(scene.get("auto_bounds", True)):
        return mins, maxs
    view = str(scene.get("view_plane", "oblique_xy")).strip().lower()
    xlim = scene.get("xlim")
    ylim = scene.get("ylim")
    zlim = scene.get("zlim")
    if isinstance(xlim, list) and len(xlim) >= 2:
        mins[0] = float(xlim[0])
        maxs[0] = float(xlim[1])
    if isinstance(ylim, list) and len(ylim) >= 2:
        if view == "xz":
            mins[2] = float(ylim[0])
            maxs[2] = float(ylim[1])
        else:
            mins[1] = float(ylim[0])
            maxs[1] = float(ylim[1])
    if isinstance(zlim, list) and len(zlim) >= 2:
        mins[2] = float(zlim[0])
        maxs[2] = float(zlim[1])
    if maxs[2] <= mins[2]:
        zc = 0.5 * (mins[2] + maxs[2])
        mins[2] = zc - 0.1
        maxs[2] = zc + 0.1
    return mins, maxs


def _style_scalar_bar(bar: vtk.vtkScalarBarActor, title: str) -> None:
    bar.SetTitle(title)
    bar.SetNumberOfLabels(5)
    bar.SetPosition(0.945, 0.16)
    bar.SetWidth(0.038)
    bar.SetHeight(0.62)
    bar.SetLabelFormat("%-#6.3g")
    title_prop = bar.GetTitleTextProperty()
    title_prop.SetFontSize(15)
    title_prop.SetColor(float(TEXT_RGB[0]), float(TEXT_RGB[1]), float(TEXT_RGB[2]))
    title_prop.BoldOff()
    label_prop = bar.GetLabelTextProperty()
    label_prop.SetFontSize(13)
    label_prop.SetColor(float(TEXT_RGB[0]), float(TEXT_RGB[1]), float(TEXT_RGB[2]))
    label_prop.BoldOff()
    bar.UnconstrainedFontSizeOn()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic ParaView/VTK renderer for dynamics trajectory.json cases.")
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("frame_dir", type=Path)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--target-seconds", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    frame_dir = args.frame_dir.expanduser().resolve()
    frame_dir.mkdir(parents=True, exist_ok=True)
    traj_path = case_dir / "results" / "trajectory.json"
    if not traj_path.exists():
        raise SystemExit(f"Missing trajectory: {traj_path}")

    data = _load_json(traj_path)
    objects = data.get("objects") if isinstance(data.get("objects"), list) else []
    scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    times = data.get("times") if isinstance(data.get("times"), list) and data.get("times") else [0.0]
    family = str(data.get("family") or case_dir.parent.name)

    _augment_speed_scalars(objects, [float(t) for t in times], family)
    metric_name, metric_range = _choose_metric(objects)
    smoother = ScalarSmoother(objects, metric_name)
    config = RenderConfig(
        family=family,
        metric_name=metric_name,
        ctf=_build_ctf(metric_range[0], metric_range[1]) if metric_name else None,
        metric_range=metric_range,
        segment_radius=0.01,
        smoother=smoother,
    )

    effective_seconds = _effective_target_seconds(family, float(times[-1]), float(args.target_seconds))
    frame_indices = _selected_indices_for_family(
        len(times),
        int(args.fps),
        effective_seconds,
        family,
        [float(t) for t in times],
        objects,
    )
    mins, maxs = _collect_bounds(objects, family, frame_indices)
    mins, maxs = _apply_scene_bounds(scene, family, mins, maxs)
    diag = math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(3)))
    config.segment_radius = max(0.003, 0.0045 * max(diag, 1.0))

    ren = vtk.vtkRenderer()
    ren.SetBackground(float(BG_RGB[0]), float(BG_RGB[1]), float(BG_RGB[2]))
    ren.SetUseDepthPeeling(1)
    ren.SetMaximumNumberOfPeels(64)
    ren.SetOcclusionRatio(0.05)

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(int(args.width), int(args.height))
    rw.SetAlphaBitPlanes(1)
    rw.SetMultiSamples(0)
    rw.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)
    iren.Initialize()

    # These families display the orientation indicator in the evaluation videos.
    if family in {
        "cantileverBeamTransient", "harmonicResponseFRF", "j2PlasticBeam",
        "lemaitreDamageBar", "reissnerShellModal", "rubberBlockCompression",
        "stressWavePropagation",
    }:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        orientation = vtk.vtkOrientationMarkerWidget()
        orientation.SetOrientationMarker(axes)
        orientation.SetInteractor(iren)
        orientation.SetViewport(0.0, 0.0, 0.18, 0.18)
        orientation.SetEnabled(1)
        orientation.InteractiveOff()

    if metric_name and config.ctf is not None:
        scene.setdefault("reserve_scalar_bar", True)
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(config.ctf)
        _style_scalar_bar(scalar_bar, _nice_label(metric_name))
        ren.AddViewProp(scalar_bar)

    actors = [TrajectoryActor(obj, ren, config) for obj in objects if isinstance(obj, dict)]
    camera = ren.GetActiveCamera()
    _configure_camera(camera, scene, mins, maxs)
    ren.ResetCameraClippingRange()

    to_image = vtk.vtkWindowToImageFilter()
    to_image.SetInput(rw)
    to_image.SetScale(1)
    to_image.SetInputBufferTypeToRGBA()
    to_image.ReadFrontBufferOff()

    writer = vtk.vtkPNGWriter()
    for out_idx, src_idx in enumerate(frame_indices):
        for actor in actors:
            actor.update(src_idx)
        _update_follow_camera(camera, scene, mins, maxs, objects, src_idx)
        rw.Render()
        ren.ResetCameraClippingRange()
        to_image.Modified()
        to_image.Update()
        writer.SetFileName(str(frame_dir / f"frame_{out_idx:04d}.png"))
        writer.SetInputConnection(to_image.GetOutputPort())
        writer.Write()


if __name__ == "__main__":
    main()
