#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BG_HEX = "#726F69"
TEXT_RGB = [0.96, 0.96, 0.94]
COLORBAR_TARGET_LABELS = 5
LOG_COLORBAR_MAX_DECADES = 3.0
COMPONENT_Z = 0.012
COMPONENT_EDGE_RGB = [0.96, 0.96, 0.94]
COMPONENT_STYLES: dict[str, dict[str, Any]] = {
    "glass": {"color": [0.72, 0.85, 0.93], "opacity": 0.08, "edge_width": 1.35},
    "metal": {"color": [0.90, 0.91, 0.94], "opacity": 0.10, "edge_width": 1.45},
    "mask": {"color": [0.31, 0.38, 0.45], "opacity": 0.28, "edge_width": 1.5},
    "polarizer": {"color": [0.52, 0.78, 0.82], "opacity": 0.10, "edge_width": 1.35},
    "polarizer_axis": {"color": [0.92, 0.98, 0.78], "opacity": 0.72, "edge_width": 1.15},
    "analyzer_axis": {"color": [1.00, 0.82, 0.35], "opacity": 0.82, "edge_width": 1.2},
    "gauge_frame": {"color": [0.96, 0.96, 0.94], "opacity": 0.18, "edge_width": 1.2},
    "intensity_gauge": {"color": [1.00, 0.86, 0.28], "opacity": 0.62, "edge_width": 0.9},
    "interface": {"color": [0.80, 0.88, 0.94], "opacity": 0.08, "edge_width": 1.3},
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _frame_indices(n_frames: int, target_count: int, *, start: int = 0, end: int | None = None) -> list[int]:
    if n_frames <= 0:
        return []
    end_idx = n_frames - 1 if end is None else min(n_frames - 1, max(0, int(end)))
    start_idx = min(end_idx, max(0, int(start)))
    if target_count <= 0:
        return list(range(start_idx, end_idx + 1))
    if target_count == 1:
        return [start_idx]
    span = float(end_idx - start_idx)
    return [int(round(start_idx + i * span / float(target_count - 1))) for i in range(target_count)]


def _case_family(meta: dict[str, Any], case_dir: Path) -> str:
    return str(meta.get("family") or meta.get("case_prefix") or case_dir.name.split("_S", 1)[0])


def _frame_power(frame: dict[str, Any]) -> float:
    summary = frame.get("summary") if isinstance(frame, dict) else None
    if not isinstance(summary, dict):
        return 0.0
    try:
        return max(0.0, float(summary.get("power", 0.0) or 0.0))
    except Exception:
        return 0.0


def _frame_peak(frame: dict[str, Any]) -> float:
    summary = frame.get("summary") if isinstance(frame, dict) else None
    if not isinstance(summary, dict):
        return 0.0
    try:
        return max(0.0, float(summary.get("peak", 0.0) or 0.0))
    except Exception:
        return 0.0


def _frame_roi_power(case_dir: Path, frame: dict[str, Any], field_name: str) -> float:
    field_name = str(field_name or "irradiance")
    field_file = str(frame.get("field_file", "") if isinstance(frame, dict) else "")
    if not field_file:
        return 0.0
    field_path = case_dir / "results" / "sensor_fields" / field_file
    if not field_path.exists():
        return 0.0
    try:
        from vtkmodules.vtkIOXML import vtkXMLImageDataReader  # type: ignore
        from vtkmodules.util.numpy_support import vtk_to_numpy  # type: ignore

        reader = vtkXMLImageDataReader()
        reader.SetFileName(str(field_path))
        reader.Update()
        data = reader.GetOutput()
        if data is None or data.GetPointData() is None:
            return 0.0
        arr = data.GetPointData().GetArray(field_name)
        if arr is None:
            return 0.0
        nx, ny, nz = [int(v) for v in data.GetDimensions()]
        values = vtk_to_numpy(arr)
        if nx <= 0 or ny <= 0 or nz <= 0 or int(values.size) != nx * ny * nz:
            return 0.0
        plane = values.reshape((nz, ny, nx)).sum(axis=0)
        x0_frac = _clamp(float(os.environ.get("OPTICS_MEEP_ROI_X0", "0.40")), 0.0, 0.98)
        x1_frac = _clamp(float(os.environ.get("OPTICS_MEEP_ROI_X1", "0.72")), x0_frac + 0.01, 1.0)
        y0_frac = _clamp(float(os.environ.get("OPTICS_MEEP_ROI_Y0", "0.04")), 0.0, 0.98)
        y1_frac = _clamp(float(os.environ.get("OPTICS_MEEP_ROI_Y1", "0.96")), y0_frac + 0.01, 1.0)
        xs = min(nx - 1, max(0, int(math.floor(nx * x0_frac))))
        xe = min(nx, max(xs + 1, int(math.ceil(nx * x1_frac))))
        ys = min(ny - 1, max(0, int(math.floor(ny * y0_frac))))
        ye = min(ny, max(ys + 1, int(math.ceil(ny * y1_frac))))
        roi = plane[ys:ye, xs:xe]
        return max(0.0, float(roi[roi > 0.0].sum()))
    except Exception:
        return 0.0


def _active_frame_window(
    frames: list[dict[str, Any]],
    family: str,
    *,
    case_dir: Path | None = None,
    field_name: str = "irradiance",
) -> tuple[int, int]:
    if not frames:
        return 0, 0
    if not family.startswith("meep"):
        return 0, len(frames) - 1
    if os.environ.get("OPTICS_TRIM_MEEP_IDLE", "1").strip().lower() in {"0", "false", "no"}:
        return 0, len(frames) - 1

    if case_dir is not None and os.environ.get("OPTICS_MEEP_ROI_ACTIVITY", "1").strip().lower() not in {"0", "false", "no"}:
        roi_powers = [_frame_roi_power(case_dir, frame, field_name) for frame in frames]
        max_roi_power = max(roi_powers) if roi_powers else 0.0
        if max_roi_power > 0.0:
            roi_frac = _clamp(float(os.environ.get("OPTICS_MEEP_ROI_ACTIVE_FRACTION", "0.02")), 1.0e-6, 0.25)
            for idx, power in enumerate(roi_powers):
                if power >= max_roi_power * roi_frac:
                    pre_roll = max(
                        0,
                        int(
                            round(
                                len(frames)
                                * _clamp(float(os.environ.get("OPTICS_MEEP_PREROLL_FRACTION", "0.02")), 0.0, 0.5)
                            )
                        ),
                    )
                    return max(0, idx - pre_roll), len(frames) - 1

    powers = [_frame_power(frame) for frame in frames]
    peaks = [_frame_peak(frame) for frame in frames]
    max_power = max(powers) if powers else 0.0
    max_peak = max(peaks) if peaks else 0.0
    if max_power <= 0.0 and max_peak <= 0.0:
        return 0, len(frames) - 1

    frac = _clamp(float(os.environ.get("OPTICS_MEEP_ACTIVE_FRACTION", "0.005")), 1.0e-6, 0.25)
    first = 0
    for idx, (power, peak) in enumerate(zip(powers, peaks)):
        power_hit = max_power > 0.0 and power >= max_power * frac
        peak_hit = max_peak > 0.0 and peak >= max_peak * frac
        if power_hit or peak_hit:
            first = idx
            break
    pre_roll = max(1, int(round(len(frames) * _clamp(float(os.environ.get("OPTICS_MEEP_PREROLL_FRACTION", "0.02")), 0.0, 0.5))))
    return max(0, first - pre_roll), len(frames) - 1


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * float(len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - float(lo)
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _sorted_positive_values(values: Iterable[float]) -> list[float]:
    out = [float(v) for v in values if float(v) > 0.0]
    out.sort()
    return out


def _parse_hex_rgb(value: str, *, default_hex: str = DEFAULT_BG_HEX) -> list[float]:
    raw = str(value or default_hex).strip()
    if not raw.startswith("#"):
        raw = f"#{raw}"
    if len(raw) != 7:
        raw = default_hex
    try:
        return [int(raw[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    except Exception:
        return [int(default_hex[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]


def _base_color_window(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = _quantile(values, 0.10)
    hi = _quantile(values, 0.995)
    if hi <= lo:
        lo = values[0]
        hi = values[-1]
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _color_window(values: list[float], color_scale: str) -> tuple[float, float]:
    base_lo, base_hi = _base_color_window(values)
    if color_scale == "log" and base_hi > 0.0:
        lo = max(base_lo, base_hi / (10.0 ** LOG_COLORBAR_MAX_DECADES))
        hi = base_hi
        if hi <= lo:
            lo = max(values[0], hi / 10.0) if values else 1.0e-6
        return float(lo), float(max(hi, lo * 10.0))
    return base_lo, base_hi


def _camera_scale(bounds: tuple[float, float, float, float], width: int, height: int) -> float:
    x0, x1, y0, y1 = bounds
    aspect = max(1.0, float(width)) / max(1.0, float(height))
    half_y = 0.5 * abs(y1 - y0)
    half_x = 0.5 * abs(x1 - x0) / aspect
    fit = os.environ.get("PV_CAMERA_FIT", "contain").strip().lower()
    if fit == "cover":
        scale = min(half_y, half_x)
    else:
        scale = max(half_y, half_x)
    return max(1.0e-6, scale) * 1.02


def _set_prop(proxy, key: str, value) -> None:
    try:
        setattr(proxy, key, value)
    except Exception:
        pass


def _choose_color_scale(color_lo: float, color_hi: float, requested: str) -> str:
    mode = (requested or "auto").strip().lower()
    if mode not in {"auto", "linear", "log"}:
        mode = "auto"
    if mode == "auto":
        if color_lo > 0.0 and color_hi > color_lo and (color_hi / color_lo) >= 64.0:
            return "log"
        return "linear"
    if mode == "log" and color_lo <= 0.0:
        return "linear"
    return mode


def _round_sig(value: float, digits: int = 2) -> float:
    if value == 0.0:
        return 0.0
    scale = 10.0 ** (digits - 1 - math.floor(math.log10(abs(value))))
    return round(value * scale) / scale


def _dedupe_labels(labels: list[float]) -> list[float]:
    out: list[float] = []
    for value in labels:
        vv = float(value)
        if not out or abs(vv - out[-1]) > max(1.0e-12, abs(vv) * 1.0e-6):
            out.append(vv)
    return out


def _linear_colorbar_labels(color_lo: float, color_hi: float, target: int = COLORBAR_TARGET_LABELS) -> list[float]:
    if color_hi <= color_lo:
        return [float(color_lo), float(color_hi)]
    if target <= 1:
        return [float(_round_sig(color_lo)), float(_round_sig(color_hi))]
    span = color_hi - color_lo
    labels = [_round_sig(color_lo + span * i / float(target - 1), digits=2) for i in range(target)]
    labels[0] = float(color_lo)
    labels[-1] = float(color_hi)
    return _dedupe_labels(labels)


def _log_colorbar_labels(color_lo: float, color_hi: float, target: int = COLORBAR_TARGET_LABELS) -> list[float]:
    if color_lo <= 0.0 or color_hi <= color_lo:
        return _linear_colorbar_labels(color_lo, color_hi, target=target)
    if target <= 1:
        return [float(color_lo), float(color_hi)]
    log_lo = math.log10(color_lo)
    log_hi = math.log10(color_hi)
    labels = [
        _round_sig(10.0 ** (log_lo + (log_hi - log_lo) * i / float(target - 1)), digits=2)
        for i in range(target)
    ]
    labels[0] = float(color_lo)
    labels[-1] = float(color_hi)
    return _dedupe_labels(labels)


def _colorbar_labels(color_lo: float, color_hi: float, color_scale: str) -> list[float]:
    if color_scale == "log":
        return _log_colorbar_labels(color_lo, color_hi)
    return _linear_colorbar_labels(color_lo, color_hi)


def _colorbar_label_format(color_scale: str) -> str:
    env_fmt = os.environ.get("PV_COLORBAR_LABEL_FORMAT", "").strip()
    if env_fmt:
        return env_fmt
    return "%.1e" if color_scale == "log" else "%-#6.3g"


def _field_label(field_name: str) -> str:
    mapping = {
        "irradiance": "Intensity (a.u.)",
        "intensity": "Intensity (a.u.)",
        "field_value": "Field Value",
    }
    return mapping.get(str(field_name), str(field_name))


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _visual_size(actual: float, span: float, lo_frac: float, hi_frac: float) -> float:
    span = max(1.0e-9, float(span))
    return _clamp(float(actual), lo_frac * span, hi_frac * span)


def _rot_xy(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    theta = math.radians(float(angle_deg))
    ct = math.cos(theta)
    st = math.sin(theta)
    return x * ct - y * st, x * st + y * ct


def _rect_from_bounds(x0: float, x1: float, y0: float, y1: float, *, z: float = COMPONENT_Z) -> list[list[float]]:
    return [
        [float(x0), float(y0), float(z)],
        [float(x1), float(y0), float(z)],
        [float(x1), float(y1), float(z)],
        [float(x0), float(y1), float(z)],
    ]


def _rectangle_polygon(
    center_x: float,
    center_y: float,
    half_w: float,
    half_h: float,
    *,
    angle_deg: float = 0.0,
    z: float = COMPONENT_Z,
) -> list[list[float]]:
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    out: list[list[float]] = []
    for xx, yy in corners:
        rx, ry = _rot_xy(xx, yy, angle_deg)
        out.append([center_x + rx, center_y + ry, z])
    return out


def _segment_polygon(
    p0: tuple[float, float],
    p1: tuple[float, float],
    thickness: float,
    *,
    z: float = COMPONENT_Z,
) -> list[list[float]]:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    length = math.hypot(dx, dy)
    if length <= 1.0e-9:
        return _rectangle_polygon(float(p0[0]), float(p0[1]), thickness * 0.5, thickness * 0.5, z=z)
    nx = -dy / length
    ny = dx / length
    hh = 0.5 * float(thickness)
    return [
        [float(p0[0]) + nx * hh, float(p0[1]) + ny * hh, z],
        [float(p1[0]) + nx * hh, float(p1[1]) + ny * hh, z],
        [float(p1[0]) - nx * hh, float(p1[1]) - ny * hh, z],
        [float(p0[0]) - nx * hh, float(p0[1]) - ny * hh, z],
    ]


def _axis_strip_polygon(
    center_x: float,
    center_y: float,
    angle_deg: float,
    length: float,
    thickness: float,
    *,
    z: float = COMPONENT_Z,
) -> list[list[float]]:
    theta = math.radians(float(angle_deg))
    dx = 0.5 * float(length) * math.cos(theta)
    dy = 0.5 * float(length) * math.sin(theta)
    return _segment_polygon(
        (float(center_x) - dx, float(center_y) - dy),
        (float(center_x) + dx, float(center_y) + dy),
        float(thickness),
        z=z,
    )


def _triangle_polygon(points: list[tuple[float, float]], *, z: float = COMPONENT_Z) -> list[list[float]]:
    return [[float(px), float(py), float(z)] for px, py in points]


def _annulus_polygon(
    center_x: float,
    center_y: float,
    inner_r: float,
    outer_r: float,
    *,
    n: int = 64,
    z: float = COMPONENT_Z,
) -> list[list[float]]:
    outer: list[list[float]] = []
    inner: list[list[float]] = []
    for i in range(max(12, int(n))):
        t = 2.0 * math.pi * float(i) / float(max(12, int(n)))
        ct = math.cos(t)
        st = math.sin(t)
        outer.append([center_x + outer_r * ct, center_y + outer_r * st, z])
    for i in range(max(12, int(n)) - 1, -1, -1):
        t = 2.0 * math.pi * float(i) / float(max(12, int(n)))
        ct = math.cos(t)
        st = math.sin(t)
        inner.append([center_x + inner_r * ct, center_y + inner_r * st, z])
    return outer + inner


def _biconvex_lens_polygon(
    center_x: float,
    center_y: float,
    half_height: float,
    thickness: float,
    *,
    samples: int = 32,
    z: float = COMPONENT_Z,
) -> list[list[float]]:
    y_vals = [(-1.0 + 2.0 * i / float(max(1, samples - 1))) * half_height for i in range(samples)]
    left_side: list[list[float]] = []
    right_side: list[list[float]] = []
    for yy in y_vals:
        blend = 1.0 - min(1.0, abs(yy) / max(half_height, 1.0e-9)) ** 1.55
        half_t = 0.5 * thickness * max(0.0, blend)
        left_side.append([center_x - half_t, center_y + yy, z])
        right_side.append([center_x + half_t, center_y + yy, z])
    return left_side + list(reversed(right_side))


def _parabolic_mirror_polygon(
    apex_x: float,
    center_y: float,
    aperture_half: float,
    sag: float,
    thickness: float,
    *,
    samples: int = 36,
    z: float = COMPONENT_Z,
) -> list[list[float]]:
    ys = [(-1.0 + 2.0 * i / float(max(1, samples - 1))) * aperture_half for i in range(samples)]
    front: list[list[float]] = []
    back: list[list[float]] = []
    for yy in ys:
        frac = yy / max(aperture_half, 1.0e-9)
        xx = apex_x - sag * (frac * frac)
        front.append([xx, center_y + yy, z])
        back.append([xx - thickness, center_y + yy, z])
    return front + list(reversed(back))


def _style_with_scale(style_name: str, span: float) -> dict[str, Any]:
    base = dict(COMPONENT_STYLES.get(style_name, COMPONENT_STYLES["glass"]))
    opacity_scale = _clamp(float(os.environ.get("OPTICS_COMPONENT_OPACITY_SCALE", "1.0")), 0.2, 2.0)
    base["opacity"] = _clamp(float(base.get("opacity", 0.18)) * opacity_scale, 0.06, 0.55)
    base["edge_color"] = list(COMPONENT_EDGE_RGB)
    base["edge_width"] = max(float(base.get("edge_width", 1.5)), 0.0018 * max(1.0, float(span)))
    return base


def _component_specs(meta: dict[str, Any], frame_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    visualization = meta.get("visualization") if isinstance(meta.get("visualization"), dict) else {}
    if int(visualization.get("ray_bundle_count", 0) or 0) <= 0:
        # Field-only cases read cleaner without floating instrument glyphs.
        return []
    field_extent = visualization.get("field_extent") if isinstance(visualization.get("field_extent"), list) else None
    if not isinstance(field_extent, list) or len(field_extent) != 4:
        field_extent = [-1.25, 1.25, -0.6, 0.6]
    x0, x1, y0, y1 = [float(v) for v in field_extent]
    w = max(1.0e-9, x1 - x0)
    h = max(1.0e-9, y1 - y0)
    s = min(w, h)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    family = str(meta.get("family") or meta.get("case_name") or meta.get("case_id") or "")
    optics = meta.get("optics") if isinstance(meta.get("optics"), dict) else {}
    specs: list[dict[str, Any]] = []

    def add(style_name: str, polygons: list[list[list[float]]], **overrides: Any) -> None:
        cleaned = [poly for poly in polygons if isinstance(poly, list) and len(poly) >= 3]
        if not cleaned:
            return
        style = _style_with_scale(style_name, s)
        specs.append({"polygons": cleaned, **style, **overrides})

    if family == "polarizerMalus":
        plate_h = 0.31 * h
        plate_w = 0.012 * w
        left_x = x0 + 0.38 * w
        right_x = x0 + 0.60 * w
        add(
            "polarizer",
            [
                _rectangle_polygon(left_x, cy, plate_w, plate_h),
                _rectangle_polygon(right_x, cy, plate_w, plate_h),
            ],
            representation="Wireframe",
            opacity=0.66,
        )
        return specs

    raw_specs = frame_meta.get("component_specs") if isinstance(frame_meta, dict) else None
    if isinstance(raw_specs, list):
        for item in raw_specs:
            if not isinstance(item, dict):
                continue
            add(
                str(item.get("style", "glass")),
                item.get("polygons") if isinstance(item.get("polygons"), list) else [],
                **{k: v for k, v in item.items() if k not in {"style", "polygons"}},
            )
        if specs:
            return specs

    if family == "planeMirrorReflection":
        add("metal", [_rectangle_polygon(cx, cy, 0.26 * w, 0.010 * h, angle_deg=45.0)], representation="Wireframe", opacity=0.72)
    elif family == "concaveMirrorFocus":
        mirror = optics.get("mirror") if isinstance(optics.get("mirror"), dict) else {}
        aperture = _visual_size(float(mirror.get("aperture_radius", 0.09)), h, 0.18, 0.34)
        add("metal", [_parabolic_mirror_polygon(0.0, cy, aperture, 0.07 * w, 0.018 * w)], representation="Wireframe", opacity=0.72)
    elif family == "prismDispersion":
        prism = optics.get("prism") if isinstance(optics.get("prism"), dict) else {}
        apex_angle = math.radians(_clamp(float(prism.get("apex_angle_deg", 48.0)), 20.0, 80.0))
        base_half = _clamp(0.22 * h, 0.16 * h, 0.30 * h)
        apex_x = 0.15
        base_dx = base_half / max(math.tan(0.5 * apex_angle), 1.0e-6)
        base_x = max(x0 + 0.12 * w, apex_x - base_dx)
        add(
            "glass",
            [[
                [apex_x, 0.0, COMPONENT_Z],
                [base_x, base_half, COMPONENT_Z],
                [base_x, -base_half, COMPONENT_Z],
            ]],
            representation="Wireframe",
            opacity=0.70,
        )
    elif family == "beamSplitterPlate":
        splitter = optics.get("beam_splitter") if isinstance(optics.get("beam_splitter"), dict) else {}
        angle = float(splitter.get("angle_deg", 45.0))
        add("glass", [_rectangle_polygon(cx, cy, 0.22 * s, 0.018 * s, angle_deg=angle)], representation="Wireframe", opacity=0.68)
    elif family == "thinLensFocus":
        lens = optics.get("lens") if isinstance(optics.get("lens"), dict) else {}
        aperture = _visual_size(float(lens.get("aperture_radius", 0.05)), h, 0.18, 0.32)
        add("glass", [_biconvex_lens_polygon(0.0, cy, aperture, 0.050 * w)], representation="Wireframe", opacity=0.68)
    elif family == "glassSphereCaustic":
        sphere = optics.get("sphere") if isinstance(optics.get("sphere"), dict) else {}
        radius = _visual_size(float(sphere.get("radius", 0.2)), s, 0.14, 0.28)
        add("glass", [_annulus_polygon(cx, cy, radius * 0.93, radius, n=72)], representation="Wireframe", opacity=0.68)
    elif family == "cornerCubeRetroreflector":
        aperture = _visual_size(float(((optics.get("corner_cube") or {}) if isinstance(optics.get("corner_cube"), dict) else {}).get("aperture_radius", 0.1)), s, 0.13, 0.24)
        tip_x = cx + 0.22 * w
        add("metal", [_triangle_polygon([(tip_x, cy), (tip_x - 1.65 * aperture, cy + aperture), (tip_x - 1.65 * aperture, cy - aperture)])], representation="Wireframe", opacity=0.70)
    elif family == "polarizerMalus":
        plate_h = 0.30 * h
        plate_w = 0.020 * w
        add(
            "polarizer",
            [
                _rectangle_polygon(-0.12, cy, plate_w, plate_h),
                _rectangle_polygon(0.12, cy, plate_w, plate_h),
            ],
            representation="Wireframe",
            opacity=0.64,
        )

    return specs


def _write_polygon_vtp(path: Path, polygons: list[list[list[float]]]) -> None:
    from vtkmodules.vtkCommonCore import vtkPoints  # type: ignore
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolygon, vtkPolyData  # type: ignore
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter  # type: ignore

    points = vtkPoints()
    polys = vtkCellArray()
    for poly in polygons:
        if len(poly) < 3:
            continue
        cell = vtkPolygon()
        cell.GetPointIds().SetNumberOfIds(len(poly))
        for idx, point in enumerate(poly):
            pid = points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else COMPONENT_Z))
            cell.GetPointIds().SetId(idx, pid)
        polys.InsertNextCell(cell)

    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write component VTP: {path}")


def _show_component(spec: dict[str, Any], path: Path, view):
    from paraview.simple import OpenDataFile, Show  # type: ignore

    _write_polygon_vtp(path, spec.get("polygons", []))
    proxy = OpenDataFile(str(path))
    display = Show(proxy, view)
    representation = str(spec.get("representation", "Surface With Edges"))
    _set_prop(display, "Representation", representation)
    _set_prop(display, "ColorArrayName", [None, ""])
    _set_prop(display, "MapScalars", 0)
    _set_prop(display, "Lighting", 0)
    _set_prop(display, "Ambient", 1.0)
    _set_prop(display, "Diffuse", 0.0)
    _set_prop(display, "Specular", 0.0)
    _set_prop(display, "Opacity", float(spec.get("opacity", 0.18)))
    _set_prop(display, "AmbientColor", spec.get("color", COMPONENT_STYLES["glass"]["color"]))
    _set_prop(display, "DiffuseColor", spec.get("color", COMPONENT_STYLES["glass"]["color"]))
    _set_prop(display, "LineWidth", float(spec.get("edge_width", 1.5)))
    _set_prop(display, "RenderLinesAsTubes", 1)
    if representation != "Wireframe":
        _set_prop(display, "EdgeColor", spec.get("edge_color", COMPONENT_EDGE_RGB))
    return proxy, display


def _preset_low_rgb(preset_name: str) -> list[float]:
    preset = (preset_name or "viridis").strip().lower()
    if preset == "plasma":
        return [0.050383, 0.029803, 0.527975]
    if preset == "inferno":
        return [0.001462, 0.000466, 0.013866]
    return [0.267004, 0.004874, 0.329415]


def _preset_rgb_points(color_lo: float, color_hi: float, preset_name: str) -> list[float]:
    preset = (preset_name or "viridis").strip().lower()
    span = max(1.0e-9, float(color_hi) - float(color_lo))
    stop_sets: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
        "viridis": [
            (0.00, (0.267004, 0.004874, 0.329415)),
            (0.25, (0.229739, 0.322361, 0.545706)),
            (0.50, (0.127568, 0.566949, 0.550556)),
            (0.75, (0.369214, 0.788888, 0.382914)),
            (1.00, (0.993248, 0.906157, 0.143936)),
        ],
        "plasma": [
            (0.00, (0.050383, 0.029803, 0.527975)),
            (0.25, (0.494877, 0.011990, 0.657865)),
            (0.50, (0.798216, 0.280197, 0.469538)),
            (0.75, (0.973416, 0.585761, 0.251540)),
            (1.00, (0.940015, 0.975158, 0.131326)),
        ],
        "inferno": [
            (0.00, (0.001462, 0.000466, 0.013866)),
            (0.25, (0.341500, 0.062325, 0.429425)),
            (0.50, (0.735683, 0.215906, 0.330245)),
            (0.75, (0.978422, 0.557937, 0.034931)),
            (1.00, (0.988362, 0.998364, 0.644924)),
        ],
    }
    stops = stop_sets.get(preset, stop_sets["viridis"])
    points: list[float] = []
    for frac, rgb in stops:
        points.extend([float(color_lo) + span * float(frac), float(rgb[0]), float(rgb[1]), float(rgb[2])])
    return points


def _style_scalar_bar(bar, title: str, color_scale: str) -> None:
    _set_prop(bar, "Title", title)
    _set_prop(bar, "ComponentTitle", "")
    label_format = _colorbar_label_format(color_scale)
    _set_prop(bar, "LabelFormat", label_format)
    _set_prop(bar, "RangeLabelFormat", label_format)
    _set_prop(bar, "WindowLocation", "Any Location")
    _set_prop(bar, "TitleFontSize", int(os.environ.get("PV_COLORBAR_TITLE_FONT_SIZE", "14")))
    _set_prop(bar, "LabelFontSize", int(os.environ.get("PV_COLORBAR_LABEL_FONT_SIZE", "12")))
    _set_prop(bar, "ScalarBarThickness", int(os.environ.get("PV_COLORBAR_THICKNESS", "14")))
    cb_pos_x = float(os.environ.get("PV_COLORBAR_POS_X", "0.875"))
    cb_pos_y = float(os.environ.get("PV_COLORBAR_POS_Y", "0.20"))
    cb_len = float(os.environ.get("PV_COLORBAR_LEN", "0.58"))
    cb_pos_x = min(0.95, max(0.01, cb_pos_x))
    cb_pos_y = min(0.95, max(0.01, cb_pos_y))
    cb_len = min(0.95 - cb_pos_y, max(0.05, cb_len))
    _set_prop(bar, "Position", [cb_pos_x, cb_pos_y])
    _set_prop(bar, "ScalarBarLength", cb_len)
    _set_prop(bar, "NumberOfLabels", COLORBAR_TARGET_LABELS)
    _set_prop(bar, "AddRangeLabels", 0)
    _set_prop(bar, "TitleColor", TEXT_RGB)
    _set_prop(bar, "LabelColor", TEXT_RGB)
    _set_prop(bar, "AutomaticLabelFormat", 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render optics VTK outputs with ParaView")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--target-seconds", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--field-name", default="irradiance")
    parser.add_argument("--color-preset", default="Viridis")
    parser.add_argument("--color-scale", default="auto", choices=["auto", "linear", "log"])
    parser.add_argument("--show-rays", action="store_true")
    parser.add_argument("--show-annotations", action="store_true")
    parser.add_argument("--hide-components", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    frame_dir = args.frame_dir.expanduser().resolve()
    frame_dir.mkdir(parents=True, exist_ok=True)

    series = _read_json(case_dir / "results" / "scan_series.json")
    frames = series.get("frames") if isinstance(series.get("frames"), list) else []
    if not frames:
        raise RuntimeError(f"No scan frames found under {case_dir / 'results' / 'scan_series.json'}")
    meta = _read_json(case_dir / "case_meta.json")
    field_extent = ((meta.get("visualization") or {}).get("field_extent") if isinstance(meta.get("visualization"), dict) else None) or [-1.25, 1.25, -0.6, 0.6]
    if not isinstance(field_extent, list) or len(field_extent) != 4:
        field_extent = [-1.25, 1.25, -0.6, 0.6]
    x0, x1, y0, y1 = [float(v) for v in field_extent]
    family = _case_family(meta, case_dir)

    target_count = max(1, int(round(max(1.0, float(args.target_seconds)) * max(1, int(args.fps)))))
    active_start, active_end = _active_frame_window(frames, family, case_dir=case_dir, field_name=args.field_name)
    chosen = _frame_indices(len(frames), target_count, start=active_start, end=active_end)

    from paraview.simple import (  # type: ignore
        ColorBy,
        Delete,
        GetActiveViewOrCreate,
        GetColorTransferFunction,
        GetScalarBar,
        HideScalarBarIfNotNeeded,
        Hide,
        OpenDataFile,
        Render,
        SaveScreenshot,
        Show,
        Text,
        _DisableFirstRenderCameraReset,
    )
    from vtkmodules.vtkIOXML import vtkXMLImageDataReader  # type: ignore
    from vtkmodules.util.numpy_support import vtk_to_numpy  # type: ignore

    _DisableFirstRenderCameraReset()
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [max(320, int(args.width)), max(240, int(args.height))]
    bg_rgb = _parse_hex_rgb(os.environ.get("PV_BG_COLOR", os.environ.get("OPTICS_BG_COLOR", DEFAULT_BG_HEX)))
    _set_prop(view, "UseColorPaletteForBackground", 0)
    _set_prop(view, "UseGradientBackground", 0)
    _set_prop(view, "BackgroundColorMode", "Single Color")
    _set_prop(view, "Background", bg_rgb)
    _set_prop(view, "Background2", bg_rgb)
    _set_prop(view, "OrientationAxesVisibility", 0)
    _set_prop(view, "CenterAxesVisibility", 0)
    _set_prop(view, "UseFXAA", 1)
    _set_prop(view, "MultiSamples", 8)
    _set_prop(view, "CameraParallelProjection", 1)
    _set_prop(view, "CameraViewUp", [0.0, 1.0, 0.0])
    right_gutter = _clamp(float(os.environ.get("PV_RIGHT_GUTTER_FRACTION", "0.32")), 0.0, 0.80)
    camera_x1 = x1 + (x1 - x0) * right_gutter
    camera_cx = 0.5 * (x0 + camera_x1)
    camera_cy = 0.5 * (y0 + y1)
    view.CameraFocalPoint = [camera_cx, camera_cy, 0.0]
    view.CameraPosition = [camera_cx, camera_cy, 8.0]
    view.CameraParallelScale = _camera_scale((x0, camera_x1, y0, y1), view.ViewSize[0], view.ViewSize[1])

    field_files = [
        case_dir / "results" / "sensor_fields" / str((frames[idx] if 0 <= idx < len(frames) else {}).get("field_file", ""))
        for idx in chosen
    ]
    fixed_values: list[float] = []
    for field_fp in field_files:
        if not field_fp.exists():
            continue
        reader = vtkXMLImageDataReader()
        reader.SetFileName(str(field_fp))
        reader.Update()
        data = reader.GetOutput()
        arr = data.GetPointData().GetArray(str(args.field_name)) if data is not None and data.GetPointData() is not None else None
        if arr is not None:
            fixed_values.extend(_sorted_positive_values(vtk_to_numpy(arr)))
    fixed_values.sort()
    base_lo, base_hi = _base_color_window(fixed_values)
    if args.color_scale == "auto" and family in {"polarizerMalus", "waveplatePolarization"}:
        color_scale = "linear"
    else:
        color_scale = _choose_color_scale(base_lo, base_hi, args.color_scale)
    color_lo, color_hi = _color_window(fixed_values, color_scale)
    if family == "polarizerMalus":
        bg_rgb = _preset_low_rgb(args.color_preset)
        _set_prop(view, "Background", bg_rgb)
        _set_prop(view, "Background2", bg_rgb)
    scalar_labels = _colorbar_labels(color_lo, color_hi, color_scale)
    component_dir = frame_dir / ".components"
    component_dir.mkdir(parents=True, exist_ok=True)
    show_components = not args.hide_components and os.environ.get("OPTICS_SHOW_COMPONENTS", "1").strip().lower() not in {"0", "false", "no"}

    for out_idx, src_idx in enumerate(chosen):
        frame_meta = frames[src_idx] if 0 <= src_idx < len(frames) else {}
        field_fp = case_dir / "results" / "sensor_fields" / str(frame_meta.get("field_file", ""))
        ray_fp = case_dir / "results" / "ray_paths" / str(frame_meta.get("ray_file", ""))
        if not field_fp.exists():
            raise FileNotFoundError(f"Missing VTI field file: {field_fp}")

        field = OpenDataFile(str(field_fp))
        field.UpdatePipeline()

        field_display = Show(field, view)
        field_display.Representation = "Surface"
        ColorBy(field_display, ("POINTS", str(args.field_name)))
        lut = GetColorTransferFunction(str(args.field_name))
        lut.RGBPoints = _preset_rgb_points(color_lo, color_hi, args.color_preset)
        lut.ColorSpace = "RGB"
        _set_prop(lut, "UseLogScale", 0)
        lut.RescaleTransferFunction(color_lo, color_hi)
        _set_prop(lut, "UseLogScale", 1 if color_scale == "log" else 0)
        field_display.LookupTable = lut
        field_display.ColorArrayName = ["POINTS", str(args.field_name)]
        field_display.MapScalars = 1
        field_display.InterpolateScalarsBeforeMapping = 1
        field_display.Opacity = 1.0
        field_display.Ambient = 1.0
        field_display.Diffuse = 0.0
        field_display.Specular = 0.0
        field_display.SetScalarBarVisibility(view, True)

        scalar_bar = GetScalarBar(lut, view)
        _style_scalar_bar(scalar_bar, _field_label(args.field_name), color_scale)
        _set_prop(scalar_bar, "UseCustomLabels", 1)
        _set_prop(scalar_bar, "CustomLabels", scalar_labels)

        component_handles: list[tuple[Any, Any]] = []
        if show_components:
            for comp_idx, spec in enumerate(_component_specs(meta, frame_meta=frame_meta)):
                comp_path = component_dir / f"frame_{out_idx:04d}_component_{comp_idx:02d}.vtp"
                component_handles.append(_show_component(spec, comp_path, view))

        ray = None
        ray_display = None
        if args.show_rays and ray_fp.exists():
            ray = OpenDataFile(str(ray_fp))
            ray_display = Show(ray, view)
            ray_display.Representation = "Wireframe"
            ray_strength = 1.0
            if family == "polarizerMalus":
                ray_strength = math.sqrt(_clamp(_frame_peak(frame_meta), 0.0, 1.0))
            ray_display.LineWidth = 2.0 + 3.0 * ray_strength
            ray_color = [1.0, 0.92, 0.58] if family == "polarizerMalus" else [0.96, 0.96, 0.96]
            ray_display.DiffuseColor = ray_color
            ray_display.AmbientColor = ray_color
            ray_display.Opacity = 0.12 + 0.82 * ray_strength

        text = None
        text_display = None
        if args.show_annotations:
            case_name = meta.get("case_name", case_dir.name)
            param = series.get("parameter", "scan_value")
            value = frame_meta.get("scan_value")
            text = Text()
            text.Text = f"{case_name}\n{param} = {value:.4g}" if isinstance(value, (int, float)) else f"{case_name}\n{param} = {value}"
            text_display = Show(text, view)
            text_display.FontSize = 16
            text_display.Color = TEXT_RGB
            text_display.WindowLocation = "Upper Left Corner"

        Render(view)
        SaveScreenshot(str(frame_dir / f"frame_{out_idx:04d}.png"), view, ImageResolution=view.ViewSize)

        if text_display is not None:
            Delete(text)
            del text_display
        if ray_display is not None:
            Delete(ray)
            del ray_display
        for proxy, display in component_handles:
            Delete(proxy)
            del display
        HideScalarBarIfNotNeeded(lut, view)
        Delete(field)
        del field_display


if __name__ == "__main__":
    main()
