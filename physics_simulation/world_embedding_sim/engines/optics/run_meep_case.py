#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cmath
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from field_utils import (
    _clamp,
    _summary,
)


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


def _scan_count(case: dict[str, Any]) -> int:
    override = os.environ.get("OPTICS_MEEP_FRAME_COUNT", "").strip()
    if override:
        try:
            return max(3, int(override))
        except ValueError:
            pass
    scan = case.get("scan") if isinstance(case.get("scan"), dict) else {}
    return max(3, int(scan.get("count", 40)))


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _meep_settings(case: dict[str, Any]) -> dict[str, Any]:
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    settings = solver.get("meep") if isinstance(solver.get("meep"), dict) else {}
    return dict(settings)


def _cell_size(case: dict[str, Any]) -> tuple[float, float]:
    raw = _meep_settings(case).get("cell_size", [7.2, 4.4])
    if not isinstance(raw, list) or len(raw) != 2:
        raw = [7.2, 4.4]
    return float(raw[0]), float(raw[1])


def _field_extent(case: dict[str, Any]) -> list[float]:
    vis = (
        case.get("visualization") if isinstance(case.get("visualization"), dict) else {}
    )
    extent = vis.get("field_extent")
    if isinstance(extent, list) and len(extent) == 4:
        return [float(x) for x in extent]
    sx, sy = _cell_size(case)
    pml = float(_meep_settings(case).get("pml_thickness", 0.55))
    return [-0.5 * sx + pml, 0.5 * sx - pml, -0.5 * sy + pml, 0.5 * sy - pml]


def _field_region(case: dict[str, Any]):
    import meep as mp

    x0, x1, y0, y1 = _field_extent(case)
    return mp.Vector3(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0), mp.Vector3(
        x1 - x0, y1 - y0, 0
    )


def _disk(
    cx: float, cy: float, radius: float, *, n: int = 72, z: float = 0.012
) -> list[list[float]]:
    return [
        [
            cx + radius * math.cos(2.0 * math.pi * i / n),
            cy + radius * math.sin(2.0 * math.pi * i / n),
            z,
        ]
        for i in range(n)
    ]


def _rect(
    cx: float, cy: float, hw: float, hh: float, *, z: float = 0.012
) -> list[list[float]]:
    return [
        [cx - hw, cy - hh, z],
        [cx + hw, cy - hh, z],
        [cx + hw, cy + hh, z],
        [cx - hw, cy + hh, z],
    ]


def _arr2d(arr: Any) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    if vals.ndim != 2:
        vals = np.reshape(vals, (max(1, vals.size), 1))
    # Meep returns x-major arrays; VTI writer expects y-major rows.
    return np.asarray(vals.T, dtype=float)


def _component_from_name(name: str):
    import meep as mp

    return mp.Hz if str(name).strip() == "Hz" else mp.Ez


def _medium(index: float):
    import meep as mp

    return mp.Medium(epsilon=float(index) ** 2)


def _metal_like_medium(case: dict[str, Any]):
    import meep as mp

    dimer = (case.get("optics") or {}).get("plasmonic_dimer") or {}
    freq = float(dimer.get("drude_frequency", 1.45))
    gamma = float(dimer.get("drude_gamma", 0.10))
    try:
        return mp.Medium(
            epsilon=1.0,
            E_susceptibilities=[
                mp.DrudeSusceptibility(frequency=freq, gamma=gamma, sigma=8.0)
            ],
        )
    except Exception:
        return mp.Medium(epsilon=10.0, D_conductivity=0.06)


def _common_source(
    case: dict[str, Any], center_x: float, size_y: float, component_name: str = "Ez"
):
    import meep as mp

    source = (case.get("optics") or {}).get("source") or {}
    wavelength = float(source.get("wavelength", 0.8))
    fcen = 1.0 / max(wavelength, 1.0e-6)
    fwidth = float(_meep_settings(case).get("source_fwidth", 0.18))
    return mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        component=_component_from_name(component_name),
        center=mp.Vector3(center_x, 0, 0),
        size=mp.Vector3(0, size_y, 0),
    )


def _build_scene(case: dict[str, Any]):
    import meep as mp

    family = str(case.get("family", ""))
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    settings = _meep_settings(case)
    sx, sy = _cell_size(case)
    pml = float(settings.get("pml_thickness", 0.55))
    src_x = -0.5 * sx + pml + 0.25
    src_height = max(0.5, sy - 2.0 * pml - 0.3)
    geometry: list[Any] = []
    sources: list[Any] = []
    components: list[dict[str, Any]] = []

    if family == "meepCylinderScattering":
        sc = (
            optics.get("scatterer") if isinstance(optics.get("scatterer"), dict) else {}
        )
        radius = float(sc.get("radius", 0.32))
        index = float(sc.get("refractive_index", 1.8))
        geometry.append(
            mp.Cylinder(radius=radius, center=mp.Vector3(), material=_medium(index))
        )
        sources.append(_common_source(case, src_x, src_height))
        components.append(
            {
                "style": "glass",
                "polygons": [_disk(0.0, 0.0, radius)],
                "representation": "Wireframe",
                "opacity": 0.52,
            }
        )

    elif family == "meepWaveguideMode":
        wg = (
            optics.get("waveguide") if isinstance(optics.get("waveguide"), dict) else {}
        )
        width = float(wg.get("core_width", 0.42))
        index = float(wg.get("core_index", 2.4))
        geometry.append(
            mp.Block(
                center=mp.Vector3(),
                size=mp.Vector3(sx, width, mp.inf),
                material=_medium(index),
            )
        )
        sources.append(_common_source(case, src_x, max(width * 0.95, 0.18)))
        components.append(
            {
                "style": "glass",
                "polygons": [_rect(0.0, 0.0, 0.5 * sx - pml, 0.5 * width)],
                "representation": "Wireframe",
                "opacity": 0.42,
            }
        )

    elif family == "meepPhotonicCrystalDefect":
        pc = (
            optics.get("photonic_crystal")
            if isinstance(optics.get("photonic_crystal"), dict)
            else {}
        )
        pitch = float(pc.get("pitch", 0.55))
        radius = float(pc.get("rod_radius", 0.11))
        index = float(pc.get("rod_index", 3.0))
        polys: list[list[list[float]]] = []
        for ix in range(-4, 5):
            for iy in range(-3, 4):
                if abs(ix) <= 1 and abs(iy) <= 1:
                    continue
                cx, cy = ix * pitch, iy * pitch
                if abs(cx) > 0.5 * sx - pml - 0.25 or abs(cy) > 0.5 * sy - pml - 0.25:
                    continue
                geometry.append(
                    mp.Cylinder(
                        radius=radius,
                        center=mp.Vector3(cx, cy),
                        material=_medium(index),
                    )
                )
                polys.append(_disk(cx, cy, radius, n=24))
        wavelength = float(((optics.get("source") or {}).get("wavelength", 0.82)))
        sources.append(
            mp.Source(
                mp.GaussianSource(frequency=1.0 / wavelength, fwidth=0.22),
                component=mp.Ez,
                center=mp.Vector3(),
            )
        )
        components.append(
            {
                "style": "mask",
                "polygons": polys,
                "representation": "Wireframe",
                "opacity": 0.30,
            }
        )

    elif family == "meepPhotonicCrystalWaveguide":
        pc = (
            optics.get("photonic_crystal_waveguide")
            if isinstance(optics.get("photonic_crystal_waveguide"), dict)
            else {}
        )
        pitch = float(pc.get("pitch", 0.54))
        radius = float(pc.get("rod_radius", 0.11))
        index = float(pc.get("rod_index", 3.0))
        defect_half = 0.5 * float(pc.get("line_defect_width", 0.65))
        polys: list[list[list[float]]] = []
        for ix in range(-5, 6):
            for iy in range(-3, 4):
                cx, cy = ix * pitch, iy * pitch
                if abs(cy) < defect_half:
                    continue
                if abs(cx) > 0.5 * sx - pml - 0.25 or abs(cy) > 0.5 * sy - pml - 0.25:
                    continue
                geometry.append(
                    mp.Cylinder(
                        radius=radius,
                        center=mp.Vector3(cx, cy),
                        material=_medium(index),
                    )
                )
                polys.append(_disk(cx, cy, radius, n=22))
        sources.append(_common_source(case, src_x, max(0.32, defect_half * 1.5)))
        components.append(
            {
                "style": "mask",
                "polygons": polys,
                "representation": "Wireframe",
                "opacity": 0.30,
            }
        )
        components.append(
            {
                "style": "glass",
                "polygons": [_rect(0.0, 0.0, 0.5 * sx - pml, defect_half)],
                "representation": "Wireframe",
                "opacity": 0.12,
            }
        )

    elif family == "meepGratingDiffraction":
        grating = (
            optics.get("grating") if isinstance(optics.get("grating"), dict) else {}
        )
        period = float(grating.get("period", 0.62))
        duty = float(grating.get("duty_cycle", 0.42))
        index = float(grating.get("bar_index", 2.6))
        thick = float(grating.get("bar_thickness", 0.18))
        bar_h = max(0.05, period * duty)
        polys: list[list[list[float]]] = []
        count = int(max(3, math.floor((sy - 2.0 * pml) / period)))
        for j in range(-count, count + 1):
            cy = j * period
            if abs(cy) > 0.5 * sy - pml - 0.1:
                continue
            geometry.append(
                mp.Block(
                    center=mp.Vector3(0, cy),
                    size=mp.Vector3(thick, bar_h, mp.inf),
                    material=_medium(index),
                )
            )
            polys.append(_rect(0.0, cy, 0.5 * thick, 0.5 * bar_h))
        sources.append(_common_source(case, src_x, src_height))
        components.append(
            {
                "style": "mask",
                "polygons": polys,
                "representation": "Surface With Edges",
                "opacity": 0.34,
            }
        )

    elif family == "meepPlasmonicDimer":
        dimer = (
            optics.get("plasmonic_dimer")
            if isinstance(optics.get("plasmonic_dimer"), dict)
            else {}
        )
        radius = float(dimer.get("particle_radius", 0.22))
        gap = float(dimer.get("gap", 0.09))
        cx = radius + 0.5 * gap
        material = _metal_like_medium(case)
        geometry.append(
            mp.Cylinder(radius=radius, center=mp.Vector3(-cx, 0), material=material)
        )
        geometry.append(
            mp.Cylinder(radius=radius, center=mp.Vector3(cx, 0), material=material)
        )
        sources.append(_common_source(case, src_x, src_height, component_name="Hz"))
        components.append(
            {
                "style": "metal",
                "polygons": [_disk(-cx, 0.0, radius), _disk(cx, 0.0, radius)],
                "representation": "Wireframe",
                "opacity": 0.58,
            }
        )

    elif family == "meepDipoleRadiation":
        dipole = optics.get("dipole") if isinstance(optics.get("dipole"), dict) else {}
        wavelength = float(dipole.get("wavelength", 0.8))
        component = _component_from_name(str(dipole.get("orientation", "Ez")))
        sources.append(
            mp.Source(
                mp.GaussianSource(frequency=1.0 / wavelength, fwidth=0.22),
                component=component,
                center=mp.Vector3(),
            )
        )
        components.append(
            {
                "style": "source",
                "polygons": [_disk(0.0, 0.0, 0.055, n=24)],
                "representation": "Surface With Edges",
                "opacity": 0.75,
            }
        )

    elif family == "meepEvanescentTIR":
        interface = (
            optics.get("interface") if isinstance(optics.get("interface"), dict) else {}
        )
        n1 = float(interface.get("n_high", 1.7))
        n2 = float(interface.get("n_low", 1.0))
        angle = math.radians(float(interface.get("incident_angle_deg", 66.0)))
        wavelength = float(((optics.get("source") or {}).get("wavelength", 0.8)))
        fcen = 1.0 / wavelength
        geometry.append(
            mp.Block(
                center=mp.Vector3(0, -0.25 * sy),
                size=mp.Vector3(sx, 0.5 * sy, mp.inf),
                material=_medium(n1),
            )
        )
        kx = 2.0 * math.pi * n1 * fcen * math.sin(angle)

        def amp_func(r):
            return cmath.exp(1j * kx * r.x)

        sources.append(
            mp.Source(
                mp.GaussianSource(frequency=fcen, fwidth=0.18),
                component=mp.Ez,
                center=mp.Vector3(0, -0.58),
                size=mp.Vector3(max(0.5, sx - 2.0 * pml - 0.6), 0, 0),
                amp_func=amp_func,
            )
        )
        components.append(
            {
                "style": "interface",
                "polygons": [_rect(0.0, 0.0, 0.5 * sx - pml, 0.018)],
                "representation": "Surface With Edges",
                "opacity": 0.28,
            }
        )
        components.append(
            {
                "style": "glass",
                "polygons": [
                    _rect(0.0, -0.25 * sy, 0.5 * sx - pml, 0.25 * sy - pml * 0.2)
                ],
                "representation": "Wireframe",
                "opacity": 0.12,
            }
        )

    else:
        raise KeyError(f"Unsupported Meep family: {family}")

    return geometry, sources, components


def _run_meep(case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import meep as mp

    if hasattr(mp, "verbosity"):
        mp.verbosity(0)
    settings = _meep_settings(case)
    sx, sy = _cell_size(case)
    resolution = int(settings.get("resolution", 16))
    pml = float(settings.get("pml_thickness", 0.55))
    count = _scan_count(case)
    run_time = float(
        settings.get("run_time", (case.get("scan") or {}).get("stop", 18.0))
    )
    record_start_fraction = _clamp(
        _float_env(
            "OPTICS_MEEP_RECORD_START_FRACTION",
            float(settings.get("record_start_fraction", 0.0)),
        ),
        0.0,
        0.95,
    )
    record_start_time_default = run_time * record_start_fraction
    if "record_start_time" in settings:
        record_start_time_default = float(
            settings.get("record_start_time", record_start_time_default)
        )
    record_start_time = _clamp(
        _float_env("OPTICS_MEEP_RECORD_START_TIME", record_start_time_default),
        0.0,
        run_time * 0.95,
    )
    record_start_fraction = record_start_time / max(run_time, 1.0e-12)
    record_span = max(run_time - record_start_time, run_time / float(max(1, count)))
    dt = record_span / float(max(1, count))
    geometry, sources, components = _build_scene(case)

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, 0),
        boundary_layers=[mp.PML(pml)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        force_complex_fields=True,
    )

    center, size = _field_region(case)
    extent = _field_extent(case)
    raw_frames: list[dict[str, Any]] = []
    global_peak = 0.0
    if record_start_time > 1.0e-12:
        sim.run(until=record_start_time)
    for idx in range(count):
        sim.run(until=dt)
        ez = _arr2d(np.real(sim.get_array(center=center, size=size, component=mp.Ez)))
        try:
            ex = _arr2d(
                np.real(sim.get_array(center=center, size=size, component=mp.Ex))
            )
        except Exception:
            ex = np.zeros_like(ez)
        try:
            ey = _arr2d(
                np.real(sim.get_array(center=center, size=size, component=mp.Ey))
            )
        except Exception:
            ey = np.zeros_like(ez)
        try:
            hz = _arr2d(
                np.real(sim.get_array(center=center, size=size, component=mp.Hz))
            )
        except Exception:
            hz = np.zeros_like(ez)
        try:
            eps = _arr2d(
                np.real(
                    sim.get_array(center=center, size=size, component=mp.Dielectric)
                )
            )
        except Exception:
            eps = np.ones_like(ez)
        raw_intensity = np.nan_to_num(
            ez * ez + ex * ex + ey * ey + 0.15 * hz * hz,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        raw_intensity[raw_intensity < 0.0] = 0.0
        global_peak = max(
            global_peak, float(np.max(raw_intensity)) if raw_intensity.size else 0.0
        )
        raw_frames.append(
            {
                "idx": idx,
                "time": float(sim.meep_time()),
                "ex": ex,
                "ey": ey,
                "ez": ez,
                "hz": hz,
                "eps": eps,
                "raw": raw_intensity,
            }
        )

    frames: list[dict[str, Any]] = []
    peaks: list[float] = []
    powers: list[float] = []
    for frame in raw_frames:
        intensity = frame["raw"] / max(global_peak, 1.0e-12)
        eps = frame["eps"]
        eps_span = float(np.nanmax(eps) - np.nanmin(eps)) if eps.size else 0.0
        material = (eps - float(np.nanmin(eps))) / max(eps_span, 1.0e-12)
        summary = _summary(intensity, extent)
        peaks.append(summary["peak"])
        powers.append(summary["power"])
        frames.append(
            {
                "value": frame["time"],
                "arrays": {
                    "irradiance": intensity,
                    "electric_field_x": frame["ex"],
                    "electric_field_y": frame["ey"],
                    "electric_field_z": frame["ez"],
                    "magnetic_field_z": frame["hz"],
                    "material_indicator": material,
                    "relative_permittivity": eps,
                },
                "points": np.zeros((0, 3), dtype=float),
                "lines": [],
                "cell_data": {},
                "summary": summary,
                "component_specs": components,
            }
        )

    metrics = _family_metrics(case, frames, extent)
    metrics.update(
        {
            "backend": "meep",
            "meep_version": getattr(mp, "__version__", "unknown"),
            "resolution": resolution,
            "run_time": run_time,
            "record_start_time": record_start_time,
            "record_start_fraction": record_start_fraction,
            "record_time_span": record_span,
            "recorded_frame_count": count,
            "pml_thickness": pml,
            "peak_irradiance_max": max(peaks) if peaks else 0.0,
            "peak_irradiance_mean": float(np.mean(peaks)) if peaks else 0.0,
            "integrated_power_mean": float(np.mean(powers)) if powers else 0.0,
            "pml_edge_leakage_fraction": _edge_leakage(
                frames[-1]["arrays"]["irradiance"]
            )
            if frames
            else 1.0,
        }
    )
    return frames, metrics


def _edge_leakage(field: np.ndarray) -> float:
    arr = np.asarray(field, dtype=float)
    if arr.size == 0:
        return 1.0
    edge = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.mean(edge) / max(float(np.mean(arr)), 1.0e-12))


def _family_metrics(
    case: dict[str, Any], frames: list[dict[str, Any]], extent: list[float]
) -> dict[str, float]:
    family = str(case.get("family", ""))
    if not frames:
        return {}
    metric_frame = max(
        frames,
        key=lambda item: float((item.get("summary") or {}).get("power", 0.0))
        + float((item.get("summary") or {}).get("peak", 0.0)),
    )
    field = np.asarray(metric_frame["arrays"]["irradiance"], dtype=float)
    material = np.asarray(metric_frame["arrays"]["material_indicator"], dtype=float)
    ny, nx = field.shape
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xg, yg = np.meshgrid(xs, ys)
    total = max(float(np.sum(field)), 1.0e-12)
    out: dict[str, float] = {}

    if family == "meepCylinderScattering":
        sc = (case.get("optics") or {}).get("scatterer") or {}
        radius = float(sc.get("radius", 0.3))
        annulus = (np.hypot(xg, yg) > radius * 1.15) & (np.hypot(xg, yg) < radius + 0.9)
        out["scattering_contrast"] = (
            float(np.std(field[annulus])) if np.any(annulus) else 0.0
        )
        out["forward_power_proxy"] = float(np.sum(field[xg > radius + 0.4]) / total)
        out["backward_power_proxy"] = float(np.sum(field[xg < -radius - 0.4]) / total)
    elif family == "meepWaveguideMode":
        width = float(
            (((case.get("optics") or {}).get("waveguide") or {}).get("core_width", 0.4))
        )
        core = np.abs(yg) <= 0.5 * width
        out["mode_confinement"] = float(np.sum(field[core]) / total)
        out["waveguide_leakage"] = float(np.sum(field[~core]) / total)
    elif family == "meepPhotonicCrystalDefect":
        center = np.hypot(xg, yg) <= 0.55
        out["localization_ratio"] = float(np.sum(field[center]) / total)
        out["lattice_material_fill"] = float(np.mean(material))
    elif family == "meepPhotonicCrystalWaveguide":
        core = np.abs(yg) <= 0.38
        out["line_defect_confinement"] = float(np.sum(field[core]) / total)
        out["right_transmission_proxy"] = float(np.sum(field[xg > 1.5]) / total)
    elif family == "meepGratingDiffraction":
        right = field[:, max(0, nx - max(3, nx // 10))]
        profile = np.asarray(right, dtype=float)
        cutoff = float(np.max(profile)) * 0.18 if profile.size else 0.0
        peaks = []
        for i in range(1, len(profile) - 1):
            if (
                profile[i] > cutoff
                and profile[i] >= profile[i - 1]
                and profile[i] >= profile[i + 1]
            ):
                peaks.append(i)
        out["grating_order_count"] = float(len(peaks))
        if len(peaks) >= 2:
            out["grating_order_spacing"] = float(np.mean(np.diff(ys[peaks])))
        else:
            out["grating_order_spacing"] = 0.0
    elif family == "meepPlasmonicDimer":
        dimer = (case.get("optics") or {}).get("plasmonic_dimer") or {}
        gap = float(dimer.get("gap", 0.1))
        radius = float(dimer.get("particle_radius", 0.2))
        gap_mask = (np.abs(xg) <= max(0.5 * gap, 0.03)) & (np.abs(yg) <= radius * 0.75)
        background = np.percentile(field, 50)
        out["hotspot_enhancement"] = (
            float(np.max(field[gap_mask]) / max(float(background), 1.0e-12))
            if np.any(gap_mask)
            else 0.0
        )
        out["gap_localization_ratio"] = (
            float(np.sum(field[gap_mask]) / total) if np.any(gap_mask) else 0.0
        )
    elif family == "meepDipoleRadiation":
        r = np.hypot(xg, yg)
        shell = (r > 0.9) & (r < 1.45)
        vals = field[shell]
        out["radiation_shell_contrast"] = (
            float(np.std(vals) / max(float(np.mean(vals)), 1.0e-12))
            if vals.size
            else 0.0
        )
        out["radiated_power_proxy"] = total
    elif family == "meepEvanescentTIR":
        upper = yg > 0.08
        if np.any(upper):
            rows = []
            yvals = []
            for j, y in enumerate(ys):
                if y <= 0.08:
                    continue
                rows.append(max(float(np.mean(field[j, :])), 1.0e-12))
                yvals.append(float(y))
            if len(rows) >= 3:
                slope, _ = np.polyfit(np.asarray(yvals), np.log(np.asarray(rows)), 1)
                out["evanescent_decay_length"] = float(
                    -1.0 / min(float(slope), -1.0e-9)
                )
        out["upper_field_fraction"] = (
            float(np.sum(field[upper]) / total) if np.any(upper) else 0.0
        )
    return out


def simulate_case(
    case: dict[str, Any], backend: str = "meep"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if backend != "meep":
        raise ValueError(f"run_meep_case only supports backend='meep', got {backend}")
    return _run_meep(case)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Meep-backed optics/EM case through the standard writer."
    )
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    import run_case as standard_run_case

    standard_run_case.run_case(
        args.case_dir,
        backend="meep",
        strict_mitsuba=False,
        force_rerun=bool(args.force_rerun),
    )


if __name__ == "__main__":
    main()
