#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from field_utils import (
    _summary,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _scan_values(case: dict[str, Any]) -> np.ndarray:
    scan = case.get("scan") if isinstance(case.get("scan"), dict) else {}
    start = float(scan.get("start", 0.0))
    stop = float(scan.get("stop", 1.0))
    count = max(2, int(scan.get("count", 48)))
    return np.linspace(start, stop, count)


def _grid(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    vis = (
        case.get("visualization") if isinstance(case.get("visualization"), dict) else {}
    )
    res = vis.get("field_resolution", [192, 120])
    extent = vis.get("field_extent", [-1.6, 1.6, -1.0, 1.0])
    if not isinstance(res, list) or len(res) != 2:
        res = [192, 120]
    if not isinstance(extent, list) or len(extent) != 4:
        extent = [-1.6, 1.6, -1.0, 1.0]
    nx = max(48, int(res[0]))
    ny = max(40, int(res[1]))
    x0, x1, y0, y1 = [float(v) for v in extent]
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    return *np.meshgrid(xs, ys), [x0, x1, y0, y1]


def _write_vti(path: Path, arrays: dict[str, np.ndarray], extent: list[float]) -> None:
    first = next(iter(arrays.values()))
    ny, nx = np.asarray(first).shape
    x0, x1, y0, y1 = [float(v) for v in extent]
    dx = (x1 - x0) / float(max(1, nx - 1))
    dy = (y1 - y0) / float(max(1, ny - 1))
    whole = f"0 {nx - 1} 0 {ny - 1} 0 0"
    blocks: list[str] = []
    for name, arr in arrays.items():
        vals = np.asarray(arr, dtype=float)
        if vals.shape != (ny, nx):
            continue
        payload = " ".join(f"{float(v):.9g}" for v in vals.reshape(-1))
        blocks.append(
            f'        <DataArray type="Float32" Name="{name}" format="ascii">{payload}</DataArray>'
        )
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{whole}" Origin="{x0:.9g} {y0:.9g} 0" Spacing="{dx:.9g} {dy:.9g} 1">\n'
        f'    <Piece Extent="{whole}">\n'
        '      <PointData Scalars="irradiance">\n'
        + "\n".join(blocks)
        + "\n      </PointData>\n"
        "      <CellData></CellData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)


def _write_empty_vtp(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        '  <PolyData><Piece NumberOfPoints="0" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n'
        "    <PointData></PointData><CellData></CellData><Points></Points><Lines></Lines>\n"
        "  </Piece></PolyData>\n"
        "</VTKFile>\n"
    )


def _write_pvd(path: Path, datasets: list[tuple[int, str]]) -> None:
    body = "\n".join(
        f'    <DataSet timestep="{i}" group="" part="0" file="{fp}"/>'
        for i, fp in datasets
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n" + body + "\n  </Collection>\n"
        "</VTKFile>\n"
    )


def _rect(
    cx: float, cy: float, hw: float, hh: float, z: float = 0.012
) -> list[list[float]]:
    return [
        [cx - hw, cy - hh, z],
        [cx + hw, cy - hh, z],
        [cx + hw, cy + hh, z],
        [cx - hw, cy + hh, z],
    ]


def _disk(
    cx: float, cy: float, r: float, n: int = 72, z: float = 0.012
) -> list[list[float]]:
    return [
        [
            cx + r * math.cos(2 * math.pi * i / n),
            cy + r * math.sin(2 * math.pi * i / n),
            z,
        ]
        for i in range(n)
    ]


def _norm(field: np.ndarray, scale: float | None = None) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0.0] = 0.0
    peak = (
        float(scale) if scale is not None else (float(np.max(arr)) if arr.size else 0.0)
    )
    return arr / max(peak, 1.0e-12)


def _family_field(
    case: dict[str, Any], xg: np.ndarray, yg: np.ndarray, phase: float
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, float]]:
    family = str(case.get("family", ""))
    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    k = 2.0 * math.pi
    comp: list[dict[str, Any]] = []
    aux: dict[str, float] = {}

    if family == "fdtdCylinderScattering":
        sc = (
            optics.get("scatterer") if isinstance(optics.get("scatterer"), dict) else {}
        )
        radius = float(sc.get("radius", 0.24))
        index = float(sc.get("refractive_index", 1.7))
        r = np.hypot(xg, yg)
        theta = np.arctan2(yg, xg)
        incident = np.cos(k * (xg + 1.2) - 2.0 * math.pi * phase)
        envelope = np.exp(-((yg / 0.42) ** 8))
        scatter_amp = min(1.2, 0.35 * radius * max(0.0, index - 1.0) / 0.2)
        scattered = (
            scatter_amp
            * np.cos(2.8 * k * r - 2.0 * math.pi * phase)
            * np.exp(-(((r - radius) / 0.55) ** 2))
            * (0.35 + 0.65 * np.cos(theta) ** 2)
        )
        ez = envelope * incident + scattered * (r > radius)
        material = (r <= radius).astype(float)
        comp.append(
            {
                "style": "glass",
                "polygons": [_disk(0.0, 0.0, radius)],
                "representation": "Wireframe",
                "opacity": 0.58,
            }
        )
        aux["scattering_contrast"] = float(
            np.std(scattered[r > radius]) / max(np.std(incident), 1.0e-9)
        )
    elif family == "waveguideModePropagation":
        wg = (
            optics.get("waveguide") if isinstance(optics.get("waveguide"), dict) else {}
        )
        width = float(wg.get("core_width", 0.24))
        bend = float(wg.get("bend_strength", 0.12))
        center = bend * np.sin(1.6 * xg + 2.0 * math.pi * phase)
        mode = np.exp(-(((yg - center) / max(width, 1.0e-6)) ** 2))
        ez = mode * (1.0 + 0.22 * np.cos(10.0 * xg - 2.0 * math.pi * phase))
        material = (np.abs(yg) <= width).astype(float)
        comp.append(
            {
                "style": "glass",
                "polygons": [_rect(0.0, 0.0, 1.45, width)],
                "representation": "Wireframe",
                "opacity": 0.46,
            }
        )
        aux["mode_confinement"] = float(
            np.sum((ez * ez) * material) / max(np.sum(ez * ez), 1.0e-12)
        )
    elif family == "photonicCrystalDefectMode":
        pc = (
            optics.get("photonic_crystal")
            if isinstance(optics.get("photonic_crystal"), dict)
            else {}
        )
        pitch = float(pc.get("pitch", 0.3))
        hole = float(pc.get("hole_radius", 0.08))
        defect = float(pc.get("defect_scale", 1.8))
        lattice = np.zeros_like(xg)
        polys: list[list[list[float]]] = []
        for ix in range(-4, 5):
            for iy in range(-3, 4):
                if abs(ix) <= 1 and abs(iy) <= 1:
                    continue
                cx = ix * pitch
                cy = iy * pitch
                rr = (xg - cx) ** 2 + (yg - cy) ** 2
                lattice += np.exp(-rr / max(hole * hole, 1.0e-8))
                if len(polys) < 60:
                    polys.append(_disk(cx, cy, hole, n=24))
        localized = np.exp(-((xg / (0.22 * defect)) ** 2 + (yg / (0.18 * defect)) ** 2))
        standing = 1.0 + 0.28 * np.cos(8.0 * xg + 2.0 * math.pi * phase) * np.cos(
            8.0 * yg
        )
        ez = localized * standing / (1.0 + 0.18 * lattice)
        material = np.clip(lattice, 0.0, 1.0)
        comp.append(
            {
                "style": "mask",
                "polygons": polys,
                "representation": "Wireframe",
                "opacity": 0.34,
            }
        )
        aux["localization_ratio"] = float(
            np.sum((ez * ez) * (np.hypot(xg, yg) < 0.28))
            / max(np.sum(ez * ez), 1.0e-12)
        )
    elif family == "plasmonicNanoDimerHotspot":
        dimer = (
            optics.get("plasmonic_dimer")
            if isinstance(optics.get("plasmonic_dimer"), dict)
            else {}
        )
        radius = float(dimer.get("particle_radius", 0.18))
        gap = float(dimer.get("gap", 0.06))
        resonance = float(dimer.get("resonance_wavelength", 0.62))
        c0 = radius + 0.5 * gap
        left = np.hypot(xg + c0, yg)
        right = np.hypot(xg - c0, yg)
        linewidth = 0.10 + 0.40 * gap
        drive_wavelength = resonance + 0.26 * (phase - 0.5)
        detuning = (drive_wavelength - resonance) / max(linewidth, 1.0e-6)
        resonant_gain = 0.22 + 1.78 * math.exp(-(detuning * detuning))
        breathing = 1.0 + 0.28 * math.sin(2.0 * math.pi * phase)
        waist_x = max(gap * 1.25 * breathing, 1.0e-6)
        waist_y = max(gap * 0.62 / max(breathing, 0.35), 1.0e-6)
        gap_hotspot = np.exp(-((xg / waist_x) ** 2 + (yg / waist_y) ** 2))
        rim = np.exp(-(((left - radius) / 0.045) ** 2)) + np.exp(
            -(((right - radius) / 0.045) ** 2)
        )
        background = 0.55 + 0.12 * math.cos(2.0 * math.pi * phase)
        ez = (
            background
            + (9.5 * resonant_gain) * gap_hotspot
            + (1.45 + 0.55 * resonant_gain) * rim
        )
        material = ((left < radius) | (right < radius)).astype(float)
        comp.append(
            {
                "style": "metal",
                "polygons": [_disk(-c0, 0.0, radius), _disk(c0, 0.0, radius)],
                "representation": "Wireframe",
                "opacity": 0.60,
            }
        )
        aux["hotspot_enhancement"] = float(
            np.max(ez * ez) / max(np.percentile(ez * ez, 50), 1.0e-12)
        )
        aux["resonant_gain"] = float(resonant_gain)
    elif family == "evanescentNearFieldTIR":
        interface = (
            optics.get("interface") if isinstance(optics.get("interface"), dict) else {}
        )
        n1 = float(interface.get("n_high", 1.55))
        n2 = float(interface.get("n_low", 1.0))
        angle = math.radians(float(interface.get("incident_angle_deg", 68.0)))
        decay = 0.10 + 0.22 / max(
            math.sqrt(max(1.0e-6, (n1 * math.sin(angle)) ** 2 - n2 * n2)), 1.0
        )
        upper = yg < 0.0
        incident = np.cos(
            k * (xg * math.sin(angle) - yg * math.cos(angle)) - 2.0 * math.pi * phase
        )
        reflected = np.cos(
            k * (xg * math.sin(angle) + yg * math.cos(angle)) - 2.0 * math.pi * phase
        )
        evanescent = np.exp(-np.maximum(yg, 0.0) / decay) * np.cos(
            k * xg * math.sin(angle) - 2.0 * math.pi * phase
        )
        ez = np.where(upper, incident + reflected, evanescent)
        material = (yg < 0.0).astype(float)
        comp.append(
            {
                "style": "interface",
                "polygons": [_rect(0.0, 0.0, 1.6, 0.012)],
                "representation": "Surface With Edges",
                "opacity": 0.36,
            }
        )
        aux["evanescent_decay_length"] = float(decay)
    else:
        ez = np.cos(k * xg - 2.0 * math.pi * phase) * np.exp(-yg * yg / 0.2)
        material = np.zeros_like(ez)

    return ez, material, comp, aux


def solve_case(case_dir: Path) -> None:
    case = _read_json(case_dir / "solver_input.json")
    values = _scan_values(case)
    xg, yg, extent = _grid(case)
    field_dir = case_dir / "results" / "sensor_fields"
    ray_dir = case_dir / "results" / "ray_paths"
    field_dir.mkdir(parents=True, exist_ok=True)
    ray_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    peaks: list[float] = []
    powers: list[float] = []
    aux_accum: dict[str, list[float]] = {}
    field_datasets: list[tuple[int, str]] = []
    ray_datasets: list[tuple[int, str]] = []
    prepared: list[
        tuple[
            int,
            float,
            np.ndarray,
            np.ndarray,
            list[dict[str, Any]],
            dict[str, float],
            np.ndarray,
        ]
    ] = []
    global_peak = 0.0
    for idx, value in enumerate(values):
        phase = float(value)
        ez, material, comps, aux = _family_field(case, xg, yg, phase)
        raw_intensity = np.nan_to_num(
            np.asarray(ez, dtype=float) ** 2, nan=0.0, posinf=0.0, neginf=0.0
        )
        raw_intensity[raw_intensity < 0.0] = 0.0
        global_peak = max(
            global_peak, float(np.max(raw_intensity)) if raw_intensity.size else 0.0
        )
        prepared.append((idx, float(value), ez, material, comps, aux, raw_intensity))

    for idx, value, ez, material, comps, aux, raw_intensity in prepared:
        intensity = _norm(raw_intensity, scale=global_peak)
        arrays = {
            "irradiance": intensity,
            "electric_field_z": np.asarray(ez, dtype=float),
            "material_indicator": np.asarray(material, dtype=float),
        }
        field_name = f"frame_{idx:04d}.vti"
        ray_name = f"frame_{idx:04d}.vtp"
        _write_vti(field_dir / field_name, arrays, extent)
        _write_empty_vtp(ray_dir / ray_name)
        summary = _summary(intensity, extent)
        peaks.append(summary["peak"])
        powers.append(summary["power"])
        for key, val in aux.items():
            aux_accum.setdefault(key, []).append(float(val))
        frames.append(
            {
                "frame": idx,
                "scan_value": float(value),
                "summary": summary,
                "field_file": field_name,
                "ray_file": ray_name,
                "component_specs": comps,
            }
        )
        field_datasets.append((idx, f"sensor_fields/{field_name}"))
        ray_datasets.append((idx, f"ray_paths/{ray_name}"))

    _write_pvd(case_dir / "results" / "sensor_fields.pvd", field_datasets)
    _write_pvd(case_dir / "results" / "ray_paths.pvd", ray_datasets)
    _write_json(
        case_dir / "results" / "scan_series.json",
        {
            "parameter": (case.get("scan") or {}).get("parameter", "drive_phase"),
            "backend": "numpy_fdtd_pilot",
            "frames": frames,
        },
    )
    metrics = {
        "backend": "numpy_fdtd_pilot",
        "n_frames": len(frames),
        "peak_irradiance_max": max(peaks) if peaks else 0.0,
        "peak_irradiance_mean": float(np.mean(peaks)) if peaks else 0.0,
        "integrated_power_mean": float(np.mean(powers)) if powers else 0.0,
    }
    for key, vals in aux_accum.items():
        metrics[key] = float(np.mean(vals)) if vals else 0.0
    _write_json(case_dir / "postProcessing" / "metrics.json", metrics)
    log = case_dir / "log.solve"
    log.write_text("BEGIN backend=numpy_fdtd_pilot\nEND backend=numpy_fdtd_pilot\n")
    print(f"[run_em_case] ok {case_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run numpy FDTD-style EM pilot case.")
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    solve_case(args.case_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
