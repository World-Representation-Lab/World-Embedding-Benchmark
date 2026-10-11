#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fe_utils import (
    _resample_path,
    _principal_2d,
    _write_pvd,
)


@dataclass
class SolveResult:
    backend: str
    n_steps: int
    times: list[float]
    max_disp: float
    max_vm_stress: float
    max_reaction_force: float
    hysteresis_area: float
    localization_index: float
    snap_indicator: float
    crack_length_final: float = 0.0
    plastic_strain_p95: float = 0.0
    strain_energy_total: float = 0.0
    hole_gradient_contrast: float = 0.0
    load_history: list[float] = field(default_factory=list)
    max_disp_history: list[float] = field(default_factory=list)
    max_vm_history: list[float] = field(default_factory=list)
    strain_energy_history: list[float] = field(default_factory=list)
    plastic_strain_history: list[float] = field(default_factory=list)
    crack_length_history: list[float] = field(default_factory=list)
    dim: str = "2D"
    nonlinear_model: str = "none"


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


def _load_case(case_dir: Path) -> dict[str, Any]:
    case = _read_json(case_dir / "solver_input.json")
    if case:
        return case
    return _read_json(case_dir / "case_meta.json")


def _case_dim(case: dict[str, Any]) -> str:
    dim_raw = str(case.get("dim", "")).strip().upper()
    if dim_raw in {"2D", "3D"}:
        return dim_raw
    geom = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    if str(geom.get("shape_3d", "")).strip():
        return "3D"
    return "2D"


def _nonlinear_model(case: dict[str, Any]) -> str:
    nl = case.get("nonlinear") if isinstance(case.get("nonlinear"), dict) else {}
    model = str(nl.get("model", "none")).strip().lower()
    return model if model else "none"


def _case_requires_fe(case: dict[str, Any]) -> bool:
    del case
    return True


def _mesh(
    case: dict[str, Any]
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]], float, float
]:
    geom = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    mesh = solver.get("mesh") if isinstance(solver.get("mesh"), dict) else {}

    lx = float(geom.get("lx", 1.0))
    ly = float(geom.get("ly", 0.3))
    nx = max(12, int(mesh.get("nx", 96)))
    ny = max(6, int(mesh.get("ny", 36)))
    family = str(case.get("family") or case.get("case_prefix") or "")
    if family == "centerNotchPlate2D":
        # Keep center notch and ligament aligned with mesh symmetry planes.
        if nx % 2 != 0:
            nx += 1
        if ny % 2 != 0:
            ny += 1

    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(-0.5 * ly, 0.5 * ly, ny + 1)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")

    points = np.column_stack([xg.ravel(), yg.ravel(), np.zeros(xg.size, dtype=float)])
    quads: list[list[int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n0 = j * stride + i
            n1 = n0 + 1
            n2 = n1 + stride
            n3 = n0 + stride
            quads.append([n0, n1, n2, n3])

    return xg, yg, points, np.asarray(quads, dtype=np.int64), quads, lx, ly


def _load_history(case: dict[str, Any], n_steps: int) -> list[float]:
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}
    raw = loading.get("load_path")
    if isinstance(raw, list) and raw:
        vals = [float(x) for x in raw]
    else:
        vals = []
    vals = _resample_path(vals, n_steps)
    return [max(0.0, min(1.0, float(x))) for x in vals]


def _gradient_fields(
    ux: np.ndarray, uy: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dux_dy, dux_dx = np.gradient(ux, dy, dx, edge_order=1)
    duy_dy, duy_dx = np.gradient(uy, dy, dx, edge_order=1)
    eps_xx = dux_dx
    eps_yy = duy_dy
    eps_xy = 0.5 * (dux_dy + duy_dx)
    return eps_xx, eps_yy, eps_xy


def _stress_from_strain(
    eps_xx: np.ndarray,
    eps_yy: np.ndarray,
    eps_xy: np.ndarray,
    E: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Plane stress constitutive relation.
    denom = max(1e-12, 1.0 - nu * nu)
    c = E / denom
    sxx = c * (eps_xx + nu * eps_yy)
    syy = c * (eps_yy + nu * eps_xx)
    txy = E / max(1e-12, 1.0 + nu) * eps_xy
    vm = np.sqrt(np.maximum(0.0, sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy))
    return sxx, syy, txy, vm


def _family_fields(
    *,
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    lx: float,
    ly: float,
    lam: float,
    load_scale: float,
    sign: float,
    material: dict[str, Any],
    geometry: dict[str, Any],
    loading: dict[str, Any],
    state: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    eps = sign * load_scale * lam
    nu = float(material.get("nu", 0.3))

    if family == "uniaxialTension" or family == "uniaxialCompression":
        ux = eps * x
        uy = -nu * eps * y
        return ux, uy

    if family == "yieldHardening":
        yld = float(material.get("yield_strain", 0.012))
        hard = float(
            material.get(
                "hardening_modulus", max(1.0, float(material.get("E", 1.0)) * 0.02)
            )
        )
        E = max(1.0, float(material.get("E", 1.0)))
        hard_ratio = min(0.35, max(0.005, hard / E))
        ee = min(abs(eps), yld)
        ep = max(0.0, abs(eps) - yld)
        e_eff = math.copysign(ee + hard_ratio * ep * 2.0, eps)
        neck_center = float(geometry.get("neck_region_center", 0.55 * lx))
        neck_w = max(1e-6, float(geometry.get("neck_region_width", 0.16 * lx)))
        band = np.exp(-(((x - neck_center) / neck_w) ** 2))
        ux = e_eff * x * (1.0 + 0.05 * band)
        uy = -nu * e_eff * y * (1.0 + 0.35 * band)
        return ux, uy

    if family == "neoHookean":
        stretch = max(0.35, 1.0 + eps)
        ux = (stretch - 1.0) * x + 0.02 * eps * (y / max(1e-9, ly)) * x
        uy = (stretch ** (-0.5) - 1.0) * y
        return ux, uy

    if family == "cantileverLocalization":
        xi = x / max(1e-9, lx)
        shape = xi * xi * (3.0 - 2.0 * xi)
        band_x = float(geometry.get("localization_band_x", 0.45 * lx))
        band = np.exp(-(((x - band_x) / max(1e-9, 0.14 * lx)) ** 2))
        tip = abs(eps) * lx
        uy = -tip * shape * (1.0 + 0.6 * band)
        ux = 0.14 * uy * (y / max(1e-9, 0.5 * ly))
        return ux, uy

    if family == "loadUnloadHysteresis":
        prev = state.get("prev_eps", 0.0)
        yld = float(material.get("yield_stress", 0.0)) / max(
            1.0, float(material.get("E", 1.0))
        )
        yld = max(0.002, min(0.08, yld))
        p = state.get("plastic", 0.0)
        if abs(eps) > yld:
            p = 0.9 * p + 0.1 * math.copysign(abs(eps) - yld, eps)
        if (eps - prev) * prev < 0.0:
            p *= 0.94
        state["plastic"] = p
        state["prev_eps"] = eps

        e_eff = eps - 0.65 * p
        ux = e_eff * x
        uy = -nu * e_eff * y
        return ux, uy

    if family == "bucklingSnapthrough":
        critical = float(loading.get("critical_load_factor", 0.55))
        imperf = float(geometry.get("imperfection", 0.002))
        xi = x / max(1e-9, lx)
        mode = np.sin(math.pi * xi)
        amp = 0.02 * lx * math.tanh(9.0 * (lam - critical))
        uy = (imperf * lx + amp) * mode
        ux = eps * x * (1.0 - 0.15 * mode * mode)
        return ux, uy

    if family == "necking":
        eps_t = abs(eps)
        center = float(geometry.get("neck_region_center", 0.55 * lx))
        width = max(1e-9, float(geometry.get("neck_region_width", 0.14 * lx)))
        loc = np.exp(-(((x - center) / width) ** 2))
        onset = float(loading.get("damage_onset", 0.55))
        neck = max(0.0, min(1.0, (lam - onset) / max(1e-6, 1.0 - onset)))
        ux = math.copysign(eps_t, eps) * x * (1.0 - 0.35 * neck * loc)
        uy = -nu * math.copysign(eps_t, eps) * y * (1.0 + 2.0 * neck * loc)
        return ux, uy

    if family == "plateWithHole":
        cx = float(geometry.get("hole_cx", 0.5 * lx))
        cy = float(geometry.get("hole_cy", 0.0))
        r0 = max(1e-6, float(geometry.get("hole_radius", 0.12 * min(lx, ly))))
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-9
        theta = np.arctan2(dy, dx)
        conc = 1.0 + (r0 * r0 / (r * r)) * np.cos(2.0 * theta)
        mask = np.exp(-((r / (2.8 * r0)) ** 2))
        ux = eps * x * (1.0 + 0.16 * conc * mask)
        uy = -nu * eps * y * (1.0 - 0.1 * conc * mask)
        return ux, uy

    if family == "pureShear":
        shear_scale = float(loading.get("shear_ratio_xy", 1.0))
        gamma = eps * shear_scale
        ux = gamma * y
        uy = 0.08 * gamma * x * (y / max(1e-9, 0.5 * ly))
        return ux, uy

    if family == "threePointBending":
        xi = x / max(1e-9, lx)
        support_span = float(geometry.get("support_span", 0.82 * lx))
        span_ratio = max(0.55, min(0.98, support_span / max(1e-9, lx)))
        curve = np.sin(math.pi * xi) ** 2
        center_drop_ratio = float(loading.get("center_drop_ratio", 1.0))
        drop = abs(eps) * lx * center_drop_ratio
        uy = -drop * curve * (0.7 + 0.6 * span_ratio)
        ux = 0.22 * uy * (y / max(1e-9, 0.5 * ly))
        return ux, uy

    if family == "torsionShear":
        xc = float(geometry.get("twist_axis_x", 0.5 * lx))
        wc = float(geometry.get("warping_coeff", 0.2))
        twist_strength = float(loading.get("twist_strength", 1.0))
        phi = eps * twist_strength
        xr = x - xc
        yr = y
        ux = -phi * yr * (1.0 + wc * (xr / max(1e-9, lx)))
        uy = phi * xr * (1.0 - 0.5 * wc * (yr / max(1e-9, 0.5 * ly)))
        return ux, uy

    if family == "biaxialTension":
        ratio = float(loading.get("biaxial_ratio", geometry.get("biaxial_ratio", 1.0)))
        ratio = max(0.2, min(2.0, ratio))
        ex = eps
        ey = eps * ratio
        ux = ex * x
        uy = ey * y
        return ux, uy

    if family == "contactPatchCompression":
        sign_c = -abs(eps)
        cx = float(geometry.get("patch_center_x", 0.5 * lx))
        hw = max(1e-9, float(geometry.get("patch_half_width", 0.12 * lx)))
        peak = float(loading.get("patch_peak_ratio", 1.0))
        band = np.exp(-(((x - cx) / hw) ** 2))
        uy = sign_c * ly * (0.12 + 0.6 * peak * band) * lam
        ux = (
            0.18
            * abs(uy)
            * np.tanh((x - cx) / max(1e-9, 0.6 * hw))
            * (y / max(1e-9, 0.5 * ly))
        )
        return ux, uy

    if family == "nearIncompressibleBlock":
        vol_w = float(loading.get("volumetric_weight", 1.0))
        ex = eps
        ey = -ex * (0.92 + 0.06 * vol_w)
        comp = ex + ey
        ux = ex * x + 0.03 * comp * (x / max(1e-9, lx)) * (y / max(1e-9, ly))
        uy = ey * y - 0.02 * comp * (x / max(1e-9, lx)) ** 2 * y
        return ux, uy

    if family == "centerNotchPlate2D":
        notch_x = float(geometry.get("notch_x", 0.5 * lx))
        notch_h = max(1e-6, float(geometry.get("notch_half_height", 0.04 * ly)))
        notch_a = max(2.0e-3 * lx, float(geometry.get("notch_half_length", 0.08 * lx)))
        crack_half_ratio = min(
            0.90, max(0.25, float(geometry.get("target_crack_half_ratio", 0.72)))
        )
        # Symmetric extension around the notch center avoids one-sided boundary bias.
        ux = eps * (x - notch_x)
        uy = -nu * eps * y
        dx_c = np.abs(x - notch_x)
        y_band = np.exp(-((y / max(1e-9, notch_h)) ** 2))
        ahead = np.maximum(0.0, dx_c - notch_a)
        lig_half = max(1e-9, 0.5 * lx - notch_a)
        tip = np.exp(
            -(
                ((dx_c - notch_a) / max(1e-9, 0.03 * lx)) ** 2
                + (y / max(1e-9, 1.2 * notch_h)) ** 2
            )
        )
        drive = np.clip((lam - 0.06) / 0.94, 0.0, 1.0)
        target_len = max(1.5 * notch_h, crack_half_ratio * drive * lig_half)
        trail = np.clip(
            (target_len + 2.0 * notch_h - ahead)
            / max(target_len + 2.0 * notch_h, 1e-9),
            0.0,
            1.0,
        )
        opening = (
            abs(eps)
            * lx
            * drive
            * (0.015 * y_band + 0.045 * y_band * np.power(trail, 0.75) + 0.02 * tip)
        )
        uy = uy + np.sign(y + 1e-12) * opening
        return ux, uy

    # Default linear elastic response.
    ux = eps * x
    uy = -nu * eps * y
    return ux, uy


def _write_vtu(
    path: Path, points: np.ndarray, quads: np.ndarray, point_data: dict[str, np.ndarray]
) -> None:
    n_points = int(points.shape[0])
    n_cells = int(quads.shape[0])
    if n_cells <= 0:
        raise RuntimeError(f"Refusing to write point-only VTU (zero cells): {path}")

    conn = quads.reshape(-1)
    offsets = np.arange(1, n_cells + 1, dtype=np.int64) * 4
    types = np.full((n_cells,), 9, dtype=np.uint8)  # VTK_QUAD

    def fmt_float_array(arr: np.ndarray) -> str:
        if arr.ndim == 1:
            return " ".join(f"{float(x):.9g}" for x in arr)
        if arr.ndim == 2:
            return " ".join(" ".join(f"{float(v):.9g}" for v in row) for row in arr)
        raise ValueError("Unsupported array shape")

    def fmt_int_array(arr: np.ndarray) -> str:
        if arr.ndim == 1:
            return " ".join(str(int(x)) for x in arr)
        if arr.ndim == 2:
            return " ".join(" ".join(str(int(v)) for v in row) for row in arr)
        raise ValueError("Unsupported array shape")

    point_data_xml = []
    for name, arr in point_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            point_data_xml.append(
                f'<DataArray type="Float64" Name="{name}" format="ascii">{fmt_float_array(arr)}</DataArray>'
            )
        elif arr.ndim == 2 and arr.shape[1] in {2, 3}:
            ncomp = int(arr.shape[1])
            point_data_xml.append(
                f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">'
                f"{fmt_float_array(arr)}"
                "</DataArray>"
            )
        else:
            raise ValueError(f"Unsupported point data array for {name}: {arr.shape}")

    xml = f"""<?xml version=\"1.0\"?>
<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">
  <UnstructuredGrid>
    <Piece NumberOfPoints=\"{n_points}\" NumberOfCells=\"{n_cells}\">
      <PointData>
        {' '.join(point_data_xml)}
      </PointData>
      <CellData></CellData>
      <Points>
        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">{fmt_float_array(points)}</DataArray>
      </Points>
      <Cells>
        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">{fmt_int_array(conn)}</DataArray>
        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">{fmt_int_array(offsets)}</DataArray>
        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">{fmt_int_array(types)}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    path.write_text(xml)


def _run_surrogate(case_dir: Path, case: dict[str, Any]) -> SolveResult:
    family = str(
        case.get("family") or case.get("case_prefix") or case_dir.name.split("_", 1)[0]
    )
    material = case.get("material") if isinstance(case.get("material"), dict) else {}
    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}

    n_steps = max(6, int(solver.get("n_steps", 20)))
    x, y, points, quads_arr, quads_list, lx, ly = _mesh(case)
    _ = quads_list

    load_scale = float(loading.get("load_factor", 0.12))
    sign = float(loading.get("load_sign", 1.0))

    E = max(1e4, float(material.get("E", 2.1e11)))
    nu = float(material.get("nu", 0.3))
    thickness = float(geometry.get("thickness", 0.01))

    results_dir = case_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    hist = _load_history(case, n_steps)
    times = [i / max(1, n_steps - 1) for i in range(n_steps)]

    dx = (
        lx / max(1, int(solver.get("mesh", {}).get("nx", 96)))
        if isinstance(solver.get("mesh"), dict)
        else lx / 96.0
    )
    dy = (
        ly / max(1, int(solver.get("mesh", {}).get("ny", 36)))
        if isinstance(solver.get("mesh"), dict)
        else ly / 36.0
    )

    vtu_files: list[str] = []
    max_disp = 0.0
    max_vm = 0.0
    max_rf = 0.0
    force_hist: list[float] = []
    disp_hist: list[float] = []
    max_disp_hist: list[float] = []
    max_vm_hist: list[float] = []
    strain_energy_hist: list[float] = []
    loc_hist: list[float] = []
    snap_hist: list[float] = []
    crack_hist: list[float] = []
    prev_u3: np.ndarray | None = None

    state: dict[str, float] = {"plastic": 0.0, "prev_eps": 0.0}

    for i, lam in enumerate(hist):
        ux, uy = _family_fields(
            family=family,
            x=x,
            y=y,
            lx=lx,
            ly=ly,
            lam=float(lam),
            load_scale=load_scale,
            sign=sign,
            material=material,
            geometry=geometry,
            loading=loading,
            state=state,
        )

        eps_xx, eps_yy, eps_xy = _gradient_fields(ux, uy, dx, dy)
        sxx, syy, txy, vm = _stress_from_strain(eps_xx, eps_yy, eps_xy, E=E, nu=nu)
        sig1, _ = _principal_2d(sxx, syy, txy)
        eps1, _ = _principal_2d(eps_xx, eps_yy, eps_xy)

        disp_mag = np.sqrt(ux * ux + uy * uy)
        max_disp = max(max_disp, float(np.max(disp_mag)))
        max_vm = max(max_vm, float(np.max(vm)))
        max_disp_hist.append(float(np.max(disp_mag)))
        max_vm_hist.append(float(np.max(vm)))

        rf = float(np.mean(sxx[:, -1]) * ly * thickness)
        max_rf = max(max_rf, abs(rf))
        force_hist.append(rf)
        disp_hist.append(float(np.mean(ux[:, -1])))
        step_energy = abs(rf * disp_hist[-1])
        prev_energy = strain_energy_hist[-1] if strain_energy_hist else 0.0
        strain_energy_hist.append(float(prev_energy + step_energy))

        loc = float(np.max(vm) / max(1e-12, np.mean(vm) + 1e-12))
        loc_hist.append(loc)
        snap_hist.append(float(np.max(np.abs(uy))))

        u3 = np.column_stack([ux.ravel(), uy.ravel(), np.zeros_like(ux).ravel()])
        if prev_u3 is None:
            v3 = np.zeros_like(u3)
        else:
            dt_tau = max(1e-9, float(times[i] - times[i - 1]))
            v3 = (u3 - prev_u3) / dt_tau
        vmag_flat = np.linalg.norm(v3, axis=1)
        vm_flat = vm.ravel()
        disp_flat = disp_mag.ravel()
        aux_flat = np.zeros_like(vm_flat)
        crack_flat = np.zeros_like(vm_flat)
        if family == "centerNotchPlate2D":
            notch_x = float(geometry.get("notch_x", 0.5 * lx))
            notch_h = max(1e-6, float(geometry.get("notch_half_height", 0.04 * ly)))
            notch_a = max(
                2.0e-3 * lx, float(geometry.get("notch_half_length", 0.08 * lx))
            )
            crack_half_ratio = min(
                0.90, max(0.25, float(geometry.get("target_crack_half_ratio", 0.72)))
            )
            dx_c = np.abs(x - notch_x)
            y_band = np.exp(-((y / max(1e-9, notch_h)) ** 2))
            ahead = np.maximum(0.0, dx_c - notch_a)
            lig_half = max(1e-9, 0.5 * lx - notch_a)
            tip = np.exp(
                -(
                    ((dx_c - notch_a) / max(1e-9, 0.03 * lx)) ** 2
                    + (y / max(1e-9, 1.15 * notch_h)) ** 2
                )
            )
            notch_core = ((dx_c <= notch_a) & (np.abs(y) <= 0.75 * notch_h)).astype(
                float
            )
            drive = np.clip((float(lam) - 0.08) / 0.92, 0.0, 1.0)
            target_len = max(1.6 * notch_h, crack_half_ratio * drive * lig_half)
            trail = np.clip(
                (target_len + 2.0 * notch_h - ahead)
                / max(target_len + 2.0 * notch_h, 1e-9),
                0.0,
                1.0,
            )
            damage = np.clip(
                0.14 * notch_core
                + 0.18 * tip
                + drive * (0.50 * y_band * np.power(trail, 0.80) + 0.34 * tip),
                0.0,
                1.0,
            )
            crack = (damage >= 0.40).astype(float)
            aux_flat = damage.ravel()
            crack_flat = crack.ravel()
            crack_len = 0.0
            if np.any(crack > 0.5):
                crack_band = (crack > 0.5) & (np.abs(y) <= 1.8 * notch_h)
                if np.any(crack_band):
                    x_band = x[crack_band]
                    ext_l = max(0.0, (notch_x - notch_a) - float(np.min(x_band)))
                    ext_r = max(0.0, float(np.max(x_band)) - (notch_x + notch_a))
                    crack_len = float(ext_l + ext_r)
                else:
                    crack_len = 0.0
            crack_hist.append(crack_len)
        else:
            crack_hist.append(0.0)

        vtu_name = f"frame_{i:04d}.vtu"
        vtu_path = results_dir / vtu_name
        _write_vtu(
            vtu_path,
            points,
            quads_arr,
            {
                "u": u3,
                "v": v3,
                "sigma_vm": vm_flat,
                "max_principal_stress": sig1.ravel(),
                "max_principal_strain": eps1.ravel(),
                "u_mag": disp_flat,
                "v_mag": vmag_flat,
                "aux_field": aux_flat,
                "damage": aux_flat,
                "crack_mask": crack_flat,
            },
        )
        vtu_files.append(vtu_name)
        prev_u3 = np.array(u3, copy=True)

    pvd_path = results_dir / "displacement.pvd"
    _write_pvd(pvd_path, times, vtu_files)

    hysteresis_area = 0.0
    if len(disp_hist) > 2:
        hysteresis_area = float(
            abs(np.trapezoid(np.asarray(force_hist), np.asarray(disp_hist)))
        )

    localization_index = float(max(loc_hist) if loc_hist else 0.0)
    snap_indicator = float(max(snap_hist) - min(snap_hist) if snap_hist else 0.0)

    return SolveResult(
        backend="surrogate",
        n_steps=n_steps,
        times=times,
        max_disp=max_disp,
        max_vm_stress=max_vm,
        max_reaction_force=max_rf,
        hysteresis_area=hysteresis_area,
        localization_index=localization_index,
        snap_indicator=snap_indicator,
        crack_length_final=float(crack_hist[-1] if crack_hist else 0.0),
        strain_energy_total=float(strain_energy_hist[-1])
        if strain_energy_hist
        else 0.0,
        load_history=[float(x) for x in hist],
        max_disp_history=[float(x) for x in max_disp_hist],
        max_vm_history=[float(x) for x in max_vm_hist],
        strain_energy_history=[float(x) for x in strain_energy_hist],
        plastic_strain_history=[],
        crack_length_history=[float(x) for x in crack_hist],
        dim=_case_dim(case),
        nonlinear_model=_nonlinear_model(case),
    )


def _backend_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _run_fenicsx(
    case_dir: Path,
    case: dict[str, Any],
    *,
    hole_adapt_target: float,
    hole_adapt_tol: float,
    hole_adapt_max_iters: int,
) -> SolveResult:
    if not _backend_available("dolfinx"):
        raise RuntimeError("backend=fenicsx requested but dolfinx is not installed")

    if _case_dim(case) == "2D":
        from fe_fenicsx_solver_2d_general import solve_case as solve_case_fe_2d

        out = solve_case_fe_2d(
            case_dir=case_dir,
            case=case,
            hole_adapt_target=hole_adapt_target,
            hole_adapt_tol=hole_adapt_tol,
            hole_adapt_max_iters=hole_adapt_max_iters,
        )
        return SolveResult(
            backend=str(out.get("backend", "fenicsx_2d")),
            n_steps=int(out.get("n_steps", 0)),
            times=list(out.get("times", [])),
            max_disp=float(out.get("max_disp", 0.0)),
            max_vm_stress=float(out.get("max_vm_stress", 0.0)),
            max_reaction_force=float(out.get("max_reaction_force", 0.0)),
            hysteresis_area=float(out.get("hysteresis_area", 0.0)),
            localization_index=float(out.get("localization_index", 0.0)),
            snap_indicator=float(out.get("snap_indicator", 0.0)),
            crack_length_final=float(out.get("crack_length_final", 0.0)),
            plastic_strain_p95=float(out.get("plastic_strain_p95", 0.0)),
            strain_energy_total=float(out.get("strain_energy_total", 0.0)),
            hole_gradient_contrast=float(out.get("hole_gradient_contrast", 0.0)),
            load_history=[float(x) for x in out.get("load_history", [])],
            max_disp_history=[float(x) for x in out.get("max_disp_history", [])],
            max_vm_history=[float(x) for x in out.get("max_vm_history", [])],
            strain_energy_history=[
                float(x) for x in out.get("strain_energy_history", [])
            ],
            plastic_strain_history=[
                float(x) for x in out.get("plastic_strain_history", [])
            ],
            crack_length_history=[
                float(x) for x in out.get("crack_length_history", [])
            ],
            dim=_case_dim(case),
            nonlinear_model=str(out.get("nonlinear_model", _nonlinear_model(case))),
        )

    from fe_fenicsx_solver import solve_case as solve_case_fe

    out = solve_case_fe(
        case_dir=case_dir,
        case=case,
        hole_adapt_target=hole_adapt_target,
        hole_adapt_tol=hole_adapt_tol,
        hole_adapt_max_iters=hole_adapt_max_iters,
    )
    return SolveResult(
        backend=str(out.get("backend", "fenicsx_proxy")),
        n_steps=int(out.get("n_steps", 0)),
        times=list(out.get("times", [])),
        max_disp=float(out.get("max_disp", 0.0)),
        max_vm_stress=float(out.get("max_vm_stress", 0.0)),
        max_reaction_force=float(out.get("max_reaction_force", 0.0)),
        hysteresis_area=float(out.get("hysteresis_area", 0.0)),
        localization_index=float(out.get("localization_index", 0.0)),
        snap_indicator=float(out.get("snap_indicator", 0.0)),
        crack_length_final=float(out.get("crack_length_final", 0.0)),
        plastic_strain_p95=float(out.get("plastic_strain_p95", 0.0)),
        strain_energy_total=float(out.get("strain_energy_total", 0.0)),
        hole_gradient_contrast=float(out.get("hole_gradient_contrast", 0.0)),
        load_history=[float(x) for x in out.get("load_history", [])],
        max_disp_history=[float(x) for x in out.get("max_disp_history", [])],
        max_vm_history=[float(x) for x in out.get("max_vm_history", [])],
        strain_energy_history=[float(x) for x in out.get("strain_energy_history", [])],
        plastic_strain_history=[
            float(x) for x in out.get("plastic_strain_history", [])
        ],
        crack_length_history=[float(x) for x in out.get("crack_length_history", [])],
        dim=_case_dim(case),
        nonlinear_model=_nonlinear_model(case),
    )


def _solve(
    case_dir: Path,
    case: dict[str, Any],
    *,
    backend: str,
    strict_fe: bool,
    hole_adapt_target: float,
    hole_adapt_tol: float,
    hole_adapt_max_iters: int,
) -> SolveResult:
    b = backend.strip().lower()
    requires_fe = _case_requires_fe(case)

    if strict_fe:
        if b == "surrogate":
            raise RuntimeError("strict-fe mode forbids backend=surrogate")
        if b == "auto":
            b = "fenicsx"
        if b != "fenicsx":
            raise RuntimeError(
                f"strict-fe mode supports only fenicsx/auto, got backend={backend}"
            )
        return _run_fenicsx(
            case_dir,
            case,
            hole_adapt_target=hole_adapt_target,
            hole_adapt_tol=hole_adapt_tol,
            hole_adapt_max_iters=hole_adapt_max_iters,
        )

    if b == "fenicsx":
        return _run_fenicsx(
            case_dir,
            case,
            hole_adapt_target=hole_adapt_target,
            hole_adapt_tol=hole_adapt_tol,
            hole_adapt_max_iters=hole_adapt_max_iters,
        )

    if b == "auto":
        if _backend_available("dolfinx"):
            return _run_fenicsx(
                case_dir,
                case,
                hole_adapt_target=hole_adapt_target,
                hole_adapt_tol=hole_adapt_tol,
                hole_adapt_max_iters=hole_adapt_max_iters,
            )
        if requires_fe:
            raise RuntimeError(
                "Case requires FE backend, but dolfinx is unavailable. "
                "Install dolfinx or disable this case."
            )
        return _run_surrogate(case_dir, case)

    if b == "surrogate":
        if requires_fe:
            raise RuntimeError("3D/nonlinear cases cannot use surrogate backend")
        return _run_surrogate(case_dir, case)

    raise RuntimeError(f"Unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one solid benchmark case")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "fenicsx", "surrogate"]
    )
    parser.add_argument(
        "--strict-fe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require fenicsx backend; fail fast if unavailable.",
    )
    parser.add_argument("--hole-adapt-target", type=float, default=3.0)
    parser.add_argument("--hole-adapt-tol", type=float, default=0.3)
    parser.add_argument("--hole-adapt-max-iters", type=int, default=4)
    args = parser.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"case_dir not found: {case_dir}")

    log_path = case_dir / "log.solve"
    force_rerun = _bool_env("FORCE_RERUN_CASES", default=False)
    if (
        (not force_rerun)
        and log_path.exists()
        and ("END" in log_path.read_text(errors="ignore"))
    ):
        print(f"[solid_bench] skip solved case: {case_dir.name}")
        return

    case = _load_case(case_dir)
    if not case:
        raise RuntimeError(f"Missing solver_input.json/case_meta.json in {case_dir}")

    try:
        result = _solve(
            case_dir,
            case,
            backend=args.backend,
            strict_fe=bool(args.strict_fe),
            hole_adapt_target=float(args.hole_adapt_target),
            hole_adapt_tol=float(args.hole_adapt_tol),
            hole_adapt_max_iters=max(1, int(args.hole_adapt_max_iters)),
        )
        # Persist any in-solver metadata adjustments (e.g., adaptive hole radius).
        (case_dir / "case_meta.json").write_text(json.dumps(case, indent=2) + "\n")
        (case_dir / "solver_input.json").write_text(json.dumps(case, indent=2) + "\n")

        metrics = {
            "status": "ok",
            "backend": result.backend,
            "dim": result.dim,
            "nonlinear_model": result.nonlinear_model,
            "n_steps": result.n_steps,
            "times": result.times,
            "max_disp": result.max_disp,
            "max_vm_stress": result.max_vm_stress,
            "max_reaction_force": result.max_reaction_force,
            "hysteresis_area": result.hysteresis_area,
            "localization_index": result.localization_index,
            "snap_indicator": result.snap_indicator,
            "crack_length_final": result.crack_length_final,
            "plastic_strain_p95": result.plastic_strain_p95,
            "strain_energy_total": result.strain_energy_total,
            "hole_gradient_contrast": result.hole_gradient_contrast,
            "load_history": result.load_history,
            "max_disp_history": result.max_disp_history,
            "max_vm_history": result.max_vm_history,
            "strain_energy_history": result.strain_energy_history,
            "plastic_strain_history": result.plastic_strain_history,
            "crack_length_history": result.crack_length_history,
        }
        (case_dir / "postProcessing").mkdir(parents=True, exist_ok=True)
        (case_dir / "postProcessing" / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )

        log_txt = (
            f"case={case_dir.name}\n"
            f"backend={result.backend}\n"
            f"dim={result.dim}\n"
            f"nonlinear_model={result.nonlinear_model}\n"
            f"n_steps={result.n_steps}\n"
            f"max_disp={result.max_disp:.6e}\n"
            f"max_vm_stress={result.max_vm_stress:.6e}\n"
            "END\n"
        )
        log_path.write_text(log_txt)
        print(
            f"[solid_bench] solved case={case_dir.name} backend={result.backend} n_steps={result.n_steps}"
        )
    except Exception:
        tb = traceback.format_exc()
        log_path.write_text(tb + "\nFAIL\n")
        raise


if __name__ == "__main__":
    main()
