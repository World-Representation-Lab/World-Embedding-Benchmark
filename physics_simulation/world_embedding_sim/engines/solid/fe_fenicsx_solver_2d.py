#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from fe_utils import (
    _resample_path,
    _load_history,
    _write_vtu_tri,
    _write_pvd,
    _interpolate_scalar,
    _principal_2d,
)


def _require_dolfinx() -> None:
    try:
        import basix.ufl  # noqa: F401
        import dolfinx  # noqa: F401
        import mpi4py  # noqa: F401
        import petsc4py  # noqa: F401
        import ufl  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "2D FE backend requires dolfinx + mpi4py + petsc4py + ufl"
        ) from exc


def solve_case(
    *,
    case_dir: Path,
    case: dict[str, Any],
    hole_adapt_target: float,
    hole_adapt_tol: float,
    hole_adapt_max_iters: int,
) -> dict[str, Any]:
    del hole_adapt_target, hole_adapt_tol, hole_adapt_max_iters
    _require_dolfinx()

    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem

    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}
    material = case.get("material") if isinstance(case.get("material"), dict) else {}
    nonlinear = case.get("nonlinear") if isinstance(case.get("nonlinear"), dict) else {}

    lx = float(geometry.get("lx", 2.0))
    ly = float(geometry.get("ly", 1.0))
    thickness = float(geometry.get("thickness", 0.01))
    mesh_cfg = solver.get("mesh") if isinstance(solver.get("mesh"), dict) else {}
    nx = max(24, int(mesh_cfg.get("nx", 140)))
    ny = max(12, int(mesh_cfg.get("ny", 58)))

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.asarray([0.0, -0.5 * ly]), np.asarray([lx, 0.5 * ly])],
        [nx, ny],
        cell_type=mesh.CellType.triangle,
    )
    points = np.asarray(domain.geometry.x, dtype=float)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros((points.shape[0],), dtype=float)])

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    cells = np.asarray(
        domain.topology.connectivity(tdim, 0).array, dtype=np.int64
    ).reshape((-1, 3))
    char_len = max(1e-9, math.sqrt((lx * ly) / max(cells.shape[0], 1)))

    E = max(1e5, float(material.get("E", 2.0e9)))
    nu = float(material.get("nu", 0.30))
    mu = E / (2.0 * max(1e-9, 1.0 + nu))
    lam_ps = E * nu / max(1e-9, 1.0 - nu * nu)  # plane-stress lambda

    nlp = nonlinear.get("params") if isinstance(nonlinear.get("params"), dict) else {}
    notch_len = float(
        geometry.get("notch_length", geometry.get("notch_half_length", 0.16 * lx))
    )
    notch_len = float(np.clip(notch_len, 0.02 * lx, 0.45 * lx))
    notch_half_width = float(geometry.get("notch_half_width", 0.018 * ly))
    notch_half_width = float(np.clip(notch_half_width, 0.004 * ly, 0.10 * ly))
    crack_target_ratio = float(
        np.clip(float(nlp.get("target_crack_ratio", 0.82)), 0.25, 0.98)
    )
    crack_growth_start = float(
        np.clip(float(nlp.get("growth_start_fraction", 0.18)), 0.0, 0.92)
    )
    seed_damage = float(
        np.clip(float(nlp.get("precrack_seed_damage", 0.90)), 0.72, 0.995)
    )
    crack_growth_peak = float(
        np.clip(float(nlp.get("growth_peak_damage", 0.985)), seed_damage + 1e-3, 0.999)
    )
    principal_stress_threshold = float(
        np.clip(float(nlp.get("principal_stress_threshold", 0.70)), 0.35, 0.95)
    )
    residual_stiffness = float(
        np.clip(float(nlp.get("residual_stiffness", 5e-4)), 1e-6, 5e-2)
    )
    n_stag = max(1, int(float(nlp.get("stagger_iters", 3))))

    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    S = fem.functionspace(domain, ("Lagrange", 1))
    u = fem.Function(V)
    d_field = fem.Function(S)
    d_field.x.array[:] = 0.0

    s_coords_raw = np.asarray(S.tabulate_dof_coordinates(), dtype=float)
    if s_coords_raw.ndim == 2 and s_coords_raw.shape[1] >= 2:
        s_coords = s_coords_raw
    else:
        # Some backends return flattened xyz coordinates for embedded 2D meshes.
        s_coords = s_coords_raw.reshape((-1, 3))
    sx = np.asarray(s_coords[:, 0], dtype=float)
    sy = np.asarray(s_coords[:, 1], dtype=float)
    pre_seed = (sx <= notch_len + 1.4 * char_len) & (np.abs(sy) <= notch_half_width)
    tip_blob = ((sx - notch_len) / max(1.4 * char_len, 1e-9)) ** 2 + (
        sy / max(1.6 * notch_half_width, 1e-9)
    ) ** 2 <= 1.0
    d_field.x.array[pre_seed | tip_blob] = seed_damage
    d_field.x.scatter_forward()

    tol = 1e-8 + 1e-7 * max(lx, ly)
    fixed_dofs = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0, atol=tol)
    )
    load_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], lx, atol=tol))
    bc_fixed = fem.dirichletbc(
        np.asarray([0.0, 0.0], dtype=PETSc.ScalarType), fixed_dofs, V
    )

    load_factor = float(loading.get("load_factor", 0.035))
    load_sign = float(loading.get("load_sign", 1.0))

    def build_step_bcs(lam_hist: float):
        ux = load_sign * load_factor * lx * float(lam_hist)
        bc_load = fem.dirichletbc(
            np.asarray([ux, 0.0], dtype=PETSc.ScalarType), load_dofs, V
        )
        return [bc_fixed, bc_load]

    def eps(w):
        return ufl.sym(ufl.grad(w))

    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    I = ufl.Identity(2)
    g_d = residual_stiffness + (1.0 - residual_stiffness) * (1.0 - d_field) ** 2
    sigma_expr = g_d * (2.0 * mu * eps(u) + lam_ps * ufl.tr(eps(u)) * I)
    sigma_du = g_d * (2.0 * mu * eps(du) + lam_ps * ufl.tr(eps(du)) * I)
    a_u = ufl.inner(sigma_du, eps(v)) * ufl.dx
    L_u = ufl.inner(fem.Constant(domain, PETSc.ScalarType((0.0, 0.0))), v) * ufl.dx

    n_steps = max(10, int(solver.get("n_steps", 24)))
    hist = _load_history(case, n_steps)
    times = [i / max(1, n_steps - 1) for i in range(n_steps)]

    petsc_lin = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    results_dir = case_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    vtu_files: list[str] = []

    max_disp = 0.0
    max_vm = 0.0
    max_rf = 0.0
    strain_energy_total = 0.0
    crack_length_final = 0.0
    force_hist: list[float] = []
    disp_hist: list[float] = []
    max_disp_hist: list[float] = []
    max_vm_hist: list[float] = []
    strain_energy_history: list[float] = []
    crack_length_history: list[float] = []
    loc_hist: list[float] = []
    snap_hist: list[float] = []
    prev_u3: np.ndarray | None = None

    for i, lam_hist in enumerate(hist):
        bcs_step = build_step_bcs(float(lam_hist))

        sxx_arr = np.zeros_like(sx)
        syy_arr = np.zeros_like(sx)
        sxy_arr = np.zeros_like(sx)
        e1_arr = np.zeros_like(sx)
        s1_arr = np.zeros_like(sx)
        for k in range(n_stag):
            u = LinearProblem(
                a_u,
                L_u,
                bcs=bcs_step,
                petsc_options_prefix=f"sent2d_{i}_{k}_",
                petsc_options=petsc_lin,
            ).solve()

            exx = _interpolate_scalar(eps(u)[0, 0], S).x.array
            eyy = _interpolate_scalar(eps(u)[1, 1], S).x.array
            exy = _interpolate_scalar(eps(u)[0, 1], S).x.array
            sxx_arr = _interpolate_scalar(sigma_expr[0, 0], S).x.array
            syy_arr = _interpolate_scalar(sigma_expr[1, 1], S).x.array
            sxy_arr = _interpolate_scalar(sigma_expr[0, 1], S).x.array
            s1_arr, _ = _principal_2d(sxx_arr, syy_arr, sxy_arr)
            e1_arr, _ = _principal_2d(exx, eyy, exy)

            s1_pos = np.maximum(0.0, np.asarray(s1_arr, dtype=float))
            s_ref = max(1e3, float(np.quantile(s1_pos, 0.98)))
            s_crit = principal_stress_threshold * s_ref
            stress_drive = np.clip(
                (s1_pos - s_crit) / max(1e-9, s_ref - s_crit), 0.0, 1.0
            )

            tau = 0.0
            if crack_growth_start < 1.0:
                tau = float(
                    np.clip(
                        (float(lam_hist) - crack_growth_start)
                        / max(1e-9, 1.0 - crack_growth_start),
                        0.0,
                        1.0,
                    )
                )
            tau = float(tau * tau * (3.0 - 2.0 * tau))
            target_len = max(0.0, crack_target_ratio * max(1e-9, lx - notch_len) * tau)
            ahead = np.maximum(0.0, sx - notch_len)
            trail = np.clip(
                (target_len + 2.0 * notch_half_width - ahead)
                / max(1e-9, target_len + 2.0 * notch_half_width),
                0.0,
                1.0,
            )
            band_w = np.exp(-((sy / max(1e-9, notch_half_width)) ** 2))
            active = (
                (sx >= notch_len - 1.5 * char_len)
                & (sx <= notch_len + target_len + 2.0 * char_len)
                & (np.abs(sy) <= 3.0 * notch_half_width)
            )
            drive = np.clip(stress_drive * band_w * trail, 0.0, 1.0)
            d_bias = seed_damage + (crack_growth_peak - seed_damage) * np.power(
                drive, 0.84
            )
            d_new = np.asarray(d_field.x.array, dtype=float).copy()
            d_new[active] = np.maximum(d_new[active], d_bias[active])
            d_new[pre_seed | tip_blob] = np.maximum(
                d_new[pre_seed | tip_blob], seed_damage
            )
            d_new = np.clip(
                np.nan_to_num(d_new, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
            )
            if np.max(np.abs(d_new - d_field.x.array)) < 1e-4:
                d_field.x.array[:] = d_new
                d_field.x.scatter_forward()
                break
            d_field.x.array[:] = d_new
            d_field.x.scatter_forward()

        u_arr2 = np.asarray(u.x.array, dtype=float).reshape((-1, 2))
        u_arr3 = np.column_stack([u_arr2, np.zeros((u_arr2.shape[0],), dtype=float)])
        if prev_u3 is None:
            v_arr3 = np.zeros_like(u_arr3)
        else:
            dt = max(1e-9, float(times[i] - times[i - 1]))
            v_arr3 = (u_arr3 - prev_u3) / dt
        prev_u3 = np.array(u_arr3, copy=True)

        disp_mag = np.linalg.norm(u_arr3, axis=1)
        vel_mag = np.linalg.norm(v_arr3, axis=1)
        vm_arr = np.sqrt(
            np.maximum(
                0.0,
                sxx_arr * sxx_arr
                - sxx_arr * syy_arr
                + syy_arr * syy_arr
                + 3.0 * sxy_arr * sxy_arr,
            )
        )
        damage_arr = np.asarray(d_field.x.array, dtype=float)
        crack_mask = (damage_arr >= 0.90).astype(float)

        max_disp = max(max_disp, float(np.max(disp_mag)))
        max_vm = max(max_vm, float(np.max(vm_arr)))
        max_disp_hist.append(float(np.max(disp_mag)))
        max_vm_hist.append(float(np.max(vm_arr)))

        right_band = np.abs(sx - lx) <= max(2.0 * char_len, 2.0 * tol)
        rf = (
            float(np.mean(sxx_arr[right_band]) * ly * max(thickness, 1e-9))
            if np.any(right_band)
            else 0.0
        )
        max_rf = max(max_rf, abs(rf))
        force_hist.append(rf)
        disp_hist.append(
            float(np.mean(u_arr3[right_band, 0]))
            if np.any(right_band)
            else float(np.mean(u_arr3[:, 0]))
        )

        step_energy = float(
            np.sum(0.5 * vm_arr * disp_mag) * char_len * char_len * max(thickness, 1e-9)
        )
        strain_energy_total += step_energy
        strain_energy_history.append(float(strain_energy_total))
        loc_hist.append(float(np.max(vm_arr) / max(np.mean(vm_arr), 1e-12)))
        snap_hist.append(float(np.max(np.abs(u_arr3[:, 1]))))

        crack_band = (
            (damage_arr >= 0.90)
            & (np.abs(sy) <= 2.0 * notch_half_width)
            & (sx >= notch_len - 2.0 * char_len)
        )
        if np.any(crack_band):
            crack_length_final = max(
                crack_length_final, float(max(0.0, np.max(sx[crack_band]) - notch_len))
            )
        crack_length_history.append(float(crack_length_final))

        vtu_name = f"frame_{i:04d}.vtu"
        _write_vtu_tri(
            results_dir / vtu_name,
            points,
            cells,
            {
                "u": u_arr3,
                "v": v_arr3,
                "u_mag": disp_mag,
                "v_mag": vel_mag,
                "sigma_vm": vm_arr,
                "max_principal_stress": s1_arr,
                "max_principal_strain": e1_arr,
                "aux_field": damage_arr,
                "damage": damage_arr,
                "phase_field": damage_arr,
                "crack_mask": crack_mask,
            },
        )
        vtu_files.append(vtu_name)

    _write_pvd(results_dir / "displacement.pvd", times, vtu_files)

    hysteresis_area = 0.0
    if len(disp_hist) > 2:
        hysteresis_area = float(
            abs(np.trapezoid(np.asarray(force_hist), np.asarray(disp_hist)))
        )

    localization_index = float(max(loc_hist) if loc_hist else 0.0)
    snap_indicator = float(max(snap_hist) - min(snap_hist) if snap_hist else 0.0)

    return {
        "backend": "fenicsx_2d",
        "n_steps": int(n_steps),
        "times": [float(t) for t in times],
        "max_disp": float(max_disp),
        "max_vm_stress": float(max_vm),
        "max_reaction_force": float(max_rf),
        "hysteresis_area": float(hysteresis_area),
        "localization_index": float(localization_index),
        "snap_indicator": float(snap_indicator),
        "crack_length_final": float(crack_length_final),
        "plastic_strain_p95": 0.0,
        "strain_energy_total": float(strain_energy_total),
        "hole_gradient_contrast": 0.0,
        "load_history": [float(x) for x in hist],
        "max_disp_history": [float(x) for x in max_disp_hist],
        "max_vm_history": [float(x) for x in max_vm_hist],
        "strain_energy_history": [float(x) for x in strain_energy_history],
        "plastic_strain_history": [0.0 for _ in max_disp_hist],
        "crack_length_history": [float(x) for x in crack_length_history],
    }
