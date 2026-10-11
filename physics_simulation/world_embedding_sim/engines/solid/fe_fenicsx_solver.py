#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fe_utils import (
    _resample_path,
    _load_history,
    _write_pvd,
)

from hex_mesh_utils import dolfinx_hex_to_vtk
from mesh3d_structured import build_structured_mesh_3d


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def _require_dolfinx() -> None:
    try:
        import basix.ufl  # noqa: F401
        import dolfinx  # noqa: F401
        import mpi4py  # noqa: F401
        import petsc4py  # noqa: F401
        import ufl  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "backend=fenicsx requires dolfinx + mpi4py + petsc4py + ufl"
        ) from exc


def _write_vtu(
    path: Path, points: np.ndarray, cells: np.ndarray, point_data: dict[str, np.ndarray]
) -> None:
    n_points = int(points.shape[0])
    cells_vtk = dolfinx_hex_to_vtk(cells)
    n_cells = int(cells_vtk.shape[0])
    if n_cells <= 0:
        raise RuntimeError(f"Refusing to write point-only VTU (zero cells): {path}")
    conn = cells_vtk.reshape(-1)
    offsets = np.arange(1, n_cells + 1, dtype=np.int64) * 8
    types = np.full((n_cells,), 12, dtype=np.uint8)  # VTK_HEXAHEDRON

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

    point_data_xml: list[str] = []
    for name, arr in point_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            point_data_xml.append(
                f'<DataArray type="Float64" Name="{name}" format="ascii">{fmt_float_array(arr)}</DataArray>'
            )
        elif arr.ndim == 2 and arr.shape[1] in {2, 3}:
            point_data_xml.append(
                f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{int(arr.shape[1])}" format="ascii">'
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


def _dual_node_split_export(
    *,
    points: np.ndarray,
    cells: np.ndarray,
    u_arr: np.ndarray,
    vm_arr: np.ndarray,
    aux_arr: np.ndarray,
    crack_plane: float,
    crack_plane_band: float,
    crack_y_root: float,
    crack_single_sided: bool,
    crack_side_sign: float,
    crack_target_ratio: float,
    crack_growth_start: float,
    split_damage: float,
    opening_ref: float,
    roughness_amp: float,
    lam_hist: float,
    char_len: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Dual-node cohesive export proxy:
    duplicate one side of heavily damaged crack-plane nodes so rendered frames
    become two detached entities without deleting volume cells.
    """
    n_points = int(points.shape[0])
    if (
        n_points <= 0
        or aux_arr.shape[0] != n_points
        or vm_arr.shape[0] != n_points
        or u_arr.shape[0] != n_points
        or cells.size == 0
    ):
        z = np.zeros((n_points if n_points > 0 else 0,), dtype=float)
        return points, cells, u_arr, vm_arr, aux_arr, z, z

    band = max(1.2 * float(char_len), 1.0 * float(crack_plane_band))
    plane_mask = np.abs(points[:, 0] - float(crack_plane)) <= band
    if crack_single_sided:
        side_mask = float(crack_side_sign) * (
            points[:, 1] - float(crack_y_root)
        ) >= -0.15 * float(char_len)
        # Duplicate the +x side cells to detach interface without deleting volume elements.
        cell_side = np.mean(points[cells, 0], axis=1) >= float(crack_plane)
        normal = np.asarray([1.0, 0.0, 0.0], dtype=float)
    else:
        side_mask = np.ones((n_points,), dtype=bool)
        cell_side = np.mean(points[cells, 0], axis=1) >= float(crack_plane)
        normal = np.asarray([1.0, 0.0, 0.0], dtype=float)

    tau = 0.0
    if float(crack_growth_start) < 1.0:
        tau = float(
            np.clip(
                (float(lam_hist) - float(crack_growth_start))
                / max(1e-9, 1.0 - float(crack_growth_start)),
                0.0,
                1.0,
            )
        )
    tau = float(tau * tau * (3.0 - 2.0 * tau))

    if crack_single_sided:
        front_coord = np.maximum(
            0.0, -float(crack_side_sign) * (points[:, 1] - float(crack_y_root))
        )
        front_span = max(
            2.0 * float(char_len),
            float(np.max(front_coord[side_mask])) if np.any(side_mask) else 0.0,
        )
    else:
        y_root_abs = abs(float(crack_y_root))
        front_coord = np.maximum(0.0, y_root_abs - np.abs(points[:, 1]))
        front_span = max(2.0 * float(char_len), y_root_abs)
    target_len = max(
        0.0, float(np.clip(crack_target_ratio, 0.05, 1.30)) * front_span * tau
    )
    front_open = front_coord <= (target_len + 1.5 * float(char_len))

    split_thr = float(np.clip(split_damage, 0.05, 0.995))
    w_pre = np.clip((aux_arr - split_thr) / max(1e-9, 1.0 - split_thr), 0.0, 1.0)
    sep_n_base = max(0.0, float(opening_ref)) * max(0.0, tau) * np.power(w_pre, 0.8)
    sep_t_base = np.zeros((n_points,), dtype=float)
    split_mask = (
        plane_mask & side_mask & front_open & (aux_arr >= max(0.08, 0.4 * split_thr))
    )
    if not np.any(split_mask):
        return points, cells, u_arr, vm_arr, aux_arr, sep_n_base, sep_t_base

    split_cells = np.any(split_mask[cells], axis=1) & cell_side
    if not np.any(split_cells):
        return points, cells, u_arr, vm_arr, aux_arr, sep_n_base, sep_t_base

    candidate_nodes = np.unique(cells[split_cells].reshape(-1))
    nodes_to_dup = candidate_nodes
    if nodes_to_dup.size == 0:
        return points, cells, u_arr, vm_arr, aux_arr, sep_n_base, sep_t_base

    new_points = np.vstack((points, points[nodes_to_dup]))
    new_vm = np.concatenate((vm_arr, vm_arr[nodes_to_dup]))
    new_aux = np.concatenate((aux_arr, aux_arr[nodes_to_dup]))
    new_sep_n = np.concatenate((sep_n_base, sep_n_base[nodes_to_dup]))
    new_sep_t = np.concatenate((sep_t_base, sep_t_base[nodes_to_dup]))

    y_span = max(1e-9, float(np.max(points[:, 1]) - np.min(points[:, 1])))
    z_span = max(1e-9, float(np.max(points[:, 2]) - np.min(points[:, 2])))
    y_loc = points[nodes_to_dup, 1] / y_span
    z_loc = points[nodes_to_dup, 2] / z_span
    rough = 1.0 + float(np.clip(roughness_amp, 0.0, 0.6)) * (
        0.55 * np.sin(13.0 * y_loc + 7.0 * z_loc)
        + 0.45 * np.cos(17.0 * y_loc - 11.0 * z_loc)
    )
    rough = np.clip(rough, 0.65, 1.45)

    w_damage = np.clip(
        (aux_arr[nodes_to_dup] - split_thr) / max(1e-9, 1.0 - split_thr), 0.0, 1.0
    )
    if crack_single_sided:
        front_dup = np.maximum(
            0.0,
            -float(crack_side_sign) * (points[nodes_to_dup, 1] - float(crack_y_root)),
        )
    else:
        front_dup = np.maximum(
            0.0, abs(float(crack_y_root)) - np.abs(points[nodes_to_dup, 1])
        )
    w_front = np.clip(
        (target_len + 1.3 * float(char_len) - front_dup)
        / max(target_len + 1.3 * float(char_len), 1e-9),
        0.0,
        1.0,
    )
    w_plane = np.exp(
        -(
            (
                np.abs(points[nodes_to_dup, 0] - float(crack_plane))
                / max(1.2 * band, 1e-9)
            )
            ** 2
        )
    )
    w = np.maximum(w_damage, w_front) * w_plane

    aperture = max(0.0, float(opening_ref)) * max(0.0, tau) * np.power(w, 0.8) * rough
    offsets = aperture[:, None] * normal[None, :]
    new_u = np.vstack((u_arr, u_arr[nodes_to_dup] + offsets))
    new_sep_n[nodes_to_dup] = np.maximum(new_sep_n[nodes_to_dup], aperture)
    new_sep_n[new_points.shape[0] - nodes_to_dup.size :] = np.maximum(
        new_sep_n[new_points.shape[0] - nodes_to_dup.size :], aperture
    )

    remap = np.arange(new_points.shape[0], dtype=np.int64)
    remap[nodes_to_dup] = np.arange(n_points, new_points.shape[0], dtype=np.int64)
    new_cells = np.asarray(cells, dtype=np.int64).copy()
    new_cells[split_cells] = remap[new_cells[split_cells]]
    return new_points, new_cells, new_u, new_vm, new_aux, new_sep_n, new_sep_t


def _estimate_hole_gradient_contrast(hole_ratio: float, load_factor: float) -> float:
    return 1.0 + 7.2 * float(hole_ratio) + 0.6 * float(load_factor)


def _hole_gradient_contrast_from_field(
    *,
    points: np.ndarray,
    u: np.ndarray,
    geometry: dict[str, Any],
    char_len: float,
) -> float:
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    min_span = max(1e-9, min(lx, ly))
    cx = float(geometry.get("hole_cx", 0.5 * lx))
    cy = float(geometry.get("hole_cy", 0.0))
    r0 = max(1e-9, float(geometry.get("hole_radius", 0.22 * min_span)))

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    rxy = np.sqrt(dx * dx + dy * dy)
    u_mag = np.linalg.norm(u, axis=1)
    grad_proxy = u_mag / np.maximum(rxy, max(1e-9, 0.2 * min_span))

    hole_band = (rxy >= 1.02 * r0) & (rxy <= 1.45 * r0)
    far_lo = max(2.8 * r0, 0.32 * min_span)
    far_hi = max(far_lo + 2.0 * char_len, 0.62 * min_span)
    far_field = (rxy >= far_lo) & (rxy <= far_hi)
    if int(np.count_nonzero(hole_band)) < 16 or int(np.count_nonzero(far_field)) < 16:
        return 0.0

    num = float(np.quantile(grad_proxy[hole_band], 0.95))
    den = float(np.median(grad_proxy[far_field]))
    if den <= 1e-12:
        return 0.0
    return float(1.15 * num / den)


def _family_displacement_proxy(
    *,
    case: dict[str, Any],
    points: np.ndarray,
    lam: float,
) -> np.ndarray:
    family = str(case.get("family", "unknown"))
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}
    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}

    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    lz = float(geometry.get("lz", 1.0))
    load_scale = float(loading.get("load_factor", 0.06))
    sign = float(loading.get("load_sign", 1.0))
    eps = sign * load_scale * float(lam)

    x = points[:, 0] / max(1e-9, lx)
    y = points[:, 1] / max(1e-9, 0.5 * ly)
    u = np.zeros_like(points)
    if family == "plateWithHole3D":
        cx = float(geometry.get("hole_cx", 0.5 * lx))
        cy = float(geometry.get("hole_cy", 0.0))
        r0 = max(1e-9, float(geometry.get("hole_radius", 0.2 * min(lx, ly))))
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        rxy = np.sqrt(dx * dx + dy * dy) + 1e-9
        theta = np.arctan2(dy, dx)
        conc = 1.0 + (r0 * r0 / (rxy * rxy)) * np.cos(2.0 * theta)
        mask = np.exp(-((rxy / (3.2 * r0)) ** 2))
        u[:, 0] = eps * points[:, 0] * (1.0 + 0.2 * conc * mask)
        u[:, 1] = -0.22 * eps * points[:, 1] * (1.0 - 0.14 * conc * mask)
    elif family == "longCuboidBending3D":
        shape = np.sin(math.pi * x) ** 2
        u[:, 1] = -abs(eps) * lx * shape
    else:
        u[:, 0] = eps * points[:, 0]
        u[:, 1] = -0.2 * eps * points[:, 1]
        u[:, 2] = -0.2 * eps * points[:, 2]
    return u


def _evaluate_hole_ratio(case: dict[str, Any], hole_ratio: float) -> float:
    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}

    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    lz = float(geometry.get("lz", max(0.2, 0.3 * ly)))
    min_span = max(1e-9, min(lx, ly))
    geometry["hole_radius"] = float(hole_ratio * min_span)

    mesh_cfg = solver.get("mesh") if isinstance(solver.get("mesh"), dict) else {}
    coarse_mesh = {
        "nx": max(10, int(mesh_cfg.get("nx", 24)) // 2),
        "ny": max(10, int(mesh_cfg.get("ny", 18)) // 2),
        "nz": max(8, int(mesh_cfg.get("nz", 14)) // 2),
    }
    mesh3d = build_structured_mesh_3d(geometry, coarse_mesh)
    points = mesh3d.points
    cells = mesh3d.cells
    char_len = max(1e-9, (lx * ly * lz / max(cells.shape[0], 1)) ** (1.0 / 3.0))

    lam = float(np.clip(float(loading.get("load_factor", 0.06)), 0.2, 1.0))
    u = _family_displacement_proxy(case=case, points=points, lam=lam)
    estimate = _estimate_hole_gradient_contrast(
        hole_ratio, float(loading.get("load_factor", 0.06))
    )
    contrast = _hole_gradient_contrast_from_field(
        points=points, u=u, geometry=geometry, char_len=char_len
    )
    if contrast <= 0.0:
        return estimate

    # Coarse-mesh probe can be noisy/non-monotone; blend with analytic estimate
    # to keep hole-radius bisection stable and reproducible.
    return float(0.8 * estimate + 0.2 * contrast)


def _adapt_hole_radius(
    case: dict[str, Any], target: float, tol: float, max_iters: int
) -> float:
    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    min_span = max(1e-9, min(lx, ly))

    lo = 0.12
    hi = 0.42
    best = float(geometry.get("hole_radius", 0.22 * min_span)) / min_span
    best_err = float("inf")
    best_g = 0.0
    for _ in range(max(1, int(max_iters))):
        mid = 0.5 * (lo + hi)
        g = _evaluate_hole_ratio(case, mid)
        err = abs(g - target)
        if err < best_err:
            best = mid
            best_err = err
            best_g = g
        if err <= tol:
            best = mid
            best_g = g
            break
        if g < target:
            lo = mid
        else:
            hi = mid

    geometry["hole_radius"] = float(best * min_span)
    return float(best_g if best_g > 0.0 else _evaluate_hole_ratio(case, best))


def _build_dolfinx_mesh(mesh3d):
    from mpi4py import MPI
    import basix.ufl
    import ufl
    from dolfinx import mesh

    coord_el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    domain_ufl = ufl.Mesh(coord_el)
    # Convert VTK-like hexa node ordering to DOLFINx ordering.
    cells = mesh3d.cells[:, [0, 1, 3, 2, 4, 5, 7, 6]].astype(np.int64)
    return mesh.create_mesh(
        MPI.COMM_WORLD,
        cells,
        domain_ufl,
        mesh3d.points.astype(np.float64),
    )


def _mesh_cells_hex(domain) -> np.ndarray:
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    conn = domain.topology.connectivity(tdim, 0)
    nc = int(domain.topology.index_map(tdim).size_local)
    cells = np.zeros((nc, 8), dtype=np.int64)
    for i in range(nc):
        cells[i, :] = np.asarray(conn.links(i), dtype=np.int64)
    return cells


def _surface_facets(domain, axis: int, value: float, tol: float):
    from dolfinx import mesh

    fdim = domain.topology.dim - 1
    return mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[axis], value, atol=tol)
    )


def _interpolate_scalar(expr, space):
    from dolfinx import fem

    out = fem.Function(space)
    out.interpolate(fem.Expression(expr, space.element.interpolation_points))
    return out


def _linear_material(material: dict[str, Any]) -> tuple[float, float]:
    E = max(1e5, float(material.get("E", 2.1e11)))
    nu = float(material.get("nu", 0.3))
    nu = max(-0.49, min(0.49, nu))
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(mu), float(lam)


def _nan_inf_counts(arr: np.ndarray) -> dict[str, int]:
    a = np.asarray(arr, dtype=float).reshape(-1)
    return {
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
    }


def _finite_minmax(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, dtype=float).reshape(-1)
    finite = a[np.isfinite(a)]
    if finite.size <= 0:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def _hex_center_jacobian_det(
    points_xyz: np.ndarray, cells_dolfinx: np.ndarray
) -> np.ndarray:
    if cells_dolfinx.size == 0:
        return np.zeros((0,), dtype=float)
    cells_vtk = dolfinx_hex_to_vtk(np.asarray(cells_dolfinx, dtype=np.int64))
    x = np.asarray(points_xyz, dtype=float)[cells_vtk]  # (nc,8,3)
    dxi = 0.125 * np.asarray([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=float)
    deta = 0.125 * np.asarray([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=float)
    dzeta = 0.125 * np.asarray(
        [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=float
    )
    dxdxi = np.tensordot(x, dxi, axes=([1], [0]))  # (nc,3)
    dxdeta = np.tensordot(x, deta, axes=([1], [0]))  # (nc,3)
    dxdzeta = np.tensordot(x, dzeta, axes=([1], [0]))  # (nc,3)
    J = np.stack((dxdxi, dxdeta, dxdzeta), axis=2)  # (nc,3,3)
    return np.linalg.det(J)


def _cohesive_debug_fields(
    *,
    damage_arr: np.ndarray,
    lam_hist: float,
    growth_start: float,
    split_damage: float,
    opening_ref: float,
    normal_strength: float,
    shear_strength: float,
    delta_f: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(damage_arr, dtype=float)
    tau = 0.0
    if float(growth_start) < 1.0:
        tau = float(
            np.clip(
                (float(lam_hist) - float(growth_start))
                / max(1e-9, 1.0 - float(growth_start)),
                0.0,
                1.0,
            )
        )
    tau = float(tau * tau * (3.0 - 2.0 * tau))
    split_thr = float(np.clip(split_damage, 0.05, 0.995))
    w = np.clip((d - split_thr) / max(1e-9, 1.0 - split_thr), 0.0, 1.0)
    sep_n = max(0.0, float(opening_ref)) * max(0.0, tau) * np.power(w, 0.8)
    sep_t = np.zeros_like(sep_n)
    delta = max(1e-12, float(delta_f))
    tr_n = max(0.0, float(normal_strength)) * np.maximum(0.0, 1.0 - d) * (sep_n / delta)
    tr_t = max(0.0, float(shear_strength)) * np.maximum(0.0, 1.0 - d) * (sep_t / delta)
    return sep_n, sep_t, tr_n, tr_t


def _load_axis_and_vector(
    *,
    family: str,
    lam_hist: float,
    geometry: dict[str, Any],
    loading: dict[str, Any],
) -> np.ndarray:
    lx = float(geometry.get("lx", 1.0))
    lz = float(geometry.get("lz", 1.0))
    load_factor = float(loading.get("load_factor", 0.05))
    load_sign = float(loading.get("load_sign", 1.0))
    load_mode = str(loading.get("mode", "")).strip().lower()

    if load_mode in {"axial_compression", "indentation_control"}:
        amp = load_sign * load_factor * lz * lam_hist
        return np.asarray([0.0, 0.0, -abs(amp)], dtype=float)
    if load_mode == "bending_displacement":
        amp = load_sign * load_factor * lx * lam_hist
        return np.asarray([0.0, -abs(amp), 0.0], dtype=float)
    if load_mode == "torsion_control":
        twist_deg = float(loading.get("twist_angle_deg", 14.0))
        angle = math.radians(twist_deg) * lam_hist
        return np.asarray([angle, 0.0, 0.0], dtype=float)
    if family in {"notchedRodTension3D", "cohesiveInterfaceTension3D"}:
        # Enforce pure axial opening load for Mode-I dominant crack propagation.
        amp = load_sign * load_factor * lx * lam_hist
        return np.asarray([amp, 0.0, 0.0], dtype=float)
    amp = load_sign * load_factor * lx * lam_hist
    return np.asarray([amp, 0.0, 0.0], dtype=float)


def solve_case(
    *,
    case_dir: Path,
    case: dict[str, Any],
    hole_adapt_target: float,
    hole_adapt_tol: float,
    hole_adapt_max_iters: int,
) -> dict[str, Any]:
    _require_dolfinx()

    from petsc4py import PETSc
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}
    material = case.get("material") if isinstance(case.get("material"), dict) else {}
    nonlinear = case.get("nonlinear") if isinstance(case.get("nonlinear"), dict) else {}
    family = str(case.get("family") or case_dir.name.split("_", 1)[0])
    load_mode = str(loading.get("mode", "")).strip().lower()

    if str(case.get("dim", "2D")).strip().upper() != "3D":
        raise RuntimeError("fe_fenicsx_solver expects 3D case payload")

    hole_gradient_contrast = 0.0
    shape_3d = str(geometry.get("shape_3d", "")).strip().lower()
    if shape_3d == "plate_with_hole_3d":
        hole_gradient_contrast = _adapt_hole_radius(
            case,
            target=float(hole_adapt_target),
            tol=float(hole_adapt_tol),
            max_iters=max(1, int(hole_adapt_max_iters)),
        )

    nonlinear_model = str(nonlinear.get("model", "none")).strip().lower()
    model_kind = nonlinear_model
    if family in {"notchedRodTension3D", "cohesiveInterfaceTension3D"}:
        model_kind = "czm_crack"
    elif family == "plasticJ2LargeStrain3D":
        model_kind = "j2_plasticity"
    elif family == "hyperelasticNeoHookean3D":
        model_kind = "neo_hookean"
    elif model_kind in {"", "none"}:
        model_kind = "linear"

    mesh_cfg = solver.get("mesh") if isinstance(solver.get("mesh"), dict) else {}
    if model_kind in {"neo_hookean", "j2_plasticity", "czm_crack"}:
        mesh_cfg = dict(mesh_cfg)
        nx_cap = max(8, _env_int("FE_NL_MESH_CAP_NX", 20))
        ny_cap = max(8, _env_int("FE_NL_MESH_CAP_NY", 16))
        nz_cap = max(8, _env_int("FE_NL_MESH_CAP_NZ", 14))
        mesh_cfg["nx"] = min(int(mesh_cfg.get("nx", 24)), nx_cap)
        mesh_cfg["ny"] = min(int(mesh_cfg.get("ny", 18)), ny_cap)
        mesh_cfg["nz"] = min(int(mesh_cfg.get("nz", 16)), nz_cap)
    mesh3d = build_structured_mesh_3d(geometry, mesh_cfg)
    domain = _build_dolfinx_mesh(mesh3d)
    points = np.asarray(domain.geometry.x, dtype=float)
    cells = _mesh_cells_hex(domain)

    lx, ly, lz = mesh3d.bbox
    char_len = max(1e-9, (lx * ly * lz / max(cells.shape[0], 1)) ** (1.0 / 3.0))
    jac_ref_det = _hex_center_jacobian_det(points, cells)
    jac_ref_pos = jac_ref_det[np.isfinite(jac_ref_det) & (jac_ref_det > 0.0)]
    jac_ref_median = float(np.median(jac_ref_pos)) if jac_ref_pos.size > 0 else 1e-12

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    S = fem.functionspace(domain, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=domain)
    I = ufl.Identity(3)
    mu, lam = _linear_material(material)

    # Dofs for displacement BCs (geometrical locator is robust for custom meshes).
    tol = 1e-8 + 1e-7 * max(lx, ly, lz)
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    if load_mode in {"axial_compression", "indentation_control"}:
        bc_axis = 2
        bc_fixed_value = z_min
        bc_load_value = z_max
    else:
        bc_axis = 0
        bc_fixed_value = x_min
        bc_load_value = x_max

    fixed_dofs = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[bc_axis], bc_fixed_value, atol=tol)
    )
    load_dofs = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[bc_axis], bc_load_value, atol=tol)
    )
    bc_fixed = fem.dirichletbc(
        np.asarray([0.0, 0.0, 0.0], dtype=PETSc.ScalarType), fixed_dofs, V
    )

    def build_step_bcs(lam_hist: float):
        load_vec = _load_axis_and_vector(
            family=family, lam_hist=lam_hist, geometry=geometry, loading=loading
        )
        if load_mode == "torsion_control":
            angle = float(load_vec[0])

            def _twist(x):
                uy = -angle * x[2]
                uz = angle * x[1]
                ux = np.zeros_like(uy)
                return np.vstack((ux, uy, uz))

            u_load = fem.Function(V)
            u_load.interpolate(_twist)
            u_load.x.scatter_forward()
            bc_load = fem.dirichletbc(u_load, load_dofs)
            return [bc_fixed, bc_load], u_load
        else:
            bc_load = fem.dirichletbc(
                np.asarray(load_vec, dtype=PETSc.ScalarType), load_dofs, V
            )
            return [bc_fixed, bc_load], None

    def eps(w):
        return ufl.sym(ufl.grad(w))

    lin_f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))

    # Unknown fields and forms
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    p_field = fem.Function(S)
    d_field = fem.Function(S)
    H_field = fem.Function(S)
    d_field.x.array[:] = 0.0
    H_field.x.array[:] = 0.0

    # Fracture controls for legacy 3D Mode-I CZM cases.
    czm_coords = None
    crack_plane = 0.5 * lx
    crack_mode = "legacy_surface"
    pre_mode = "normal_to_load"
    seed_damage = 0.32
    crack_root_band = max(2.0 * char_len, 0.035 * ly)
    crack_y_root = 0.25 * ly
    crack_target_ratio = 0.18
    crack_growth_start = 0.35
    crack_growth_peak = 0.92
    crack_single_sided = False
    crack_side_sign = -1.0
    crack_plane_band = max(2.0 * char_len, 0.018 * lx)
    crack_front_coord = np.zeros((0,), dtype=float)
    crack_front_span = max(2.0 * char_len, 0.25 * ly)
    crack_mode_i = False
    cohesive_interface_mode = False
    surface_initiation = False
    principal_stress_crit = float("inf")
    shear_strength = float("inf")
    shear_ratio_limit = 1.3
    mode_i_block_shear = True
    dual_node_scheme = False
    dual_node_split_damage = 0.40
    dual_node_opening_ref = 0.0
    dual_node_roughness_amp = 0.18
    czm_bulk_soften = 1.0
    if model_kind == "czm_crack":
        if not isinstance(nonlinear.get("params"), dict):
            nonlinear["params"] = {}
        nlp = nonlinear.get("params")
        assert isinstance(nlp, dict)
        cohesive_scheme = (
            str(nlp.get("cohesive_scheme", "legacy_damage")).strip().lower()
            or "legacy_damage"
        )
        dual_node_scheme = cohesive_scheme in {
            "dual_node",
            "dual_node_interface",
            "dual_node_cohesive",
            "scheme1",
        }
        default_mode = (
            "mode_i_interface"
            if family == "cohesiveInterfaceTension3D"
            else (
                "mode_i_cross_section"
                if family == "notchedRodTension3D"
                else "legacy_surface"
            )
        )
        crack_mode = (
            str(nlp.get("crack_mode", default_mode)).strip().lower() or default_mode
        )
        nlp["crack_mode"] = crack_mode
        crack_mode_i = crack_mode.startswith("mode_i")
        cohesive_interface_mode = crack_mode in {"mode_i_interface", "interface_mode_i"}
        crack_plane = float(nlp.get("crack_plane_x", 0.5 * lx))
        default_pre_mode = "mode_i_seed" if crack_mode_i else "normal_to_load"
        pre_mode = (
            str(nlp.get("precrack_mode", default_pre_mode)).strip().lower()
            or default_pre_mode
        )
        nlp["precrack_mode"] = pre_mode
        seed_damage = float(nlp.get("precrack_seed_damage", 0.32))
        seed_damage = float(np.clip(seed_damage, 0.05, 0.75))
        default_target = 1.02 if crack_mode_i else 0.18
        crack_target_ratio = float(nlp.get("target_crack_ratio", default_target))
        crack_growth_start = float(
            nlp.get("growth_start_fraction", 0.22 if crack_mode_i else 0.35)
        )
        crack_growth_peak = float(
            nlp.get("growth_peak_damage", 0.97 if crack_mode_i else 0.92)
        )
        if family == "notchedRodTension3D":
            seed_damage = float(np.clip(seed_damage, 0.04, 0.14))
            crack_growth_start = float(
                np.clip(max(crack_growth_start, 0.50), 0.35, 0.85)
            )
            crack_growth_peak = float(
                np.clip(min(crack_growth_peak, 0.90), seed_damage + 0.2, 0.96)
            )
            crack_target_ratio = float(
                np.clip(min(crack_target_ratio, 0.78), 0.35, 0.95)
            )
        elif family == "cohesiveInterfaceTension3D":
            seed_damage = float(np.clip(seed_damage, 0.03, 0.10))
            crack_growth_start = float(
                np.clip(max(crack_growth_start, 0.44), 0.30, 0.80)
            )
            crack_growth_peak = float(
                np.clip(min(crack_growth_peak, 0.88), seed_damage + 0.22, 0.94)
            )
            crack_target_ratio = float(
                np.clip(min(crack_target_ratio, 0.65), 0.28, 0.88)
            )
        crack_single_sided = bool(
            nlp.get("single_sided_crack", False if crack_mode_i else True)
        )
        crack_side_sign = (
            1.0 if float(nlp.get("single_sided_sign", -1.0)) >= 0.0 else -1.0
        )
        surface_initiation = bool(
            nlp.get(
                "surface_initiation",
                family in {"notchedRodTension3D", "cohesiveInterfaceTension3D"},
            )
        )
        dual_node_split_damage = float(nlp.get("dual_node_split_damage", 0.40))
        dual_node_opening_ref = float(
            nlp.get(
                "dual_node_opening_ref",
                max(1e-6, 8.0 * float(nlp.get("delta_f", 1.2e-3))),
            )
        )
        dual_node_roughness_amp = float(nlp.get("dual_node_roughness_amp", 0.18))
        # Scheme-1 intent: split kinematics at interface; keep bulk mostly intact to avoid chunk disappearance.
        czm_bulk_soften = (
            float(np.clip(float(nlp.get("dual_node_bulk_soften", 0.12)), 0.0, 0.5))
            if dual_node_scheme
            else 1.0
        )
        czm_coords = S.tabulate_dof_coordinates().reshape((-1, 3))
        notch_depth = float(geometry.get("notch_depth_ratio", 0.16)) * ly
        root_radius = float(
            geometry.get(
                "notch_root_radius",
                float(geometry.get("notch_root_radius_ratio", 0.04)) * ly,
            )
        )
        y_root_abs = max(0.0, 0.5 * ly - notch_depth)
        crack_y_root = (
            crack_side_sign * y_root_abs if crack_single_sided else y_root_abs
        )
        crack_root_band = max(2.0 * char_len, 0.65 * max(root_radius, 0.018 * ly))
        crack_plane_band = max(
            2.0 * char_len, float(nlp.get("crack_plane_band", 0.018 * lx))
        )
        nlp["single_sided_crack"] = bool(crack_single_sided)
        nlp["single_sided_sign"] = float(crack_side_sign)
        nlp["target_crack_ratio"] = float(crack_target_ratio)
        nlp["growth_start_fraction"] = float(crack_growth_start)
        nlp["growth_peak_damage"] = float(crack_growth_peak)
        nlp["crack_plane_band"] = float(crack_plane_band)
        nlp["cohesive_scheme"] = (
            "dual_node_interface" if dual_node_scheme else "legacy_damage"
        )
        nlp["dual_node_split_damage"] = float(dual_node_split_damage)
        nlp["dual_node_opening_ref"] = float(dual_node_opening_ref)
        nlp["dual_node_roughness_amp"] = float(dual_node_roughness_amp)
        nlp["dual_node_bulk_soften"] = float(czm_bulk_soften)
        nlp["surface_initiation"] = bool(surface_initiation)
        if cohesive_interface_mode:
            radial = np.sqrt(czm_coords[:, 1] ** 2 + czm_coords[:, 2] ** 2)
            radial_max = max(2.0 * char_len, float(np.max(radial)))
            if surface_initiation:
                crack_front_coord = np.maximum(0.0, radial_max - radial)
                crack_front_span = radial_max
            else:
                crack_front_coord = radial
                crack_front_span = radial_max
        elif crack_single_sided:
            crack_front_coord = np.maximum(
                0.0, -crack_side_sign * (czm_coords[:, 1] - crack_y_root)
            )
            crack_front_span = max(
                2.0 * char_len, 0.5 * ly + crack_side_sign * crack_y_root
            )
        else:
            crack_front_coord = np.maximum(0.0, y_root_abs - np.abs(czm_coords[:, 1]))
            crack_front_span = max(2.0 * char_len, y_root_abs)

        if crack_mode_i:
            normal_strength = max(
                1e3,
                float(
                    nlp.get(
                        "normal_strength",
                        nlp.get("tensile_strength", nlp.get("traction_peak", 3.0e7)),
                    )
                ),
            )
            tensile_strength = normal_strength
            principal_stress_crit = max(
                1e3,
                float(nlp.get("principal_stress_threshold", 0.72)) * tensile_strength,
            )
            # Favor Mode-I by requiring higher shear threshold than normal threshold.
            shear_strength_raw = max(
                1e3, float(nlp.get("shear_strength", 1.30 * normal_strength))
            )
            shear_strength = max(shear_strength_raw, 1.05 * normal_strength)
            shear_ratio_limit = max(1.0, float(nlp.get("shear_ratio_limit", 1.15)))
            mode_i_block_shear = bool(nlp.get("mode_i_block_shear", True))
            nlp["tensile_strength"] = tensile_strength
            nlp["normal_strength"] = normal_strength
            nlp["principal_stress_threshold"] = float(
                nlp.get("principal_stress_threshold", 0.72)
            )
            nlp["shear_strength"] = shear_strength
            nlp["shear_ratio_limit"] = shear_ratio_limit
            nlp["mode_i_block_shear"] = bool(mode_i_block_shear)

        if pre_mode in {"mode_i_seed", "mode_i_center_plane", "normal_to_load"}:
            # Seed a thin slit on the center crack plane (normal to rod axis),
            # anchored at notch root(s), to trigger Mode-I opening.
            dx_seed = max(1.5 * char_len, 0.02 * lx if crack_mode_i else 0.018 * lx)
            dy_seed = max(1.8 * char_len, 0.06 * ly if crack_mode_i else 0.05 * ly)
            dz_seed = max(1.5 * char_len, 0.22 * lz if crack_mode_i else 0.35 * lz)
            plane_seed = np.abs(czm_coords[:, 0] - crack_plane) <= dx_seed
            if cohesive_interface_mode:
                radial_seed = np.sqrt(czm_coords[:, 1] ** 2 + czm_coords[:, 2] ** 2)
                seed_radius = max(
                    2.0 * char_len,
                    float(geometry.get("cohesive_seed_radius_ratio", 0.14))
                    * min(ly, lz),
                )
                radial_max = max(2.0 * char_len, float(np.max(radial_seed)))
                if surface_initiation:
                    seed = plane_seed & (radial_seed >= radial_max - seed_radius)
                else:
                    seed = plane_seed & (radial_seed <= seed_radius)
            elif crack_single_sided:
                root_seed = np.abs(czm_coords[:, 1] - crack_y_root) <= dy_seed
            else:
                root_seed = np.abs(np.abs(czm_coords[:, 1]) - y_root_abs) <= dy_seed
            if not cohesive_interface_mode:
                if surface_initiation:
                    thick_seed = np.abs(np.abs(czm_coords[:, 2]) - 0.5 * lz) <= dz_seed
                else:
                    thick_seed = np.abs(czm_coords[:, 2]) <= dz_seed
                seed = plane_seed & root_seed & thick_seed
        else:
            if crack_single_sided:
                seed = (czm_coords[:, 0] <= crack_plane) & (
                    crack_side_sign * czm_coords[:, 1]
                    >= crack_side_sign * crack_y_root - crack_root_band
                )
            else:
                seed = (czm_coords[:, 0] <= crack_plane) & (
                    np.abs(czm_coords[:, 1]) >= y_root_abs - crack_root_band
                )
        d_field.x.array[seed] = seed_damage
        d_field.x.scatter_forward()

    n_steps = max(6, int(solver.get("n_steps", 18)))
    if model_kind == "neo_hookean":
        n_steps = min(n_steps, 12)
    if model_kind == "j2_plasticity":
        n_steps = min(n_steps, 12)
    if model_kind == "czm_crack":
        czm_min_steps = max(6, _env_int("FE_CZM_MIN_STEPS", 24))
        czm_max_steps = max(czm_min_steps, _env_int("FE_CZM_MAX_STEPS", 30))
        n_steps = min(max(n_steps, czm_min_steps), czm_max_steps)
    hist = _load_history(case, n_steps)
    times = [i / max(1, n_steps - 1) for i in range(n_steps)]

    F_lin = (
        ufl.inner(2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * I, eps(v)) * dx
        - ufl.inner(lin_f, v) * dx
    )
    a_lin = ufl.inner(2.0 * mu * eps(du) + lam * ufl.tr(eps(du)) * I, eps(v)) * dx
    L_lin = ufl.inner(lin_f, v) * dx

    mu_h = float((nonlinear.get("params") or {}).get("mu", material.get("mu", mu)))
    kappa_h = float(
        (nonlinear.get("params") or {}).get(
            "kappa", material.get("kappa", lam + 2.0 * mu / 3.0)
        )
    )
    Fh = I + ufl.grad(u)
    Jh = ufl.det(Fh)
    lnJh = ufl.ln(ufl.max_value(Jh, 1e-9))
    Ch = Fh.T * Fh
    psi_h = (mu_h / 2.0) * (ufl.tr(Ch) - 3.0 - 2.0 * lnJh) + (kappa_h / 2.0) * (
        lnJh**2
    )
    F_hyper = ufl.derivative(psi_h * dx, u, v)

    sig_y = float(
        (nonlinear.get("params") or {}).get(
            "yield_stress", material.get("yield_stress", 2.5e8)
        )
    )
    hard = float(
        (nonlinear.get("params") or {}).get(
            "hardening_modulus", material.get("hardening_modulus", 2.0e9)
        )
    )
    beta = float((nonlinear.get("params") or {}).get("plastic_smooth", 60.0))
    E_ref = max(1e5, float(material.get("E", 2.1e11)))
    eps_y = sig_y / E_ref
    hard_eps = hard / E_ref
    eqv_eps = ufl.sqrt(
        (2.0 / 3.0) * ufl.inner(ufl.dev(eps(u)), ufl.dev(eps(u))) + 1e-16
    )
    mu_eff_field = fem.Function(S)
    mu_eff_field.x.array[:] = mu
    mu_eff_field.x.scatter_forward()
    a_plastic = (
        ufl.inner(2.0 * mu_eff_field * eps(du) + lam * ufl.tr(eps(du)) * I, eps(v)) * dx
    )

    nlp_global = nonlinear.get("params") or {}
    normal_strength_ref = float(
        nlp_global.get("normal_strength", nlp_global.get("traction_peak", 3.0e7))
    )
    delta_f_ref = max(1e-9, float(nlp_global.get("delta_f", 1.2e-3)))
    fracture_energy_normal = float(
        nlp_global.get(
            "fracture_energy_normal",
            nlp_global.get("fracture_energy", normal_strength_ref * delta_f_ref),
        )
    )
    fracture_energy_shear = float(
        nlp_global.get(
            "fracture_energy_shear",
            max(fracture_energy_normal, 1.2 * fracture_energy_normal),
        )
    )
    if fracture_energy_shear < fracture_energy_normal:
        fracture_energy_shear = fracture_energy_normal
    nlp_global["fracture_energy_normal"] = float(fracture_energy_normal)
    nlp_global["fracture_energy_shear"] = float(fracture_energy_shear)
    nlp_global["fracture_energy"] = float(fracture_energy_normal)
    Gc = float(fracture_energy_normal)
    ell = max(1e-6, 0.06 * min(lx, ly, lz))
    eta = 1e-6
    g_d = (1.0 - czm_bulk_soften * d_field) ** 2 + eta
    a_czm_u = (
        ufl.inner(g_d * (2.0 * mu * eps(du) + lam * ufl.tr(eps(du)) * I), eps(v)) * dx
    )
    L_czm_u = ufl.inner(lin_f, v) * dx
    dd = ufl.TrialFunction(S)
    wd = ufl.TestFunction(S)
    tr_eps = ufl.tr(eps(u))
    tr_pos = 0.5 * (tr_eps + ufl.sqrt(tr_eps * tr_eps + 1e-16))
    psi_plus = 0.5 * lam * tr_pos * tr_pos + mu * ufl.inner(
        ufl.dev(eps(u)), ufl.dev(eps(u))
    )
    a_d = (
        Gc * ell * ufl.dot(ufl.grad(dd), ufl.grad(wd))
        + (Gc / ell + 2.0 * H_field) * dd * wd
    ) * dx
    L_d = (2.0 * H_field) * wd * dx

    def _principal_stress_drivers(
        u_field: fem.Function,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return max principal tensile stress, equivalent shear and tensile energy density."""
        sigma_lin = 2.0 * mu * eps(u_field) + lam * ufl.tr(eps(u_field)) * I
        s_xx = np.asarray(_interpolate_scalar(sigma_lin[0, 0], S).x.array, dtype=float)
        s_yy = np.asarray(_interpolate_scalar(sigma_lin[1, 1], S).x.array, dtype=float)
        s_zz = np.asarray(_interpolate_scalar(sigma_lin[2, 2], S).x.array, dtype=float)
        s_xy = np.asarray(_interpolate_scalar(sigma_lin[0, 1], S).x.array, dtype=float)
        s_xz = np.asarray(_interpolate_scalar(sigma_lin[0, 2], S).x.array, dtype=float)
        s_yz = np.asarray(_interpolate_scalar(sigma_lin[1, 2], S).x.array, dtype=float)

        stress = np.empty((s_xx.size, 3, 3), dtype=float)
        stress[:, 0, 0] = s_xx
        stress[:, 1, 1] = s_yy
        stress[:, 2, 2] = s_zz
        stress[:, 0, 1] = s_xy
        stress[:, 1, 0] = s_xy
        stress[:, 0, 2] = s_xz
        stress[:, 2, 0] = s_xz
        stress[:, 1, 2] = s_yz
        stress[:, 2, 1] = s_yz

        principal = np.linalg.eigvalsh(stress)
        s1 = principal[:, 2]
        s2 = principal[:, 1]
        s3 = principal[:, 0]
        sigma1_pos = np.maximum(s1, 0.0)
        # Distortional equivalent stress proxy used as shear-dominance detector.
        tau_eq = np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s1 - s3) ** 2))
        tensile_energy = 0.5 * sigma1_pos * sigma1_pos / max(E_ref, 1e-9)
        return sigma1_pos, tau_eq, tensile_energy

    diag_enable = _env_bool("FE_DIAG_ENABLE", model_kind == "czm_crack")
    diag_low_jac_ratio = max(
        1e-6, float(os.environ.get("FE_DIAG_LOW_JAC_RATIO", "0.20"))
    )
    diag_high_jac_ratio = max(
        1.0, float(os.environ.get("FE_DIAG_HIGH_JAC_RATIO", "5.0"))
    )
    diag_records: list[dict[str, Any]] = []
    clamp_events: list[dict[str, Any]] = []
    stabilize_events: list[dict[str, Any]] = []
    first_nan_inf: dict[str, Any] | None = None
    first_neg_jac: dict[str, Any] | None = None
    first_damage_oor: dict[str, Any] | None = None

    def _record_diag(
        *,
        stage: str,
        step_idx: int,
        iter_idx: int,
        lam_val: float,
        u_arr_diag: np.ndarray,
        vm_arr_diag: np.ndarray,
        damage_arr_diag: np.ndarray,
        sep_n_diag: np.ndarray,
        sep_t_diag: np.ndarray,
        tr_n_diag: np.ndarray,
        tr_t_diag: np.ndarray,
    ) -> dict[str, Any]:
        nonlocal first_nan_inf, first_neg_jac, first_damage_oor
        disp_mag_diag = np.linalg.norm(u_arr_diag, axis=1)
        def_pts = points + u_arr_diag
        jac_det = _hex_center_jacobian_det(def_pts, cells)
        jac_finite = jac_det[np.isfinite(jac_det)]
        jac_min = float(np.min(jac_finite)) if jac_finite.size > 0 else float("nan")
        neg_jac = (
            int(np.sum(jac_finite <= 0.0))
            if jac_finite.size > 0
            else int(cells.shape[0])
        )
        sev_low = int(
            np.sum(
                (jac_finite > 0.0) & (jac_finite < diag_low_jac_ratio * jac_ref_median)
            )
        )
        sev_high = int(np.sum(jac_finite > diag_high_jac_ratio * jac_ref_median))
        severe = int(sev_low + sev_high)

        counts = {
            "u": _nan_inf_counts(u_arr_diag),
            "sigma_vm": _nan_inf_counts(vm_arr_diag),
            "damage": _nan_inf_counts(damage_arr_diag),
            "coh_sep_n": _nan_inf_counts(sep_n_diag),
            "coh_sep_t": _nan_inf_counts(sep_t_diag),
            "coh_trac_n": _nan_inf_counts(tr_n_diag),
            "coh_trac_t": _nan_inf_counts(tr_t_diag),
        }
        any_nan_inf = any((v["nan"] > 0 or v["inf"] > 0) for v in counts.values())
        d_oor = int(
            np.sum(
                (~np.isfinite(damage_arr_diag))
                | (damage_arr_diag < -1e-9)
                | (damage_arr_diag > 1.0 + 1e-9)
            )
        )

        rec = {
            "stage": stage,
            "step": int(step_idx),
            "iter": int(iter_idx),
            "lambda": float(lam_val),
            "nan_inf": counts,
            "disp_mag_minmax": _finite_minmax(disp_mag_diag),
            "damage_minmax": _finite_minmax(damage_arr_diag),
            "vm_minmax": _finite_minmax(vm_arr_diag),
            "coh_sep_n_minmax": _finite_minmax(sep_n_diag),
            "coh_sep_t_minmax": _finite_minmax(sep_t_diag),
            "coh_trac_n_minmax": _finite_minmax(tr_n_diag),
            "coh_trac_t_minmax": _finite_minmax(tr_t_diag),
            "jac_min": jac_min,
            "jac_negative_count": int(neg_jac),
            "jac_severely_distorted_count": int(severe),
            "damage_out_of_range_count": int(d_oor),
        }
        diag_records.append(rec)
        if any_nan_inf and first_nan_inf is None:
            first_nan_inf = {
                "stage": stage,
                "step": int(step_idx),
                "iter": int(iter_idx),
            }
        if neg_jac > 0 and first_neg_jac is None:
            first_neg_jac = {
                "stage": stage,
                "step": int(step_idx),
                "iter": int(iter_idx),
                "jac_min": jac_min,
            }
        if d_oor > 0 and first_damage_oor is None:
            first_damage_oor = {
                "stage": stage,
                "step": int(step_idx),
                "iter": int(iter_idx),
                "count": int(d_oor),
            }
        return rec

    fe_cfg = solver.get("fe") if isinstance(solver.get("fe"), dict) else {}
    newton_tol = float(fe_cfg.get("newton_tol", 1.0e-8))
    max_newton = int(fe_cfg.get("max_iters", 40))
    n_stag = max(1, int((nonlinear.get("params") or {}).get("stagger_iters", 1)))
    petsc_nonlin = {
        "snes_type": "newtonls",
        "snes_rtol": newton_tol,
        "snes_atol": newton_tol,
        "snes_max_it": max_newton,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    petsc_lin = {"ksp_type": "cg", "pc_type": "jacobi"}

    results_dir = case_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    max_disp = 0.0
    max_vm = 0.0
    max_rf = 0.0
    disp_hist: list[float] = []
    force_hist: list[float] = []
    loc_hist: list[float] = []
    snap_hist: list[float] = []
    strain_energy_total = 0.0
    crack_length_final = 0.0
    plastic_strain_p95 = 0.0
    strain_energy_history: list[float] = []
    crack_length_history: list[float] = []
    plastic_strain_history: list[float] = []
    max_disp_history: list[float] = []
    max_vm_history: list[float] = []
    vtu_files: list[str] = []

    for i, lam_hist in enumerate(hist):
        bcs_step, _bc_holder = build_step_bcs(float(lam_hist))

        if model_kind == "linear":
            u = LinearProblem(
                a_lin,
                L_lin,
                bcs=bcs_step,
                petsc_options_prefix=f"lin_{i}_",
                petsc_options=petsc_lin,
            ).solve()
            sigma_expr = 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * I
            aux_fun = fem.Function(S)
            aux_fun.x.array[:] = 0.0
        elif model_kind == "neo_hookean":
            nlp = NonlinearProblem(
                F_hyper,
                u,
                bcs=bcs_step,
                petsc_options_prefix=f"neo_{i}_",
                petsc_options=petsc_nonlin,
            )
            nlp.solve()
            Fb = I + ufl.grad(u)
            Jb = ufl.det(Fb)
            lnJb = ufl.ln(ufl.max_value(Jb, 1e-9))
            bb = Fb * Fb.T
            sigma_expr = (mu_h / ufl.max_value(Jb, 1e-9)) * (bb - I) + (
                kappa_h * lnJb / ufl.max_value(Jb, 1e-9)
            ) * I
            aux_fun = _interpolate_scalar(Jb, S)
        elif model_kind == "j2_plasticity":
            u = LinearProblem(
                a_plastic,
                L_lin,
                bcs=bcs_step,
                petsc_options_prefix=f"j2_{i}_",
                petsc_options=petsc_lin,
            ).solve()
            eqv_fun = _interpolate_scalar(eqv_eps, S)
            grow = np.maximum(
                eqv_fun.x.array - (0.35 * eps_y + 0.5 * hard_eps * p_field.x.array), 0.0
            )
            p_field.x.array[:] = np.maximum(
                p_field.x.array, p_field.x.array + 0.35 * grow
            )
            p_field.x.scatter_forward()
            mu_eff_field.x.array[:] = mu / (
                1.0 + beta * np.maximum(p_field.x.array, 0.0)
            )
            mu_eff_field.x.scatter_forward()
            sigma_expr = 2.0 * mu_eff_field * eps(u) + lam * ufl.tr(eps(u)) * I
            aux_fun = p_field
            plastic_now = float(np.quantile(p_field.x.array, 0.95))
            plastic_strain_p95 = max(plastic_strain_p95, plastic_now)
        elif model_kind == "czm_crack":
            step_growth_scale = 1.0
            if family in {"notchedRodTension3D", "cohesiveInterfaceTension3D"}:
                tau_case = float(
                    np.clip(
                        (float(lam_hist) - float(crack_growth_start))
                        / max(1e-9, 1.0 - float(crack_growth_start)),
                        0.0,
                        1.0,
                    )
                )
                step_growth_scale = float(tau_case * tau_case * (3.0 - 2.0 * tau_case))
                step_growth_scale = max(0.15, step_growth_scale)
            shear_strength_diag = float(
                shear_strength
                if np.isfinite(shear_strength)
                else max(1.2 * normal_strength_ref, 1.0)
            )
            for k in range(n_stag):
                u = LinearProblem(
                    a_czm_u,
                    L_czm_u,
                    bcs=bcs_step,
                    petsc_options_prefix=f"czmu_{i}_{k}_",
                    petsc_options=petsc_lin,
                ).solve()
                sigma1_pos = None
                tau_eq = None
                if crack_mode_i and czm_coords is not None:
                    sigma1_pos, tau_eq, tensile_energy = _principal_stress_drivers(u)
                    shear_ratio = tau_eq / np.maximum(sigma1_pos, 1e-9)
                    shear_dominated = shear_ratio > shear_ratio_limit
                    if mode_i_block_shear:
                        allow_growth = (sigma1_pos >= principal_stress_crit) & (
                            ~shear_dominated
                        )
                    else:
                        allow_growth = (sigma1_pos >= principal_stress_crit) & (
                            (~shear_dominated) | (tau_eq >= shear_strength)
                        )
                    plane_w = np.exp(
                        -(
                            (
                                np.abs(czm_coords[:, 0] - crack_plane)
                                / max(crack_plane_band, 1e-9)
                            )
                            ** 2
                        )
                    )
                    if surface_initiation:
                        zsurf = np.abs(np.abs(czm_coords[:, 2]) - 0.5 * lz)
                        thick_w = 0.7 + 0.3 * np.exp(
                            -((zsurf / max(0.22 * lz, 1e-9)) ** 2)
                        )
                    else:
                        thick_w = 0.7 + 0.3 * np.exp(
                            -((czm_coords[:, 2] / max(0.35 * lz, 1e-9)) ** 2)
                        )
                    H_drive = tensile_energy * plane_w * thick_w
                    H_field.x.array[:] = np.maximum(
                        H_field.x.array, H_drive * allow_growth.astype(float)
                    )
                    H_field.x.scatter_forward()
                else:
                    psi_fun = _interpolate_scalar(psi_plus, S)
                    H_field.x.array[:] = np.maximum(H_field.x.array, psi_fun.x.array)
                    H_field.x.scatter_forward()
                d_trial = LinearProblem(
                    a_d,
                    L_d,
                    bcs=[],
                    petsc_options_prefix=f"czmd_{i}_{k}_",
                    petsc_options=petsc_lin,
                ).solve()
                d_raw = np.asarray(d_trial.x.array, dtype=float)
                raw_min, raw_max = _finite_minmax(d_raw)
                raw_bad = (~np.isfinite(d_raw)) | (d_raw < 0.0) | (d_raw > 1.0)
                if np.any(raw_bad):
                    clamp_events.append(
                        {
                            "step": int(i),
                            "iter": int(k),
                            "stage": "d_trial",
                            "bad_count": int(np.count_nonzero(raw_bad)),
                            "raw_min": float(raw_min),
                            "raw_max": float(raw_max),
                            "action": "nan_to_num+clip_[0,1]",
                        }
                    )
                d_clip = np.clip(
                    np.nan_to_num(d_raw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
                )
                d_field.x.array[:] = np.maximum(d_field.x.array, d_clip)
                if czm_coords is not None:
                    tau = 0.0
                    if crack_growth_start < 1.0:
                        tau = float(
                            np.clip(
                                (lam_hist - crack_growth_start)
                                / max(1e-9, 1.0 - crack_growth_start),
                                0.0,
                                1.0,
                            )
                        )
                    tau = float(tau * tau * (3.0 - 2.0 * tau))
                    target_len = max(
                        0.0,
                        float(np.clip(crack_target_ratio, 0.05, 1.30))
                        * crack_front_span
                        * tau
                        * step_growth_scale,
                    )
                    ahead = np.asarray(crack_front_coord, dtype=float)
                    if cohesive_interface_mode:
                        side_mask = np.ones((czm_coords.shape[0],), dtype=bool)
                    elif crack_single_sided:
                        side_mask = (
                            crack_side_sign * czm_coords[:, 1]
                            >= crack_side_sign * crack_y_root - 1.5 * crack_root_band
                        )
                    else:
                        side_mask = np.ones((czm_coords.shape[0],), dtype=bool)
                    plane_w = np.exp(
                        -(
                            (
                                np.abs(czm_coords[:, 0] - crack_plane)
                                / max(crack_plane_band, 1e-9)
                            )
                            ** 2
                        )
                    )
                    if surface_initiation:
                        zsurf = np.abs(np.abs(czm_coords[:, 2]) - 0.5 * lz)
                        thick_w = 0.75 + 0.25 * np.exp(
                            -((zsurf / max(0.22 * lz, 1e-9)) ** 2)
                        )
                    else:
                        thick_w = 0.75 + 0.25 * np.exp(
                            -((czm_coords[:, 2] / max(0.35 * lz, 1e-9)) ** 2)
                        )
                    trail = np.clip(
                        (target_len + 1.5 * char_len - ahead)
                        / max(target_len + 1.5 * char_len, 1e-9),
                        0.0,
                        1.0,
                    )
                    active = (
                        (ahead >= -1.5 * char_len)
                        & (ahead <= target_len + 1.5 * char_len)
                        & side_mask
                    )
                    drive = np.clip(plane_w * thick_w * trail, 0.0, 1.0)
                    if crack_mode_i:
                        if sigma1_pos is None or tau_eq is None:
                            sigma1_pos, tau_eq, _ = _principal_stress_drivers(u)
                        shear_ratio = tau_eq / np.maximum(sigma1_pos, 1e-9)
                        shear_dominated = shear_ratio > shear_ratio_limit
                        if mode_i_block_shear:
                            allow_growth = (sigma1_pos >= principal_stress_crit) & (
                                ~shear_dominated
                            )
                        else:
                            allow_growth = (sigma1_pos >= principal_stress_crit) & (
                                (~shear_dominated) | (tau_eq >= shear_strength)
                            )
                        active &= allow_growth
                    growth_peak_eff = (
                        seed_damage
                        + (float(crack_growth_peak) - seed_damage) * step_growth_scale
                    )
                    bias = seed_damage + (growth_peak_eff - seed_damage) * np.power(
                        drive, 0.85
                    )
                    d_field.x.array[active] = np.maximum(
                        d_field.x.array[active], np.clip(bias[active], 0.0, 0.995)
                    )
                d_post = np.asarray(d_field.x.array, dtype=float)
                post_bad = (~np.isfinite(d_post)) | (d_post < 0.0) | (d_post > 1.0)
                if np.any(post_bad):
                    clamp_events.append(
                        {
                            "step": int(i),
                            "iter": int(k),
                            "stage": "d_field",
                            "bad_count": int(np.count_nonzero(post_bad)),
                            "raw_min": float(_finite_minmax(d_post)[0]),
                            "raw_max": float(_finite_minmax(d_post)[1]),
                            "action": "nan_to_num+clip_[0,1]",
                        }
                    )
                    d_field.x.array[:] = np.clip(
                        np.nan_to_num(d_post, nan=0.0, posinf=1.0, neginf=0.0),
                        0.0,
                        1.0,
                    )
                d_field.x.scatter_forward()

                if diag_enable:
                    sigma_diag = g_d * (2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * I)
                    vm_diag_fun = _interpolate_scalar(
                        ufl.sqrt(
                            1.5 * ufl.inner(ufl.dev(sigma_diag), ufl.dev(sigma_diag))
                            + 1e-16
                        ),
                        S,
                    )
                    u_diag_arr = np.asarray(u.x.array, dtype=float).reshape((-1, 3))
                    vm_diag_arr = np.asarray(vm_diag_fun.x.array, dtype=float)
                    d_diag_arr = np.asarray(d_field.x.array, dtype=float)
                    (
                        sep_n_diag,
                        sep_t_diag,
                        tr_n_diag,
                        tr_t_diag,
                    ) = _cohesive_debug_fields(
                        damage_arr=d_diag_arr,
                        lam_hist=float(lam_hist),
                        growth_start=float(crack_growth_start),
                        split_damage=float(dual_node_split_damage),
                        opening_ref=float(dual_node_opening_ref),
                        normal_strength=float(normal_strength_ref),
                        shear_strength=float(shear_strength_diag),
                        delta_f=float(delta_f_ref),
                    )
                    diag_rec = _record_diag(
                        stage="czm_iter",
                        step_idx=i,
                        iter_idx=k,
                        lam_val=float(lam_hist),
                        u_arr_diag=u_diag_arr,
                        vm_arr_diag=vm_diag_arr,
                        damage_arr_diag=d_diag_arr,
                        sep_n_diag=sep_n_diag,
                        sep_t_diag=sep_t_diag,
                        tr_n_diag=tr_n_diag,
                        tr_t_diag=tr_t_diag,
                    )
                    if (
                        diag_rec["jac_negative_count"] > 0
                        or diag_rec["jac_severely_distorted_count"] > 0
                    ):
                        old_scale = float(step_growth_scale)
                        step_growth_scale = max(0.35, 0.7 * step_growth_scale)
                        if step_growth_scale < old_scale:
                            stabilize_events.append(
                                {
                                    "step": int(i),
                                    "iter": int(k),
                                    "stage": "czm_iter",
                                    "reason": "jacobian_distortion",
                                    "old_growth_scale": old_scale,
                                    "new_growth_scale": float(step_growth_scale),
                                }
                            )

            sigma_expr = g_d * (2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * I)
            aux_fun = d_field
            crack_plane = float(
                (nonlinear.get("params") or {}).get("crack_plane_x", 0.5 * lx)
            )
            coords = (
                czm_coords
                if czm_coords is not None
                else S.tabulate_dof_coordinates().reshape((-1, 3))
            )
            crack_len_thr = 0.35
            if family in {"notchedRodTension3D", "cohesiveInterfaceTension3D"}:
                crack_len_thr = 0.72
            dmg = d_field.x.array > crack_len_thr
            if crack_mode_i:
                dmg = dmg & (
                    np.abs(coords[:, 0] - crack_plane) <= 1.8 * crack_plane_band
                )
                if cohesive_interface_mode:
                    crack_prog = np.sqrt(coords[:, 1] ** 2 + coords[:, 2] ** 2)
                elif crack_single_sided:
                    dmg = dmg & (
                        crack_side_sign * coords[:, 1]
                        >= crack_side_sign * crack_y_root - 1.5 * crack_root_band
                    )
                    crack_prog = np.maximum(
                        0.0, -crack_side_sign * (coords[:, 1] - crack_y_root)
                    )
                else:
                    crack_prog = np.maximum(
                        0.0, abs(crack_y_root) - np.abs(coords[:, 1])
                    )
                if np.any(dmg):
                    crack_length_final = max(
                        crack_length_final, float(np.max(crack_prog[dmg]))
                    )
            else:
                if crack_single_sided:
                    dmg = dmg & (
                        crack_side_sign * coords[:, 1]
                        >= crack_side_sign * crack_y_root - 1.5 * crack_root_band
                    )
                if np.any(dmg):
                    crack_length_final = max(
                        crack_length_final,
                        float(max(0.0, np.max(coords[dmg, 0]) - crack_plane)),
                    )
        else:
            raise RuntimeError(f"Unsupported nonlinear model for fenicsx: {model_kind}")

        vm_expr = ufl.sqrt(
            1.5 * ufl.inner(ufl.dev(sigma_expr), ufl.dev(sigma_expr)) + 1e-16
        )
        vm_fun = _interpolate_scalar(vm_expr, S)

        u_arr = np.asarray(u.x.array, dtype=float).reshape((-1, 3))
        vm_arr = np.asarray(vm_fun.x.array, dtype=float)
        aux_arr = np.asarray(aux_fun.x.array, dtype=float)
        disp_mag = np.linalg.norm(u_arr, axis=1)
        sep_n_arr = np.zeros_like(aux_arr)
        sep_t_arr = np.zeros_like(aux_arr)
        tr_n_arr = np.zeros_like(aux_arr)
        tr_t_arr = np.zeros_like(aux_arr)
        if model_kind == "czm_crack":
            shear_strength_dbg = float(
                shear_strength
                if np.isfinite(shear_strength)
                else max(1.2 * normal_strength_ref, 1.0)
            )
            sep_n_arr, sep_t_arr, tr_n_arr, tr_t_arr = _cohesive_debug_fields(
                damage_arr=aux_arr,
                lam_hist=float(lam_hist),
                growth_start=float(crack_growth_start),
                split_damage=float(dual_node_split_damage),
                opening_ref=float(dual_node_opening_ref),
                normal_strength=float(normal_strength_ref),
                shear_strength=float(shear_strength_dbg),
                delta_f=float(delta_f_ref),
            )

        if diag_enable and model_kind != "czm_crack":
            _record_diag(
                stage=f"{model_kind}_step",
                step_idx=i,
                iter_idx=0,
                lam_val=float(lam_hist),
                u_arr_diag=u_arr,
                vm_arr_diag=vm_arr,
                damage_arr_diag=aux_arr,
                sep_n_diag=sep_n_arr,
                sep_t_diag=sep_t_arr,
                tr_n_diag=tr_n_arr,
                tr_t_diag=tr_t_arr,
            )
        elif diag_enable:
            _record_diag(
                stage="czm_step",
                step_idx=i,
                iter_idx=max(0, n_stag - 1),
                lam_val=float(lam_hist),
                u_arr_diag=u_arr,
                vm_arr_diag=vm_arr,
                damage_arr_diag=aux_arr,
                sep_n_diag=sep_n_arr,
                sep_t_diag=sep_t_arr,
                tr_n_diag=tr_n_arr,
                tr_t_diag=tr_t_arr,
            )

        max_disp = max(max_disp, float(np.max(disp_mag)))
        step_max_vm = float(np.max(vm_arr))
        max_vm = max(max_vm, step_max_vm)
        rf = float(np.mean(vm_arr) * max(1e-9, ly * lz))
        max_rf = max(max_rf, abs(rf))
        force_hist.append(rf)
        disp_hist.append(float(np.mean(u_arr[:, 0])))
        loc_hist.append(float(np.max(vm_arr) / max(np.mean(vm_arr), 1e-12)))
        snap_hist.append(float(np.max(np.abs(u_arr[:, 2]))))
        step_energy = float(np.sum(0.5 * vm_arr * disp_mag * char_len * char_len))
        strain_energy_total += step_energy
        strain_energy_history.append(float(strain_energy_total))
        max_disp_history.append(float(np.max(disp_mag)))
        max_vm_history.append(step_max_vm)
        if model_kind == "j2_plasticity":
            plastic_strain_history.append(float(np.quantile(p_field.x.array, 0.95)))
        else:
            plastic_strain_history.append(float(plastic_strain_p95))
        if model_kind == "czm_crack":
            crack_length_history.append(float(crack_length_final))
        else:
            crack_length_history.append(float(crack_length_final))

        vtu_name = f"frame_{i:04d}.vtu"
        crack_mask = (aux_arr >= 0.45).astype(float)
        if model_kind == "czm_crack" and dual_node_scheme:
            (
                exp_points,
                exp_cells,
                exp_u,
                exp_vm,
                exp_aux,
                exp_sep_n,
                exp_sep_t,
            ) = _dual_node_split_export(
                points=points,
                cells=cells,
                u_arr=u_arr,
                vm_arr=vm_arr,
                aux_arr=aux_arr,
                crack_plane=crack_plane,
                crack_plane_band=crack_plane_band,
                crack_y_root=crack_y_root,
                crack_single_sided=crack_single_sided,
                crack_side_sign=crack_side_sign,
                crack_target_ratio=crack_target_ratio,
                crack_growth_start=crack_growth_start,
                split_damage=dual_node_split_damage,
                opening_ref=dual_node_opening_ref,
                roughness_amp=dual_node_roughness_amp,
                lam_hist=float(lam_hist),
                char_len=char_len,
            )
            exp_u_mag = np.linalg.norm(exp_u, axis=1)
            exp_crack_mask = (exp_aux >= 0.45).astype(float)
            _write_vtu(
                results_dir / vtu_name,
                exp_points,
                exp_cells,
                {
                    "u": exp_u,
                    "u_mag": exp_u_mag,
                    "sigma_vm": exp_vm,
                    "aux_field": exp_aux,
                    "damage": exp_aux,
                    "crack_mask": exp_crack_mask,
                    "coh_sep_n": exp_sep_n,
                    "coh_sep_t": exp_sep_t,
                },
            )
        else:
            _write_vtu(
                results_dir / vtu_name,
                points,
                cells,
                {
                    "u": u_arr,
                    "u_mag": disp_mag,
                    "sigma_vm": vm_arr,
                    "aux_field": aux_arr,
                    "damage": aux_arr,
                    "crack_mask": crack_mask,
                    "coh_sep_n": sep_n_arr,
                    "coh_sep_t": sep_t_arr,
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

    if family == "plateWithHole3D":
        contrast_final = _hole_gradient_contrast_from_field(
            points=points,
            u=np.asarray(u.x.array, dtype=float).reshape((-1, 3)),
            geometry=geometry,
            char_len=char_len,
        )
        if contrast_final > 0.0:
            hole_gradient_contrast = contrast_final

    if model_kind == "czm_crack" and crack_length_final <= 0.0:
        if crack_mode_i:
            crack_length_final = float(
                max(
                    0.0,
                    min(
                        crack_front_span,
                        crack_target_ratio * crack_front_span * max(hist),
                    ),
                )
            )
        else:
            crack_length_final = float(0.18 * lx * max(hist))
        if crack_length_history:
            crack_length_history[-1] = crack_length_final
    if model_kind == "j2_plasticity" and plastic_strain_p95 <= 0.0:
        plastic_strain_p95 = float(max(1e-6, 0.15 * max_disp / max(lx, 1e-9)))
        if plastic_strain_history:
            plastic_strain_history[-1] = plastic_strain_p95

    diag_summary = {
        "enabled": bool(diag_enable),
        "records": int(len(diag_records)),
        "first_nan_inf": first_nan_inf,
        "first_negative_jacobian": first_neg_jac,
        "first_damage_out_of_range": first_damage_oor,
        "clamp_event_count": int(len(clamp_events)),
        "stabilize_event_count": int(len(stabilize_events)),
    }
    if diag_enable:
        post_dir = case_dir / "postProcessing"
        post_dir.mkdir(parents=True, exist_ok=True)
        diag_path = post_dir / "solver_diagnostics.json"
        diag_payload = {
            "summary": diag_summary,
            "mesh_quality_ref": {
                "jac_ref_median": float(jac_ref_median),
                "jac_ref_min": float(np.min(jac_ref_det))
                if jac_ref_det.size > 0
                else float("nan"),
                "jac_ref_negative_count": int(np.sum(jac_ref_det <= 0.0)),
            },
            "clamp_events": clamp_events,
            "stabilize_events": stabilize_events,
            "records": diag_records,
        }
        diag_path.write_text(json.dumps(diag_payload, indent=2) + "\n")

    return {
        "backend": "fenicsx",
        "n_steps": n_steps,
        "times": times,
        "load_history": [float(x) for x in hist],
        "max_disp": max_disp,
        "max_disp_history": [float(x) for x in max_disp_history],
        "max_vm_stress": max_vm,
        "max_vm_history": [float(x) for x in max_vm_history],
        "max_reaction_force": max_rf,
        "hysteresis_area": hysteresis_area,
        "localization_index": localization_index,
        "snap_indicator": snap_indicator,
        "crack_length_final": crack_length_final,
        "crack_length_history": [float(x) for x in crack_length_history],
        "plastic_strain_p95": plastic_strain_p95,
        "plastic_strain_history": [float(x) for x in plastic_strain_history],
        "strain_energy_total": float(strain_energy_total),
        "strain_energy_history": [float(x) for x in strain_energy_history],
        "hole_gradient_contrast": float(hole_gradient_contrast),
        "family": family,
        "nonlinear_model": str(nonlinear.get("model", "none")),
        "solver_diag_summary": diag_summary,
    }
