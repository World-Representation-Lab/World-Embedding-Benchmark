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
    _principal_2d,
    _interpolate_scalar,
)


def _require_dolfinx() -> None:
    try:
        import basix.ufl  # noqa: F401
        import dolfinx  # noqa: F401
        import gmsh  # noqa: F401
        import mpi4py  # noqa: F401
        import petsc4py  # noqa: F401
        import ufl  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "2D FE backend requires dolfinx + gmsh + mpi4py + petsc4py + ufl"
        ) from exc


def _plane_stress_material(material: dict[str, Any]) -> tuple[float, float, float]:
    E = max(1e5, float(material.get("E", 2.0e9)))
    nu = float(material.get("nu", 0.30))
    nu = max(-0.45, min(0.49, nu))
    mu = E / (2.0 * max(1e-9, 1.0 + nu))
    lam_ps = E * nu / max(1e-9, 1.0 - nu * nu)
    return float(E), float(mu), float(lam_ps)


def _triangle_cells(domain) -> np.ndarray:
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    conn = domain.topology.connectivity(tdim, 0)
    nc = int(domain.topology.index_map(tdim).size_local)
    cells = np.zeros((nc, 3), dtype=np.int64)
    for i in range(nc):
        cells[i, :] = np.asarray(conn.links(i), dtype=np.int64)
    return cells


def _dogbone_polygon(
    lx: float, ly: float, center: float, neck_len: float
) -> list[tuple[float, float]]:
    shoulder = 0.5 * ly
    neck = 0.34 * ly
    trans = max(0.08 * lx, 0.75 * neck_len)
    x1 = max(0.08 * lx, center - 0.5 * neck_len - trans)
    x2 = max(x1 + 0.04 * lx, center - 0.5 * neck_len)
    x3 = min(lx - 0.04 * lx, center + 0.5 * neck_len)
    x4 = min(lx - 0.08 * lx, center + 0.5 * neck_len + trans)
    if x3 <= x2:
        mid = center
        x2 = max(0.18 * lx, mid - 0.03 * lx)
        x3 = min(0.82 * lx, mid + 0.03 * lx)
    if x4 <= x3:
        x4 = min(lx - 0.05 * lx, x3 + 0.06 * lx)
    pts = [
        (0.0, shoulder),
        (x1, shoulder),
        (x2, neck),
        (x3, neck),
        (x4, shoulder),
        (lx, shoulder),
        (lx, -shoulder),
        (x4, -shoulder),
        (x3, -neck),
        (x2, -neck),
        (x1, -shoulder),
        (0.0, -shoulder),
    ]
    return pts


def _build_gmsh_domain(case: dict[str, Any]):
    import gmsh
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    family = str(case.get("family") or case.get("case_prefix") or "")
    kind = str(geometry.get("kind", "prism")).strip().lower()
    lx = float(geometry.get("lx", 2.0))
    ly = float(geometry.get("ly", 1.0))
    mesh_cfg = solver.get("mesh") if isinstance(solver.get("mesh"), dict) else {}
    nx = max(24, int(mesh_cfg.get("nx", 120)))
    ny = max(12, int(mesh_cfg.get("ny", 48)))
    h = max(1e-4, min(lx / float(nx), ly / float(ny)))

    comm = MPI.COMM_WORLD
    rank = comm.rank
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if rank == 0:
        gmsh.model.add(f"{family or 'solid2d'}")
        occ = gmsh.model.occ
        if kind == "plate_with_hole":
            cx = float(geometry.get("hole_cx", 0.5 * lx))
            cy = float(geometry.get("hole_cy", 0.0))
            r0 = max(1e-6, float(geometry.get("hole_radius", 0.16 * min(lx, ly))))
            rect = occ.addRectangle(0.0, -0.5 * ly, 0.0, lx, ly)
            hole = occ.addDisk(cx, cy, 0.0, r0, r0)
            cut = occ.cut([(2, rect)], [(2, hole)], removeObject=True, removeTool=True)
            surf_tags = [tag for dim, tag in cut[0] if dim == 2]
        elif kind == "dogbone":
            center = float(
                np.clip(
                    float(geometry.get("neck_region_center", 0.5 * lx)),
                    0.2 * lx,
                    0.8 * lx,
                )
            )
            neck_len = float(
                np.clip(
                    float(geometry.get("neck_region_width", 0.18 * lx)),
                    0.08 * lx,
                    0.35 * lx,
                )
            )
            pts = _dogbone_polygon(lx, ly, center, neck_len)
            p_tags = [occ.addPoint(px, py, 0.0, h) for px, py in pts]
            l_tags = [
                occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)])
                for i in range(len(p_tags))
            ]
            loop = occ.addCurveLoop(l_tags)
            surf_tags = [occ.addPlaneSurface([loop])]
        else:
            surf_tags = [occ.addRectangle(0.0, -0.5 * ly, 0.0, lx, ly)]
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2, surf_tags, 1)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        gmsh.model.mesh.generate(2)
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    domain = mesh_data.mesh
    gmsh.finalize()
    return domain


def _build_domain(case: dict[str, Any]):
    from mpi4py import MPI
    from dolfinx import mesh

    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    family = str(case.get("family") or case.get("case_prefix") or "")
    kind = str(geometry.get("kind", "prism")).strip().lower()
    lx = float(geometry.get("lx", 2.0))
    ly = float(geometry.get("ly", 1.0))
    mesh_cfg = solver.get("mesh") if isinstance(solver.get("mesh"), dict) else {}
    nx = max(24, int(mesh_cfg.get("nx", 120)))
    ny = max(12, int(mesh_cfg.get("ny", 48)))

    if kind in {"plate_with_hole", "dogbone"}:
        domain = _build_gmsh_domain(case)
    else:
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.asarray([0.0, -0.5 * ly]), np.asarray([lx, 0.5 * ly])],
            [nx, ny],
            cell_type=mesh.CellType.triangle,
        )

    if family == "bucklingSnapthrough":
        amp = float(geometry.get("imperfection", 0.002)) * lx
        coords = domain.geometry.x
        coords[:, 1] += amp * np.sin(math.pi * coords[:, 0] / max(lx, 1e-9))

    return domain


def _model_kind(case: dict[str, Any]) -> str:
    family = str(case.get("family") or case.get("case_prefix") or "")
    material = case.get("material") if isinstance(case.get("material"), dict) else {}
    nonlinear = case.get("nonlinear") if isinstance(case.get("nonlinear"), dict) else {}
    model = str(nonlinear.get("model", "none")).strip().lower()
    mat_model = str(material.get("model", "")).strip().lower()
    if family == "centerNotchPlate2D":
        return "sent_damage_2d"
    if family == "bucklingSnapthrough":
        return "geometric_nonlinear"
    if (
        model == "neo_hookean"
        or mat_model == "neo_hookean"
        or family in {"neoHookean", "nearIncompressibleBlock"}
    ):
        return "neo_hookean"
    if (
        model in {"j2_plasticity", "kinematic_hardening"}
        or mat_model in {"j2_plastic", "kinematic_hardening"}
        or family in {"yieldHardening", "loadUnloadHysteresis", "necking"}
    ):
        return "j2_plasticity"
    return "linear"


def _make_spatial_field(S, fill: float = 1.0):
    from dolfinx import fem

    f = fem.Function(S)
    f.x.array[:] = float(fill)
    f.x.scatter_forward()
    return f


def _family_scalar_fields(
    *, S, family: str, geometry: dict[str, Any], material: dict[str, Any]
):
    coords = np.asarray(S.tabulate_dof_coordinates(), dtype=float).reshape((-1, 3))
    x = coords[:, 0]
    y = coords[:, 1]
    lx = float(geometry.get("lx", max(float(np.max(x)), 1.0)))
    ly = float(geometry.get("ly", max(float(np.max(y) - np.min(y)), 1.0)))

    mat_scale = _make_spatial_field(S, 1.0)
    yield_scale = _make_spatial_field(S, 1.0)

    if family == "cantileverLocalization":
        center = float(geometry.get("localization_band_x", 0.45 * lx))
        width = max(1e-6, 0.12 * lx)
        band = np.exp(-(((x - center) / width) ** 2))
        mat_scale.x.array[:] = 1.0 - 0.28 * band
        mat_scale.x.scatter_forward()
    elif family == "necking":
        center = float(geometry.get("neck_region_center", 0.55 * lx))
        width = max(1e-6, float(geometry.get("neck_region_width", 0.16 * lx)))
        band = np.exp(-(((x - center) / width) ** 2))
        yield_scale.x.array[:] = 1.0 - 0.18 * band
        yield_scale.x.scatter_forward()
    elif family == "yieldHardening":
        center = float(geometry.get("neck_region_center", 0.52 * lx))
        width = max(1e-6, float(geometry.get("neck_region_width", 0.18 * lx)))
        band = np.exp(-(((x - center) / width) ** 2))
        yield_scale.x.array[:] = 1.0 - 0.08 * band
        yield_scale.x.scatter_forward()

    return coords, x, y, mat_scale, yield_scale


def _locate_component_bc(V, component: int, locator, value):
    from dolfinx import fem

    Vc, _ = V.sub(component).collapse()
    dofs = fem.locate_dofs_geometrical((V.sub(component), Vc), locator)
    fun = fem.Function(Vc)
    if np.isscalar(value):
        fun.x.array[:] = float(value)
    else:
        fun.interpolate(value)
    fun.x.scatter_forward()
    return fem.dirichletbc(fun, dofs, V.sub(component)), fun


def _locate_vector_bc(V, locator, value):
    from petsc4py import PETSc
    from dolfinx import fem

    dofs = fem.locate_dofs_geometrical(V, locator)
    if isinstance(value, np.ndarray):
        return fem.dirichletbc(np.asarray(value, dtype=PETSc.ScalarType), dofs, V), None
    fun = fem.Function(V)
    fun.interpolate(value)
    fun.x.scatter_forward()
    return fem.dirichletbc(fun, dofs), fun


def _build_step_bcs(
    *,
    V,
    family: str,
    geometry: dict[str, Any],
    loading: dict[str, Any],
    lam_hist: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    tol: float,
):
    lx = float(geometry.get("lx", x_max - x_min))
    ly = float(geometry.get("ly", y_max - y_min))
    load_factor = float(loading.get("load_factor", 0.04))
    load_sign = float(loading.get("load_sign", 1.0))
    support_span = float(geometry.get("support_span", 0.82 * lx))
    load_width = max(1e-6, float(geometry.get("load_width", 0.08 * lx)))
    patch_center_x = float(geometry.get("patch_center_x", 0.5 * lx))
    patch_half_width = max(1e-6, float(geometry.get("patch_half_width", 0.12 * lx)))
    biaxial_ratio = float(
        loading.get("biaxial_ratio", geometry.get("biaxial_ratio", 1.0))
    )
    center_drop_ratio = float(loading.get("center_drop_ratio", 1.0))
    patch_peak_ratio = float(loading.get("patch_peak_ratio", 1.0))
    twist_strength = float(loading.get("twist_strength", 1.0))
    volumetric_weight = float(loading.get("volumetric_weight", 1.0))

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    pin_tol = max(2.0 * tol, 5.0e-4 * max(lx, ly))

    left = lambda x: np.isclose(x[0], x_min, atol=tol)
    right = lambda x: np.isclose(x[0], x_max, atol=tol)
    top = lambda x: np.isclose(x[1], y_max, atol=tol)
    bottom = lambda x: np.isclose(x[1], y_min, atol=tol)
    left_mid = lambda x: np.isclose(x[0], x_min, atol=tol) & np.isclose(
        x[1], y_mid, atol=pin_tol
    )
    right_mid = lambda x: np.isclose(x[0], x_max, atol=tol) & np.isclose(
        x[1], y_mid, atol=pin_tol
    )
    left_bottom = lambda x: np.isclose(x[0], x_min, atol=tol) & np.isclose(
        x[1], y_min, atol=pin_tol
    )
    bottom_center = lambda x: np.isclose(x[0], x_mid, atol=pin_tol) & np.isclose(
        x[1], y_min, atol=tol
    )

    support_xl = 0.5 * (lx - support_span)
    support_xr = 0.5 * (lx + support_span)
    support_hw = 0.5 * load_width
    left_support = lambda x: np.isclose(x[1], y_min, atol=tol) & (
        np.abs(x[0] - support_xl) <= support_hw
    )
    right_support = lambda x: np.isclose(x[1], y_min, atol=tol) & (
        np.abs(x[0] - support_xr) <= support_hw
    )
    top_mid_patch = lambda x: np.isclose(x[1], y_max, atol=tol) & (
        np.abs(x[0] - x_mid) <= 0.5 * load_width
    )
    top_patch = lambda x: np.isclose(x[1], y_max, atol=tol) & (
        np.abs(x[0] - patch_center_x) <= patch_half_width
    )

    bcs = []
    holders: list[Any] = []
    monitor: dict[str, Any] = {"kind": "none"}

    def add_comp(comp: int, locator, value):
        bc, holder = _locate_component_bc(V, comp, locator, value)
        bcs.append(bc)
        if holder is not None:
            holders.append(holder)

    def add_vec(locator, value):
        bc, holder = _locate_vector_bc(V, locator, value)
        bcs.append(bc)
        if holder is not None:
            holders.append(holder)

    if family in {
        "uniaxialTension",
        "uniaxialCompression",
        "yieldHardening",
        "loadUnloadHysteresis",
        "necking",
        "plateWithHole",
        "neoHookean",
    }:
        ux = load_sign * load_factor * lx * float(lam_hist)
        add_comp(0, left, 0.0)
        add_comp(1, left_bottom, 0.0)
        add_comp(0, right, ux)
        monitor = {
            "kind": "edge",
            "axis": 0,
            "value": x_max,
            "component": 0,
            "span": ly,
        }
    elif family == "cantileverLocalization":
        uy = -abs(load_factor) * ly * float(lam_hist)
        add_vec(left, np.asarray([0.0, 0.0], dtype=float))
        add_comp(1, right, uy)
        monitor = {
            "kind": "edge",
            "axis": 0,
            "value": x_max,
            "component": 1,
            "span": ly,
        }
    elif family == "loadUnloadHysteresis":
        ux = load_sign * load_factor * lx * float(lam_hist)
        add_comp(0, left, 0.0)
        add_comp(1, left_bottom, 0.0)
        add_comp(0, right, ux)
        monitor = {
            "kind": "edge",
            "axis": 0,
            "value": x_max,
            "component": 0,
            "span": ly,
        }
    elif family == "bucklingSnapthrough":
        ux = -abs(load_factor) * lx * float(lam_hist)
        add_comp(0, left, 0.0)
        add_comp(0, right, ux)
        add_comp(1, left_mid, 0.0)
        add_comp(1, right_mid, 0.0)
        monitor = {
            "kind": "edge",
            "axis": 0,
            "value": x_max,
            "component": 0,
            "span": ly,
        }
    elif family == "pureShear":
        gamma = (
            load_sign
            * load_factor
            * float(loading.get("shear_ratio_xy", 1.0))
            * float(lam_hist)
        )
        add_vec(bottom, np.asarray([0.0, 0.0], dtype=float))
        add_comp(0, top, gamma * ly)
        add_comp(1, top, 0.0)
        monitor = {
            "kind": "edge",
            "axis": 1,
            "value": y_max,
            "component": 0,
            "span": lx,
        }
    elif family == "threePointBending":
        uy = -abs(load_factor) * ly * center_drop_ratio * float(lam_hist)
        add_comp(1, left_support, 0.0)
        add_comp(1, right_support, 0.0)
        add_comp(0, left_mid, 0.0)
        add_comp(1, top_mid_patch, uy)
        monitor = {
            "kind": "patch",
            "axis": 1,
            "value": y_max,
            "component": 1,
            "x_center": x_mid,
            "half_width": 0.5 * load_width,
            "span": load_width,
        }
    elif family == "torsionShear":
        theta = load_factor * twist_strength * float(lam_hist)
        add_vec(left, np.asarray([0.0, 0.0], dtype=float))
        add_comp(1, right, 0.0)
        add_comp(0, right, lambda x: -theta * (x[1] - y_mid))
        monitor = {
            "kind": "edge",
            "axis": 0,
            "value": x_max,
            "component": 0,
            "span": ly,
        }
    elif family == "biaxialTension":
        ux = abs(load_factor) * lx * float(lam_hist)
        uy = abs(load_factor) * biaxial_ratio * ly * float(lam_hist)
        add_comp(0, left, 0.0)
        add_comp(0, right, ux)
        add_comp(1, bottom, -0.5 * uy)
        add_comp(1, top, 0.5 * uy)
        monitor = {
            "kind": "biaxial",
            "x_value": x_max,
            "y_value": y_max,
            "ux": ux,
            "uy": uy,
            "x_span": ly,
            "y_span": lx,
        }
    elif family == "contactPatchCompression":
        uy = -abs(load_factor) * patch_peak_ratio * ly * float(lam_hist)
        add_comp(1, bottom, 0.0)
        add_comp(0, bottom_center, 0.0)
        add_comp(1, top_patch, uy)
        monitor = {
            "kind": "patch",
            "axis": 1,
            "value": y_max,
            "component": 1,
            "x_center": patch_center_x,
            "half_width": patch_half_width,
            "span": 2.0 * patch_half_width,
        }
    elif family == "nearIncompressibleBlock":
        ux = abs(load_factor) * lx * float(lam_hist)
        uy = -0.35 * abs(load_factor) * volumetric_weight * ly * float(lam_hist)
        add_comp(0, left, 0.0)
        add_comp(1, bottom, 0.0)
        add_comp(0, right, ux)
        add_comp(1, top, uy)
        monitor = {
            "kind": "biaxial",
            "x_value": x_max,
            "y_value": y_max,
            "ux": ux,
            "uy": uy,
            "x_span": ly,
            "y_span": lx,
        }
    else:
        ux = load_sign * load_factor * lx * float(lam_hist)
        add_comp(0, left, 0.0)
        add_comp(1, left_bottom, 0.0)
        add_comp(0, right, ux)
        monitor = {
            "kind": "edge",
            "axis": 0,
            "value": x_max,
            "component": 0,
            "span": ly,
        }

    return bcs, holders, monitor


def _edge_mask(
    coords: np.ndarray,
    *,
    axis: int,
    value: float,
    tol: float,
    x_center: float | None = None,
    half_width: float | None = None,
) -> np.ndarray:
    mask = np.isclose(coords[:, axis], value, atol=tol)
    if axis == 1 and x_center is not None and half_width is not None:
        mask &= np.abs(coords[:, 0] - x_center) <= half_width + tol
    return mask


def _reaction_monitor(
    monitor: dict[str, Any],
    *,
    coords: np.ndarray,
    sxx: np.ndarray,
    syy: np.ndarray,
    sxy: np.ndarray,
    u_arr: np.ndarray,
    tol: float,
    thickness: float,
) -> tuple[float, float]:
    kind = str(monitor.get("kind", "none"))
    if kind == "edge":
        mask = _edge_mask(
            coords, axis=int(monitor["axis"]), value=float(monitor["value"]), tol=tol
        )
        if not np.any(mask):
            return 0.0, 0.0
        comp = int(monitor.get("component", 0))
        if int(monitor["axis"]) == 0:
            traction = sxx if comp == 0 else sxy
        else:
            traction = sxy if comp == 0 else syy
        rf = float(
            np.mean(traction[mask]) * float(monitor.get("span", 1.0)) * thickness
        )
        disp = float(np.mean(u_arr[mask, comp]))
        return rf, disp
    if kind == "patch":
        mask = _edge_mask(
            coords,
            axis=int(monitor["axis"]),
            value=float(monitor["value"]),
            tol=tol,
            x_center=float(monitor.get("x_center", 0.0)),
            half_width=float(monitor.get("half_width", 0.0)),
        )
        if not np.any(mask):
            return 0.0, 0.0
        comp = int(monitor.get("component", 1))
        traction = sxy if comp == 0 else syy
        rf = float(
            np.mean(traction[mask]) * float(monitor.get("span", 1.0)) * thickness
        )
        disp = float(np.mean(u_arr[mask, comp]))
        return rf, disp
    if kind == "biaxial":
        mask_x = _edge_mask(
            coords, axis=0, value=float(monitor.get("x_value", 0.0)), tol=tol
        )
        mask_y = _edge_mask(
            coords, axis=1, value=float(monitor.get("y_value", 0.0)), tol=tol
        )
        rfx = (
            float(np.mean(sxx[mask_x]) * float(monitor.get("x_span", 1.0)) * thickness)
            if np.any(mask_x)
            else 0.0
        )
        rfy = (
            float(np.mean(syy[mask_y]) * float(monitor.get("y_span", 1.0)) * thickness)
            if np.any(mask_y)
            else 0.0
        )
        disp_x = (
            float(np.mean(u_arr[mask_x, 0]))
            if np.any(mask_x)
            else float(monitor.get("ux", 0.0))
        )
        disp_y = (
            float(np.mean(u_arr[mask_y, 1]))
            if np.any(mask_y)
            else float(monitor.get("uy", 0.0))
        )
        return float(np.hypot(rfx, rfy)), float(np.hypot(disp_x, disp_y))
    return 0.0, 0.0


def _hole_gradient_contrast(
    *, coords: np.ndarray, vm: np.ndarray, geometry: dict[str, Any], char_len: float
) -> float:
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    cx = float(geometry.get("hole_cx", 0.5 * lx))
    cy = float(geometry.get("hole_cy", 0.0))
    r0 = max(1e-9, float(geometry.get("hole_radius", 0.2 * min(lx, ly))))
    dx = coords[:, 0] - cx
    dy = coords[:, 1] - cy
    r = np.sqrt(dx * dx + dy * dy)
    near = (r >= 1.02 * r0) & (r <= 1.45 * r0)
    far_lo = max(2.8 * r0, 0.32 * min(lx, ly))
    far_hi = max(far_lo + 2.0 * char_len, 0.62 * min(lx, ly))
    far = (r >= far_lo) & (r <= far_hi)
    if int(np.count_nonzero(near)) < 8 or int(np.count_nonzero(far)) < 8:
        return 0.0
    num = float(np.quantile(vm[near], 0.95))
    den = float(np.median(vm[far]))
    if den <= 1e-12:
        return 0.0
    return float(num / den)


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
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

    geometry = case.get("geometry") if isinstance(case.get("geometry"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}
    material = case.get("material") if isinstance(case.get("material"), dict) else {}
    nonlinear = case.get("nonlinear") if isinstance(case.get("nonlinear"), dict) else {}
    family = str(
        case.get("family") or case.get("case_prefix") or case_dir.name.split("_", 1)[0]
    )
    model_kind = _model_kind(case)

    domain = _build_domain(case)
    pts_raw = np.asarray(domain.geometry.x, dtype=float)
    points = (
        pts_raw
        if pts_raw.shape[1] == 3
        else np.column_stack([pts_raw, np.zeros((pts_raw.shape[0],), dtype=float)])
    )
    cells = _triangle_cells(domain)
    lx = float(geometry.get("lx", np.max(points[:, 0]) - np.min(points[:, 0])))
    ly = float(geometry.get("ly", np.max(points[:, 1]) - np.min(points[:, 1])))
    thickness = float(geometry.get("thickness", 0.01))
    char_len = max(1e-9, math.sqrt((lx * ly) / max(cells.shape[0], 1)))

    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    tol = 2.5 * char_len

    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    S = fem.functionspace(domain, ("Lagrange", 1))
    coords_s, sx, sy, mat_scale, yield_scale = _family_scalar_fields(
        S=S, family=family, geometry=geometry, material=material
    )

    n_steps = max(8, int(solver.get("n_steps", 24)))
    if model_kind in {"neo_hookean", "geometric_nonlinear"}:
        n_steps = min(max(n_steps, 10), 32)
    hist = _load_history(case, n_steps)
    times = [i / max(1, n_steps - 1) for i in range(n_steps)]

    E, mu, lam_ps = _plane_stress_material(material)
    mu_h = float((nonlinear.get("params") or {}).get("mu", material.get("mu", mu)))
    if model_kind == "geometric_nonlinear":
        mu_h = mu
    kappa_h = float(
        (nonlinear.get("params") or {}).get("kappa", material.get("kappa", 8.0 * mu_h))
    )
    if model_kind == "neo_hookean" and family == "nearIncompressibleBlock":
        kappa_h = min(kappa_h, 80.0 * mu_h)

    u = fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    I = ufl.Identity(2)
    eps = lambda w: ufl.sym(ufl.grad(w))
    zero = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))

    p_field = fem.Function(S)
    p_field.x.array[:] = 0.0
    p_field.x.scatter_forward()
    mu_eff_field = fem.Function(S)
    mu_eff_field.x.array[:] = mu
    mu_eff_field.x.scatter_forward()

    sigma_lin_expr = mat_scale * (2.0 * mu * eps(u) + lam_ps * ufl.tr(eps(u)) * I)
    sigma_lin_du = mat_scale * (2.0 * mu * eps(du) + lam_ps * ufl.tr(eps(du)) * I)
    a_lin = ufl.inner(sigma_lin_du, eps(v)) * ufl.dx
    L_lin = ufl.inner(zero, v) * ufl.dx

    Fh = I + ufl.grad(u)
    Jh = ufl.det(Fh)
    Ch = Fh.T * Fh
    lnJh = ufl.ln(ufl.max_value(Jh, 1.0e-9))
    psi_h = (mu_h / 2.0) * (ufl.tr(Ch) - 2.0 - 2.0 * lnJh) + (kappa_h / 2.0) * (
        lnJh**2
    )
    F_hyper = ufl.derivative(psi_h * ufl.dx, u, v)

    eqv_eps = ufl.sqrt(
        (2.0 / 3.0) * ufl.inner(ufl.dev(eps(u)), ufl.dev(eps(u))) + 1e-16
    )
    sigma_plastic_expr = mat_scale * (
        2.0 * mu_eff_field * eps(u) + lam_ps * ufl.tr(eps(u)) * I
    )
    sigma_plastic_du = mat_scale * (
        2.0 * mu_eff_field * eps(du) + lam_ps * ufl.tr(eps(du)) * I
    )
    a_plastic = ufl.inner(sigma_plastic_du, eps(v)) * ufl.dx

    fe_cfg = solver.get("fe") if isinstance(solver.get("fe"), dict) else {}
    newton_tol = float(fe_cfg.get("newton_tol", 1.0e-8))
    max_newton = max(12, int(fe_cfg.get("max_iters", 28)))
    petsc_lin = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_nonlin = {
        "snes_type": "newtonls",
        "snes_rtol": newton_tol,
        "snes_atol": newton_tol,
        "snes_max_it": max_newton,
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
    plastic_strain_p95 = 0.0
    force_hist: list[float] = []
    disp_hist: list[float] = []
    max_disp_hist: list[float] = []
    max_vm_hist: list[float] = []
    strain_energy_history: list[float] = []
    plastic_strain_history: list[float] = []
    crack_length_history: list[float] = []
    loc_hist: list[float] = []
    snap_hist: list[float] = []
    prev_u3: np.ndarray | None = None
    hole_gradient_contrast = 0.0

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
    beta = float((nonlinear.get("params") or {}).get("plastic_smooth", 48.0))
    eps_y_base = sig_y / max(E, 1e-9)
    hard_eps = hard / max(E, 1e-9)
    yield_strain = float(material.get("yield_strain", eps_y_base))
    yield_field = np.asarray(yield_scale.x.array, dtype=float) * max(
        eps_y_base, yield_strain
    )
    n_stag = max(1, int((nonlinear.get("params") or {}).get("stagger_iters", 2)))

    for i, lam_hist in enumerate(hist):
        bcs_step, holders, monitor = _build_step_bcs(
            V=V,
            family=family,
            geometry=geometry,
            loading=loading,
            lam_hist=float(lam_hist),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tol=tol,
        )
        _ = holders

        if model_kind == "linear":
            u_sol = LinearProblem(
                a_lin,
                L_lin,
                bcs=bcs_step,
                petsc_options_prefix=f"lin2d_{i}_",
                petsc_options=petsc_lin,
            ).solve()
            u.x.array[:] = u_sol.x.array
            u.x.scatter_forward()
            sigma_expr = sigma_lin_expr
            aux_arr = np.zeros_like(sx)
        elif model_kind in {"neo_hookean", "geometric_nonlinear"}:
            nlp = NonlinearProblem(
                F_hyper,
                u,
                bcs=bcs_step,
                petsc_options_prefix=f"neo2d_{i}_",
                petsc_options=petsc_nonlin,
            )
            nlp.solve()
            Fb = I + ufl.grad(u)
            Jb = ufl.det(Fb)
            Cb = Fb.T * Fb
            lnJb = ufl.ln(ufl.max_value(Jb, 1.0e-9))
            bb = Fb * Fb.T
            sigma_expr = (mu_h / ufl.max_value(Jb, 1.0e-9)) * (bb - I) + (
                kappa_h * lnJb / ufl.max_value(Jb, 1.0e-9)
            ) * I
            aux_arr = np.asarray(_interpolate_scalar(Jb, S).x.array, dtype=float)
        elif model_kind == "j2_plasticity":
            for k in range(n_stag):
                u_sol = LinearProblem(
                    a_plastic,
                    L_lin,
                    bcs=bcs_step,
                    petsc_options_prefix=f"j22d_{i}_{k}_",
                    petsc_options=petsc_lin,
                ).solve()
                u.x.array[:] = u_sol.x.array
                u.x.scatter_forward()
                eqv_fun = _interpolate_scalar(eqv_eps, S)
                eqv_arr = np.asarray(eqv_fun.x.array, dtype=float)
                grow = np.maximum(
                    eqv_arr
                    - (
                        yield_field
                        + 0.55 * hard_eps * np.asarray(p_field.x.array, dtype=float)
                    ),
                    0.0,
                )
                if family == "loadUnloadHysteresis":
                    grow *= 0.9
                p_new = np.maximum(
                    np.asarray(p_field.x.array, dtype=float),
                    np.asarray(p_field.x.array, dtype=float) + 0.38 * grow,
                )
                if np.max(np.abs(p_new - p_field.x.array)) < 1e-6:
                    p_field.x.array[:] = p_new
                    p_field.x.scatter_forward()
                    break
                p_field.x.array[:] = p_new
                p_field.x.scatter_forward()
                mu_eff_field.x.array[:] = mu / (1.0 + beta * np.maximum(p_new, 0.0))
                mu_eff_field.x.scatter_forward()
            mu_eff_field.x.array[:] = mu / (
                1.0 + beta * np.maximum(np.asarray(p_field.x.array, dtype=float), 0.0)
            )
            mu_eff_field.x.scatter_forward()
            sigma_expr = sigma_plastic_expr
            aux_arr = np.asarray(p_field.x.array, dtype=float)
            plastic_now = float(np.quantile(aux_arr, 0.95)) if aux_arr.size else 0.0
            plastic_strain_p95 = max(plastic_strain_p95, plastic_now)
        else:
            raise RuntimeError(f"Unsupported 2D FE model kind: {model_kind}")

        exx = np.asarray(_interpolate_scalar(eps(u)[0, 0], S).x.array, dtype=float)
        eyy = np.asarray(_interpolate_scalar(eps(u)[1, 1], S).x.array, dtype=float)
        exy = np.asarray(_interpolate_scalar(eps(u)[0, 1], S).x.array, dtype=float)
        sxx_arr = np.asarray(
            _interpolate_scalar(sigma_expr[0, 0], S).x.array, dtype=float
        )
        syy_arr = np.asarray(
            _interpolate_scalar(sigma_expr[1, 1], S).x.array, dtype=float
        )
        sxy_arr = np.asarray(
            _interpolate_scalar(sigma_expr[0, 1], S).x.array, dtype=float
        )
        s1_arr, _ = _principal_2d(sxx_arr, syy_arr, sxy_arr)
        e1_arr, _ = _principal_2d(exx, eyy, exy)

        vm_arr = np.sqrt(
            np.maximum(
                0.0,
                sxx_arr * sxx_arr
                - sxx_arr * syy_arr
                + syy_arr * syy_arr
                + 3.0 * sxy_arr * sxy_arr,
            )
        )
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
        damage = np.zeros_like(aux_arr)
        crack_mask = np.zeros_like(aux_arr)

        max_disp = max(max_disp, float(np.max(disp_mag)))
        max_vm = max(max_vm, float(np.max(vm_arr)))
        max_disp_hist.append(float(np.max(disp_mag)))
        max_vm_hist.append(float(np.max(vm_arr)))

        rf, disp_mon = _reaction_monitor(
            monitor,
            coords=coords_s,
            sxx=sxx_arr,
            syy=syy_arr,
            sxy=sxy_arr,
            u_arr=u_arr2,
            tol=tol,
            thickness=max(thickness, 1e-9),
        )
        max_rf = max(max_rf, abs(rf))
        force_hist.append(float(rf))
        disp_hist.append(float(disp_mon))

        sigma_energy = 0.5 * (sxx_arr * exx + syy_arr * eyy + 2.0 * sxy_arr * exy)
        step_energy = float(
            np.sum(np.maximum(0.0, sigma_energy))
            * char_len
            * char_len
            * max(thickness, 1e-9)
        )
        strain_energy_total += step_energy
        strain_energy_history.append(float(strain_energy_total))

        loc_hist.append(float(np.max(vm_arr) / max(np.mean(vm_arr), 1e-12)))
        if family == "bucklingSnapthrough":
            mid_band = np.abs(sx - 0.5 * (x_min + x_max)) <= 2.0 * char_len
            snap_hist.append(
                float(
                    np.max(np.abs(u_arr2[mid_band, 1]))
                    if np.any(mid_band)
                    else np.max(np.abs(u_arr2[:, 1]))
                )
            )
        else:
            snap_hist.append(float(np.max(np.abs(u_arr2[:, 1]))))

        if family == "plateWithHole":
            hole_gradient_contrast = max(
                hole_gradient_contrast,
                _hole_gradient_contrast(
                    coords=coords_s, vm=vm_arr, geometry=geometry, char_len=char_len
                ),
            )

        if model_kind == "j2_plasticity":
            plastic_strain_history.append(
                float(np.quantile(np.asarray(p_field.x.array, dtype=float), 0.95))
            )
        else:
            plastic_strain_history.append(0.0)
        crack_length_history.append(0.0)

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
                "aux_field": aux_arr,
                "damage": damage,
                "phase_field": damage,
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

    nonlinear_model = model_kind
    if model_kind == "geometric_nonlinear":
        nonlinear_model = "geometric_nonlinear"

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
        "crack_length_final": 0.0,
        "plastic_strain_p95": float(plastic_strain_p95),
        "strain_energy_total": float(strain_energy_total),
        "hole_gradient_contrast": float(hole_gradient_contrast),
        "load_history": [float(x) for x in hist],
        "max_disp_history": [float(x) for x in max_disp_hist],
        "max_vm_history": [float(x) for x in max_vm_hist],
        "strain_energy_history": [float(x) for x in strain_energy_history],
        "plastic_strain_history": [float(x) for x in plastic_strain_history],
        "crack_length_history": [float(x) for x in crack_length_history],
        "nonlinear_model": nonlinear_model,
    }
