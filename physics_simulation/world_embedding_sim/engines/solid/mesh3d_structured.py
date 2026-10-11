#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Mesh3D:
    points: np.ndarray  # (N, 3)
    cells: np.ndarray  # (M, 8) hexa connectivity
    cell_centers: np.ndarray  # (M, 3)
    shape_3d: str
    bbox: tuple[float, float, float]
    hole_ratio: float | None = None


def _grid_points(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int) -> np.ndarray:
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(-0.5 * ly, 0.5 * ly, ny + 1)
    zs = np.linspace(-0.5 * lz, 0.5 * lz, nz + 1)
    # Keep x as the fastest-varying index to match _grid_cells connectivity.
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return pts


def _grid_cells(nx: int, ny: int, nz: int) -> np.ndarray:
    stride_x = 1
    stride_y = nx + 1
    stride_z = (nx + 1) * (ny + 1)
    cells: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = i * stride_x + j * stride_y + k * stride_z
                n100 = n000 + stride_x
                n110 = n100 + stride_y
                n010 = n000 + stride_y
                n001 = n000 + stride_z
                n101 = n100 + stride_z
                n111 = n110 + stride_z
                n011 = n010 + stride_z
                cells.append([n000, n100, n110, n010, n001, n101, n111, n011])
    return np.asarray(cells, dtype=np.int64)


def _cell_centers(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return np.mean(points[cells], axis=1)


def _sdf_mask_all(centers: np.ndarray) -> np.ndarray:
    return np.ones((centers.shape[0],), dtype=bool)


def _sdf_mask_cylinder(centers: np.ndarray, geometry: dict) -> np.ndarray:
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    lz = float(geometry.get("lz", 1.0))
    r = float(geometry.get("cylinder_radius", 0.42 * min(ly, lz)))
    cx = float(geometry.get("cylinder_cx", 0.5 * lx))
    cy = float(geometry.get("cylinder_cy", 0.0))
    cz = float(geometry.get("cylinder_cz", 0.0))
    dx = centers[:, 0] - cx
    dy = centers[:, 1] - cy
    dz = centers[:, 2] - cz
    radial = np.sqrt(dy * dy + dz * dz)
    return (radial <= r) & (np.abs(dx) <= 0.5 * lx)


def _sdf_mask_sphere(centers: np.ndarray, geometry: dict) -> np.ndarray:
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    lz = float(geometry.get("lz", 1.0))
    r = float(geometry.get("sphere_radius", 0.45 * min(lx, ly, lz)))
    cx = float(geometry.get("sphere_cx", 0.5 * lx))
    cy = float(geometry.get("sphere_cy", 0.0))
    cz = float(geometry.get("sphere_cz", 0.0))
    d2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2 + (centers[:, 2] - cz) ** 2
    return d2 <= r * r


def _sdf_mask_plate_with_hole(centers: np.ndarray, geometry: dict) -> np.ndarray:
    # Keep plate block and remove through-hole along z direction.
    cx = float(geometry.get("hole_cx", 0.5 * float(geometry.get("lx", 1.0))))
    cy = float(geometry.get("hole_cy", 0.0))
    r = float(geometry.get("hole_radius", 0.2 * min(float(geometry.get("lx", 1.0)), float(geometry.get("ly", 1.0)))))
    dy = centers[:, 1] - cy
    dx = centers[:, 0] - cx
    radial = np.sqrt(dx * dx + dy * dy)
    return radial >= r


def _sdf_mask_notched_bar(centers: np.ndarray, geometry: dict) -> np.ndarray:
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    notch_depth = float(geometry.get("notch_depth_ratio", 0.16)) * ly
    notch_width = float(geometry.get("notch_width_ratio", 0.12)) * lx
    notch_plane = float(geometry.get("notch_plane_x", 0.5 * lx))
    root_radius = float(geometry.get("notch_root_radius", float(geometry.get("notch_root_radius_ratio", 0.04)) * ly))
    root_radius = max(0.0, min(root_radius, 0.48 * notch_depth, 0.48 * notch_width))
    x = centers[:, 0]
    y = centers[:, 1]
    y_abs = np.abs(y)
    y_root = 0.5 * ly - notch_depth
    dx = np.abs(x - notch_plane)
    if root_radius <= 1e-12:
        notch_band = dx <= notch_width
        notch_cut = notch_band & (y_abs >= y_root)
        return ~notch_cut

    # Rounded notch root removes the elastic singular corner present in the
    # sharp rectangular cut while preserving the overall ligament width.
    rect_core = (dx <= max(notch_width - root_radius, 0.0)) & (y_abs >= y_root)
    upper_core = (dx <= notch_width) & (y_abs >= y_root + root_radius)
    cx = max(notch_width - root_radius, 0.0)
    cy = y_root + root_radius
    corner_round = ((dx - cx) ** 2 + (y_abs - cy) ** 2 <= root_radius * root_radius) & (y_abs >= y_root)
    notch_cut = rect_core | upper_core | corner_round
    return ~notch_cut


def _mask_dispatch(shape_3d: str) -> Callable[[np.ndarray, dict], np.ndarray]:
    s = shape_3d.strip().lower()
    if s == "cylinder":
        return _sdf_mask_cylinder
    if s == "sphere":
        return _sdf_mask_sphere
    if s == "plate_with_hole_3d":
        return _sdf_mask_plate_with_hole
    if s == "notched_bar_3d":
        return _sdf_mask_notched_bar
    return lambda centers, _geometry: _sdf_mask_all(centers)


def _extract_active_submesh(points: np.ndarray, cells: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active_cells = cells[active]
    if active_cells.size == 0:
        raise RuntimeError("No active cells after geometry mask; cannot build 3D mesh.")

    used = np.unique(active_cells.reshape(-1))
    remap = np.full((points.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    new_points = points[used]
    new_cells = remap[active_cells]
    return new_points, new_cells


def build_structured_mesh_3d(geometry: dict, mesh: dict) -> Mesh3D:
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 0.5))
    lz = float(geometry.get("lz", 0.5))
    nx = max(8, int(mesh.get("nx", 28)))
    ny = max(8, int(mesh.get("ny", 20)))
    nz = max(8, int(mesh.get("nz", 20)))
    shape_3d = str(geometry.get("shape_3d", "cuboid"))

    points = _grid_points(lx, ly, lz, nx, ny, nz)
    cells = _grid_cells(nx, ny, nz)
    centers = _cell_centers(points, cells)

    mask_fn = _mask_dispatch(shape_3d)
    active = mask_fn(centers, geometry)
    points_sub, cells_sub = _extract_active_submesh(points, cells, active)
    centers_sub = _cell_centers(points_sub, cells_sub)

    hole_ratio = None
    if shape_3d == "plate_with_hole_3d":
        hole_ratio = float(geometry.get("hole_radius", 0.2 * min(lx, ly))) / max(1e-9, min(lx, ly))

    return Mesh3D(
        points=points_sub,
        cells=cells_sub,
        cell_centers=centers_sub,
        shape_3d=shape_3d,
        bbox=(lx, ly, lz),
        hole_ratio=hole_ratio,
    )
