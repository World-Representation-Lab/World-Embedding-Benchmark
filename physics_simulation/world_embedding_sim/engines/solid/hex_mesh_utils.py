from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# DOLFINx uses a lexicographic hexahedron vertex order; VTK expects the quad
# rings to be [0,1,2,3] and [4,5,6,7]. This permutation is its own inverse.
DOLFINX_HEX_TO_VTK = np.asarray([0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int64)
VTK_HEX_FACES = np.asarray(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ],
    dtype=np.int64,
)


def dolfinx_hex_to_vtk(cells: np.ndarray) -> np.ndarray:
    arr = np.asarray(cells, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"Expected (n, 8) hexahedra, got {arr.shape}")
    return arr[:, DOLFINX_HEX_TO_VTK]


def _quad_order_score(face_points: np.ndarray) -> np.ndarray:
    t0 = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
    t1 = np.cross(face_points[:, 2] - face_points[:, 0], face_points[:, 3] - face_points[:, 0])
    n0 = np.linalg.norm(t0, axis=1)
    n1 = np.linalg.norm(t1, axis=1)
    good = (n0 > 1e-14) & (n1 > 1e-14)
    score = np.full((face_points.shape[0],), -1.0, dtype=float)
    if np.any(good):
        score[good] = np.sum(t0[good] * t1[good], axis=1) / (n0[good] * n1[good])
    return score


def _hex_order_score(points: np.ndarray, cells: np.ndarray, sample_size: int = 128) -> float:
    if cells.size == 0:
        return 0.0
    n_cells = int(cells.shape[0])
    if n_cells > sample_size:
        sample_idx = np.linspace(0, n_cells - 1, sample_size, dtype=np.int64)
        cells = cells[sample_idx]
    face_points = points[cells[:, VTK_HEX_FACES].reshape((-1, 4))]
    return float(np.mean(_quad_order_score(face_points)))


def infer_hex_order(points: np.ndarray, cells: np.ndarray) -> str:
    arr = np.asarray(cells, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 8 or arr.size == 0:
        return "vtk"
    vtk_score = _hex_order_score(points, arr)
    dolfinx_score = _hex_order_score(points, dolfinx_hex_to_vtk(arr))
    return "dolfinx" if dolfinx_score > vtk_score + 1e-6 else "vtk"


def ensure_vtk_hex_order(points: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, str]:
    order = infer_hex_order(points, cells)
    if order == "dolfinx":
        return dolfinx_hex_to_vtk(cells), order
    return np.asarray(cells, dtype=np.int64), order


def _parse_vtu_hexa(path: Path) -> tuple[ET.ElementTree, ET.Element, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = ET.parse(path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise RuntimeError(f"Invalid VTU file: {path}")

    pts_da = piece.find("./Points/DataArray")
    conn_da = piece.find("./Cells/DataArray[@Name='connectivity']")
    off_da = piece.find("./Cells/DataArray[@Name='offsets']")
    typ_da = piece.find("./Cells/DataArray[@Name='types']")
    if pts_da is None or conn_da is None or off_da is None or typ_da is None:
        raise RuntimeError(f"Missing VTU arrays in {path}")

    points = np.fromstring(pts_da.text or "", sep=" ", dtype=float).reshape((-1, 3))
    conn = np.fromstring(conn_da.text or "", sep=" ", dtype=np.int64)
    offsets = np.fromstring(off_da.text or "", sep=" ", dtype=np.int64)
    types = np.fromstring(typ_da.text or "", sep=" ", dtype=np.int64)
    return tree, piece, points, conn, offsets, types


def vtu_has_dolfinx_hex_order(path: Path) -> bool:
    _, _, points, conn, offsets, types = _parse_vtu_hexa(path)
    cells: list[np.ndarray] = []
    start = 0
    for i, end in enumerate(offsets):
        stop = int(end)
        if int(types[i]) == 12:
            cell = conn[start:stop]
            if cell.size == 8:
                cells.append(cell.astype(np.int64))
        start = stop
    if not cells:
        return False
    return infer_hex_order(points, np.asarray(cells, dtype=np.int64)) == "dolfinx"


def rewrite_vtu_with_vtk_hex_cells(src: Path, dst: Path) -> bool:
    tree, piece, points, conn, offsets, types = _parse_vtu_hexa(src)
    conn_da = piece.find("./Cells/DataArray[@Name='connectivity']")
    if conn_da is None:
        raise RuntimeError(f"Missing connectivity array in {src}")

    cells: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    start = 0
    for i, end in enumerate(offsets):
        stop = int(end)
        if int(types[i]) == 12:
            cell = conn[start:stop]
            if cell.size == 8:
                cells.append(cell.astype(np.int64))
                slices.append((start, stop))
        start = stop

    changed = False
    if cells:
        cells_arr = np.asarray(cells, dtype=np.int64)
        fixed_cells, order = ensure_vtk_hex_order(points, cells_arr)
        changed = order == "dolfinx"
        if changed:
            new_conn = conn.copy()
            for fixed, (start, stop) in zip(fixed_cells, slices):
                new_conn[start:stop] = fixed
            conn_da.text = " ".join(str(int(x)) for x in new_conn)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if changed:
        tree.write(dst, encoding="unicode")
    elif src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return changed
