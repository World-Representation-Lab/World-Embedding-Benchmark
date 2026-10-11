"""Shared loading histories, finite-element field helpers, and VTK output."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _resample_path(vals: list[float], n_steps: int) -> list[float]:
    if not vals:
        if n_steps <= 1:
            return [1.0]
        return [i / float(n_steps - 1) for i in range(n_steps)]
    if len(vals) == n_steps:
        return [float(x) for x in vals]
    src_x = np.linspace(0.0, 1.0, len(vals))
    dst_x = np.linspace(0.0, 1.0, n_steps)
    out = np.interp(dst_x, src_x, np.asarray(vals, dtype=float))
    return [float(x) for x in out]


def _load_history(case: dict[str, Any], n_steps: int) -> list[float]:
    loading = case.get("loading") if isinstance(case.get("loading"), dict) else {}
    raw = loading.get("load_path")
    vals = [float(x) for x in raw] if isinstance(raw, list) and raw else []
    vals = _resample_path(vals, n_steps)
    return [max(0.0, min(1.0, float(x))) for x in vals]


def _write_vtu_tri(
    path: Path, points: np.ndarray, cells: np.ndarray, point_data: dict[str, np.ndarray]
) -> None:
    n_points = int(points.shape[0])
    n_cells = int(cells.shape[0])
    if n_cells <= 0:
        raise RuntimeError(f"Refusing to write point-only VTU (zero cells): {path}")
    conn = cells.reshape(-1)
    offsets = np.arange(1, n_cells + 1, dtype=np.int64) * 3
    types = np.full((n_cells,), 5, dtype=np.uint8)  # VTK_TRIANGLE

    def _fmt_float_array(arr: np.ndarray) -> str:
        if arr.ndim == 1:
            return " ".join(f"{float(x):.9g}" for x in arr)
        if arr.ndim == 2:
            return " ".join(" ".join(f"{float(v):.9g}" for v in row) for row in arr)
        raise ValueError("Unsupported array shape")

    def _fmt_int_array(arr: np.ndarray) -> str:
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
                f'<DataArray type="Float64" Name="{name}" format="ascii">{_fmt_float_array(arr)}</DataArray>'
            )
        elif arr.ndim == 2 and arr.shape[1] in {2, 3}:
            point_data_xml.append(
                f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{int(arr.shape[1])}" format="ascii">'
                f"{_fmt_float_array(arr)}"
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
        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">{_fmt_float_array(points)}</DataArray>
      </Points>
      <Cells>
        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">{_fmt_int_array(conn)}</DataArray>
        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">{_fmt_int_array(offsets)}</DataArray>
        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">{_fmt_int_array(types)}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    path.write_text(xml)


def _write_pvd(path: Path, times: list[float], files: list[str]) -> None:
    rows = [
        f'    <DataSet timestep="{float(t):.9g}" group="" part="0" file="{f}"/>'
        for t, f in zip(times, files)
    ]
    txt = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n" + "\n".join(rows) + "\n  </Collection>\n"
        "</VTKFile>\n"
    )
    path.write_text(txt)


def _interpolate_scalar(expr, S):
    from dolfinx import fem

    fun = fem.Function(S)
    ip = S.element.interpolation_points
    pts = ip() if callable(ip) else ip
    fun.interpolate(fem.Expression(expr, pts))
    return fun


def _principal_2d(
    xx: np.ndarray, yy: np.ndarray, xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = 0.5 * (xx + yy)
    rad = np.sqrt(np.maximum(0.0, 0.25 * (xx - yy) * (xx - yy) + xy * xy))
    return mean + rad, mean - rad
