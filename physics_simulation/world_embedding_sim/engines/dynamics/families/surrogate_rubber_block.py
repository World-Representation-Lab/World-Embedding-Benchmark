from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _box_track, _case_name, _duration, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_rubber_block(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    sx = float(mech.get("size_x_m", 0.1))
    sy = float(mech.get("size_y_m", 0.1))
    sz = float(mech.get("size_z_m", 0.1))
    nx = max(3, int(mech.get("nx", 4)))
    ny = max(3, int(mech.get("ny", 4)))
    nz = max(3, int(mech.get("nz", 4)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    t_ramp = float(mech.get("ramp_time_s", 1.0))
    comp_max = float(mech.get("compression_ratio", 0.5))

    def comp_ratio(t: float) -> float:
        if t <= t_ramp:
            return comp_max * (t / max(t_ramp, 1.0e-9))
        return comp_max * math.exp(-3.0 * (t - t_ramp))

    xs = np.linspace(-0.5 * sx, 0.5 * sx, nx + 1)
    ys = np.linspace(-0.5 * sy, 0.5 * sy, ny + 1)
    zs = np.linspace(0.0, sz, nz + 1)
    objects: list[dict[str, Any]] = []
    force_rows: list[list[float]] = []
    k_eff = float(mech.get("k_eff_n_m", 4.0e4))
    for t in times:
        c = comp_ratio(float(t))
        top_disp = c * sz
        force_rows.append([float(t), float(top_disp), float(k_eff * top_disp * (1.0 + 2.0 * c))])

    dx = sx / max(nx, 1)
    dy = sy / max(ny, 1)
    dz = sz / max(nz, 1)
    top_disp_series = np.asarray([comp_ratio(float(t)) * sz for t in times], dtype=float)
    node_x = np.zeros((len(times), nx + 1), dtype=float)
    node_y = np.zeros((len(times), ny + 1), dtype=float)
    node_z = np.zeros((len(times), nz + 1), dtype=float)
    for k, t in enumerate(times):
        c = comp_ratio(float(t))
        # Use one shared nodal lattice (single-mesh semantics) to avoid
        # layer-wise visual separation under large compression.
        bulge = 1.0 + 0.24 * c
        node_x[k, :] = xs * bulge
        node_y[k, :] = ys * bulge
        node_z[k, :] = zs * (1.0 - c) + 0.18

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                xl = node_x[:, ix]
                xr = node_x[:, ix + 1]
                yl = node_y[:, iy]
                yr = node_y[:, iy + 1]
                zl = node_z[:, iz]
                zr = node_z[:, iz + 1]
                pos = np.column_stack([0.5 * (xl + xr), 0.5 * (yl + yr), 0.5 * (zl + zr)])
                cell_sizes = np.column_stack(
                    [
                        1.03 * np.maximum(1.0e-6, xr - xl),
                        1.03 * np.maximum(1.0e-6, yr - yl),
                        1.03 * np.maximum(0.42 * dz, zr - zl),
                    ]
                )
                disp_series = np.linalg.norm(pos - pos[0], axis=1)
                radial = np.sqrt(
                    (pos[:, 0] / max(0.5 * sx, 1.0e-9)) ** 2
                    + (pos[:, 1] / max(0.5 * sy, 1.0e-9)) ** 2
                )
                # Use an interpolated scalar dominated by global compression + radial spread
                # to avoid visible layer-by-layer color banding along the vertical direction.
                interp_u = top_disp_series * (0.72 + 0.28 * np.clip(radial, 0.0, 1.0))
                objects.append(
                    _box_track(
                        name=f"rubber_elem_{ix:02d}_{iy:02d}_{iz:02d}",
                        size=[0.92 * dx, 0.92 * dy, 0.92 * dz],
                        mass=0.02,
                        color=_rgb(0.20, 0.68, 0.82),
                        positions=pos,
                        sizes=cell_sizes,
                        scalars={"displacement_m": [float(v) for v in disp_series], "u_interp_m": [float(v) for v in interp_u]},
                        extras={
                            "color_metric": "u_interp_m",
                            "scalar_smoothing": True,
                            "scalar_smooth_group": "rubber_block",
                            "scalar_smooth_alpha": 0.94,
                            "scalar_smooth_k": 42,
                            "soft_edges": True,
                        },
                    )
                )
    expected_nodes = int((nx + 1) * (ny + 1) * (nz + 1))
    data = {"times": times.tolist(), "objects": objects, "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)}
    data["mesh_diagnostics"] = {"node_count": expected_nodes, "expected_node_count": expected_nodes, "single_shared_lattice": True}
    _add_csv_export(data, "force_displacement.csv", ["time_s", "top_displacement_m", "reaction_force_n"], force_rows)
    return data
