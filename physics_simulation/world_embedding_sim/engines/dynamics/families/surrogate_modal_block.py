from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _box_track, _case_name, _duration, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_modal_block(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    a = float(mech.get("length_x_m", 0.5))
    b = float(mech.get("length_y_m", 0.1))
    c = float(mech.get("length_z_m", 0.05))
    nx = max(4, int(mech.get("nx", 10)))
    ny = max(2, int(mech.get("ny", 2)))
    nz = max(2, int(mech.get("nz", 2)))
    mode_table = [(1, 1, 1, 72.0), (2, 1, 1, 118.0), (1, 2, 1, 146.0), (3, 1, 1, 191.0), (2, 2, 1, 232.0), (1, 1, 2, 278.0)]
    mode_id = max(1, min(len(mode_table), int(mech.get("mode_index", 1)))) - 1
    m, n, p, f_mode = mode_table[mode_id]
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    amp = float(mech.get("mode_amplitude_m", 0.003))
    damping = float(mech.get("modal_decay", 0.24))

    xs = np.linspace(0.0, a, nx + 1)
    ys = np.linspace(-0.5 * b, 0.5 * b, ny + 1)
    zs = np.linspace(0.0, c, nz + 1)
    dx = a / max(nx, 1)
    dy = b / max(ny, 1)
    dz = c / max(nz, 1)
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    zc = 0.5 * (zs[:-1] + zs[1:])
    objects: list[dict[str, Any]] = []
    for ix, xv in enumerate(xc):
        for iy, yv in enumerate(yc):
            for iz, zv in enumerate(zc):
                shape = math.sin(m * math.pi * xv / max(a, 1.0e-9)) * math.cos(n * math.pi * yv / max(b, 1.0e-9)) * math.sin(p * math.pi * (zv + 0.5 * c) / max(c, 1.0e-9))
                shape *= (xv / max(a, 1.0e-9))
                disp = amp * float(shape) * np.exp(-damping * times) * np.sin(2.0 * math.pi * f_mode * times)
                pos = np.column_stack([np.full_like(times, xv), np.full_like(times, yv), np.full_like(times, zv) + disp + 0.25])
                disp_series = np.abs(disp)
                objects.append(
                    _box_track(
                        name=f"block_elem_{ix:02d}_{iy:02d}_{iz:02d}",
                        size=[0.92 * dx, 0.92 * dy, 0.92 * dz],
                        mass=0.2,
                        color=_rgb(0.18, 0.62, 0.86),
                        positions=pos,
                        scalars={"displacement_m": [float(v) for v in disp_series]},
                        extras={"color_metric": "displacement_m", "scalar_smoothing": True, "scalar_smooth_group": "modal_block"},
                    )
                )
    data = {"times": times.tolist(), "objects": objects, "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=None)}
    _add_csv_export(data, "modal_frequencies.csv", ["mode", "frequency_hz"], [[float(i + 1), float(item[3])] for i, item in enumerate(mode_table)])
    return data
