from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _box_track, _case_name, _duration, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_reissner_shell_modal(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    a = float(mech.get("a_m", 0.5))
    b = float(mech.get("b_m", 0.5))
    t_plate = float(mech.get("thickness_m", 0.01))
    E = float(mech.get("young_modulus_pa", 2.0e11))
    nu = float(mech.get("poisson", 0.3))
    rho = float(mech.get("density_kg_m3", 7800.0))
    nx = max(8, int(mech.get("nx", 16)))
    ny = max(8, int(mech.get("ny", 16)))
    m = max(1, int(mech.get("mode_m", 1)))
    n = max(1, int(mech.get("mode_n", 1)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    D = E * (t_plate**3) / max(12.0 * (1.0 - nu * nu), 1.0e-9)
    f11 = (math.pi / 2.0) * math.sqrt(max(D, 1.0e-12) / max(rho * t_plate, 1.0e-12)) * (((1.0 / a) ** 2) + ((1.0 / b) ** 2))
    f_mode = f11 * math.sqrt(((m / a) ** 2 + (n / b) ** 2) / (((1.0 / a) ** 2) + ((1.0 / b) ** 2)))
    amp = float(mech.get("mode_amplitude_m", 0.006))

    xs = np.linspace(0.0, a, nx + 1)
    ys = np.linspace(0.0, b, ny + 1)
    dx = a / max(nx, 1)
    dy = b / max(ny, 1)
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    objects: list[dict[str, Any]] = []
    for ix, xv in enumerate(xc):
        for iy, yv in enumerate(yc):
            shape = math.sin(m * math.pi * xv / max(a, 1.0e-9)) * math.sin(n * math.pi * yv / max(b, 1.0e-9))
            uz = amp * float(shape) * np.sin(2.0 * math.pi * f_mode * times) * np.exp(-0.12 * times)
            pos = np.column_stack([xv - 0.5 * a + np.zeros_like(times), yv - 0.5 * b + np.zeros_like(times), 0.20 + uz])
            objects.append(
                _box_track(
                    name=f"shell_elem_{ix:02d}_{iy:02d}",
                    size=[0.93 * dx, 0.93 * dy, max(0.0018, 0.75 * t_plate)],
                    mass=0.01,
                    color=_rgb(0.24, 0.70, 0.84),
                    positions=pos,
                    scalars={"Uz": [float(v) for v in uz]},
                    extras={"color_metric": "Uz", "scalar_smoothing": True, "scalar_smooth_group": "reissner_shell"},
                )
            )
    data = {"times": times.tolist(), "objects": objects, "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=None)}
    mode_list = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
    rows = []
    for im, inn in mode_list:
        fmn = f11 * math.sqrt(((im / a) ** 2 + (inn / b) ** 2) / (((1.0 / a) ** 2) + ((1.0 / b) ** 2)))
        rows.append([float(im), float(inn), float(fmn)])
    _add_csv_export(data, "shell_eigenmodes.csv", ["m", "n", "frequency_hz"], rows)
    return data
