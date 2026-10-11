from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _capsule_track, _case_name, _duration, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_stress_wave(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    L = float(mech.get("length_m", 1.0))
    E = float(mech.get("young_modulus_pa", 2.0e11))
    rho = float(mech.get("density_kg_m3", 7800.0))
    sigma_max = float(mech.get("sigma_max_pa", 1.0e8))
    t_rise = float(mech.get("rise_time_s", 20.0e-6))
    t_fall = float(mech.get("fall_time_s", 40.0e-6))
    c_l = math.sqrt(max(E, 1.0e-9) / max(rho, 1.0e-9))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    x = np.linspace(0.0, L, max(40, int(mech.get("monitor_nodes", 80))))

    def tri(tt: float) -> float:
        if tt < 0.0 or tt > t_fall:
            return 0.0
        if tt <= t_rise:
            return sigma_max * tt / max(t_rise, 1.0e-12)
        return sigma_max * max(0.0, 1.0 - (tt - t_rise) / max(t_fall - t_rise, 1.0e-12))

    node_pos: list[np.ndarray] = []
    node_sigma: list[np.ndarray] = []
    for i, xi in enumerate(x):
        sigma_series: list[float] = []
        u_series: list[float] = []
        for t in times:
            s_inc = tri(float(t - xi / c_l))
            s_ref = tri(float(t - (2.0 * L - xi) / c_l))
            sigma = s_inc - s_ref
            sigma_series.append(abs(float(sigma)) / 1.0e6)
            u_series.append(2.0e-4 * float(sigma / max(E, 1.0e-9)))
        pos = np.column_stack([np.full_like(times, xi) + np.asarray(u_series, dtype=float), np.zeros_like(times), np.full_like(times, 0.22)])
        node_pos.append(pos)
        node_sigma.append(np.asarray(sigma_series, dtype=float))

    objects: list[dict[str, Any]] = []
    for i in range(len(node_pos) - 1):
        pa = node_pos[i]
        pb = node_pos[i + 1]
        sig = 0.5 * (node_sigma[i] + node_sigma[i + 1])
        seg = _capsule_track(
            name=f"wave_elem_{i:03d}",
            color=_rgb(0.24, 0.58, 0.86),
            points_a=pa,
            points_b=pb,
            radius=max(0.010, 0.5 * float(mech.get("diameter_m", 0.02))),
            mass=0.03,
            scalars={"sigma11_mpa": [float(v) for v in sig]},
            extras={"scalar_smoothing": True, "scalar_smooth_group": "stress_wave"},
        )
        seg["color_metric"] = "sigma11_mpa"
        objects.append(seg)

    probe_x = [0.2, 0.5, 0.8]
    rows: list[list[float]] = []
    for t in times:
        vals = [float(t)]
        for xp in probe_x:
            s_inc = tri(float(t - xp / c_l))
            s_ref = tri(float(t - (2.0 * L - xp) / c_l))
            vals.append(float((s_inc - s_ref) / 1.0e6))
        rows.append(vals)
    data = {
        "times": times.tolist(),
        "objects": objects,
        "scene": {
            "view_plane": "xz",
            "ground_value": None,
            "title": _case_name(case, Path(".")),
            "camera_zoom": float(mech.get("camera_zoom", 1.20)),
        },
    }
    _add_csv_export(data, "stress_history.csv", ["time_s", "sigma_x0.2_mpa", "sigma_x0.5_mpa", "sigma_x0.8_mpa"], rows)
    return data
