from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _capsule_track, _case_name, _duration, _mech_cfg, _rgb, _sample_count, _step_size, math, np)

def _simulate_surrogate_lemaitre_damage(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    L = float(mech.get("length_m", 0.8))
    n_nodes = max(20, int(mech.get("nodes", 61)))
    f = float(mech.get("cycle_frequency_hz", 10.0))
    u0 = float(mech.get("disp_amplitude_m", 5.0e-4))
    start_cycle = float(mech.get("damage_start_cycle", 10.0))
    fail_cycle = max(start_cycle + 1.0, float(mech.get("damage_fail_cycle", 40.0)))
    duration = max(_duration(case), (fail_cycle + 1.25) / max(f, 1.0e-6))
    times = np.linspace(0.0, duration, max(280, _sample_count(case)))
    dt = float(times[1] - times[0]) if len(times) > 1 else _step_size(case)
    x = np.linspace(0.0, L, n_nodes)
    notch_w = float(mech.get("notch_width_m", 0.06))
    z0 = float(mech.get("base_z_m", 0.24))

    node_pos: list[np.ndarray] = []
    node_d: list[np.ndarray] = []
    max_d_rows: list[list[float]] = []
    for i, xi in enumerate(x):
        s = xi / max(L, 1.0e-9)
        gauss = math.exp(-((xi - 0.5 * L) ** 2) / max(2.0 * notch_w * notch_w, 1.0e-12))
        pos = np.zeros((len(times), 3), dtype=float)
        d_series: list[float] = []
        peeq_series: list[float] = []
        for k, t in enumerate(times):
            cyc = t * f
            dmax = max(0.0, min(0.995, (cyc - start_cycle) / max(fail_cycle - start_cycle, 1.0e-9)))
            dval = dmax * gauss
            peeq = 0.14 * dval + 0.01 * abs(math.sin(2.0 * math.pi * f * t))
            u = u0 * math.sin(2.0 * math.pi * f * t) * s * (1.0 - 0.55 * dval)
            pos[k] = [xi + u, 0.0, z0]
            d_series.append(float(dval))
            peeq_series.append(float(peeq))
            if i == n_nodes // 2 and abs((cyc % 1.0) - 0.0) < (dt * f):
                max_d_rows.append([float(cyc), float(dmax)])
        node_pos.append(pos)
        node_d.append(np.asarray(d_series, dtype=float))
    objects: list[dict[str, Any]] = []
    for i in range(len(node_pos) - 1):
        seg = _capsule_track(
            name=f"damage_elem_{i:03d}",
            color=_rgb(0.24, 0.62, 0.86),
            points_a=node_pos[i],
            points_b=node_pos[i + 1],
            radius=max(0.010, 0.010 * L),
            mass=0.03,
            scalars={"D": [float(v) for v in (0.5 * (node_d[i] + node_d[i + 1]))]},
            extras={"scalar_smoothing": True, "scalar_smooth_group": "lemaitre_damage"},
        )
        seg["color_metric"] = "D"
        objects.append(seg)
    data = {"times": times.tolist(), "objects": objects, "scene": {"view_plane": "xz", "ground_value": None, "title": _case_name(case, Path("."))}}
    if max_d_rows:
        _add_csv_export(data, "damage_evolution.csv", ["cycle", "max_D"], max_d_rows)
    return data
