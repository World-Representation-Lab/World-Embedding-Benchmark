from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _box_track, _cantilever_shape, _capsule_track, _case_name, _duration, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_j2_plastic_beam(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    L = float(mech.get("length_m", 1.0))
    n_elem = max(8, int(mech.get("elements", 20)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    x = np.linspace(0.0, L, n_elem + 1)
    s = x / max(L, 1.0e-9)
    phi = _cantilever_shape(s)
    z0 = float(mech.get("base_z_m", 0.28))

    a_max = float(mech.get("max_tip_deflection_m", 0.020))
    a_res = float(mech.get("residual_tip_deflection_m", 0.006))
    peeq_max = float(mech.get("max_peeq", 0.12))
    amp = np.zeros_like(times)
    peeq_t = np.zeros_like(times)
    for i, t in enumerate(times):
        if t <= 0.1:
            amp[i] = a_max * (t / 0.1)
            peeq_t[i] = peeq_max * 0.55 * (t / 0.1)
        elif t <= 0.5:
            amp[i] = a_max
            peeq_t[i] = peeq_max * (0.55 + 0.45 * (t - 0.1) / 0.4)
        else:
            tau = t - 0.5
            amp[i] = a_res + (a_max - a_res) * math.exp(-5.0 * tau)
            peeq_t[i] = peeq_max

    node_pos: list[np.ndarray] = []
    node_peeq: list[np.ndarray] = []
    for i, (xi, ph) in enumerate(zip(x, phi)):
        z = z0 + amp * float(ph)
        peeq = (1.0 - float(s[i])) ** 2 * peeq_t
        pos = np.column_stack([np.full_like(times, xi), np.zeros_like(times), z])
        node_pos.append(pos)
        node_peeq.append(np.asarray(peeq, dtype=float))
    objects: list[dict[str, Any]] = []
    for i in range(len(node_pos) - 1):
        pa = node_pos[i]
        pb = node_pos[i + 1]
        peeq_seg = 0.5 * (node_peeq[i] + node_peeq[i + 1])
        seg = _capsule_track(
            name=f"plastic_elem_{i:02d}",
            color=_rgb(0.28, 0.66, 0.86),
            points_a=pa,
            points_b=pb,
            radius=0.0084,
            mass=0.06,
            scalars={"PEEQ": [float(v) for v in peeq_seg]},
            extras={"scalar_smoothing": True, "scalar_smooth_group": "j2_plastic"},
        )
        seg["color_metric"] = "PEEQ"
        objects.append(seg)
    clamp_pos = np.repeat(np.asarray([[0.0, 0.0, z0]], dtype=float), len(times), axis=0)
    objects.append(_box_track(name="plastic_clamp", size=[0.03, 0.03, 0.10], mass=0.0, color=_rgb(0.42, 0.44, 0.50), positions=clamp_pos, extras={"color_mode": "fixed"}))
    data = {"times": times.tolist(), "objects": objects, "scene": {"view_plane": "xz", "ground_value": None, "title": _case_name(case, Path("."))}}
    _add_csv_export(data, "peeq_tip_history.csv", ["time_s", "tip_displacement_m", "tip_peeq"], [[float(t), float(node_pos[-1][k, 2] - z0), float(node_peeq[-1][k])] for k, t in enumerate(times)])
    return data
