from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _cantilever_shape, _capsule_track, _case_name, _duration, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_harmonic_frf(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    f_start = float(mech.get("f_start_hz", 10.0))
    f_end = float(mech.get("f_end_hz", 500.0))
    f_step = float(mech.get("f_step_hz", 5.0))
    f1 = float(mech.get("f1_hz", 82.0))
    F0 = float(mech.get("force_n", 10.0))
    beta_r = float(mech.get("rayleigh_beta", 1.0e-5))
    drive_hz = float(mech.get("drive_frequency_hz", f1))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    z0 = float(mech.get("base_z_m", 0.35))
    L = float(mech.get("length_m", 0.5))
    x = np.linspace(0.0, L, 17)
    s = x / max(L, 1.0e-9)
    phi = _cantilever_shape(s)

    k = 1.0e5
    wn = 2.0 * math.pi * f1
    m_eff = k / max(wn * wn, 1.0e-9)
    c_eff = beta_r * k
    sweep = np.arange(f_start, f_end + 0.5 * f_step, f_step)
    amps: list[float] = []
    for f in sweep:
        w = 2.0 * math.pi * float(f)
        denom = math.sqrt((k - m_eff * w * w) ** 2 + (c_eff * w) ** 2)
        amps.append(float((F0 / max(denom, 1.0e-9)) * 25.0))

    w_drive = 2.0 * math.pi * drive_hz
    denom_drive = math.sqrt((k - m_eff * w_drive * w_drive) ** 2 + (c_eff * w_drive) ** 2)
    amp_drive = min(float((F0 / max(denom_drive, 1.0e-9)) * 25.0), 0.04 * L)
    trans = 1.0 - np.exp(-2.2 * times)
    a_t = amp_drive * trans * np.sin(w_drive * times)
    physical_peak = max(float(np.max(np.abs(a_t))), 1.0e-12)
    target_visual_peak = float(mech.get("visual_tip_amplitude_ratio", 0.12)) * L
    visual_scale = min(120.0, max(1.0, target_visual_peak / physical_peak))
    a_visual = a_t * visual_scale

    node_pos: list[np.ndarray] = []
    node_u: list[np.ndarray] = []
    for xi, ph in zip(x, phi):
        node_pos.append(np.column_stack([np.full_like(times, xi), np.zeros_like(times), z0 + a_visual * float(ph)]))
        node_u.append(np.abs(a_t * float(ph)))
    objects: list[dict[str, Any]] = []
    for i in range(len(node_pos) - 1):
        pa = node_pos[i]
        pb = node_pos[i + 1]
        u_seg = 0.5 * (node_u[i] + node_u[i + 1])
        seg = _capsule_track(
            name=f"frf_elem_{i:02d}",
            color=_rgb(0.22, 0.72, 0.86),
            points_a=pa,
            points_b=pb,
            radius=0.0078,
            mass=0.08,
            scalars={"u_mag": [float(v) for v in u_seg]},
            extras={"scalar_smoothing": True, "scalar_smooth_group": "harmonic_frf"},
        )
        seg["color_metric"] = "u_mag"
        objects.append(seg)

    data = {
        "times": times.tolist(),
        "objects": objects,
        "scene": {
            "view_plane": "xz",
            "ground_value": None,
            "title": _case_name(case, Path(".")),
            "camera_zoom": float(mech.get("camera_zoom", 1.55)),
            "reserve_scalar_bar": True,
        },
    }
    _add_csv_export(data, "frf.csv", ["frequency_hz", "tip_amplitude_m"], [[float(f), float(a)] for f, a in zip(sweep, amps)])
    if amps:
        idx_max = int(np.argmax(np.asarray(amps, dtype=float)))
        _add_csv_export(data, "resonance_point.csv", ["frequency_hz", "tip_amplitude_m"], [[float(sweep[idx_max]), float(amps[idx_max])]])
    data["visualization"] = {"deformation_scale": float(visual_scale), "physical_scalar": "u_mag"}
    return data
