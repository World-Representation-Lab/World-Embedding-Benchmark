from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _box_track, _cantilever_shape, _capsule_track, _case_name, _duration, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_cantilever_beam_transient(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    L = float(mech.get("length_m", 1.0))
    b = float(mech.get("width_m", 0.02))
    h = float(mech.get("height_m", 0.02))
    E = float(mech.get("young_modulus_pa", 2.0e11))
    rho = float(mech.get("density_kg_m3", 7800.0))
    F = float(mech.get("tip_load_n", 100.0))
    t_remove = float(mech.get("load_remove_time_s", 0.5))
    n_elem = max(8, int(mech.get("elements", 20)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    x = np.linspace(0.0, L, n_elem + 1)
    s = x / max(L, 1.0e-9)
    phi = _cantilever_shape(s)
    I = b * (h**3) / 12.0
    A = b * h
    delta_static = F * (L**3) / max(3.0 * E * I, 1.0e-9)
    f1 = ((1.875**2) / (2.0 * math.pi)) * math.sqrt(max(E * I, 1.0e-12) / max(rho * A * (L**4), 1.0e-12))

    amp = np.zeros_like(times)
    for i, t in enumerate(times):
        if t <= t_remove:
            amp[i] = delta_static * (1.0 - math.exp(-12.0 * t))
        else:
            tau = t - t_remove
            amp[i] = delta_static * math.exp(-1.6 * tau) * math.cos(2.0 * math.pi * f1 * tau)

    physical_peak = max(float(np.max(np.abs(amp))), 1.0e-12)
    target_visual_peak = float(mech.get("visual_tip_amplitude_ratio", 0.10)) * L
    visual_scale = min(250.0, max(1.0, target_visual_peak / physical_peak))
    amp_visual = amp * visual_scale
    z0 = float(mech.get("base_z_m", 0.30))
    mass_node = rho * A * L / max(n_elem, 1)
    node_pos: list[np.ndarray] = []
    node_u: list[np.ndarray] = []
    for xi, ph in zip(x, phi):
        z = z0 + amp_visual * float(ph)
        node_pos.append(np.column_stack([np.full_like(times, xi), np.zeros_like(times), z]))
        node_u.append(np.abs(amp * float(ph)))

    objects: list[dict[str, Any]] = []
    for i in range(len(node_pos) - 1):
        pa = node_pos[i]
        pb = node_pos[i + 1]
        u_seg = 0.5 * (node_u[i] + node_u[i + 1])
        seg = _capsule_track(
            name=f"beam_elem_{i:02d}",
            color=_rgb(0.24, 0.66, 0.86),
            points_a=pa,
            points_b=pb,
            radius=max(0.0080, 0.55 * h),
            mass=float(mass_node),
            scalars={"u_mag": [float(v) for v in u_seg]},
            extras={"scalar_smoothing": True, "scalar_smooth_group": "cantilever"},
        )
        seg["color_metric"] = "u_mag"
        objects.append(seg)
    clamp_pos = np.repeat(np.asarray([[0.0, 0.0, z0]], dtype=float), len(times), axis=0)
    objects.append(
        _box_track(
            name="beam_clamp",
            size=[0.03, 0.03, 0.10],
            mass=0.0,
            color=_rgb(0.42, 0.44, 0.50),
            positions=clamp_pos,
            extras={"color_mode": "fixed"},
        )
    )

    data = {
        "times": times.tolist(),
        "objects": objects,
        "scene": {
            "view_plane": "xz",
            "ground_value": None,
            "title": _case_name(case, Path(".")),
            "camera_zoom": float(mech.get("camera_zoom", 1.35)),
            "reserve_scalar_bar": True,
        },
    }
    _add_csv_export(
        data,
        "tip_displacement.csv",
        ["time_s", "tip_displacement_m"],
        [[float(t), float(amp[k])] for k, t in enumerate(times)],
    )
    _add_csv_export(data, "beam_modal_validation.csv", ["mode_index", "f_hz"], [[1.0, float(f1)]])
    data["visualization"] = {"deformation_scale": float(visual_scale), "physical_scalar": "u_mag"}
    return data
