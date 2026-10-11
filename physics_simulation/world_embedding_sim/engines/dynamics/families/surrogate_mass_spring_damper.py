from __future__ import annotations
from run_case import (Any, Path, _box_track, _case_name, _duration, _mech_cfg, _rgb, _segment_track, _step_size, _times, math, np)

def _simulate_surrogate_mass_spring_damper(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    n = max(1, min(4, int(mech.get("dof_count", 1))))
    masses = np.asarray(mech.get("masses", [1.0] * n), dtype=float).reshape(-1)
    masses = np.pad(masses, (0, max(0, n - len(masses))), constant_values=float(masses[-1] if len(masses) else 1.0))[:n]
    masses = np.maximum(masses, 1.0e-4)
    wall_k = np.asarray(mech.get("wall_stiffness", [42.0] + [0.0] * max(0, n - 1)), dtype=float).reshape(-1)
    wall_k = np.pad(wall_k, (0, max(0, n - len(wall_k))), constant_values=0.0)[:n]
    wall_c = np.asarray(mech.get("wall_damping", [0.55] + [0.0] * max(0, n - 1)), dtype=float).reshape(-1)
    wall_c = np.pad(wall_c, (0, max(0, n - len(wall_c))), constant_values=0.0)[:n]
    link_k = np.asarray(mech.get("link_stiffness", [28.0] * max(0, n - 1)), dtype=float).reshape(-1)
    link_k = np.pad(link_k, (0, max(0, n - 1 - len(link_k))), constant_values=float(link_k[-1] if len(link_k) else 28.0))[: max(0, n - 1)]
    link_c = np.asarray(mech.get("link_damping", [0.35] * max(0, n - 1)), dtype=float).reshape(-1)
    link_c = np.pad(link_c, (0, max(0, n - 1 - len(link_c))), constant_values=float(link_c[-1] if len(link_c) else 0.35))[: max(0, n - 1)]
    q = np.asarray(mech.get("initial_displacement", [0.28] + [0.0] * max(0, n - 1)), dtype=float).reshape(-1)
    q = np.pad(q, (0, max(0, n - len(q))), constant_values=0.0)[:n]
    v = np.asarray(mech.get("initial_velocity", [0.0] * n), dtype=float).reshape(-1)
    v = np.pad(v, (0, max(0, n - len(v))), constant_values=0.0)[:n]
    forcing = mech.get("forcing") if isinstance(mech.get("forcing"), dict) else {}
    force_amp = float(forcing.get("amplitude_n", 0.0))
    force_freq = float(forcing.get("frequency_hz", 1.0))
    force_target = max(0, min(n - 1, int(forcing.get("target", n - 1))))
    drive_label = str(mech.get("mode", "free_response")).strip() or "free_response"
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    q_hist = np.zeros((len(times), n), dtype=float)
    v_hist = np.zeros((len(times), n), dtype=float)

    def accel(q_in: np.ndarray, v_in: np.ndarray, t_in: float) -> np.ndarray:
        force = -wall_k * q_in - wall_c * v_in
        for j in range(n - 1):
            dq = q_in[j + 1] - q_in[j]
            dv = v_in[j + 1] - v_in[j]
            f = link_k[j] * dq + link_c[j] * dv
            force[j] += f
            force[j + 1] -= f
        if abs(force_amp) > 1.0e-12:
            force[force_target] += force_amp * math.sin(2.0 * math.pi * force_freq * t_in)
        return force / masses

    for i, t in enumerate(times):
        q_hist[i] = q
        v_hist[i] = v
        if i == len(times) - 1:
            break
        # RK4 on the first-order mass-spring-damper state gives stable, visibly
        # credible motion without relying on optional multibody contact backends.
        k1q, k1v = v, accel(q, v, float(t))
        k2q, k2v = v + 0.5 * dt * k1v, accel(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v, float(t) + 0.5 * dt)
        k3q, k3v = v + 0.5 * dt * k2v, accel(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v, float(t) + 0.5 * dt)
        k4q, k4v = v + dt * k3v, accel(q + dt * k3q, v + dt * k3v, float(t) + dt)
        q = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
        v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    spacing = float(mech.get("mass_spacing", 0.72))
    x_base = np.asarray([spacing * i for i in range(n)], dtype=float)
    y_mass = 0.0
    wall_x = -0.52
    objects: list[dict[str, Any]] = []
    mass_size = [0.34, 0.24, 0.14]
    rail_height = 0.06
    track_y = y_mass - 0.5 * mass_size[1] - 0.5 * rail_height
    track_pos = np.repeat(np.asarray([[0.5 * (wall_x + x_base[-1] + 0.7), track_y, -0.015]], dtype=float), len(times), axis=0)
    objects.append(
        _box_track(
            name="guide_rail",
            size=[max(1.0, x_base[-1] - wall_x + 1.0), rail_height, 0.035],
            mass=0.0,
            color=_rgb(0.57, 0.60, 0.64),
            positions=track_pos,
            extras={"color_mode": "fixed", "opacity": 0.72},
        )
    )
    wall_pos = np.repeat(np.asarray([[wall_x - 0.06, y_mass, 0.0]], dtype=float), len(times), axis=0)
    objects.append(
        _box_track(
            name="fixed_wall",
            size=[0.08, 0.52, 0.08],
            mass=0.0,
            color=_rgb(0.39, 0.42, 0.45),
            positions=wall_pos,
            extras={"color_mode": "fixed"},
        )
    )

    mass_centers = []
    for i in range(n):
        pos = np.column_stack([x_base[i] + q_hist[:, i], np.full(len(times), y_mass), np.zeros(len(times))])
        mass_centers.append(pos)
        objects.append(
            _box_track(
                name=f"mass_{i + 1}",
                size=mass_size,
                mass=float(masses[i]),
                color=_rgb(0.18 + 0.12 * (i % 3), 0.48, 0.78 - 0.08 * i),
                positions=pos,
                scalars={"mass_speed_m_s": [float(abs(x)) for x in v_hist[:, i]]},
                extras={"color_metric": "mass_speed_m_s"},
            )
        )

    def add_spring(name: str, left: np.ndarray, right: np.ndarray, y: float, amp: float, turns: int) -> None:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        for seg in range(turns * 2 + 1):
            a = seg / float(turns * 2 + 1)
            b = (seg + 1) / float(turns * 2 + 1)
            pa = (1.0 - a) * left + a * right
            pb = (1.0 - b) * left + b * right
            ya = y + (amp if seg % 2 == 0 else -amp)
            yb = y + (-amp if seg % 2 == 0 else amp)
            pa = pa.copy()
            pb = pb.copy()
            pa[:, 1] = ya
            pb[:, 1] = yb
            spring_seg = _segment_track(name=f"{name}_{seg:02d}", color=_rgb(0.17, 0.18, 0.20), points_a=pa, points_b=pb, width=2.25)
            spring_seg["color_mode"] = "fixed"
            objects.append(spring_seg)

    wall_anchor = np.repeat(np.asarray([[wall_x, y_mass + 0.16, 0.0]], dtype=float), len(times), axis=0)
    mass_left = mass_centers[0].copy()
    mass_left[:, 0] -= 0.18
    mass_left[:, 1] = y_mass + 0.18
    add_spring("spring_wall", wall_anchor, mass_left, y_mass + 0.18, 0.052, 6)
    for i in range(n - 1):
        left = mass_centers[i].copy()
        right = mass_centers[i + 1].copy()
        left[:, 0] += 0.18
        right[:, 0] -= 0.18
        left[:, 1] = y_mass + 0.18
        right[:, 1] = y_mass + 0.18
        add_spring(f"spring_{i + 1}_{i + 2}", left, right, y_mass + 0.18, 0.048, 5)

    response = np.linalg.norm(q_hist, axis=1)
    energy = 0.5 * np.sum(masses.reshape(1, -1) * v_hist * v_hist, axis=1)
    energy += 0.5 * np.sum(wall_k.reshape(1, -1) * q_hist * q_hist, axis=1)
    if n > 1:
        dq = q_hist[:, 1:] - q_hist[:, :-1]
        energy += 0.5 * np.sum(link_k.reshape(1, -1) * dq * dq, axis=1)
    scene = {"view_plane": "xy", "ground_value": None, "title": _case_name(case, Path(".")), "case_description": drive_label}
    x_values = np.concatenate([centers[:, 0] for centers in mass_centers] + [np.asarray([wall_x - 0.10])])
    scene.update(
        {
            "xlim": [float(np.min(x_values) - 0.36), float(np.max(x_values) + 0.36)],
            "ylim": [float(track_y - 0.08), float(y_mass + 0.36)],
            "auto_bounds": False,
        }
    )
    return {
        "times": times.tolist(),
        "objects": objects,
        "global_series": {
            "displacement_norm_m": [float(x) for x in response],
            "mechanical_energy_j": [float(x) for x in energy],
        },
        "scene": scene,
    }
