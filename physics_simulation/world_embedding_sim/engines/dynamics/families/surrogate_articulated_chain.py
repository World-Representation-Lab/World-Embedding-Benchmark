from __future__ import annotations
from run_case import (Any, Path, _box_track, _capsule_track, _case_name, _circle_track, _compute_speed_series, _duration, _gravity, _mech_cfg, _rgb, _scene_bounds, _step_size, _times, math, np)

def _simulate_surrogate_articulated_chain(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    n = max(2, min(8, int(mech.get("link_count", 4))))
    lengths = np.asarray(mech.get("lengths", [0.42] * n), dtype=float).reshape(-1)
    lengths = np.pad(lengths, (0, max(0, n - len(lengths))), constant_values=float(lengths[-1] if len(lengths) else 0.42))[:n]
    lengths = np.maximum(lengths, 0.08)
    initial_deg = np.asarray(mech.get("initial_angles_deg", [38.0] + [18.0 * (-1) ** i for i in range(1, n)]), dtype=float).reshape(-1)
    initial_deg = np.pad(initial_deg, (0, max(0, n - len(initial_deg))), constant_values=0.0)[:n]
    drive = mech.get("base_drive") if isinstance(mech.get("base_drive"), dict) else {}
    drive_amp = math.radians(float(drive.get("amplitude_deg", 0.0)))
    drive_freq = float(drive.get("frequency_hz", 1.0))
    has_drive = abs(drive_amp) > 1.0e-12
    damping = float(mech.get("damping", 0.18))
    phase_lag = math.radians(float(mech.get("phase_lag_deg", 18.0)))
    terminal_mass = float(mech.get("terminal_mass", 0.45))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    g = abs(float(_gravity(case)[2]))
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 1.35]), dtype=float)

    theta_hist = np.zeros((len(times), n), dtype=float)
    joint = np.zeros((len(times), n + 1, 3), dtype=float)
    for i, t in enumerate(times):
        theta_abs = np.zeros(n, dtype=float)
        drive_phase = 2.0 * math.pi * drive_freq * t
        base_angle = drive_amp * math.sin(drive_phase) if has_drive else 0.0
        for j in range(n):
            natural = math.sqrt(g / max(float(np.mean(lengths[: j + 1])), 1.0e-6)) * (0.78 + 0.10 * j)
            transient = math.radians(float(initial_deg[j])) * math.exp(-damping * t * (0.75 + 0.08 * j)) * math.cos(natural * t + 0.35 * j)
            if has_drive:
                driven = drive_amp * (0.82 ** j) * math.sin(drive_phase - phase_lag * j)
                coupling = 0.18 * (theta_abs[j - 1] if j > 0 else base_angle)
                if j == 0:
                    theta_abs[j] = base_angle + 0.12 * transient
                else:
                    theta_abs[j] = transient + driven + coupling
            else:
                coupling = 0.16 * (theta_abs[j - 1] if j > 0 else 0.0)
                theta_abs[j] = transient + coupling
        theta_hist[i] = theta_abs
        joint[i, 0] = pivot
        cur = pivot.copy()
        for j in range(n):
            cur = cur + np.asarray([lengths[j] * math.sin(theta_abs[j]), 0.0, -lengths[j] * math.cos(theta_abs[j])], dtype=float)
            joint[i, j + 1] = cur

    objects: list[dict[str, Any]] = [
        _box_track(
            name="fixed_support",
            size=[0.34, 0.12, 0.08],
            mass=0.0,
            color=_rgb(0.36, 0.38, 0.42),
            positions=np.repeat((pivot + np.asarray([0.0, 0.0, 0.05])).reshape(1, 3), len(times), axis=0),
            extras={"color_mode": "fixed"},
        )
    ]
    if has_drive:
        base_angle_series = drive_amp * np.sin(2.0 * np.pi * drive_freq * times)
        crank_len = float(mech.get("drive_crank_radius", 0.145))
        pivot_series = np.repeat(pivot.reshape(1, 3), len(times), axis=0)
        indicator_center = pivot + np.asarray([0.0, 0.0, 0.18 + 0.25 * crank_len])
        indicator_series = np.repeat(indicator_center.reshape(1, 3), len(times), axis=0)
        crank_tip = indicator_series + np.column_stack(
            [
                crank_len * np.sin(base_angle_series),
                np.zeros_like(base_angle_series),
                crank_len * np.cos(base_angle_series),
            ]
        )
        objects.append(
            _capsule_track(
                name="drive_shaft_to_root",
                color=_rgb(0.18, 0.20, 0.23),
                points_a=pivot_series,
                points_b=indicator_series,
                radius=0.010,
                mass=0.0,
                extras={"color_mode": "fixed"},
            )
        )
        objects.append(
            _circle_track(
                name="drive_hub",
                radius=max(0.055, 0.16 * float(lengths[0])),
                mass=0.0,
                color=_rgb(0.12, 0.14, 0.16),
                positions=indicator_series,
                extras={"color_mode": "fixed"},
            )
        )
        objects.append(
            _capsule_track(
                name="visible_base_drive_arm",
                color=_rgb(0.95, 0.58, 0.16),
                points_a=indicator_series,
                points_b=crank_tip,
                radius=0.015,
                mass=0.0,
                extras={"color_mode": "fixed"},
            )
        )
    for j in range(n):
        speed_scalar = np.linalg.norm(_compute_speed_series(times, joint[:, j + 1, :]), axis=1)
        objects.append(
            _capsule_track(
                name=f"link_{j + 1}",
                color=_rgb(0.22 + 0.06 * (j % 3), 0.48, 0.75 - 0.04 * j),
                points_a=joint[:, j, :],
                points_b=joint[:, j + 1, :],
                radius=max(0.018, 0.035 - 0.0015 * j),
                mass=0.18,
                scalars={"link_speed_m_s": [float(x) for x in speed_scalar]},
                extras={"color_metric": "link_speed_m_s"},
            )
        )
        objects.append(
            _circle_track(
                name=f"joint_{j + 1}",
                radius=max(0.026, 0.038 - 0.001 * j),
                mass=0.04,
                color=_rgb(0.92, 0.50, 0.22),
                positions=joint[:, j + 1, :],
                scalars={"joint_speed_m_s": [float(x) for x in speed_scalar]},
                extras={"color_metric": "joint_speed_m_s"},
            )
        )
    if terminal_mass > 1.0e-9:
        objects.append(
            _circle_track(
                name="terminal_mass",
                radius=float(mech.get("terminal_radius", 0.07)),
                mass=terminal_mass,
                color=_rgb(0.82, 0.26, 0.24),
                positions=joint[:, -1, :],
            )
        )
    scene = {"view_plane": "xz", "ground_value": None, "title": _case_name(case, Path("."))}
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    return {
        "times": times.tolist(),
        "objects": objects,
        "scene": scene,
        "global_series": {
            "chain_tip_x_m": [float(x) for x in joint[:, -1, 0]],
            "chain_tip_z_m": [float(z) for z in joint[:, -1, 2]],
            "base_drive_angle_deg": [float(math.degrees(x)) for x in (drive_amp * np.sin(2.0 * np.pi * drive_freq * times) if has_drive else np.zeros_like(times))],
        },
    }
