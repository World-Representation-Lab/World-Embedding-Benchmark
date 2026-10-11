from __future__ import annotations
from run_case import (Any, Path, _box_track, _capsule_track, _case_name, _circle_track, _duration, _mech_cfg, _rgb, _scene_bounds, _scene_oblique, _segment_track, _step_size, _times, math, np)

def _simulate_surrogate_unbalanced_rotor(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    shaft_length = float(mech.get("shaft_length", 1.10))
    rotor_radius = float(mech.get("rotor_radius", 0.18))
    mass = max(0.05, float(mech.get("mass", 1.2)))
    critical = max(5.0, float(mech.get("critical_speed_rad_s", 150.0)))
    if "spin_start_ratio" in mech:
        spin_start = critical * abs(float(mech.get("spin_start_ratio", 0.6)))
    else:
        spin_start = abs(float(mech.get("spin_rad_s", mech.get("initial_spin_rad_s", 140.0))))
    if "spin_end_ratio" in mech:
        spin_end = critical * abs(float(mech.get("spin_end_ratio", spin_start / critical)))
    else:
        spin_end = abs(float(mech.get("spin_end_rad_s", mech.get("spin_end", spin_start))))
    zeta = max(0.005, float(mech.get("damping_ratio", 0.055)))
    eccentricity = max(0.0, float(mech.get("eccentricity_m", mech.get("imbalance_offset", 0.025))))
    amp_gain = float(mech.get("whirl_visual_gain", 1.0))
    axis_height = float(mech.get("axis_height", 0.55))
    bearing_radius = max(rotor_radius * 0.68, float(mech.get("bearing_radius", 0.13)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    nominal_center = np.asarray([0.0, 0.0, axis_height], dtype=float)
    left_bearing = np.asarray([-0.5 * shaft_length, 0.0, axis_height], dtype=float)
    right_bearing = np.asarray([0.5 * shaft_length, 0.0, axis_height], dtype=float)
    left = np.repeat(left_bearing.reshape(1, 3), len(times), axis=0)
    right = np.repeat(right_bearing.reshape(1, 3), len(times), axis=0)

    rotor = np.zeros((len(times), 3), dtype=float)
    marker = np.zeros((len(times), 3), dtype=float)
    disp = np.zeros(len(times), dtype=float)
    whirl_speed = np.zeros(len(times), dtype=float)
    spin_series: list[float] = []
    critical_ratio_series: list[float] = []
    bearing_force_series: list[float] = []
    ramp_time = max(float(times[-1]) * float(mech.get("ramp_time_fraction", 0.78)), dt)
    amp_tau = max(2.0 * dt, float(mech.get("amplitude_time_constant_s", 0.16)))
    max_whirl = float(mech.get("max_whirl_radius", 0.18))

    def response_target(ratio: float) -> tuple[float, float]:
        denom = math.sqrt((1.0 - ratio * ratio) ** 2 + (2.0 * zeta * ratio) ** 2)
        amp = eccentricity * ratio * ratio / max(denom, 1.0e-6)
        amp = float(np.clip(amp * amp_gain, 0.004, max_whirl))
        lag = math.atan2(2.0 * zeta * ratio, 1.0 - ratio * ratio)
        return amp, lag

    phase = 0.0
    amp, phase_lag = response_target(spin_start / critical)
    amp *= 0.55
    for i, t in enumerate(times):
        u = min(1.0, max(0.0, float(t) / ramp_time))
        smooth = u * u * (3.0 - 2.0 * u)
        spin_now = spin_start + (spin_end - spin_start) * smooth
        ratio = spin_now / critical
        target_amp, target_lag = response_target(ratio)
        if i > 0:
            local_dt = float(times[i] - times[i - 1])
            phase += spin_now * local_dt
            amp += (target_amp - amp) * min(1.0, local_dt / amp_tau)
        phase_lag = target_lag
        whirl_phase = phase - phase_lag
        offset = np.asarray([0.0, amp * math.cos(whirl_phase), amp * math.sin(whirl_phase)], dtype=float)
        rotor[i] = nominal_center + offset
        disp[i] = float(np.linalg.norm(offset[1:]))
        whirl_speed[i] = abs(amp * spin_now)
        spin_series.append(float(spin_now))
        critical_ratio_series.append(float(ratio))
        bearing_force_series.append(float(mass * spin_now * spin_now * (eccentricity + disp[i])))
        spin_phase = phase
        marker[i] = rotor[i] + np.asarray([0.0, rotor_radius * math.cos(spin_phase), rotor_radius * math.sin(spin_phase)], dtype=float)

    orbit_segments: list[dict[str, Any]] = []
    if bool(mech.get("show_orbit_reference", False)) and bool(mech.get("allow_auxiliary_guides", False)):
        response_amp = float(np.max(disp)) if len(disp) else max_whirl
        orbit_pts = []
        for a in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False):
            orbit_pts.append(nominal_center + np.asarray([0.0, response_amp * math.cos(a), response_amp * math.sin(a)], dtype=float))
        for j, pa0 in enumerate(orbit_pts):
            pb0 = orbit_pts[(j + 1) % len(orbit_pts)]
            seg = _segment_track(
                name=f"whirl_orbit_reference_{j:02d}",
                color=_rgb(0.55, 0.57, 0.60),
                points_a=np.repeat(pa0.reshape(1, 3), len(times), axis=0),
                points_b=np.repeat(pb0.reshape(1, 3), len(times), axis=0),
                width=0.75,
            )
            seg.update({"color_mode": "fixed", "opacity": 0.55})
            orbit_segments.append(seg)

    base_z = axis_height - bearing_radius - 0.12
    left_shaft_disp = [[0.0, float(x)] for x in disp]
    right_shaft_disp = [[float(x), 0.0] for x in disp]
    bearing_force_pair = [[0.0, float(x)] for x in bearing_force_series]
    bearing_force_pair_rev = [[float(x), 0.0] for x in bearing_force_series]
    objects: list[dict[str, Any]] = orbit_segments + [
        _box_track(
            name="left_bearing_pedestal",
            size=[0.16, 0.18, 0.24],
            mass=0.0,
            color=_rgb(0.38, 0.40, 0.43),
            positions=np.repeat(np.asarray([[left_bearing[0], 0.0, base_z]], dtype=float), len(times), axis=0),
            extras={"color_mode": "fixed"},
        ),
        _box_track(
            name="right_bearing_pedestal",
            size=[0.16, 0.18, 0.24],
            mass=0.0,
            color=_rgb(0.38, 0.40, 0.43),
            positions=np.repeat(np.asarray([[right_bearing[0], 0.0, base_z]], dtype=float), len(times), axis=0),
            extras={"color_mode": "fixed"},
        ),
        _circle_track(
            name="left_bearing_ring",
            radius=bearing_radius,
            mass=0.0,
            color=_rgb(0.50, 0.53, 0.57),
            positions=left,
            scalars={"visible": [1.0 for _ in times]},
            extras={"color_mode": "fixed"},
        ),
        _circle_track(
            name="right_bearing_ring",
            radius=bearing_radius,
            mass=0.0,
            color=_rgb(0.50, 0.53, 0.57),
            positions=right,
            scalars={"visible": [1.0 for _ in times]},
            extras={"color_mode": "fixed"},
        ),
        _capsule_track(
            name="left_shaft_span",
            color=_rgb(0.70, 0.73, 0.78),
            points_a=left,
            points_b=rotor,
            radius=max(0.014, 0.09 * rotor_radius),
            scalars={"lateral_displacement_m": left_shaft_disp, "bearing_force_n": bearing_force_pair},
            extras={"color_metric": "lateral_displacement_m"},
        ),
        _capsule_track(
            name="right_shaft_span",
            color=_rgb(0.70, 0.73, 0.78),
            points_a=rotor,
            points_b=right,
            radius=max(0.014, 0.09 * rotor_radius),
            scalars={"lateral_displacement_m": right_shaft_disp, "bearing_force_n": bearing_force_pair_rev},
            extras={"color_metric": "lateral_displacement_m"},
        ),
        _circle_track(
            name="rotor_disk",
            radius=rotor_radius,
            mass=mass,
            color=_rgb(0.86, 0.47, 0.18),
            positions=rotor,
            scalars={
                "lateral_displacement_m": [float(x) for x in disp],
                "whirl_speed_m_s": [float(x) for x in whirl_speed],
                "spin_rad_s": [float(x) for x in spin_series],
                "critical_ratio": [float(x) for x in critical_ratio_series],
                "bearing_force_n": [float(x) for x in bearing_force_series],
            },
            extras={"color_metric": "lateral_displacement_m", "surface_scalar_profile": "rotor_eccentric"},
        ),
        _circle_track(
            name="eccentric_mass_marker",
            radius=max(0.030, 0.20 * rotor_radius),
            mass=0.08 * mass,
            color=_rgb(0.10, 0.13, 0.17),
            positions=marker,
            scalars={"lateral_displacement_m": [float(x) for x in disp], "critical_ratio": [float(x) for x in critical_ratio_series]},
            extras={"color_metric": "lateral_displacement_m", "draw_on_top": True},
        ),
    ]
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=None)
    scene["case_description"] = "unbalanced rotor speed ramp showing critical-speed whirl amplification"
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 34.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 21.0))
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    return {
        "times": times.tolist(),
        "objects": objects,
        "global_series": {
            "whirl_radius_m": [float(x) for x in disp],
            "spin_rad_s": [float(x) for x in spin_series],
            "critical_ratio": [float(x) for x in critical_ratio_series],
            "bearing_force_n": [float(x) for x in bearing_force_series],
        },
        "scene": scene,
    }
