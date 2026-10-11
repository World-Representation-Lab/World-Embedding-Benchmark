from __future__ import annotations
from run_case import (Any, Path, _capsule_track, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _scene_bounds, _step_size, _times, math, np)

def _simulate_surrogate_rolling(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    g = abs(float(_gravity(case)[2]))
    radius = float(mech.get("radius", 0.25))
    width = float(mech.get("width", 0.18))
    mass = float(mech.get("mass", 1.0))
    v = float(mech.get("initial_speed", 3.0))
    omega = float(mech.get("initial_omega_rad_s", 0.0))
    mu = float(mech.get("friction", 0.35))
    mu_r = float(mech.get("rolling_friction", 0.0))
    inertia_factor = float(mech.get("inertia_factor", 0.5))
    slope_deg = float(mech.get("slope_angle_deg", 0.0))
    slope = math.radians(slope_deg)
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    s = 0.0
    theta_spin = 0.0
    pos = np.zeros((len(times), 3), dtype=float)
    spoke_a = np.zeros((len(times), 3), dtype=float)
    spoke_b = np.zeros((len(times), 3), dtype=float)
    spoke_c = np.zeros((len(times), 3), dtype=float)
    rim_marker = np.zeros((len(times), 3), dtype=float)
    omega_series: list[float] = []
    slip_ratio_series: list[float] = []
    slip_speed_series: list[float] = []
    signed_slip_series: list[float] = []
    speed_series: list[float] = []
    surface_speed_series: list[float] = []
    surface_point_speed_series: list[list[float]] = []
    friction_power_series: list[float] = []
    slope_len = max(2.0, float(mech.get("track_length", 3.6)))
    x0 = -0.55 * slope_len
    z0 = radius + max(0.0, math.sin(slope) * slope_len)
    tangent = np.asarray([math.cos(slope), 0.0, -math.sin(slope)], dtype=float)
    normal = np.asarray([math.sin(slope), 0.0, math.cos(slope)], dtype=float)
    beta = max(0.05, inertia_factor)
    rolling = False
    for i, _ in enumerate(times):
        center = np.asarray([x0, 0.0, z0], dtype=float) + tangent * s + normal * radius
        pos[i] = center
        theta_spin += omega * dt if i else 0.0
        visual_spin = -theta_spin
        rim = center + radius * (math.cos(visual_spin) * tangent + math.sin(visual_spin) * normal)
        rim_quarter = center + radius * (math.cos(visual_spin + 0.5 * math.pi) * tangent + math.sin(visual_spin + 0.5 * math.pi) * normal)
        spoke_a[i] = center
        spoke_b[i] = rim
        spoke_c[i] = rim_quarter
        rim_marker[i] = center + 0.86 * radius * (math.cos(visual_spin + 0.20) * tangent + math.sin(visual_spin + 0.20) * normal)
        omega_series.append(float(omega))
        speed_series.append(float(abs(v)))
        slip = v - omega * radius
        slip_speed = abs(slip)
        denom = max(abs(v), abs(omega * radius), 1.0e-6)
        slip_ratio_series.append(float(abs(slip) / denom))
        slip_speed_series.append(float(slip_speed))
        signed_slip_series.append(float(slip))
        surface_speed_series.append(float(abs(v) + abs(omega) * radius))
        surface_point_speed_series.append([0.0, float(abs(v) + abs(omega) * radius)])
        normal_load = mass * g * max(0.0, math.cos(slope))
        active_slip = slip_speed > 0.025 and not rolling
        friction_power_series.append(float((mu * normal_load if active_slip else 0.0) * slip_speed))
        if i == len(times) - 1:
            break
        if rolling or abs(slip) <= 1.0e-4:
            rolling = True
            a_roll = g * math.sin(slope) / (1.0 + beta)
            if abs(v) > 1.0e-6 and mu_r > 0.0:
                a_roll -= mu_r * g * math.cos(slope) * math.copysign(1.0, v)
            v += a_roll * dt
            omega = v / max(radius, 1.0e-6)
        else:
            sign = math.copysign(1.0, slip)
            a = g * math.sin(slope) - mu * g * math.cos(slope) * sign
            alpha = (mu * g * math.cos(slope) / max(beta * radius, 1.0e-6)) * sign
            v_new = v + a * dt
            omega_new = omega + alpha * dt
            if (v_new - omega_new * radius) * slip < 0.0:
                avg = 0.5 * (v_new + omega_new * radius)
                v = avg
                omega = avg / max(radius, 1.0e-6)
                rolling = True
            else:
                v = v_new
                omega = omega_new
        s = max(0.0, s + v * dt)

    start_center = np.asarray([x0, 0.0, z0], dtype=float) + normal * radius
    used_s = np.asarray([(p - start_center).dot(tangent) for p in pos], dtype=float)
    visual_len = max(slope_len, float(np.max(used_s)) + 0.45)
    ramp_a = np.repeat((np.asarray([x0, 0.0, z0], dtype=float)).reshape(1, 3), len(times), axis=0)
    ramp_b = np.repeat((np.asarray([x0, 0.0, z0], dtype=float) + tangent * visual_len).reshape(1, 3), len(times), axis=0)

    objects = [
        _capsule_track(
            name="inclined_track",
            color=_rgb(0.42, 0.44, 0.46),
            points_a=ramp_a,
            points_b=ramp_b,
            radius=0.025,
            extras={"color_mode": "fixed"},
        ),
        _circle_track(
            name="roller",
            radius=radius,
            mass=mass,
            color=_rgb(0.18, 0.61, 0.5),
            positions=pos,
            scalars={
                "omega_y": omega_series,
                "slip_ratio": slip_ratio_series,
                "slip_speed_m_s": slip_speed_series,
                "signed_slip_m_s": signed_slip_series,
                "speed_m_s": speed_series,
                "surface_speed_m_s": surface_speed_series,
                "surface_point_speed_m_s": surface_point_speed_series,
                "friction_power_w": friction_power_series,
            },
            extras={
                "color_metric": "surface_point_speed_m_s",
                "surface_scalar_profile": "rolling_surface_point_speed",
                "surface_slope_angle_deg": slope_deg,
            },
        ),
        _capsule_track(name="rotation_spoke", color=_rgb(0.08, 0.10, 0.12), points_a=spoke_a, points_b=spoke_b, radius=max(0.008, 0.04 * radius), extras={"color_mode": "fixed", "draw_on_top": True}),
        _capsule_track(name="rotation_spoke_quarter", color=_rgb(0.20, 0.22, 0.24), points_a=spoke_a, points_b=spoke_c, radius=max(0.006, 0.028 * radius), extras={"color_mode": "fixed", "draw_on_top": True}),
        _circle_track(name="rim_marker", radius=max(0.018, 0.105 * radius), mass=0.0, color=_rgb(0.04, 0.045, 0.05), positions=rim_marker, extras={"color_mode": "fixed", "draw_on_top": True}),
    ]
    scene = {
        "view_plane": "xz",
        "ground_value": None,
        "title": _case_name(case, Path(".")),
        "case_description": "sliding-to-rolling transition colored by rolling body surface point speed",
        "projected_depth": width,
        "camera_zoom": float(mech.get("camera_zoom", 1.60)),
        "reserve_scalar_bar": True,
        "scalar_bar_shift_fraction": 0.0,
        "auto_bounds": False,
        "bounds_margin_fraction": 0.08,
    }
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))

    return {
        "times": times.tolist(),
        "objects": objects,
        "scene": scene,
        "global_series": {
            "slip_speed_m_s": [float(x) for x in slip_speed_series],
            "surface_point_speed_m_s": [float(max(x)) for x in surface_point_speed_series],
            "friction_power_w": [float(x) for x in friction_power_series],
        },
    }
