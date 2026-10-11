from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _box_track, _capsule_track, _case_name, _circle_track, _duration, _mech_cfg, _rgb, _scene_oblique, _segment_track, _step_size, _times, math, np)

def _simulate_surrogate_flexible_crank_slider(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    r = float(mech.get("crank_radius_m", 0.1))
    l = max(float(mech.get("rod_length_m", 0.3)), r + 1.0e-3)
    omega = float(mech.get("omega_rad_s", 20.0))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    theta0 = math.radians(float(mech.get("theta0_deg", 0.0)))
    z0 = float(mech.get("base_z_m", 0.24))
    y_amp = float(mech.get("mid_whip_amplitude_m", 0.012))
    axis_angle = math.radians(float(mech.get("slider_axis_angle_deg", 0.0)))
    axis_c = math.cos(axis_angle)
    axis_s = math.sin(axis_angle)

    def rotate_xy(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=float, copy=True)
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = axis_c * x - axis_s * y
        out[:, 1] = axis_s * x + axis_c * y
        return out

    pin = np.zeros((len(times), 3), dtype=float)
    slider = np.zeros((len(times), 3), dtype=float)
    mid = np.zeros((len(times), 3), dtype=float)
    theta_deg: list[float] = []
    for i, t in enumerate(times):
        th = theta0 + omega * t
        x_pin = r * math.cos(th)
        y_pin = r * math.sin(th)
        x_slider = x_pin + math.sqrt(max(l * l - (r * math.sin(th)) ** 2, 1.0e-9))
        y_mid = 0.5 * y_pin + y_amp * math.sin(2.0 * th) * (1.0 - math.exp(-1.2 * t))
        pin[i] = [x_pin, y_pin, z0]
        slider[i] = [x_slider, 0.0, z0]
        mid[i] = [0.5 * (x_pin + x_slider), y_mid, z0]
        theta_deg.append(float(math.degrees(th)))

    vx_slider = np.gradient(slider[:, 0], times, edge_order=1)
    hub = np.repeat(np.asarray([[0.0, 0.0, z0]], dtype=float), len(times), axis=0)
    rail_x0 = float(np.min(slider[:, 0]) - 0.14)
    rail_x1 = float(np.max(slider[:, 0]) + 0.14)
    rail_gap = 0.035
    rail_a = np.repeat(np.asarray([[rail_x0, rail_gap, z0]], dtype=float), len(times), axis=0)
    rail_b = np.repeat(np.asarray([[rail_x1, rail_gap, z0]], dtype=float), len(times), axis=0)
    rail_c = np.repeat(np.asarray([[rail_x0, -rail_gap, z0]], dtype=float), len(times), axis=0)
    rail_d = np.repeat(np.asarray([[rail_x1, -rail_gap, z0]], dtype=float), len(times), axis=0)
    pin_render = rotate_xy(pin)
    slider_render = rotate_xy(slider)
    hub_render = rotate_xy(hub)
    mid_render = rotate_xy(mid)
    rail_a_render = rotate_xy(rail_a)
    rail_b_render = rotate_xy(rail_b)
    rail_c_render = rotate_xy(rail_c)
    rail_d_render = rotate_xy(rail_d)
    crank_speed = np.linalg.norm(np.gradient(pin_render, times, axis=0), axis=1)
    rod_curve = np.abs(mid[:, 1])
    crank = _capsule_track(name="crank_arm", color=_rgb(0.90, 0.48, 0.18), points_a=hub_render, points_b=pin_render, radius=0.0085, mass=0.4, scalars={"speed_m_s": [float(v) for v in crank_speed]})
    crank["color_metric"] = "speed_m_s"
    rod = _capsule_track(name="connecting_rod", color=_rgb(0.78, 0.72, 0.30), points_a=pin_render, points_b=slider_render, radius=0.0080, mass=0.6, scalars={"curve_m": [float(v) for v in rod_curve]})
    rod["color_metric"] = "curve_m"
    objects = [
        _circle_track(name="crank_hub", radius=0.015, mass=1.0, color=_rgb(0.40, 0.44, 0.56), positions=hub_render),
        _circle_track(name="crank_pin", radius=0.012, mass=0.4, color=_rgb(0.92, 0.46, 0.16), positions=pin_render),
        _box_track(name="slider", size=[0.05, 0.035, 0.035], mass=float(mech.get("slider_mass_kg", 1.0)), color=_rgb(0.22, 0.62, 0.86), positions=slider_render),
        crank,
        rod,
        _segment_track(name="slider_rail_top", color=_rgb(0.54, 0.56, 0.62), points_a=rail_a_render, points_b=rail_b_render, width=1.8),
        _segment_track(name="slider_rail_bottom", color=_rgb(0.54, 0.56, 0.62), points_a=rail_c_render, points_b=rail_d_render, width=1.8),
    ]
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=None)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 42.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 24.0))
    scene["camera_zoom"] = float(mech.get("camera_zoom", 1.20))
    data = {"times": times.tolist(), "objects": objects, "scene": scene}
    _add_csv_export(
        data,
        "crank_results.csv",
        ["time_s", "crank_angle_deg", "slider_velocity_m_s", "mid_deflection_m"],
        [[float(times[i]), float(theta_deg[i]), float(vx_slider[i]), float(mid[i, 1])] for i in range(len(times))],
    )
    return data
