from __future__ import annotations
from run_case import (Any, Path, _box_track, _capsule_track, _case_name, _duration, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_slider_crank(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.42]), dtype=float)
    r = float(mech.get("crank_radius", 0.25))
    l = max(float(mech.get("rod_length", 0.82)), r + 0.05)
    omega = float(mech.get("motor_speed_rad_s", 4.2))
    theta0 = math.radians(float(mech.get("initial_angle_deg", 0.0)))
    slider_size = [0.22, 0.18, 0.16]
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    crank_a = np.repeat(pivot.reshape(1, 3), len(times), axis=0)
    crank_b = np.zeros((len(times), 3), dtype=float)
    rod_b = np.zeros((len(times), 3), dtype=float)
    slider = np.zeros((len(times), 3), dtype=float)
    for i, t in enumerate(times):
        theta = theta0 + omega * t
        crank_b[i] = pivot + np.asarray([r * math.cos(theta), r * math.sin(theta), 0.0], dtype=float)
        slider_x = pivot[0] + r * math.cos(theta) + math.sqrt(max(l * l - (r * math.sin(theta)) ** 2, 1.0e-9))
        rod_b[i] = [slider_x, 0.0, pivot[2]]
        slider[i] = [slider_x, 0.0, pivot[2]]

    return {
        "times": times.tolist(),
        "objects": [
            _capsule_track(name="crank", color=_rgb(0.78, 0.67, 0.40), points_a=crank_a, points_b=crank_b, radius=0.04),
            _capsule_track(name="rod", color=_rgb(0.54, 0.65, 0.80), points_a=crank_b, points_b=rod_b, radius=0.032),
            _box_track(name="slider", size=slider_size, mass=float(mech.get("slider_mass", 1.2)), color=_rgb(0.86, 0.34, 0.24), positions=slider),
        ],
        "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=None),
    }
