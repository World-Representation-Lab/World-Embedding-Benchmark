from __future__ import annotations
from run_case import (Any, Path, _capsule_track, _case_name, _circle_track, _duration, _fourbar_joint_points, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_four_bar(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    omega = float(mech.get("motor_speed_rad_s", 2.4))
    theta0 = math.radians(float(mech.get("initial_angle_deg", 28.0)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    anchor_a = np.zeros((len(times), 3), dtype=float)
    joint_b = np.zeros((len(times), 3), dtype=float)
    joint_c = np.zeros((len(times), 3), dtype=float)
    anchor_d = np.zeros((len(times), 3), dtype=float)
    coupler_marker = np.zeros((len(times), 3), dtype=float)
    for i, t in enumerate(times):
        a, b, c, d = _fourbar_joint_points(mech, theta0 + omega * t)
        anchor_a[i] = a
        joint_b[i] = b
        joint_c[i] = c
        anchor_d[i] = d
        coupler_marker[i] = 0.5 * (b + c)

    return {
        "times": times.tolist(),
        "objects": [
            _capsule_track(name="ground_link", color=_rgb(0.40, 0.44, 0.52), points_a=anchor_a, points_b=anchor_d, radius=0.035),
            _capsule_track(name="crank", color=_rgb(0.82, 0.69, 0.34), points_a=anchor_a, points_b=joint_b, radius=0.035),
            _capsule_track(name="coupler", color=_rgb(0.28, 0.62, 0.78), points_a=joint_b, points_b=joint_c, radius=0.03),
            _capsule_track(name="rocker", color=_rgb(0.86, 0.40, 0.24), points_a=anchor_d, points_b=joint_c, radius=0.035),
            _circle_track(name="coupler_marker", radius=0.055, mass=0.4, color=_rgb(0.90, 0.48, 0.18), positions=coupler_marker),
        ],
        "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=None),
    }
