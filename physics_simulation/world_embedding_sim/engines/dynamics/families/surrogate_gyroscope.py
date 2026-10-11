from __future__ import annotations
from run_case import (Any, Path, _box_track, _capsule_track, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _scene_bounds, _scene_oblique, _segment_track, _step_size, _times, math, np)

def _simulate_surrogate_gyroscope(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
    shaft_length = float(mech.get("shaft_length", 0.65))
    rotor_radius = float(mech.get("rotor_radius", 0.16))
    mass = float(mech.get("mass", 1.1))
    tilt0 = math.radians(float(mech.get("initial_tilt_deg", 18.0)))
    phi0 = math.radians(float(mech.get("initial_azimuth_deg", 0.0)))
    spin = float(mech.get("initial_spin_rad_s", 140.0))
    nutation_amp_deg = float(mech.get("nutation_amp_deg", 2.4))
    imbalance = float(mech.get("imbalance_offset", 0.035))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    g = abs(float(_gravity(case)[2]))
    precession_rate = float(mech.get("precession_rate_rad_s", 0.0))
    if abs(precession_rate) <= 1.0e-9:
        precession_rate = max(0.18, (2.0 * mass * g * max(shaft_length * 0.5, 1.0e-6)) / max(mass * rotor_radius * rotor_radius * abs(spin), 1.0e-6))

    rotor = np.zeros((len(times), 3), dtype=float)
    head = np.zeros((len(times), 3), dtype=float)
    eccentric = np.zeros((len(times), 3), dtype=float)
    tip = np.repeat(pivot.reshape(1, 3), len(times), axis=0)
    ring_segments: list[dict[str, Any]] = []
    ring_radius = shaft_length * math.sin(max(tilt0, math.radians(4.0)))
    ring_z = pivot[2] + shaft_length * math.cos(max(tilt0, math.radians(4.0)))
    ring_pts = []
    for a in np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False):
        ring_pts.append(pivot + np.asarray([ring_radius * math.cos(a), ring_radius * math.sin(a), ring_z - pivot[2]], dtype=float))
    for j in range(len(ring_pts)):
        pa = np.repeat(ring_pts[j].reshape(1, 3), len(times), axis=0)
        pb = np.repeat(ring_pts[(j + 1) % len(ring_pts)].reshape(1, 3), len(times), axis=0)
        ring_segments.append(_segment_track(name=f"precession_reference_{j:02d}", color=_rgb(0.62, 0.65, 0.68), points_a=pa, points_b=pb, width=0.65))
    for i, t in enumerate(times):
        nutation = math.radians(nutation_amp_deg) * math.exp(-0.20 * t) * math.sin(2.7 * t)
        tilt = max(math.radians(4.0), tilt0 + nutation)
        phi = phi0 + precession_rate * t
        axis = np.asarray(
            [
                math.sin(tilt) * math.cos(phi),
                math.sin(tilt) * math.sin(phi),
                math.cos(tilt),
            ],
            dtype=float,
        )
        rotor[i] = pivot + 0.5 * shaft_length * axis
        head[i] = pivot + shaft_length * axis
        u = np.asarray([-math.sin(phi), math.cos(phi), 0.0], dtype=float)
        v_perp = np.cross(axis, u)
        v_norm = float(np.linalg.norm(v_perp))
        if v_norm > 1.0e-9:
            v_perp /= v_norm
        spin_phase = spin * t
        eccentric[i] = rotor[i] + imbalance * (math.cos(spin_phase) * u + math.sin(spin_phase) * v_perp)

    objects = ring_segments + [
        _box_track(
            name="pivot_base",
            size=[0.18, 0.18, 0.07],
            mass=0.0,
            color=_rgb(0.38, 0.40, 0.43),
            positions=np.repeat((pivot + np.asarray([0.0, 0.0, -0.035])).reshape(1, 3), len(times), axis=0),
            extras={"color_mode": "fixed"},
        ),
        _capsule_track(name="shaft", color=_rgb(0.66, 0.70, 0.78), points_a=tip, points_b=head, radius=max(0.018, 0.18 * rotor_radius), scalars={"angular_speed_rad_s": [abs(float(spin)) for _ in times]}, extras={"color_metric": "angular_speed_rad_s"}),
        _circle_track(name="rotor", radius=rotor_radius, mass=mass, color=_rgb(0.88, 0.46, 0.20), positions=rotor, scalars={"angular_speed_rad_s": [abs(float(spin)) for _ in times]}, extras={"color_metric": "angular_speed_rad_s"}),
        _circle_track(name="eccentric_mass", radius=max(0.025, 0.18 * rotor_radius), mass=0.08 * mass, color=_rgb(0.12, 0.18, 0.24), positions=eccentric),
    ]
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=None)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 36.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 24.0))
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    return {
        "times": times.tolist(),
        "objects": objects,
        "scene": scene,
    }
