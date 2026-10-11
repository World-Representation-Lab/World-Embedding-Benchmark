from __future__ import annotations
from run_case import (Any, Path, _capsule_track, _case_name, _circle_track, _duration, _gear_motor_profile, _integrate_angular_speed, _mech_cfg, _rgb, _scene_bounds, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_gear_pair(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    speed_unit = str(mech.get("speed_unit", "angular_velocity")).strip().lower()
    gear_count = max(2, int(mech.get("gear_count", 2)))
    center_a = np.asarray(mech.get("center_a", [0.0, 0.0, 0.36]), dtype=float)
    radius_a = float(mech.get("radius_a", 0.32))
    radius_b = float(mech.get("radius_b", 0.58))
    radii_cfg = mech.get("radii") if isinstance(mech.get("radii"), list) else None
    teeth_cfg = mech.get("teeth") if isinstance(mech.get("teeth"), list) else None
    masses_cfg = mech.get("masses") if isinstance(mech.get("masses"), list) else None
    radii = [float(v) for v in (radii_cfg[:gear_count] if radii_cfg else [radius_a, radius_b])]
    teeth = [int(v) for v in (teeth_cfg[:gear_count] if teeth_cfg else [int(mech.get("teeth_a", 48)), int(mech.get("teeth_b", 100))])]
    masses = [float(v) for v in (masses_cfg[:gear_count] if masses_cfg else [float(mech.get("mass_a", 1.0)), float(mech.get("mass_b", 1.2))])]
    while len(radii) < gear_count:
        radii.append(float(radii[-1]))
    while len(teeth) < gear_count:
        teeth.append(int(teeth[-1]))
    while len(masses) < gear_count:
        masses.append(float(masses[-1]))
    thickness = float(mech.get("thickness", 0.105))
    compound_mode = bool(mech.get("compound_gears", gear_count >= 4))
    compound_plane_gap = float(mech.get("compound_plane_gap", 0.44))
    mesh_clearance = float(mech.get("mesh_clearance_m", 0.002))
    theta0 = math.radians(float(mech.get("initial_angle_deg", 0.0)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    omega_drive = _gear_motor_profile(mech, times, default_speed=float(mech.get("motor_speed_rad_s", 3.6)))
    centers: list[np.ndarray] = [center_a]
    mesh_edges: list[tuple[int, int]] = []
    if gear_count >= 2:
        center_b = np.asarray(mech.get("center_b", [center_a[0] + radii[0] + radii[1] + mesh_clearance, center_a[1], center_a[2]]), dtype=float)
        centers.append(center_b)
        mesh_edges.append((0, 1))
    compound_pairs: list[tuple[int, int]] = []
    if gear_count >= 3:
        if compound_mode:
            # The linked gear sits on a second plane directly behind the first one,
            # sharing the same spin axis through a common shaft.
            centers.append(np.asarray([centers[1][0], centers[1][1] - compound_plane_gap, centers[1][2]], dtype=float))
            compound_pairs.append((1, 2))
            prev = 2
        else:
            centers.append(np.asarray([centers[1][0] + radii[1] + radii[2] + mesh_clearance, centers[1][1], centers[1][2]], dtype=float))
            mesh_edges.append((1, 2))
            prev = 2
        for gi in range(3, gear_count):
            centers.append(np.asarray([centers[prev][0] + radii[prev] + radii[gi] + mesh_clearance, centers[prev][1], centers[prev][2]], dtype=float))
            mesh_edges.append((prev, gi))
            prev = gi

    pos_series = [np.repeat(np.asarray(c, dtype=float).reshape(1, 3), len(times), axis=0) for c in centers]
    spin_rad = [np.zeros((len(times),), dtype=float) for _ in range(gear_count)]
    omega_series = [np.zeros((len(times),), dtype=float) for _ in range(gear_count)]
    spin_rad[0][:] = _integrate_angular_speed(times, omega_drive, theta0)
    omega_series[0][:] = omega_drive
    resolved = {0}
    pending_mesh = list(mesh_edges)
    pending_compound = list(compound_pairs)
    while pending_mesh or pending_compound:
        progressed = False
        next_compound: list[tuple[int, int]] = []
        for a, b in pending_compound:
            if a in resolved and b not in resolved:
                spin_rad[b][:] = spin_rad[a]
                omega_series[b][:] = omega_series[a]
                resolved.add(b)
                progressed = True
            elif b in resolved and a not in resolved:
                spin_rad[a][:] = spin_rad[b]
                omega_series[a][:] = omega_series[b]
                resolved.add(a)
                progressed = True
            else:
                next_compound.append((a, b))
        pending_compound = next_compound

        next_mesh: list[tuple[int, int]] = []
        for a, b in pending_mesh:
            if a in resolved and b not in resolved:
                ratio = float(teeth[a]) / max(float(teeth[b]), 1.0e-9)
                spin_rad[b][:] = -ratio * (spin_rad[a] - theta0) + theta0
                omega_series[b][:] = -ratio * omega_series[a]
                resolved.add(b)
                progressed = True
            elif b in resolved and a not in resolved:
                ratio = float(teeth[b]) / max(float(teeth[a]), 1.0e-9)
                spin_rad[a][:] = -ratio * (spin_rad[b] - theta0) + theta0
                omega_series[a][:] = -ratio * omega_series[b]
                resolved.add(a)
                progressed = True
            else:
                next_mesh.append((a, b))
        pending_mesh = next_mesh
        if not progressed:
            break

    hole_radius = float(mech.get("inner_hole_ratio", 0.16)) * min(radii)
    objects: list[dict[str, Any]] = []
    palette = [
        _rgb(0.82, 0.36, 0.20),
        _rgb(0.95, 0.60, 0.20),
        _rgb(0.82, 0.36, 0.20),
        _rgb(0.24, 0.58, 0.80),
        _rgb(0.28, 0.72, 0.54),
    ]
    for gi in range(gear_count):
        objects.append(
            _circle_track(
                name=f"gear_{chr(ord('a') + gi)}",
                radius=float(radii[gi]),
                mass=float(masses[gi]),
                color=palette[gi % len(palette)],
                positions=pos_series[gi],
                scalars={
                    "spin_deg": np.degrees(spin_rad[gi]).tolist(),
                    "omega_spin_rad_s": omega_series[gi].tolist(),
                    "omega_spin_abs_rad_s": np.abs(omega_series[gi]).tolist(),
                },
                extras={
                    "render_shape": "gear",
                    "color_metric": "omega_spin_abs_rad_s",
                    "teeth": int(teeth[gi]),
                    "speed_unit": speed_unit,
                    "thickness": float(thickness),
                    "inner_hole_ratio": float(mech.get("inner_hole_ratio", 0.16)),
                    "inner_hole_radius_abs": float(hole_radius),
                    "pressure_angle_deg": float(mech.get("pressure_angle_deg", 20.0)),
                    "render_spin_sign": 1.0,
                },
            )
        )
    for a, b in compound_pairs:
        shaft_radius = max(0.92 * hole_radius, 0.16 * min(float(radii[a]), float(radii[b])))
        pa = np.repeat(np.asarray([centers[a][0], centers[a][1], centers[a][2]], dtype=float).reshape(1, 3), len(times), axis=0)
        pb = np.repeat(np.asarray([centers[b][0], centers[b][1], centers[b][2]], dtype=float).reshape(1, 3), len(times), axis=0)
        objects.append(_capsule_track(name=f"shaft_{chr(ord('a') + a)}{chr(ord('a') + b)}", color=_rgb(0.06, 0.06, 0.06), points_a=pa, points_b=pb, radius=float(shaft_radius), mass=0.0))
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=None)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 40.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 22.0))
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    return {"times": times.tolist(), "objects": objects, "scene": scene}
