from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_density_for_box, _chrono_material, _chrono_point_local_to_world, _chrono_quat_z, _chrono_sampling_loop, _chrono_vec, _mech_cfg, _rgb, _scene_oblique, math, np)

def _simulate_chrono_slider_crank(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
    mech = _mech_cfg(case)
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.42]), dtype=float)
    r = float(mech.get("crank_radius", 0.25))
    l = max(float(mech.get("rod_length", 0.82)), r + 0.05)
    width = float(mech.get("body_width", 0.11))
    thickness = float(mech.get("body_thickness", 0.08))
    theta0 = math.radians(float(mech.get("initial_angle_deg", 0.0)))
    omega = float(mech.get("motor_speed_rad_s", 4.2))

    ground = chrono.ChBody()
    ground.SetFixed(True)
    ground.EnableCollision(False)
    system.Add(ground)
    mat = _chrono_material(chrono, friction=0.04, restitution=0.0)
    slider_x0 = pivot[0] + r * math.cos(theta0) + math.sqrt(max(l * l - (r * math.sin(theta0)) ** 2, 1.0e-9))
    crank = chrono.ChBodyEasyBox(r, width, thickness, _chrono_density_for_box(float(mech.get("crank_mass", 0.9)), [r, width, thickness]), True, False, mat)
    crank.SetPos(_chrono_vec(chrono, [pivot[0] + 0.5 * r, pivot[1], pivot[2]]))
    system.Add(crank)
    rod = chrono.ChBodyEasyBox(l, 0.85 * width, 0.9 * thickness, _chrono_density_for_box(float(mech.get("rod_mass", 0.8)), [l, 0.85 * width, 0.9 * thickness]), True, False, mat)
    rod.SetPos(_chrono_vec(chrono, [pivot[0] + r + 0.5 * l, pivot[1], pivot[2]]))
    system.Add(rod)

    motor = chrono.ChLinkMotorRotationSpeed()
    motor.Initialize(ground, crank, chrono.ChFramed(_chrono_vec(chrono, pivot), _chrono_quat_z(chrono, 0.0)))
    motor.SetSpeedFunction(chrono.ChFunctionConst(omega))
    system.Add(motor)

    j_bc = chrono.ChLinkLockRevolute()
    j_bc.Initialize(crank, rod, chrono.ChFramed(_chrono_vec(chrono, [pivot[0] + r, pivot[1], pivot[2]]), _chrono_quat_z(chrono, 0.0)))
    system.Add(j_bc)
    j_ca = chrono.ChLinkLockPointLine()
    j_ca.Initialize(rod, ground, chrono.ChFramed(_chrono_vec(chrono, [slider_x0, pivot[1], pivot[2]]), _chrono_quat_z(chrono, 0.0)))
    system.Add(j_ca)
    rail_pad = max(0.22, 0.35 * r)
    rail_x0 = float(pivot[0] - r - rail_pad)
    rail_x1 = float(pivot[0] + r + l + rail_pad)
    rail_offset = 0.75 * width

    trackers = [
        {
            "name": "slider_rail_upper",
            "kind": "segment",
            "color": _rgb(0.44, 0.48, 0.56),
            "width": 1.8,
            "sample": lambda: {
                "point_a": [rail_x0, float(pivot[1] + rail_offset), float(pivot[2])],
                "point_b": [rail_x1, float(pivot[1] + rail_offset), float(pivot[2])],
            },
        },
        {
            "name": "slider_rail_lower",
            "kind": "segment",
            "color": _rgb(0.44, 0.48, 0.56),
            "width": 1.8,
            "sample": lambda: {
                "point_a": [rail_x0, float(pivot[1] - rail_offset), float(pivot[2])],
                "point_b": [rail_x1, float(pivot[1] - rail_offset), float(pivot[2])],
            },
        },
        {
            "name": "slider_rail_center",
            "kind": "segment",
            "color": _rgb(0.88, 0.74, 0.30),
            "width": 1.2,
            "sample": lambda: {
                "point_a": [rail_x0, float(pivot[1]), float(pivot[2])],
                "point_b": [rail_x1, float(pivot[1]), float(pivot[2])],
            },
        },
        {
            "name": "crank",
            "kind": "capsule",
            "size": [0.5 * width],
            "mass": 0.0,
            "color": _rgb(0.78, 0.67, 0.40),
            "scalars": {},
            "sample": lambda: {
                "point_a": pivot.tolist(),
                "point_b": _chrono_point_local_to_world(crank, chrono, [0.5 * r, 0.0, 0.0]).tolist(),
                "position": (0.5 * (pivot + _chrono_point_local_to_world(crank, chrono, [0.5 * r, 0.0, 0.0]))).tolist(),
                "scalars": {},
            },
        },
        {
            "name": "rod",
            "kind": "capsule",
            "size": [0.45 * width],
            "mass": 0.0,
            "color": _rgb(0.54, 0.65, 0.80),
            "scalars": {},
            "sample": lambda: {
                "point_a": _chrono_point_local_to_world(rod, chrono, [-0.5 * l, 0.0, 0.0]).tolist(),
                "point_b": _chrono_point_local_to_world(rod, chrono, [0.5 * l, 0.0, 0.0]).tolist(),
                "position": (0.5 * (_chrono_point_local_to_world(rod, chrono, [-0.5 * l, 0.0, 0.0]) + _chrono_point_local_to_world(rod, chrono, [0.5 * l, 0.0, 0.0]))).tolist(),
                "scalars": {},
            },
        },
        {
            "name": "slider",
            "kind": "circle",
            "size": [0.08],
            "mass": float(mech.get("slider_mass", 1.2)),
            "color": _rgb(0.86, 0.34, 0.24),
            "scalars": {},
            "sample": lambda: {"position": _chrono_point_local_to_world(rod, chrono, [0.5 * l, 0.0, 0.0]).tolist(), "scalars": {}},
        },
    ]
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=None)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 38.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 27.0))
    return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene=scene)
