from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_density_for_cylinder, _chrono_material, _chrono_point_local_to_world, _chrono_sampling_loop, _chrono_vec, _gravity, _mech_cfg, _rgb, _scene_oblique, math, np)

def _simulate_chrono_gyroscope(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))

    mech = _mech_cfg(case)
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
    shaft_length = float(mech.get("shaft_length", 0.65))
    shaft_radius = float(mech.get("shaft_radius", 0.04))
    rotor_radius = float(mech.get("rotor_radius", 0.16))
    mass = float(mech.get("mass", 1.1))
    tilt = math.radians(float(mech.get("initial_tilt_deg", 18.0)))
    azimuth = math.radians(float(mech.get("initial_azimuth_deg", 0.0)))
    spin = float(mech.get("initial_spin_rad_s", 140.0))
    axis = np.asarray(
        [
            math.sin(tilt) * math.cos(azimuth),
            math.sin(tilt) * math.sin(azimuth),
            math.cos(tilt),
        ],
        dtype=float,
    )
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-9)
    quat = chrono.QuatFromAngleZ(azimuth) * chrono.QuatFromAngleY(tilt)
    body = chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, rotor_radius, shaft_length, _chrono_density_for_cylinder(mass, rotor_radius, shaft_length), True, True, _chrono_material(chrono, friction=0.05, restitution=0.0))
    body.SetPos(_chrono_vec(chrono, pivot + 0.5 * shaft_length * axis))
    body.SetRot(quat)
    body.SetAngVelParent(_chrono_vec(chrono, axis * spin))
    body.EnableCollision(False)
    system.Add(body)

    anchor = chrono.ChBody()
    anchor.SetFixed(True)
    anchor.EnableCollision(False)
    system.Add(anchor)
    joint = chrono.ChLinkLockSpherical()
    joint.Initialize(anchor, body, chrono.ChFramed(_chrono_vec(chrono, pivot), chrono.QuatFromAngleZ(0.0)))
    system.Add(joint)

    tip_local = [0.0, 0.0, -0.5 * shaft_length]
    head_local = [0.0, 0.0, 0.5 * shaft_length]
    rotor_local = [0.0, 0.0, 0.0]
    ref_axis_top = (pivot + np.asarray([0.0, 0.0, shaft_length], dtype=float)).tolist()
    trackers = [
        {
            "name": "spin_axis_ref",
            "kind": "segment",
            "color": _rgb(0.48, 0.52, 0.60),
            "width": 1.4,
            "sample": lambda: {"point_a": pivot.tolist(), "point_b": ref_axis_top},
        },
        {
            "name": "precession_projection",
            "kind": "segment",
            "color": _rgb(0.78, 0.56, 0.22),
            "width": 1.6,
            "sample": lambda: {
                "point_a": pivot.tolist(),
                "point_b": [
                    float(_chrono_point_local_to_world(body, chrono, head_local)[0]),
                    float(_chrono_point_local_to_world(body, chrono, head_local)[1]),
                    float(pivot[2]),
                ],
            },
        },
        {
            "name": "shaft",
            "kind": "segment",
            "color": _rgb(0.66, 0.70, 0.78),
            "width": 2.2,
            "sample": lambda: {
                "point_a": _chrono_point_local_to_world(body, chrono, tip_local).tolist(),
                "point_b": _chrono_point_local_to_world(body, chrono, head_local).tolist(),
            },
        },
        {
            "name": "rotor",
            "kind": "circle",
            "size": [shaft_radius + rotor_radius],
            "mass": mass,
            "color": _rgb(0.88, 0.46, 0.20),
            "scalars": {},
            "sample": lambda: {"position": _chrono_point_local_to_world(body, chrono, rotor_local).tolist(), "scalars": {}},
        },
    ]
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=None)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 36.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 26.0))
    return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene=scene)
