from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_body_box_between, _chrono_material, _chrono_point_local_to_world, _chrono_quat_z, _chrono_sampling_loop, _chrono_vec, _fourbar_joint_points, _mech_cfg, _rgb, _scene_oblique, math, np)

def _simulate_chrono_four_bar(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
    mech = _mech_cfg(case)
    a0, b0, c0, d0 = _fourbar_joint_points(mech, math.radians(float(mech.get("initial_angle_deg", 28.0))))
    width = float(mech.get("link_width", 0.07))
    thickness = float(mech.get("link_thickness", 0.06))
    omega = float(mech.get("motor_speed_rad_s", 2.4))
    render_radius = max(0.006, float(mech.get("render_link_radius", 0.24 * width)))
    theta_init = math.radians(float(mech.get("initial_angle_deg", 28.0)))
    # Use a small render-only z split to avoid apparent interpenetration in 2D projection.
    # Pin connectors are added so the ground and moving links still look connected.
    render_plane_gap = max(0.0, float(mech.get("render_plane_gap", 0.032)))
    z_offset_ground = -0.5 * render_plane_gap
    z_offset_moving = 0.5 * render_plane_gap
    mat = _chrono_material(chrono, friction=0.03, restitution=0.0)

    ground = chrono.ChBody()
    ground.SetFixed(True)
    ground.EnableCollision(False)
    system.Add(ground)
    crank = _chrono_body_box_between(system, chrono, point_a=a0, point_b=b0, width=width, thickness=thickness, mass=0.8, material=mat)
    coupler = _chrono_body_box_between(system, chrono, point_a=b0, point_b=c0, width=0.9 * width, thickness=thickness, mass=0.9, material=mat)
    rocker = _chrono_body_box_between(system, chrono, point_a=d0, point_b=c0, width=width, thickness=thickness, mass=0.85, material=mat)
    for body in (crank, coupler, rocker):
        if hasattr(body, "EnableCollision"):
            body.EnableCollision(False)

    crank_len = float(np.linalg.norm(b0 - a0))
    coupler_len = float(np.linalg.norm(c0 - b0))
    rocker_len = float(np.linalg.norm(c0 - d0))

    def _link_endpoints(body: Any, length: float) -> tuple[np.ndarray, np.ndarray]:
        pa = _chrono_point_local_to_world(body, chrono, [-0.5 * float(length), 0.0, 0.0])
        pb = _chrono_point_local_to_world(body, chrono, [0.5 * float(length), 0.0, 0.0])
        return pa, pb

    def _joint_points_now() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ca, cb = _link_endpoints(crank, crank_len)
        rb0, rc = _link_endpoints(rocker, rocker_len)
        kb, kc = _link_endpoints(coupler, coupler_len)
        a = ca
        b = 0.5 * (cb + kb)
        c = 0.5 * (kc + rc)
        d = rb0
        return a, b, c, d

    def _sample_crank() -> dict[str, Any]:
        a, b = _link_endpoints(crank, crank_len)
        return _with_z_offset({"point_a": a.tolist(), "point_b": b.tolist(), "position": (0.5 * (a + b)).tolist(), "scalars": {}}, z_offset_moving)

    def _sample_coupler() -> dict[str, Any]:
        b, c = _link_endpoints(coupler, coupler_len)
        return _with_z_offset({"point_a": b.tolist(), "point_b": c.tolist(), "position": (0.5 * (b + c)).tolist(), "scalars": {}}, z_offset_moving)

    def _sample_rocker() -> dict[str, Any]:
        d, c = _link_endpoints(rocker, rocker_len)
        return _with_z_offset({"point_a": d.tolist(), "point_b": c.tolist(), "position": (0.5 * (d + c)).tolist(), "scalars": {}}, z_offset_moving)

    def _with_z_offset(sample: dict[str, Any], z_offset: float) -> dict[str, Any]:
        if abs(z_offset) <= 1.0e-12:
            return sample
        out = dict(sample)
        if "point_a" in out and isinstance(out["point_a"], list) and len(out["point_a"]) >= 3:
            pa = list(out["point_a"])
            pa[2] = float(pa[2]) + float(z_offset)
            out["point_a"] = pa
        if "point_b" in out and isinstance(out["point_b"], list) and len(out["point_b"]) >= 3:
            pb = list(out["point_b"])
            pb[2] = float(pb[2]) + float(z_offset)
            out["point_b"] = pb
        if "position" in out and isinstance(out["position"], list) and len(out["position"]) >= 3:
            pos = list(out["position"])
            pos[2] = float(pos[2]) + float(z_offset)
            out["position"] = pos
        return out

    def _sample_ground() -> dict[str, Any]:
        sample = {"point_a": a0.tolist(), "point_b": d0.tolist(), "position": (0.5 * (a0 + d0)).tolist(), "scalars": {}}
        return _with_z_offset(sample, z_offset_ground)

    def _sample_pin_a() -> dict[str, Any]:
        a, _, _, _ = _joint_points_now()
        pa = np.asarray([float(a[0]), float(a[1]), float(a[2] + z_offset_ground)], dtype=float)
        pb = np.asarray([float(a[0]), float(a[1]), float(a[2] + z_offset_moving)], dtype=float)
        return {"point_a": pa.tolist(), "point_b": pb.tolist(), "position": (0.5 * (pa + pb)).tolist(), "scalars": {}}

    def _sample_pin_d() -> dict[str, Any]:
        _, _, _, d = _joint_points_now()
        pa = np.asarray([float(d[0]), float(d[1]), float(d[2] + z_offset_ground)], dtype=float)
        pb = np.asarray([float(d[0]), float(d[1]), float(d[2] + z_offset_moving)], dtype=float)
        return {"point_a": pa.tolist(), "point_b": pb.tolist(), "position": (0.5 * (pa + pb)).tolist(), "scalars": {}}

    motor = chrono.ChLinkMotorRotationSpeed()
    motor.Initialize(ground, crank, chrono.ChFramed(_chrono_vec(chrono, a0), _chrono_quat_z(chrono, 0.0)))
    if hasattr(motor, "SetSpeedFunction"):
        motor.SetSpeedFunction(chrono.ChFunctionConst(omega))
    else:
        motor.SetMotorFunction(chrono.ChFunctionConst(omega))
    system.Add(motor)

    for body1, body2, pt in [(crank, coupler, b0), (coupler, rocker, c0), (ground, rocker, d0)]:
        joint = chrono.ChLinkLockRevolute()
        joint.Initialize(body1, body2, chrono.ChFramed(_chrono_vec(chrono, pt), _chrono_quat_z(chrono, 0.0)))
        system.Add(joint)

    trackers = [
        {
            "name": "ground_link",
            "kind": "capsule",
            "size": [0.90 * render_radius],
            "mass": 0.0,
            "color": _rgb(0.40, 0.44, 0.52),
            "scalars": {},
            "sample": _sample_ground,
        },
        {
            "name": "crank",
            "kind": "capsule",
            "size": [render_radius],
            "mass": 0.0,
            "color": _rgb(0.82, 0.69, 0.34),
            "scalars": {},
            "sample": _sample_crank,
        },
        {
            "name": "coupler",
            "kind": "capsule",
            "size": [0.92 * render_radius],
            "mass": 0.0,
            "color": _rgb(0.28, 0.62, 0.78),
            "scalars": {},
            "sample": _sample_coupler,
        },
        {
            "name": "rocker",
            "kind": "capsule",
            "size": [render_radius],
            "mass": 0.0,
            "color": _rgb(0.86, 0.40, 0.24),
            "scalars": {},
            "sample": _sample_rocker,
        },
        {
            "name": "pin_a",
            "kind": "capsule",
            "size": [max(0.004, 0.42 * render_radius)],
            "mass": 0.0,
            "color": _rgb(0.24, 0.24, 0.26),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": _sample_pin_a,
        },
        {
            "name": "pin_d",
            "kind": "capsule",
            "size": [max(0.004, 0.42 * render_radius)],
            "mass": 0.0,
            "color": _rgb(0.24, 0.24, 0.26),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": _sample_pin_d,
        },
        {
            "name": "joint_a",
            "kind": "circle",
            "size": [max(0.010, 0.72 * render_radius)],
            "mass": 0.0,
            "color": _rgb(0.18, 0.18, 0.18),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": lambda: {"position": (np.asarray(_joint_points_now()[0], dtype=float) + np.asarray([0.0, 0.0, z_offset_moving], dtype=float)).tolist(), "scalars": {}},
        },
        {
            "name": "joint_b",
            "kind": "circle",
            "size": [max(0.010, 0.72 * render_radius)],
            "mass": 0.0,
            "color": _rgb(0.18, 0.18, 0.18),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": lambda: {"position": (np.asarray(_joint_points_now()[1], dtype=float) + np.asarray([0.0, 0.0, z_offset_moving], dtype=float)).tolist(), "scalars": {}},
        },
        {
            "name": "joint_c",
            "kind": "circle",
            "size": [max(0.010, 0.72 * render_radius)],
            "mass": 0.0,
            "color": _rgb(0.18, 0.18, 0.18),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": lambda: {"position": (np.asarray(_joint_points_now()[2], dtype=float) + np.asarray([0.0, 0.0, z_offset_moving], dtype=float)).tolist(), "scalars": {}},
        },
        {
            "name": "joint_d",
            "kind": "circle",
            "size": [max(0.010, 0.72 * render_radius)],
            "mass": 0.0,
            "color": _rgb(0.18, 0.18, 0.18),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": lambda: {"position": (np.asarray(_joint_points_now()[3], dtype=float) + np.asarray([0.0, 0.0, z_offset_moving], dtype=float)).tolist(), "scalars": {}},
        },
    ]
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=None)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 52.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 27.0))
    return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene=scene)
