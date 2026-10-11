from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_add_ground, _chrono_density_for_sphere, _chrono_material, _chrono_vec, _duration, _gravity, _maze_gravity_xy, _maze_schedule, _maze_wall_boxes, _mech_cfg, _polyline_arclength, _polyline_point_at, _polyline_points, _polyline_progress, _rgb, _sample_count, _scene_bounds, _scene_oblique, _step_size, math, np)

def _simulate_chrono_ball_maze(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    base_g = _gravity(case)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, base_g))
    _chrono_add_ground(system, chrono, case)

    mech = _mech_cfg(case)
    ball_cfg = mech.get("ball") if isinstance(mech.get("ball"), dict) else {}
    radius = float(ball_cfg.get("radius", 0.16))
    mass = float(ball_cfg.get("mass", 0.8))
    guide_path = _polyline_points(
        mech.get("guide_path"),
        default=[[-8.0, -6.0], [-8.0, 2.0], [-3.4, 2.0], [-3.4, -1.0], [1.6, -1.0]],
    )
    corridor_width = float(mech.get("corridor_width", 2.0))
    wall_thickness = float(mech.get("wall_thickness", 0.24))
    wall_height = float(mech.get("wall_height", 0.6))
    wall_boxes = _maze_wall_boxes(guide_path, corridor_width=corridor_width, wall_thickness=wall_thickness, wall_height=wall_height)
    wall_mat = _chrono_material(chrono, friction=float(mech.get("wall_friction", 0.12)), restitution=0.02)

    trackers: list[dict[str, Any]] = []
    for idx, wall in enumerate(wall_boxes):
        body = chrono.ChBodyEasyBox(
            float(wall["size"][0]),
            float(wall["size"][1]),
            float(wall["size"][2]),
            1000.0,
            True,
            True,
            wall_mat,
        )
        body.SetPos(_chrono_vec(chrono, wall["pos"]))
        body.SetFixed(True)
        system.Add(body)
        trackers.append(
            {
                "name": f"wall_{idx:02d}",
                "kind": "box",
                "size": [float(wall["size"][0]), float(wall["size"][1]), float(wall["size"][2])],
                "mass": 0.0,
                "color": _rgb(0.34, 0.40, 0.52),
                "scalars": {},
                "sample": lambda body=body: {"position": [body.GetPos().x, body.GetPos().y, body.GetPos().z], "scalars": {}},
            }
        )

    ball_mat = _chrono_material(
        chrono,
        friction=float(mech.get("floor_friction", 0.24)),
        restitution=0.06,
        rolling_friction=float(mech.get("rolling_friction", 0.0)),
    )
    ball = chrono.ChBodyEasySphere(radius, _chrono_density_for_sphere(mass, radius), True, True, ball_mat)
    start_pos = np.asarray(ball_cfg.get("position", [guide_path[0, 0], guide_path[0, 1], radius]), dtype=float)
    if start_pos.shape[0] >= 3:
        start_pos[2] = max(float(start_pos[2]), radius)
    ball.SetPos(_chrono_vec(chrono, start_pos))
    ball.SetPosDt(_chrono_vec(chrono, ball_cfg.get("velocity", [0.0, 0.0, 0.0])))
    if hasattr(ball, "SetAngVelParent"):
        ball.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
    if hasattr(ball, "SetLinDamping"):
        ball.SetLinDamping(float(mech.get("linear_damping", 0.02)))
    if hasattr(ball, "SetAngDamping"):
        ball.SetAngDamping(float(mech.get("angular_damping", 0.02)))
    system.Add(ball)
    trackers.append(
        {
            "name": "ball",
            "kind": "circle",
            "size": [radius],
            "mass": mass,
            "color": _rgb(0.18, 0.74, 0.86),
            "scalars": {},
            "sample": lambda: {"position": [ball.GetPos().x, ball.GetPos().y, ball.GetPos().z], "scalars": {}},
        }
    )

    duration = _duration(case)
    export_interval = max(1.0e-4, float(mech.get("export_interval_s", 0.005)))
    step = _step_size(case)
    total_steps = max(2, int(math.ceil(duration / max(step, 1.0e-6))))
    stride = max(1, total_steps // max(2, _sample_count(case) - 1))
    schedule = _maze_schedule(mech.get("tilt_schedule"), duration)
    max_g_xy = float(mech.get("max_tilt_gravity_xy", 2.8))
    controller_enabled = bool(mech.get("use_waypoint_tilt_controller", False))
    controller_wp = np.asarray(guide_path, dtype=float) if controller_enabled else np.zeros((0, 2), dtype=float)
    _, controller_total_len = _polyline_arclength(controller_wp) if controller_enabled else (np.asarray([0.0], dtype=float), 0.0)
    controller_tol = float(mech.get("waypoint_tolerance", max(0.24, 0.24 * corridor_width)))
    controller_kp = float(mech.get("tilt_kp", 0.90))
    controller_kd = float(mech.get("tilt_kd", 0.45))
    controller_lookahead = float(mech.get("tilt_lookahead", max(0.9, 0.55 * corridor_width)))
    times: list[float] = []
    objects: list[dict[str, Any]] = []
    for tr in trackers:
        objects.append(
            {
                "name": tr["name"],
                "kind": tr["kind"],
                "size": tr["size"],
                "mass": float(tr["mass"]),
                "color": tr["color"],
                "positions": [],
                "scalars": {},
            }
        )

    for step_idx in range(total_steps + 1):
        if step_idx == 0 or step_idx == total_steps or step_idx % stride == 0:
            times.append(float(system.GetChTime()))
            for out, tr in zip(objects, trackers):
                sample = tr["sample"]()
                out["positions"].append(sample["position"])
        if step_idx == total_steps:
            break
        if controller_enabled and controller_wp.shape[0] >= 2:
            pos_xy = np.asarray([ball.GetPos().x, ball.GetPos().y], dtype=float)
            vel_xy = np.asarray([ball.GetPosDt().x, ball.GetPosDt().y], dtype=float)
            progress, _ = _polyline_progress(pos_xy, controller_wp)
            target_s = min(controller_total_len, progress + max(0.0, controller_lookahead))
            target_xy = _polyline_point_at(controller_wp, target_s)
            err_xy = target_xy - pos_xy
            if target_s >= controller_total_len - 1.0e-6 and float(np.linalg.norm(controller_wp[-1] - pos_xy)) <= controller_tol:
                gravity_xy = np.zeros(2, dtype=float)
            else:
                gravity_xy = controller_kp * err_xy - controller_kd * vel_xy
        else:
            gravity_xy = _maze_gravity_xy(schedule, float(system.GetChTime()))
        gxy_norm = float(np.linalg.norm(gravity_xy))
        if gxy_norm > max_g_xy > 1.0e-9:
            gravity_xy = gravity_xy * (max_g_xy / gxy_norm)
        system.SetGravitationalAcceleration(_chrono_vec(chrono, [float(gravity_xy[0]), float(gravity_xy[1]), float(base_g[2])]))
        system.DoStepDynamics(step)

    view_plane_raw = str(mech.get("view_plane", "xy")).strip().lower()
    if view_plane_raw in {"oblique", "oblique_xy", "3d"}:
        scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
        scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 42.0))
        scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 24.0))
    else:
        scene = {"view_plane": "xy", "ground_value": None, "title": _case_name(case, Path("."))}
    scene.update(_scene_bounds(objects, scene.get("view_plane", "xy"), scene))
    return {"times": times, "objects": objects, "scene": scene}
