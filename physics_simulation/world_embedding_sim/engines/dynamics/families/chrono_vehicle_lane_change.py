from __future__ import annotations
from run_case import (Any, Optional, Path, _case_name, _chrono_add_bmw_vehicle, _chrono_init_bmw_vehicle_scene, _chrono_vec, _chrono_vehicle_pose, _chrono_vehicle_visual_offset, _driver_sequence_entries, _mech_cfg, _rgb, _sample_count, _scene_bounds, _scene_oblique, _vehicle_object_template, math, np)

def _simulate_chrono_vehicle_lane_change(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    mech = _mech_cfg(case)
    veh, system, terrain, spec, step, duration, tire_enum, collision_enum = _chrono_init_bmw_vehicle_scene(case, chrono)
    terrain.Initialize()
    wrapper = _chrono_add_bmw_vehicle(veh, chrono, system, step, tire_enum, collision_enum, mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {})
    driver = veh.ChDataDriver(wrapper.GetVehicle(), _driver_sequence_entries(veh, mech.get("driver_sequence") if isinstance(mech.get("driver_sequence"), list) else [], duration))
    driver.Initialize()
    vehicle_ref = wrapper.GetVehicle()
    visual_offset = _chrono_vehicle_visual_offset(vehicle_ref)
    chassis_body = vehicle_ref.GetChassisBody()
    vehicle_cfg = mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {}
    target_vel = np.asarray(vehicle_cfg.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    lane_speed_hold = bool(mech.get("lane_speed_hold", True))
    lane_speed_hold_until = float(mech.get("lane_speed_hold_until_t", duration))
    total_steps = max(2, int(math.ceil(duration / max(step, 1.0e-6))))
    stride = max(1, total_steps // max(2, _sample_count(case) - 1))
    obj = _vehicle_object_template(
        "vehicle",
        spec,
        _rgb(0.18, 0.49, 0.82),
        "bmw_e90",
        visual_offset=visual_offset,
        wheel_width_scale=float(mech.get("wheel_width_scale", 1.0)),
    )
    lane_width = float(mech.get("lane_width", 3.6))
    vehicle_cfg = mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {}
    pos0 = np.asarray(vehicle_cfg.get("position", [-28.0, 0.0, 0.6]), dtype=float)
    vel0 = np.asarray(vehicle_cfg.get("velocity", [16.0, 0.0, 0.0]), dtype=float)
    road_x0 = float(mech.get("road_x0", float(pos0[0]) - 6.0))
    road_x1 = float(mech.get("road_x1", float(pos0[0]) + max(38.0, float(np.linalg.norm(vel0)) * duration + 10.0)))
    lane_y0 = float(mech.get("lane_center_y", 0.0))
    lane_segments = [
        {
            "name": "lane_center",
            "kind": "segment",
            "color": _rgb(0.80, 0.80, 0.82),
            "width": 1.4,
            "color_mode": "fixed",
            "points_a": [],
            "points_b": [],
        },
        {
            "name": "lane_left",
            "kind": "segment",
            "color": _rgb(0.45, 0.47, 0.51),
            "width": 1.6,
            "color_mode": "fixed",
            "points_a": [],
            "points_b": [],
        },
        {
            "name": "lane_right",
            "kind": "segment",
            "color": _rgb(0.45, 0.47, 0.51),
            "width": 1.6,
            "color_mode": "fixed",
            "points_a": [],
            "points_b": [],
        },
    ]
    visual_obstacle_cfg = mech.get("visual_obstacle") if isinstance(mech.get("visual_obstacle"), dict) else {
        "enabled": True,
        "position": [float(pos0[0]) + 15.0, lane_y0, 0.45],
        "size": [1.3, 1.05, 0.9],
    }
    obstacle_obj: Optional[dict[str, Any]] = None
    if bool(visual_obstacle_cfg.get("enabled", True)):
        obs_pos = visual_obstacle_cfg.get("position", [float(pos0[0]) + 15.0, lane_y0, 0.45])
        obs_size = visual_obstacle_cfg.get("size", [1.3, 1.05, 0.9])
        obs_shape = str(visual_obstacle_cfg.get("shape", "box")).strip().lower()
        obstacle_obj = {
            "name": "lane_obstacle_ref",
            "kind": "box",
            "size": [float(obs_size[0]), float(obs_size[1]), float(obs_size[2])],
            "mass": 0.0,
            "color": _rgb(0.66, 0.28, 0.22),
            "render_shape": "speed_bump" if obs_shape in {"speed_bump", "bump", "trapezoid"} else "box",
            "top_width": 0.68 * float(obs_size[0]) if obs_shape in {"speed_bump", "bump", "trapezoid"} else 0.0,
            "color_mode": "fixed",
            "positions": [],
            "scalars": {},
            "_static_pos": [float(obs_pos[0]), float(obs_pos[1]), float(obs_pos[2])],
        }
    lane_path_controller = bool(mech.get("lane_path_controller", False))
    lane_path_cfg = mech.get("lane_path_points")
    if isinstance(lane_path_cfg, list) and lane_path_cfg:
        lane_path_xy = np.asarray(
            [
                [float(pt[0]), float(pt[1])]
                for pt in lane_path_cfg
                if isinstance(pt, list) and len(pt) >= 2
            ],
            dtype=float,
        )
    else:
        lane_path_xy = np.asarray(
            [
                [float(pos0[0]), float(pos0[1])],
                [float(pos0[0]) + 9.5, lane_y0 + 0.95],
                [float(pos0[0]) + 19.0, lane_y0 + 0.10],
                [road_x1, lane_y0],
            ],
            dtype=float,
        )
    lane_path_kp = float(mech.get("lane_path_kp", 1.05))
    lane_path_kd = float(mech.get("lane_path_kd", 0.58))
    lane_path_vy_max = float(mech.get("lane_path_vy_max", 5.8))
    lane_path_until_t = float(mech.get("lane_path_controller_until_t", duration))
    times: list[float] = []

    for step_idx in range(total_steps + 1):
        if (
            lane_path_controller
            and lane_path_xy.shape[0] >= 2
            and float(system.GetChTime()) <= lane_path_until_t
        ):
            vel = chassis_body.GetPosDt()
            pos = chassis_body.GetPos()
            target_y = float(np.interp(float(pos.x), lane_path_xy[:, 0], lane_path_xy[:, 1], left=lane_path_xy[0, 1], right=lane_path_xy[-1, 1]))
            vy_cmd = lane_path_kp * (target_y - float(pos.y)) - lane_path_kd * float(vel.y)
            vy_cmd = float(max(-lane_path_vy_max, min(lane_path_vy_max, vy_cmd)))
            vx_cmd = float(target_vel[0]) if lane_speed_hold and float(system.GetChTime()) <= lane_speed_hold_until else float(vel.x)
            chassis_body.SetPosDt(_chrono_vec(chrono, [vx_cmd, vy_cmd, vel.z]))
            if hasattr(chassis_body, "SetAngVelParent"):
                chassis_body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
        elif lane_speed_hold and float(system.GetChTime()) <= lane_speed_hold_until:
            vel = chassis_body.GetPosDt()
            chassis_body.SetPosDt(_chrono_vec(chrono, [float(target_vel[0]), vel.y, vel.z]))
        if step_idx == 0 or step_idx == total_steps or step_idx % stride == 0:
            times.append(float(system.GetChTime()))
            pos, yaw = _chrono_vehicle_pose(vehicle_ref)
            obj["positions"].append(pos)
            obj["scalars"]["yaw_deg"].append(yaw)
            lane_segments[0]["points_a"].append([road_x0, lane_y0, 0.01])
            lane_segments[0]["points_b"].append([road_x1, lane_y0, 0.01])
            lane_segments[1]["points_a"].append([road_x0, lane_y0 + 0.5 * lane_width, 0.01])
            lane_segments[1]["points_b"].append([road_x1, lane_y0 + 0.5 * lane_width, 0.01])
            lane_segments[2]["points_a"].append([road_x0, lane_y0 - 0.5 * lane_width, 0.01])
            lane_segments[2]["points_b"].append([road_x1, lane_y0 - 0.5 * lane_width, 0.01])
            if obstacle_obj is not None:
                obstacle_obj["positions"].append(list(obstacle_obj["_static_pos"]))
        if step_idx == total_steps:
            break
        sim_time = float(system.GetChTime())
        inputs = driver.GetInputs()
        driver.Synchronize(sim_time)
        wrapper.Synchronize(sim_time, inputs, terrain)
        terrain.Synchronize(sim_time)
        driver.Advance(step)
        wrapper.Advance(step)
        terrain.Advance(step)
        system.DoStepDynamics(step)

    objects: list[dict[str, Any]] = [obj] + lane_segments
    if obstacle_obj is not None:
        obstacle_obj.pop("_static_pos", None)
        objects.append(obstacle_obj)
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 34.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 25.0))
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    return {"times": times, "objects": objects, "scene": scene}
