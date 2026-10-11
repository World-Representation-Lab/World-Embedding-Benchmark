from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_add_bmw_vehicle, _chrono_init_bmw_vehicle_scene, _chrono_vec, _chrono_vehicle_pose, _chrono_vehicle_visual_offset, _driver_sequence_entries, _env_cfg, _mech_cfg, _rgb, _sample_count, _scene_bounds, _scene_oblique, _vehicle_object_template, math, np)

def _simulate_chrono_vehicle_obstacle_crossing(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    mech = _mech_cfg(case)
    veh, system, terrain, spec, step, duration, tire_enum, collision_enum = _chrono_init_bmw_vehicle_scene(case, chrono)
    obs_mat = chrono.ChContactMaterialNSC()
    if hasattr(obs_mat, "SetFriction"):
        obs_mat.SetFriction(float(_env_cfg(case).get("ground", {}).get("friction", 0.9) if isinstance(_env_cfg(case).get("ground"), dict) else 0.9))
    if hasattr(obs_mat, "SetRestitution"):
        obs_mat.SetRestitution(0.01)

    obstacle_cfgs = mech.get("obstacles") if isinstance(mech.get("obstacles"), list) else []
    obstacle_bodies: list[dict[str, Any]] = []
    for obs in obstacle_cfgs:
        if not isinstance(obs, dict):
            continue
        size = obs.get("size", [0.8, 3.5, 0.12])
        pos = obs.get("position", [0.0, 0.0, 0.06])
        shape = str(obs.get("shape", "speed_bump")).strip().lower()
        if shape in {"speed_bump", "bump", "trapezoid"}:
            length = float(size[0])
            width = float(size[1])
            height = float(size[2])
            top_width = max(0.12 * length, min(float(obs.get("top_width", 0.28 * length)), 0.82 * length))
            x_center = float(pos[0])
            y_center = float(pos[1])
            ramp_len = max(0.0, 0.5 * (length - top_width))
            if ramp_len > 1.0e-4 and height > 1.0e-4:
                angle = math.atan2(height, ramp_len)
                up_x = x_center - 0.5 * top_width - 0.5 * ramp_len
                dn_x = x_center + 0.5 * top_width + 0.5 * ramp_len
                terrain.AddPatch(
                    obs_mat,
                    chrono.ChCoordsysd(_chrono_vec(chrono, [up_x, y_center, 0.5 * height]), chrono.QuatFromAngleY(-angle)),
                    ramp_len,
                    width,
                )
                terrain.AddPatch(
                    obs_mat,
                    chrono.ChCoordsysd(_chrono_vec(chrono, [dn_x, y_center, 0.5 * height]), chrono.QuatFromAngleY(angle)),
                    ramp_len,
                    width,
                )
            if top_width > 1.0e-4 and height > 1.0e-4:
                terrain.AddPatch(
                    obs_mat,
                    chrono.ChCoordsysd(_chrono_vec(chrono, [x_center, y_center, height]), chrono.QuatFromAngleY(0.0)),
                    top_width,
                    width,
                )
            obstacle_bodies.append(
                {
                    "shape": "speed_bump",
                    "parts": [],
                    "size": [length, width, height],
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "top_width": float(top_width),
                }
            )
            continue
        body = chrono.ChBodyEasyBox(float(size[0]), float(size[1]), float(size[2]), 1000.0, True, True, obs_mat)
        body.SetPos(_chrono_vec(chrono, pos))
        body.SetFixed(True)
        system.Add(body)
        obstacle_bodies.append({"shape": "box", "parts": [body], "size": [float(size[0]), float(size[1]), float(size[2])], "position": [float(pos[0]), float(pos[1]), float(pos[2])]})

    terrain.Initialize()

    wrapper = _chrono_add_bmw_vehicle(veh, chrono, system, step, tire_enum, collision_enum, mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {})
    driver = veh.ChDataDriver(wrapper.GetVehicle(), _driver_sequence_entries(veh, mech.get("driver_sequence") if isinstance(mech.get("driver_sequence"), list) else [], duration))
    driver.Initialize()
    vehicle_ref = wrapper.GetVehicle()
    visual_offset = _chrono_vehicle_visual_offset(vehicle_ref)
    chassis_body = vehicle_ref.GetChassisBody()
    vehicle_cfg = mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {}
    target_vel = np.asarray(vehicle_cfg.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    hold_enabled = bool(mech.get("pre_obstacle_speed_hold", True))
    hold_margin = float(mech.get("pre_obstacle_hold_margin", 1.35))
    explicit_hold_until_x = mech.get("pre_obstacle_hold_until_x")
    if isinstance(explicit_hold_until_x, (int, float)):
        hold_until_x = float(explicit_hold_until_x)
    elif obstacle_cfgs:
        first_obs = min(
            [obs for obs in obstacle_cfgs if isinstance(obs, dict)],
            key=lambda x: float((x.get("position") or [0.0])[0]),
        )
        first_pos = first_obs.get("position", [0.0, 0.0, 0.0]) if isinstance(first_obs.get("position"), list) else [0.0, 0.0, 0.0]
        first_size = first_obs.get("size", [0.8, 3.5, 0.12]) if isinstance(first_obs.get("size"), list) else [0.8, 3.5, 0.12]
        hold_until_x = float(first_pos[0]) - 0.5 * float(first_size[0]) - hold_margin
    else:
        hold_until_x = float("inf")
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
    obstacles: list[dict[str, Any]] = []
    for idx, obs_data in enumerate(obstacle_bodies):
        obstacles.append(
            {
                "name": f"obstacle_{idx:02d}",
                "kind": "box",
                "size": list(obs_data.get("size", [0.8, 3.5, 0.12])),
                "mass": 0.0,
                "color": _rgb(0.44, 0.46, 0.50),
                "positions": [],
                "render_shape": "speed_bump" if str(obs_data.get("shape", "box")) == "speed_bump" else None,
                "top_width": float(obs_data.get("top_width", 0.0)),
            }
        )
    times: list[float] = []

    for step_idx in range(total_steps + 1):
        if hold_enabled and float(chassis_body.GetPos().x) < hold_until_x:
            vel = chassis_body.GetPosDt()
            chassis_body.SetPosDt(_chrono_vec(chrono, [float(target_vel[0]), float(target_vel[1]), vel.z]))
            if hasattr(chassis_body, "SetAngVelParent"):
                chassis_body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
        if step_idx == 0 or step_idx == total_steps or step_idx % stride == 0:
            times.append(float(system.GetChTime()))
            pos, yaw = _chrono_vehicle_pose(vehicle_ref)
            obj["positions"].append(pos)
            obj["scalars"]["yaw_deg"].append(yaw)
            for out, ref in zip(obstacles, obstacle_bodies):
                out["positions"].append(list(ref.get("position", [0.0, 0.0, 0.0])))
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

    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 34.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 26.0))
    objects = [obj] + obstacles
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    return {"times": times, "objects": objects, "scene": scene}
