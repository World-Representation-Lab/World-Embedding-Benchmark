from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_add_bmw_vehicle, _chrono_init_bmw_vehicle_scene, _chrono_vec, _chrono_vehicle_pose, _chrono_vehicle_visual_offset, _driver_sequence_entries, _env_cfg, _mech_cfg, _rgb, _sample_count, _scene_bounds, _vehicle_object_template, math, np)

def _simulate_chrono_ramp_jump(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    mech = _mech_cfg(case)
    veh, system, terrain, spec, step, duration, tire_enum, collision_enum = _chrono_init_bmw_vehicle_scene(case, chrono)
    ramp = mech.get("ramp") if isinstance(mech.get("ramp"), dict) else {}
    ramp_mat = chrono.ChContactMaterialNSC()
    if hasattr(ramp_mat, "SetFriction"):
        ramp_mat.SetFriction(float(_env_cfg(case).get("ground", {}).get("friction", 0.9) if isinstance(_env_cfg(case).get("ground"), dict) else 0.9))
    if hasattr(ramp_mat, "SetRestitution"):
        ramp_mat.SetRestitution(0.01)
    ramp_angle = math.radians(float(ramp.get("angle_deg", 12.0)))
    ramp_length = float(ramp.get("length", 5.0))
    ramp_width = float(ramp.get("width", 5.0))
    ramp_start_x = float(ramp.get("start_x", -6.0))
    ramp_height = float(ramp.get("height", 1.2))
    ramp_center_x = ramp_start_x + 0.5 * ramp_length * math.cos(ramp_angle)
    ramp_center_z = 0.5 * ramp_height
    terrain.AddPatch(
        ramp_mat,
        chrono.ChCoordsysd(chrono.ChVector3d(ramp_center_x, 0.0, ramp_center_z), chrono.QuatFromAngleY(-ramp_angle)),
        ramp_length,
        ramp_width,
    )
    terrain.Initialize()

    wrapper = _chrono_add_bmw_vehicle(veh, chrono, system, step, tire_enum, collision_enum, mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {})
    driver = veh.ChDataDriver(wrapper.GetVehicle(), _driver_sequence_entries(veh, mech.get("driver_sequence") if isinstance(mech.get("driver_sequence"), list) else [], duration))
    driver.Initialize()
    vehicle_ref = wrapper.GetVehicle()
    visual_offset = _chrono_vehicle_visual_offset(vehicle_ref)
    chassis_body = vehicle_ref.GetChassisBody()
    vehicle_cfg = mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {}
    target_vel = np.asarray(vehicle_cfg.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    pre_ramp_hold = bool(mech.get("pre_ramp_speed_hold", True))
    hold_until_x = float(mech.get("pre_ramp_hold_until_x", ramp_start_x - 0.85))
    total_steps = max(2, int(math.ceil(duration / max(step, 1.0e-6))))
    stride = max(1, total_steps // max(2, _sample_count(case) - 1))
    obj = _vehicle_object_template(
        "vehicle",
        spec,
        _rgb(0.20, 0.50, 0.82),
        "bmw_e90",
        visual_offset=visual_offset,
        wheel_width_scale=float(mech.get("wheel_width_scale", 1.0)),
    )
    ramp_profile = {
        "name": "ramp_profile",
        "kind": "segment",
        "color": _rgb(0.66, 0.62, 0.44),
        "width": 2.0,
        "color_mode": "fixed",
        "points_a": [],
        "points_b": [],
    }
    takeoff_x = float(ramp_start_x + ramp_length)
    landing_x = float(takeoff_x + float(mech.get("landing_target_dx", 8.0)))
    landing_ref = {
        "name": "landing_target",
        "kind": "segment",
        "color": _rgb(0.76, 0.40, 0.24),
        "width": 1.8,
        "color_mode": "fixed",
        "points_a": [],
        "points_b": [],
    }
    times: list[float] = []

    for step_idx in range(total_steps + 1):
        if pre_ramp_hold and float(chassis_body.GetPos().x) < hold_until_x:
            vel = chassis_body.GetPosDt()
            chassis_body.SetPosDt(_chrono_vec(chrono, [float(target_vel[0]), float(target_vel[1]), vel.z]))
            if hasattr(chassis_body, "SetAngVelParent"):
                chassis_body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
        if step_idx == 0 or step_idx == total_steps or step_idx % stride == 0:
            times.append(float(system.GetChTime()))
            pos, yaw = _chrono_vehicle_pose(vehicle_ref)
            obj["positions"].append(pos)
            obj["scalars"]["yaw_deg"].append(yaw)
            ramp_profile["points_a"].append([ramp_start_x, 0.0, 0.0])
            ramp_profile["points_b"].append([takeoff_x, 0.0, ramp_height])
            landing_ref["points_a"].append([landing_x, 0.0, 0.0])
            landing_ref["points_b"].append([landing_x, 0.0, max(0.6, 0.52 * ramp_height + 0.4)])
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

    objects: list[dict[str, Any]] = [obj, ramp_profile, landing_ref]
    scene = {"view_plane": "xz", "ground_value": 0.0, "title": _case_name(case, Path("."))}
    scene.update(_scene_bounds(objects, "xz"))
    return {"times": times, "objects": objects, "scene": scene}
