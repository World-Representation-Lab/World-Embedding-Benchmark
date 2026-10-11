from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_density_for_sphere, _chrono_material, _chrono_vec, _duration, _env_cfg, _gravity, _mech_cfg, _rgb, _sample_count, _scene_bounds, _scene_oblique, _step_size, math)

def _simulate_chrono_granular_collapse(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))
    mech = _mech_cfg(case)
    env_ground = _env_cfg(case).get("ground") if isinstance(_env_cfg(case).get("ground"), dict) else {}
    mat = _chrono_material(chrono, friction=float(mech.get("friction", env_ground.get("friction", 0.28))), restitution=float(mech.get("restitution", 0.06)))
    if hasattr(system, "SetMaxPenetrationRecoverySpeed"):
        system.SetMaxPenetrationRecoverySpeed(2.5)

    ground_size = env_ground.get("size", [24.0, 16.0]) if isinstance(env_ground.get("size"), list) else [24.0, 16.0]
    ground = chrono.ChBodyEasyBox(float(ground_size[0]), float(ground_size[1]), 0.25, 1000.0, True, True, mat)
    ground.SetPos(_chrono_vec(chrono, [0.0, 0.0, -0.125]))
    ground.SetFixed(True)
    system.Add(ground)

    radius = float(mech.get("particle_radius", 0.15))
    mass = float(mech.get("particle_mass", 0.28))
    nx = max(2, int(mech.get("nx", 4)))
    ny = max(1, int(mech.get("ny", 2)))
    nz = max(2, int(mech.get("nz", 5)))
    spacing = float(mech.get("spacing", 0.015))
    release = float(mech.get("release_time", 0.25))
    box_width = float(mech.get("box_width", 2.5))
    retainer_x = float(mech.get("retainer_x", 0.15))
    retainer_height = float(mech.get("retainer_height", 1.5))

    side_y = 0.5 * box_width + radius
    wall_specs = [
        {"name": "wall_back", "size": [0.18, box_width + 2.0 * radius, retainer_height], "pos": [retainer_x - nx * (2.0 * radius + spacing) - 0.35, 0.0, 0.5 * retainer_height]},
        {"name": "wall_left", "size": [nx * (2.0 * radius + spacing) + 0.7, 0.18, retainer_height], "pos": [retainer_x - 0.5 * nx * (2.0 * radius + spacing), side_y, 0.5 * retainer_height]},
        {"name": "wall_right", "size": [nx * (2.0 * radius + spacing) + 0.7, 0.18, retainer_height], "pos": [retainer_x - 0.5 * nx * (2.0 * radius + spacing), -side_y, 0.5 * retainer_height]},
    ]
    wall_bodies: list[Any] = []
    wall_trackers: list[dict[str, Any]] = []
    for spec in wall_specs:
        wall = chrono.ChBodyEasyBox(float(spec["size"][0]), float(spec["size"][1]), float(spec["size"][2]), 1000.0, True, True, mat)
        wall.SetPos(_chrono_vec(chrono, spec["pos"]))
        wall.SetFixed(True)
        system.Add(wall)
        wall_bodies.append(wall)
        wall_trackers.append(
            {
                "name": str(spec["name"]),
                "kind": "box",
                "size": [float(spec["size"][0]), float(spec["size"][1]), float(spec["size"][2])],
                "mass": 0.0,
                "color": _rgb(0.42, 0.45, 0.50),
                "render_shape": "box",
                "color_mode": "fixed",
                "positions": [],
            }
        )
    retainer = chrono.ChBodyEasyBox(0.18, box_width + 2.0 * radius, retainer_height, 1000.0, True, True, mat)
    retainer.SetPos(_chrono_vec(chrono, [retainer_x, 0.0, 0.5 * retainer_height]))
    retainer.SetFixed(True)
    system.Add(retainer)
    show_retainer_visual = bool(mech.get("show_retainer_visual", True))
    retainer_tracker = {
        "name": "retainer_wall",
        "kind": "box",
        "size": [0.18, box_width + 2.0 * radius, retainer_height],
        "mass": 0.0,
        "color": _rgb(0.46, 0.50, 0.56),
        "render_shape": "box",
        "color_mode": "fixed",
        "positions": [],
    }

    grains: list[Any] = []
    trackers: list[dict[str, Any]] = []
    x0 = retainer_x - (nx - 0.5) * (2.0 * radius + spacing) - 0.2
    y0 = -0.5 * (ny - 1) * (2.0 * radius + spacing)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                pos = [x0 + ix * (2.0 * radius + spacing), y0 + iy * (2.0 * radius + spacing), radius + iz * (2.0 * radius + spacing)]
                grain = chrono.ChBodyEasySphere(radius, _chrono_density_for_sphere(mass, radius), True, True, mat)
                grain.SetPos(_chrono_vec(chrono, pos))
                system.Add(grain)
                grains.append(grain)
                trackers.append({"name": f"grain_{ix}_{iy}_{iz}", "kind": "circle", "size": [radius], "mass": mass, "color": _rgb(0.18 + 0.02 * iz, 0.56 + 0.03 * iy, 0.78 - 0.03 * ix), "positions": []})

    total_steps = max(2, int(math.ceil(_duration(case) / max(_step_size(case), 1.0e-6))))
    stride = max(1, total_steps // max(2, _sample_count(case) - 1))
    retainer_removed = False
    show_retainer_ref = bool(mech.get("show_retainer_ref", True))
    times: list[float] = []
    objects = list(trackers) + wall_trackers
    if show_retainer_visual:
        objects.append(retainer_tracker)
    show_boundary_ref = bool(mech.get("show_boundary_ref", True))
    boundary_refs: list[dict[str, Any]] = []
    if show_boundary_ref:
        back_x = float(wall_specs[0]["pos"][0])
        front_x = float(retainer_x + 0.78 * box_width)
        y_neg = float(-side_y)
        y_pos = float(side_y)
        for name, pa, pb in [
            ("boundary_back", [back_x, y_neg, 0.01], [back_x, y_pos, 0.01]),
            ("boundary_front", [front_x, y_neg, 0.01], [front_x, y_pos, 0.01]),
            ("boundary_left", [back_x, y_pos, 0.01], [front_x, y_pos, 0.01]),
            ("boundary_right", [back_x, y_neg, 0.01], [front_x, y_neg, 0.01]),
        ]:
            boundary_refs.append(
                {
                    "name": name,
                    "kind": "segment",
                    "color": _rgb(0.62, 0.66, 0.72),
                    "width": 1.2,
                    "color_mode": "fixed",
                    "points_a": [],
                    "points_b": [],
                    "_point_a": pa,
                    "_point_b": pb,
                }
            )
        objects.extend(boundary_refs)
    retainer_ref = {
        "name": "retainer_ref",
        "kind": "segment",
        "color": _rgb(0.46, 0.50, 0.56),
        "width": 1.4,
        "color_mode": "fixed",
        "points_a": [],
        "points_b": [],
    }
    if show_retainer_ref:
        objects.append(retainer_ref)

    for step_idx in range(total_steps + 1):
        t = float(system.GetChTime())
        if (not retainer_removed) and t >= release:
            if hasattr(system, "Remove"):
                system.Remove(retainer)
            else:
                retainer.EnableCollision(False)
                retainer.SetPos(_chrono_vec(chrono, [retainer_x + 1000.0, 0.0, 0.0]))
            retainer_removed = True
        if step_idx == 0 or step_idx == total_steps or step_idx % stride == 0:
            times.append(t)
            for out, grain in zip(objects[: len(grains)], grains):
                out["positions"].append([grain.GetPos().x, grain.GetPos().y, grain.GetPos().z])
            wall_start = len(grains)
            for wall_idx, wall in enumerate(wall_bodies):
                objects[wall_start + wall_idx]["positions"].append([wall.GetPos().x, wall.GetPos().y, wall.GetPos().z])
            if show_retainer_visual:
                retainer_tracker["positions"].append([retainer_x, 0.0, 0.5 * retainer_height])
            for seg in boundary_refs:
                seg["points_a"].append(list(seg["_point_a"]))
                seg["points_b"].append(list(seg["_point_b"]))
            if show_retainer_ref:
                if not retainer_removed:
                    retainer_ref["points_a"].append([retainer_x, -0.5 * box_width, 0.0])
                    retainer_ref["points_b"].append([retainer_x, 0.5 * box_width, 0.0])
                else:
                    retainer_ref["points_a"].append([retainer_x, -0.5 * box_width, 0.0])
                    retainer_ref["points_b"].append([retainer_x, 0.5 * box_width, 0.0])
        if step_idx == total_steps:
            break
        system.DoStepDynamics(_step_size(case))

    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 28.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 24.0))
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    zoom = float(mech.get("camera_zoom", 1.0))
    if zoom > 1.0 and isinstance(scene.get("xlim"), list) and isinstance(scene.get("ylim"), list):
        xlim = scene["xlim"]
        ylim = scene["ylim"]
        if len(xlim) >= 2 and len(ylim) >= 2:
            cx = 0.5 * (float(xlim[0]) + float(xlim[1]))
            cy = 0.5 * (float(ylim[0]) + float(ylim[1]))
            hx = 0.5 * (float(xlim[1]) - float(xlim[0])) / zoom
            hy = 0.5 * (float(ylim[1]) - float(ylim[0])) / zoom
            scene["xlim"] = [cx - hx, cx + hx]
            scene["ylim"] = [cy - hy, cy + hy]
    return {"times": times, "objects": objects, "scene": scene}
