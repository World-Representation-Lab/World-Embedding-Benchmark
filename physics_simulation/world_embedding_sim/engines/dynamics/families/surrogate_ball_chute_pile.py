from __future__ import annotations
from run_case import (Any, Path, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _scene_bounds, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_ball_chute_pile(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    n_balls = max(10, int(mech.get("ball_count", 120)))
    radius = max(0.01, float(mech.get("ball_radius", 0.04)))
    radius_jitter = max(0.0, float(mech.get("radius_jitter", 0.12)))
    spawn_dt = max(0.005, float(mech.get("spawn_interval_s", 0.04)))
    chute_angle = math.radians(float(mech.get("chute_angle_deg", 45.0)))
    rng = np.random.default_rng(int(mech.get("random_seed", int(case.get("seed", 0)))))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    render_limit = max(1, min(n_balls, int(mech.get("render_ball_limit", n_balls))))

    top = np.asarray(mech.get("spawn_top", [-0.95, 0.0, 1.20]), dtype=float)
    speed0 = max(1.5, float(mech.get("chute_exit_speed_m_s", 2.8)))
    g = abs(float(_gravity(case)[2]))
    objects: list[dict[str, Any]] = []

    for i in range(min(render_limit, n_balls)):
        ri = max(0.6 * radius, radius * (1.0 + radius_jitter * float(rng.uniform(-1.0, 1.0))))
        t_spawn = i * spawn_dt
        pos = np.zeros((len(times), 3), dtype=float)
        speed = np.zeros((len(times),), dtype=float)
        contact_mag = np.zeros((len(times),), dtype=float)
        in_flight = np.zeros((len(times),), dtype=float)
        y = float(top[1] + rng.uniform(-0.08, 0.08))
        for k, t in enumerate(times):
            if t < t_spawn:
                pos[k] = [top[0], y, top[2]]
                continue
            tau = t - t_spawn
            t_roll = 0.42
            if tau <= t_roll:
                s = tau / max(t_roll, 1.0e-9)
                x = top[0] + (math.cos(chute_angle) * 1.08) * s
                z = top[2] - (math.sin(chute_angle) * 1.08) * s
                vx = (math.cos(chute_angle) * 1.08) / max(t_roll, 1.0e-9)
                vz = -(math.sin(chute_angle) * 1.08) / max(t_roll, 1.0e-9)
                pos[k] = [x, y, z]
                speed[k] = float(math.hypot(vx, vz))
                contact_mag[k] = 12.0 + 8.0 * math.sin(8.0 * s)
                in_flight[k] = 0.0
            else:
                tf = tau - t_roll
                x0 = top[0] + math.cos(chute_angle) * 1.08
                z0 = top[2] - math.sin(chute_angle) * 1.08
                vx0 = speed0 * math.cos(chute_angle)
                vz0 = speed0 * math.sin(chute_angle)
                x = x0 + vx0 * tf
                z = z0 + vz0 * tf - 0.5 * g * tf * tf
                floor_z = ri + 0.02
                if z <= floor_z:
                    z = floor_z
                    vx0 *= 0.55
                    vz0 *= -0.35
                    speed[k] = abs(vx0)
                    contact_mag[k] = 42.0 + 16.0 * math.sin(25.0 * tf)
                    in_flight[k] = 0.0
                else:
                    speed[k] = float(math.hypot(vx0, vz0 - g * tf))
                    contact_mag[k] = 0.0
                    in_flight[k] = 1.0
                pos[k] = [x, y, z]

        objects.append(
            _circle_track(
                name=f"ball_{i:03d}",
                radius=float(ri),
                mass=float((4.0 / 3.0) * math.pi * ri**3 * 2500.0),
                color=_rgb(0.28, 0.65, 0.86),
                positions=pos,
                scalars={
                    "speed_mag": [float(v) for v in speed],
                    "contact_force_mag": [float(v) for v in contact_mag],
                    "in_flight": [float(v) for v in in_flight],
                    "visual_scalar": [float(0.30 * speed[j] + (10.0 if in_flight[j] < 0.5 else 0.0) + 0.12 * contact_mag[j]) for j in range(len(times))],
                },
                extras={"color_metric": "visual_scalar"},
            )
        )

    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 36.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 25.0))
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    zoom = float(mech.get("camera_zoom", 1.85))
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
    return {"times": times.tolist(), "objects": objects, "scene": scene}
