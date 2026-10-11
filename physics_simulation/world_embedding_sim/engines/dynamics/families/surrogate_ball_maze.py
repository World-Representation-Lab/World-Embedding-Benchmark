from __future__ import annotations
from run_case import (Any, Path, _box_track, _case_name, _circle_track, _duration, _maze_wall_boxes, _mech_cfg, _polyline_arclength, _polyline_point_at, _polyline_points, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_ball_maze(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    ball = mech.get("ball") if isinstance(mech.get("ball"), dict) else {}
    radius = float(ball.get("radius", 0.16))
    mass = float(ball.get("mass", 0.8))
    guide_path = _polyline_points(
        mech.get("guide_path"),
        default=[[-8.0, -6.0], [-8.0, 2.0], [-3.4, 2.0], [-3.4, -1.0], [1.6, -1.0]],
    )
    wall_boxes = _maze_wall_boxes(
        guide_path,
        corridor_width=float(mech.get("corridor_width", 2.0)),
        wall_thickness=float(mech.get("wall_thickness", 0.24)),
        wall_height=float(mech.get("wall_height", 0.6)),
    )
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    _, total_len = _polyline_arclength(guide_path)

    positions = np.zeros((len(times), 3), dtype=float)
    yaw = np.zeros((len(times),), dtype=float)
    for i, t in enumerate(times):
        tau = float(t / max(float(times[-1]), 1.0e-6))
        eased = 0.5 - 0.5 * math.cos(math.pi * tau)
        point = _polyline_point_at(guide_path, eased * total_len)
        positions[i] = [float(point[0]), float(point[1]), radius]
        if i > 0:
            delta = positions[i, :2] - positions[i - 1, :2]
            yaw[i] = math.degrees(math.atan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1.0e-9 else yaw[i - 1]

    objects: list[dict[str, Any]] = []
    for idx, wall in enumerate(wall_boxes):
        wall_pos = np.repeat(np.asarray(wall["pos"], dtype=float).reshape(1, 3), len(times), axis=0)
        objects.append(_box_track(name=f"wall_{idx:02d}", size=wall["size"], mass=0.0, color=_rgb(0.34, 0.40, 0.52), positions=wall_pos))
    objects.append(_circle_track(name="ball", radius=radius, mass=mass, color=_rgb(0.18, 0.74, 0.86), positions=positions, scalars={"yaw_deg": yaw.tolist()}))
    return {"times": times.tolist(), "objects": objects, "scene": {"view_plane": "xy", "ground_value": None, "title": _case_name(case, Path("."))}}
