from __future__ import annotations
from run_case import (Any, Path, _box_track, _case_name, _duration, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_block_tower(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    levels = max(3, int(mech.get("levels", 5)))
    layers = max(2, int(mech.get("layers", 3)))
    size = np.asarray(mech.get("block_size", [1.1, 0.28, 0.22]), dtype=float)
    gap = float(mech.get("gap", 0.0))
    block_mass = float(mech.get("block_mass", 1.2))
    impactor_cfg = mech.get("impactor") if isinstance(mech.get("impactor"), dict) else {}
    impactor_size = np.asarray(impactor_cfg.get("size", [0.36, 0.36, 0.28]), dtype=float)
    impactor_mass = float(impactor_cfg.get("mass", 2.5))
    impactor_speed = float(impactor_cfg.get("speed", 2.0))
    impactor_y = float(impactor_cfg.get("y_offset", 0.0))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    hit_time = 0.65

    objects: list[dict[str, Any]] = []
    for level in range(levels):
        for layer in range(layers):
            positions = np.zeros((len(times), 3), dtype=float)
            orient_x = (level % 2) == 0
            spacing = (size[1] if orient_x else size[0]) * 1.05
            x0 = (layer - 0.5 * (layers - 1)) * (size[0] if orient_x else size[1]) * 0.72
            y0 = 0.0 if orient_x else (layer - 0.5 * (layers - 1)) * spacing
            if orient_x:
                y0 = (layer - 0.5 * (layers - 1)) * spacing
            z0 = 0.5 * size[2] + level * (size[2] + gap)
            amp = 0.08 * impactor_speed * math.exp(-0.12 * level) * (1.0 + 0.15 * abs(y0))
            drift_y = 0.0 if orient_x else 0.06 * (level + 1)
            for i, t in enumerate(times):
                if t <= hit_time:
                    positions[i] = [x0, y0, z0]
                else:
                    tau = t - hit_time
                    shake_x = amp * math.exp(-0.9 * tau) * math.sin((5.0 + 0.25 * level) * tau)
                    shake_y = 0.55 * amp * math.exp(-0.7 * tau) * math.sin((4.0 + 0.18 * layer) * tau)
                    topple = max(0.0, impactor_speed - 1.2) * 0.02 * (level + 1) * tau * tau
                    positions[i] = [x0 + shake_x, y0 + shake_y + drift_y * tau, max(0.5 * size[2], z0 - topple)]
            objects.append(
                _box_track(
                    name=f"tower_block_{level}_{layer}",
                    size=size.tolist() if orient_x else [float(size[1]), float(size[0]), float(size[2])],
                    mass=block_mass,
                    color=_rgb(0.26 + 0.04 * level, 0.42, max(0.18, 0.72 - 0.05 * level)),
                    positions=positions,
                )
            )

    impactor = np.zeros((len(times), 3), dtype=float)
    x_start = -8.0
    for i, t in enumerate(times):
        x = x_start + impactor_speed * min(t, hit_time)
        if t > hit_time:
            x += 0.18 * impactor_speed * (t - hit_time)
        impactor[i] = [x, impactor_y, 0.5 * impactor_size[2] + size[2]]
    objects.append(
        _box_track(name="impactor", size=impactor_size.tolist(), mass=impactor_mass, color=_rgb(0.9, 0.3, 0.22), positions=impactor)
    )

    return {"times": times.tolist(), "objects": objects, "scene": {"view_plane": "xz", "ground_value": 0.0, "title": _case_name(case, Path("."))}}
