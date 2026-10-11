from __future__ import annotations
from run_case import (Any, Path, _box_track, _case_name, _duration, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_stack(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    levels = max(2, int(mech.get("levels", 4)))
    size = np.asarray(mech.get("box_size", [0.35, 0.35, 0.28]), dtype=float)
    box_mass = float(mech.get("box_mass", 2.0))
    gap = float(mech.get("gap", 0.0))
    impactor_cfg = mech.get("impactor") if isinstance(mech.get("impactor"), dict) else {}
    impactor_size = np.asarray(impactor_cfg.get("size", [0.2, 0.2, 0.2]), dtype=float)
    impactor_mass = float(impactor_cfg.get("mass", 1.0))
    impactor_speed = float(impactor_cfg.get("speed", 1.5))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    impactor_x0 = -3.2 * (size[0] + impactor_size[0])
    t_hit = max(0.15, (0.5 * (size[0] + impactor_size[0]) - impactor_x0) / max(impactor_speed, 1.0e-6))
    amp_top = 0.18 * impactor_speed * levels
    topple_strength = max(0.0, (amp_top - 0.55 * size[0]) / max(0.2 * size[0], 1.0e-6))

    objects: list[dict[str, Any]] = []
    for level in range(levels):
        positions = np.zeros((len(times), 3), dtype=float)
        x0 = 0.0
        z0 = 0.5 * size[2] + level * (size[2] + gap)
        for i, t in enumerate(times):
            if t <= t_hit:
                x = x0
                z = z0
            else:
                tau = t - t_hit
                amp = 0.14 * impactor_speed * math.exp(-0.38 * level)
                shake = amp * math.exp(-(1.0 + 0.18 * level) * tau) * math.sin((4.6 + 0.35 * level) * tau)
                drift = 0.0
                sag = 0.0
                if topple_strength > 0.0:
                    drift = topple_strength * (level + 1) / levels * 0.12 * tau * tau
                    sag = min(0.55 * size[2], topple_strength * (level + 1) / levels * 0.06 * tau * tau)
                x = x0 + shake + drift
                z = max(0.5 * size[2], z0 - sag)
            positions[i] = [x, 0.0, z]
        objects.append(
            _box_track(
                name=f"box_{level}",
                size=size.tolist(),
                mass=box_mass,
                color=_rgb(0.22 + 0.08 * level, 0.44, max(0.18, 0.78 - 0.08 * level)),
                positions=positions,
            )
        )

    impactor = np.zeros((len(times), 3), dtype=float)
    for i, t in enumerate(times):
        x = impactor_x0 + impactor_speed * min(t, t_hit)
        if t > t_hit:
            x += 0.2 * impactor_speed * (t - t_hit)
        impactor[i] = [x, 0.0, 0.5 * impactor_size[2]]
    objects.append(
        _box_track(
            name="impactor",
            size=impactor_size.tolist(),
            mass=impactor_mass,
            color=_rgb(0.9, 0.3, 0.25),
            positions=impactor,
        )
    )

    return {
        "times": times.tolist(),
        "objects": objects,
        "scene": {
            "view_plane": "xz",
            "ground_value": 0.0,
            "title": _case_name(case, Path(".")),
        },
    }
