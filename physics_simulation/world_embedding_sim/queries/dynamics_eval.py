"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

import re

from typing import Any, Mapping

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

FAMILY_SIGNATURE_KEYS: dict[str, tuple[str, ...]] = {
    "articulatedChain": ("drive_mode_bin", "link_count_bin", "terminal_mass_bin"),
    "blockTower3D": ("impactor_row_bin", "tower_height_bin", "tower_layers_bin"),
    "cantileverBeamTransient": ("beam_aspect_bin", "load_duration_bin", "tip_load_bin"),
    "dropWeightImpactSMC": ("drop_height_bin", "plate_size_bin", "sphere_radius_bin"),
    "flexibleCrankSlider": ("crank_radius_bin", "rod_flex_bin", "slider_axis_bin"),
    "gearPair3D": ("gear_count_bin", "motor_profile_bin", "tooth_ratio_bin"),
    "harmonicResponseFRF": ("drive_frequency_bin", "forcing_bin", "sweep_band_bin"),
    "j2PlasticBeam": (
        "beam_length_bin",
        "plastic_strain_bin",
        "residual_deflection_bin",
    ),
    "lemaitreDamageBar": (
        "cyclic_amplitude_bin",
        "damage_cycle_bin",
        "notch_width_bin",
    ),
    "massSpringDamper": ("excitation_bin", "initial_condition_bin", "layout_bin"),
    "pendulum": ("initial_angle_bin", "link_length_bin", "mechanism_variant"),
    "reissnerShellModal": (
        "mode_amplitude_bin",
        "mode_pattern_bin",
        "plate_aspect_bin",
    ),
    "rigidCollision": ("body_size_bin", "impact_offset_bin", "impact_speed_bin"),
    "rollingSlidingTransition": (
        "body_inertia_bin",
        "initial_spin_bin",
        "sliding_phase_bin",
        "track_slope_bin",
    ),
    "rubberBlockCompression": (
        "block_aspect_bin",
        "compression_bin",
        "rubber_stiffness_bin",
    ),
    "sphereDrop3D": ("drop_height_bin", "lateral_velocity_bin", "sphere_radius_bin"),
    "stackStability": ("block_size_bin", "impactor_speed_bin", "stack_height_bin"),
    "stressWavePropagation": (
        "pulse_duration_bin",
        "rod_diameter_bin",
        "rod_length_bin",
    ),
    "unbalancedRotor3D": ("bearing_span_bin", "eccentricity_bin", "speed_regime_bin"),
}

FAMILY_FIXED_MECHANISMS = {
    "blockTower3D": "block_tower_3d",
    "cantileverBeamTransient": "cantilever_beam_transient",
    "dropWeightImpactSMC": "drop_weight_impact_smc",
    "flexibleCrankSlider": "flexible_crank_slider",
    "gearPair3D": "gear_pair_3d",
    "harmonicResponseFRF": "harmonic_response_frf",
    "j2PlasticBeam": "j2_plastic_beam",
    "lemaitreDamageBar": "lemaitre_damage_bar",
    "reissnerShellModal": "reissner_shell_modal",
    "rigidCollision": "rigid_collision",
    "rubberBlockCompression": "rubber_block_compression",
    "sphereDrop3D": "sphere_drop_3d",
    "stackStability": "stack_stability",
    "stressWavePropagation": "stress_wave_propagation",
}

OBSERVABLE_PATTERNS: dict[str, str] = {
    "max_speed": rf"fastest visible motion reaches about ({NUMBER}) m/s",
    "max_height": rf"trajectory or deformation reaches a height near ({NUMBER}) m",
    "max_angle_deg": (
        rf"(?:gears rotate through roughly|swing reaches an angular excursion near|"
        rf"mechanism rotates through about) ({NUMBER}) degrees"
    ),
    "max_relative_angle_deg": rf"linked bodies reach a relative angle near ({NUMBER}) degrees",
    "impact_speed": rf"impact occurs at about ({NUMBER}) m/s",
    "post_impact_speed": rf"post-impact motion remains near ({NUMBER}) m/s",
    "contact_events": rf"scene contains about ({NUMBER}) contact",
    "settling_time": rf"motion settles after roughly ({NUMBER}) s",
    "roll_capture_time": rf"rolling behavior is captured after roughly ({NUMBER}) s",
    "slip_distance": rf"sliding travel is about ({NUMBER}) m",
    "roll_distance": rf"rolling travel is about ({NUMBER}) m",
    "response_amplitude_m": rf"mass-spring response amplitude is about ({NUMBER}) m",
    "duration_sec": rf"recorded motion covers roughly ({NUMBER}) s",
}


def _extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _label(key: str) -> str:
    return re.sub(r"[-_]+", " ", key).strip()


def _parse_signature(family: str, text: str) -> dict[str, str]:
    segment = _extract(r"visible setup is distinguished by (.*?), colors encode", text)
    if segment is None:
        return {}
    parts = [
        part.strip()
        for part in segment.replace(", and ", ", ").replace(" and ", ", ").split(",")
        if part.strip()
    ]
    keys = FAMILY_SIGNATURE_KEYS[family]
    signature: dict[str, str] = {}
    for part in parts:
        matches = [
            key for key in keys if part.lower().startswith(_label(key).lower() + " ")
        ]
        if len(matches) != 1:
            raise ValueError(f"{family}: cannot parse signature component {part!r}")
        key = matches[0]
        signature[key] = part[len(_label(key)) :].strip()
    missing = sorted(set(keys) - set(signature))
    if missing:
        raise ValueError(f"{family}: missing signature keys {missing}")
    return signature


def case_from_eval_row(family: str, row: Mapping[str, Any]) -> dict[str, Any]:
    raw_text = str(row.get("raw_text") or "")
    case_id = str(row.get("case_id") or "")
    dim = _extract(r"the view and geometry are ([23]D)", raw_text)
    mechanism = _extract(r"the motion system is ([^,]+)", raw_text)
    if dim is None or mechanism is None:
        raise ValueError(f"{case_id}: missing dimension or mechanism")

    observables: dict[str, Any] = {}
    for key, pattern in OBSERVABLE_PATTERNS.items():
        value = _extract(pattern, raw_text)
        if value is not None:
            observables[key] = float(value)
    stack_state = _extract(r"final stack state is ([^,.]+)", raw_text)
    if stack_state is not None:
        observables["stack_toppled"] = stack_state.lower() in {
            "true",
            "toppled",
            "collapsed",
        }
    if "contact_events" in observables:
        observables["contact_events"] = int(round(observables["contact_events"]))

    # The fifth sentence uses settling_time as its displayed
    # duration. Families without a separate settling metric use clip duration.
    duration = observables.pop("duration_sec", None)
    if "settling_time" not in observables and duration is not None:
        observables["settling_time"] = duration

    if "Earth-scale gravity acts downward" in raw_text:
        gravity = [0.0, 0.0, -9.81]
    elif "gravity is absent" in raw_text:
        gravity = [0.0, 0.0, 0.0]
    else:
        raise ValueError(f"{case_id}: unsupported gravity phrase")
    ground_enabled = (
        "ground contact or support is part of the visible boundary condition"
        in raw_text
    )

    return {
        "case_id": case_id,
        "family": family,
        "params": {
            "dim": dim,
            "mechanism_type": FAMILY_FIXED_MECHANISMS.get(family, mechanism),
            "visual_signature": _parse_signature(family, raw_text),
            "gravity": gravity,
            "ground_enabled": ground_enabled,
            "duration_sec": duration,
        },
        "observables": observables,
    }
