"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

from dataclasses import dataclass

import re

from typing import Any, Mapping


@dataclass(frozen=True)
class FamilyNarrative:
    scene: str
    physics: str
    field_sentence: str


FAMILY_NARRATIVES: dict[str, FamilyNarrative] = {
    "articulatedChain": FamilyNarrative(
        "a chain of hinged rigid links swinging under gravity or a root drive",
        "joint constraints transmit motion along the chain, creating phase lag, folding, or free-swing response",
        "The color map represents link and joint speed magnitude along the chain, so speed colors show which parts move fastest.",
    ),
    "blockTower3D": FamilyNarrative(
        "a block tower struck by a moving impactor",
        "the impactor transfers momentum into the tower, so contact forces and gravity decide whether blocks slide or fall",
        "The color map represents block speed magnitude during tower impact, so speed colors show which parts move fastest.",
    ),
    "cantileverBeamTransient": FamilyNarrative(
        "a cantilever beam bending after a transient tip load",
        "the beam stores elastic strain energy and oscillates after the load change",
        "The color map represents beam displacement magnitude during transient bending, so colors show where deformation or motion is largest.",
    ),
    "dropWeightImpactSMC": FamilyNarrative(
        "a sphere dropping onto a deforming plate",
        "the falling sphere transfers kinetic energy into plate deformation and rebound motion",
        "The color map represents impact speed and plate response field, so speed colors show which parts move fastest.",
    ),
    "flexibleCrankSlider": FamilyNarrative(
        "a crank-slider mechanism with a visibly flexible connecting rod",
        "the crank input drives slider motion while rod flexibility creates visible bending and phase lag",
        "The color map represents mechanism speed magnitude and flexible rod motion, so speed colors show which parts move fastest.",
    ),
    "gearPair3D": FamilyNarrative(
        "interlocking spur gears driven by an angular input",
        "gear constraints transmit angular motion through meshing teeth, so angular speeds follow the gear ratio",
        "The color map represents absolute angular speed on opaque interlocking gears, so colors show the angular-rate state of the rotating body.",
    ),
    "harmonicResponseFRF": FamilyNarrative(
        "a structure vibrating under harmonic forcing near resonance",
        "the forcing frequency controls the vibration amplitude, with larger response near structural resonance",
        "The color map represents harmonic displacement response near resonance, so colors show where deformation or motion is largest.",
    ),
    "j2PlasticBeam": FamilyNarrative(
        "a beam undergoing plastic bending and residual deformation",
        "yielding localizes near high bending stress, leaving permanent deformation after unloading",
        "The color map represents plastic deformation and displacement response, so colors show where deformation or motion is largest.",
    ),
    "lemaitreDamageBar": FamilyNarrative(
        "a notched bar where damage localizes under cyclic loading",
        "plastic straining and damage concentrate around the notch as the cyclic response evolves",
        "The color map represents damage or plastic strain localization field, so colors show where material degradation is strongest.",
    ),
    "massSpringDamper": FamilyNarrative(
        "sliding blocks oscillating through springs and dampers",
        "spring stiffness and damping set the oscillation frequency, decay rate, and relative block motion",
        "The color map represents sliding block speed magnitude in a mass spring damper chain, so speed colors show which parts move fastest.",
    ),
    "pendulum": FamilyNarrative(
        "one or more linked bobs swinging under gravity",
        "gravity and joint constraints exchange potential and kinetic energy, producing single-, double-, or triple-link swing patterns",
        "The color map represents bob speed magnitude on moving pendulum bodies, so speed colors show which parts move fastest.",
    ),
    "reissnerShellModal": FamilyNarrative(
        "a shell or plate vibrating in a standing mode shape",
        "the visible deformation corresponds to a structural vibration mode with nodal lines and antinodes",
        "The color map represents shell mode displacement magnitude, so colors show where deformation or motion is largest.",
    ),
    "rigidCollision": FamilyNarrative(
        "rigid bodies translating, rotating, and separating after contact",
        "linear and angular momentum determine the post-impact translation and rotation after a nearly inelastic contact",
        "The color map represents rigid body speed magnitude during contact, so speed colors show which parts move fastest.",
    ),
    "rollingSlidingTransition": FamilyNarrative(
        "a roller on an inclined track transitioning between slip and rolling",
        "friction changes translational and angular velocity until slip reduces or rolling dominates",
        "The color map represents rolling body surface point speed during the sliding to rolling transition, so speed colors show which parts move fastest.",
    ),
    "rubberBlockCompression": FamilyNarrative(
        "a rubber-like block undergoing compression and bulging",
        "large-strain elastic deformation produces compression, recovery, and lateral bulging",
        "The color map represents large deformation displacement magnitude, so colors show where deformation or motion is largest.",
    ),
    "sphereDrop3D": FamilyNarrative(
        "a sphere falling, impacting the ground, and rebounding",
        "gravity accelerates the sphere downward, contact reverses part of the velocity, and damping reduces later rebounds",
        "The color map represents sphere speed magnitude through fall and rebound, so speed colors show which parts move fastest.",
    ),
    "stackStability": FamilyNarrative(
        "stacked blocks responding to an imposed disturbance",
        "friction and gravity determine whether the stacked bodies settle, slide, or topple after the disturbance",
        "The color map represents block speed magnitude during stack response, so speed colors show which parts move fastest.",
    ),
    "stressWavePropagation": FamilyNarrative(
        "a rod carrying a traveling and reflecting stress front",
        "an axial stress pulse travels along the rod and reflects at the boundary",
        "The color map represents axial stress wave magnitude along the rod, so colors show the stress concentration and wave front.",
    ),
    "unbalancedRotor3D": FamilyNarrative(
        "an unbalanced rotor whose vibration changes with spin speed",
        "mass eccentricity produces bearing reaction and whirl that vary with spin speed relative to critical speed",
        "The color map represents lateral displacement gradient during unbalanced rotor run up or coast down through critical speed, so colors show where deformation or motion is largest.",
    ),
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _plain_token(value: Any, fallback: str = "unknown") -> str:
    text = re.sub(r"[-_]+", " ", str(value or "")).strip().lower()
    return text or fallback


def _display_token(value: Any, fallback: str = "unknown") -> str:
    text = re.sub(r"[-_]+", " ", str(value or "")).strip()
    return text or fallback


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _speed(value: Any) -> str:
    number = _number(value)
    return f"{number:.2f}" if abs(number) >= 10.0 else f"{number:.3f}"


def _angle(value: Any) -> str:
    number = _number(value)
    return f"{number:.1f}" if abs(number) >= 100.0 else f"{number:.2f}"


def _case_dimension(case: Mapping[str, Any]) -> str:
    params = _mapping(case.get("params"))
    scene = _mapping(case.get("scene_spec"))
    value = params.get("dim") or case.get("dimension") or scene.get("dim") or "3D"
    text = str(value).strip().upper()
    return text if text in {"2D", "3D"} else str(value).strip()


def _mechanism(case: Mapping[str, Any], family: str) -> str:
    params = _mapping(case.get("params"))
    scene = _mapping(case.get("scene_spec"))
    mechanism = _mapping(scene.get("mechanism"))
    value = (
        params.get("mechanism_type")
        or case.get("scenario")
        or mechanism.get("type")
        or family
    )
    return _display_token(value)


def _signature(case: Mapping[str, Any]) -> Mapping[str, Any]:
    params = _mapping(case.get("params"))
    value = params.get("visual_signature")
    if isinstance(value, Mapping) and value:
        return value
    scene = _mapping(case.get("scene_spec"))
    variation = _mapping(scene.get("visual_variation"))
    value = variation.get("signature")
    return value if isinstance(value, Mapping) else {}


def _gravity(case: Mapping[str, Any]) -> list[float]:
    params = _mapping(case.get("params"))
    value = params.get("gravity")
    if not isinstance(value, (list, tuple)):
        scene = _mapping(case.get("scene_spec"))
        environment = _mapping(scene.get("environment"))
        value = environment.get("gravity")
    if not isinstance(value, (list, tuple)):
        return []
    return [_number(item) for item in value]


def _ground_enabled(case: Mapping[str, Any]) -> bool:
    params = _mapping(case.get("params"))
    if "ground_enabled" in params:
        return bool(params["ground_enabled"])
    scene = _mapping(case.get("scene_spec"))
    environment = _mapping(scene.get("environment"))
    ground = _mapping(environment.get("ground"))
    return bool(ground.get("enabled", False))


def _signature_phrase(key: str, value: Any) -> str:
    label = _plain_token(str(key).removesuffix("_bin"))
    shown = _display_token(value)
    return f"{label} set to {shown}"


def _variant_sentence(case: Mapping[str, Any]) -> str:
    signature = _signature(case)
    parts = [_signature_phrase(key, signature[key]) for key in sorted(signature)]
    if not parts:
        variant = "the recorded geometry and initial-condition setting"
    elif len(parts) == 1:
        variant = parts[0]
    else:
        variant = ", ".join(parts)
    force = (
        "Earth-scale gravity acts downward"
        if any(abs(value) > 1.0 for value in _gravity(case))
        else "the configured force field drives the motion"
    )
    ground = (
        "a ground or support boundary is visible"
        if _ground_enabled(case)
        else "no separate ground plane is part of the visible setup"
    )
    return f"Use the variant with {variant}; {force}; {ground}."


_DETAIL_TEMPLATES = {
    "max_angle": lambda o, c: f"the mechanism rotates through about {_angle(o.get('max_angle_deg'))} degrees",
    "gear_angle": lambda o, c: f"the mechanism rotates through about {_number(o.get('max_angle_deg')):.1f} degrees",
    "relative_angle": lambda o, c: f"linked bodies reach a relative angle near {_angle(o.get('max_relative_angle_deg'))} degrees",
    "height": lambda o, c: f"the visible height or deformation reaches about {_number(o.get('max_height')):.3f} m",
    "duration": lambda o, c: f"the recorded motion covers roughly {_number(o.get('settling_time')):.2f} s",
    "speed": lambda o, c: f"the fastest visible motion reaches about {_speed(o.get('max_speed'))} m/s",
    "impact": lambda o, c: f"impact occurs at about {_speed(o.get('impact_speed'))} m/s",
    "post_impact": lambda o, c: f"post-impact motion remains near {_speed(o.get('post_impact_speed'))} m/s",
    "stack": lambda o, c: f"the final stack state is {bool(o.get('stack_toppled'))}",
    "contacts": lambda o, c: f"the clip records about {int(_number(o.get('contact_events')))} contact events",
    "slip": lambda o, c: f"the sliding travel is about {_speed(o.get('slip_distance'))} m",
    "roll": lambda o, c: f"the rolling travel is about {_speed(o.get('roll_distance'))} m",
    "amplitude": lambda o, c: f"the response amplitude is about {_number(o.get('response_amplitude_m')):.4f} m",
}

_FAMILY_DETAILS = {
    "articulatedChain": ["max_angle", "relative_angle", "speed", "duration"],
    "blockTower3D": ["contacts", "stack", "speed", "duration"],
    "cantileverBeamTransient": ["height", "speed", "duration"],
    "dropWeightImpactSMC": ["impact", "speed", "duration"],
    "flexibleCrankSlider": ["max_angle", "speed", "duration"],
    "gearPair3D": ["gear_angle", "speed"],
    "harmonicResponseFRF": ["height", "speed", "duration"],
    "j2PlasticBeam": ["height", "speed", "duration"],
    "lemaitreDamageBar": ["height", "speed", "duration"],
    "massSpringDamper": ["amplitude", "speed", "duration"],
    "pendulum": ["max_angle", "relative_angle", "speed", "duration"],
    "reissnerShellModal": ["height", "speed", "duration"],
    "rigidCollision": ["impact", "post_impact", "speed", "duration"],
    "rollingSlidingTransition": ["slip", "roll", "speed", "duration"],
    "rubberBlockCompression": ["height", "speed", "duration"],
    "sphereDrop3D": ["impact", "post_impact", "speed", "duration"],
    "stackStability": ["contacts", "stack", "speed", "duration"],
    "stressWavePropagation": ["height", "speed"],
    "unbalancedRotor3D": ["amplitude", "height", "speed", "duration"],
}


def _observable_sentence(case: Mapping[str, Any], family: str) -> str:
    obs = _mapping(case.get("observables"))
    keys = _FAMILY_DETAILS.get(family, ["speed", "duration"])
    source_fields = {
        "max_angle": "max_angle_deg",
        "gear_angle": "max_angle_deg",
        "relative_angle": "max_relative_angle_deg",
        "height": "max_height",
        "duration": "settling_time",
        "speed": "max_speed",
        "impact": "impact_speed",
        "post_impact": "post_impact_speed",
        "stack": "stack_toppled",
        "contacts": "contact_events",
        "slip": "slip_distance",
        "roll": "roll_distance",
        "amplitude": "response_amplitude_m",
    }
    keys = [
        key
        for key in keys
        if source_fields.get(key) in obs and obs.get(source_fields[key]) is not None
        if not (key == "duration" and _number(obs.get("settling_time")) <= 0.0)
        and not (
            key == "relative_angle"
            and abs(_number(obs.get("max_relative_angle_deg"))) <= 0.0
        )
    ]
    details = [_DETAIL_TEMPLATES[k](obs, case) for k in keys]
    body = ", ".join(details)
    return f"To identify the exact video, look for {body}."


def generate_template_text(case: Mapping[str, Any]) -> str:
    family = str(case.get("family") or "").strip()
    narrative = FAMILY_NARRATIVES.get(family)
    if narrative is None:
        raise KeyError(f"No parsed-text narrative registered for family {family!r}")
    sentences = [
        (
            f"Find the {_case_dimension(case)} dynamics video of "
            f"{_mechanism(case, family)}, showing {narrative.scene}."
        ),
        f"Physically, {narrative.physics}.",
        narrative.field_sentence,
        _variant_sentence(case),
        _observable_sentence(case, family),
    ]
    return " ".join(sentences)
