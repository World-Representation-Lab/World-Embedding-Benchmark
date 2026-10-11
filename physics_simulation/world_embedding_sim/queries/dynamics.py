"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

import json

import re

from pathlib import Path

from typing import Any, Optional

FORBIDDEN_QUERY_TERMS = (
    "solver",
    "backend",
    "mesh",
    "grid",
    "element",
    "timestep",
    "step size",
    "chrono",
    "pychrono",
    "pardiso",
    "sparseqr",
    "nsc",
    "smc",
)

IMPLEMENTATION_DETAIL_KEYS = {
    "backend",
    "solver",
    "solver_name",
    "mesh",
    "mesh_resolution",
    "mesh_density",
    "node_count",
    "grid",
    "element",
    "element_type",
    "internal_timestep",
    "timestep",
    "step_size",
    "manual_contact_patch",
}


def _case_dim(meta: dict[str, Any]) -> str:
    dim = str(meta.get("dim", "")).strip().upper()
    return dim if dim in {"2D", "3D"} else "2D"


def _video_for_case(case_dir: Path) -> Optional[str]:
    case_name = case_dir.name
    video_dir = case_dir.parents[2] / "videos"
    cand = video_dir / f"{case_name}_dyn.mp4"
    return str(cand) if cand.exists() else None


def _fmt(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}g}"
    except Exception:
        return "n/a"


def _usable(v: Any) -> bool:
    s = str(v).strip().lower()
    return s not in {"", "n/a", "none", "null", "unknown", "not recorded", "[]", "{}"}


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _style_index(key: str, mod: int) -> int:
    base = sum((i + 1) * ord(ch) for i, ch in enumerate(str(key)))
    return base % max(1, int(mod))


def _pick(options: list[str], style: int) -> str:
    return options[int(style) % len(options)]


def _humanize_name(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return "dynamics case"
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", s)
    s = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", s)
    s = re.sub(r"\b([23])\s*d\b", lambda m: f"{m.group(1)}D", s, flags=re.IGNORECASE)
    s = re.sub(r"\bj\s*2\b", "J2", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def _is_impl_detail_key(key: Any) -> bool:
    text = str(key or "").strip().lower().replace("-", "_")
    if text in IMPLEMENTATION_DETAIL_KEYS:
        return True
    return any(
        tok in text
        for tok in (
            "solver",
            "backend",
            "mesh",
            "grid",
            "element",
            "timestep",
            "step_size",
        )
    )


def _clean_query_fragment(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else fallback).strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".")


def _public_label(value: Any) -> str:
    text = _humanize_name(value)
    replacements = {
        "drop Weight Impact SMC": "drop weight impact",
        "Drop Weight Impact SMC": "drop weight impact",
        "gear Pair 3D": "gear pair 3D",
        "rigid Collision": "rigid collision",
        "sphere Drop 3D": "sphere drop 3D",
        "stack Stability": "stack stability",
        "block Tower 3D": "block tower 3D",
        "flexible Crank Slider": "flexible crank slider",
        "cantilever Beam Transient": "cantilever beam transient",
        "harmonic Response FRF": "harmonic response",
        "reissner Shell Modal": "Reissner shell modal response",
        "stress Wave Propagation": "stress wave propagation",
        "rubber Block Compression": "rubber block compression",
        "j2 Plastic Beam": "J2 plastic beam",
        "J2 Plastic Beam": "J2 plastic beam",
        "lemaitre Damage Bar": "Lemaitre damage bar",
        "mass Spring Damper": "mass-spring-damper",
        "rolling Sliding Transition": "rolling-sliding transition",
        "gyroscope Rotor 3D": "3D gyroscope rotor",
        "articulated Chain": "articulated chain",
    }
    text = replacements.get(text, text)
    text = re.sub(
        r"\b(SMC|NSC|Chrono|PyChrono|Pardiso|SparseQR)\b", "", text, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", text).strip()


def _assert_query_text_allowed(text: str) -> None:
    lower = text.lower()
    for term in FORBIDDEN_QUERY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lower):
            raise ValueError(
                f"Forbidden implementation detail {term!r} found in query: {text}"
            )


def _rendered_field_for_family(family: str) -> str:
    fields = {
        "pendulum": "bob speed magnitude on moving pendulum bodies",
        "rigidCollision": "rigid-body speed magnitude during contact",
        "sphereDrop3D": "sphere speed magnitude through fall and rebound",
        "stackStability": "block speed magnitude during stack response",
        "blockTower3D": "block speed magnitude during tower impact",
        "dropWeightImpactSMC": "impact speed and plate response field",
        "flexibleCrankSlider": "mechanism speed magnitude and flexible rod motion",
        "gearPair3D": "absolute angular speed on opaque interlocking gears",
        "cantileverBeamTransient": "beam displacement magnitude during transient bending",
        "harmonicResponseFRF": "harmonic displacement response near resonance",
        "reissnerShellModal": "shell mode displacement magnitude",
        "stressWavePropagation": "axial stress wave magnitude along the rod",
        "rubberBlockCompression": "large-deformation displacement magnitude",
        "j2PlasticBeam": "plastic deformation and displacement response",
        "lemaitreDamageBar": "damage or plastic-strain localization field",
        "massSpringDamper": "sliding block speed magnitude in a mass-spring-damper chain",
        "rollingSlidingTransition": "rolling body surface-point speed during the sliding-to-rolling transition",
        "gyroscopeRotor3D": "angular speed on the spinning rotor and shaft",
        "unbalancedRotor3D": "lateral displacement gradient during unbalanced rotor run-up or coast-down through critical speed",
        "articulatedChain": "link and joint speed magnitude along the chain",
    }
    return fields.get(family, "rendered dynamics response field")


def _visual_signature_text(signature: Any) -> str:
    if not isinstance(signature, dict):
        return "not recorded"
    parts = [
        f"{_clean_query_fragment(k)}={_clean_query_fragment(v)}"
        for k, v in sorted(signature.items())
        if _usable(v) and not _is_impl_detail_key(k)
    ]
    return "; ".join(parts) if parts else "not recorded"


def _join_phrases(parts: list[str]) -> str:
    vals = [p.strip() for p in parts if p and p.strip()]
    if not vals:
        return "the recorded benchmark attributes"
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return f"{', '.join(vals[:-1])}, and {vals[-1]}"


def _is_informative_field(field: str, value: Any) -> bool:
    if not _usable(value):
        return False
    fv = _as_float(value)
    if field in {
        "max_speed",
        "max_angle_deg",
        "max_relative_angle_deg",
        "impact_speed",
        "post_impact_speed",
        "contact_events",
        "roll_capture_time",
        "slip_distance",
        "roll_distance",
        "response_amplitude_m",
        "precession_turns",
        "lane_offset_final",
        "max_relative_angle_deg",
        "max_height",
    }:
        return fv is None or abs(fv) > 1e-9
    return True


def _query_field_pool(vals: dict[str, str]) -> tuple[list[str], list[str]]:
    family = str(vals.get("family", ""))
    base_fields = [
        "family",
        "dim",
        "mechanism_type",
        "rendered_field",
        "visual_signature",
        "duration",
        "primary_length",
        "gravity",
        "ground_enabled",
    ]
    family_fields: dict[str, list[str]] = {
        "pendulum": ["max_speed", "max_angle_deg", "energy_drift_pct"],
        "rigidCollision": [
            "max_speed",
            "impact_speed",
            "post_impact_speed",
            "contact_events",
            "settling_time",
        ],
        "sphereDrop3D": [
            "max_speed",
            "max_height",
            "impact_speed",
            "post_impact_speed",
            "contact_events",
            "settling_time",
        ],
        "ballChutePile3D": [
            "max_speed",
            "max_height",
            "contact_events",
            "settling_time",
            "roll_capture_time",
        ],
        "stackStability": [
            "max_speed",
            "contact_events",
            "settling_time",
            "stack_toppled",
        ],
        "blockTower3D": [
            "max_speed",
            "contact_events",
            "settling_time",
            "stack_toppled",
        ],
        "gearPair3D": ["max_speed", "max_angle_deg", "contact_events"],
        "dropWeightImpactSMC": [
            "max_speed",
            "impact_speed",
            "post_impact_speed",
            "contact_events",
            "settling_time",
        ],
        "flexibleCrankSlider": ["max_speed", "max_height", "max_angle_deg"],
        "cantileverBeamTransient": ["max_speed", "max_height"],
        "harmonicResponseFRF": ["max_speed"],
        "reissnerShellModal": ["max_speed", "max_height"],
        "stressWavePropagation": ["max_speed"],
        "rubberBlockCompression": ["max_speed", "max_height"],
        "j2PlasticBeam": ["max_speed", "max_height"],
        "lemaitreDamageBar": ["max_speed"],
        "massSpringDamper": ["max_speed", "response_amplitude_m", "settling_time"],
        "rollingSlidingTransition": [
            "max_speed",
            "roll_capture_time",
            "slip_distance",
            "roll_distance",
        ],
        "gyroscopeRotor3D": ["max_speed", "max_angle_deg", "precession_turns"],
        "unbalancedRotor3D": ["max_speed", "max_height", "response_amplitude_m"],
        "articulatedChain": ["max_speed", "max_angle_deg", "max_relative_angle_deg"],
    }
    extras = family_fields.get(
        family,
        [
            "max_speed",
            "max_height",
            "max_angle_deg",
            "impact_speed",
            "post_impact_speed",
            "contact_events",
            "settling_time",
            "roll_capture_time",
            "energy_drift_pct",
        ],
    )
    base = [f for f in base_fields if _is_informative_field(f, vals.get(f, ""))]
    extras_clean = [f for f in extras if _is_informative_field(f, vals.get(f, ""))]
    return base, extras_clean


def _case_record(
    *,
    case_dir: Path,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    metadata_row: dict[str, Any],
) -> dict[str, Any]:
    family = str(
        meta.get("family") or meta.get("case_prefix") or case_dir.name.split("_", 1)[0]
    )
    solver = meta.get("solver") if isinstance(meta.get("solver"), dict) else {}
    mechanism = meta.get("mechanism") if isinstance(meta.get("mechanism"), dict) else {}
    environment = (
        meta.get("environment") if isinstance(meta.get("environment"), dict) else {}
    )
    scale = meta.get("scale") if isinstance(meta.get("scale"), dict) else {}
    visual_variation = (
        scale.get("visual_variation")
        if isinstance(scale.get("visual_variation"), dict)
        else {}
    )
    visual_signature = (
        visual_variation.get("signature")
        if isinstance(visual_variation.get("signature"), dict)
        else {}
    )

    return {
        "case_id": case_dir.name,
        "case_path": str(case_dir.resolve()),
        "family": family,
        "params": {
            "dim": _case_dim(meta),
            "backend": metrics.get("backend", metadata_row.get("backend", "unknown")),
            "duration_sec": solver.get("duration"),
            "step_size": solver.get("step_size"),
            "mechanism_type": mechanism.get("type", family),
            "rendered_field": _rendered_field_for_family(family),
            "visual_signature": visual_signature,
            "ground_enabled": (
                (environment.get("ground") or {}).get("enabled")
                if isinstance(environment.get("ground"), dict)
                else None
            ),
            "gravity": environment.get("gravity"),
        },
        "scene_spec": {
            "family": family,
            "dim": _case_dim(meta),
            "mechanism": mechanism,
            "environment": environment,
            "solver": "chrono_mbs",
            "visual_variation": visual_variation,
        },
        "assets": {
            "video_path": _video_for_case(case_dir),
            "trajectory_path": str(case_dir / "results" / "trajectory.json"),
            "metrics_path": str(case_dir / "postProcessing" / "metrics.json"),
        },
        "observables": {
            "max_speed": metrics.get("max_speed"),
            "max_height": metrics.get("max_height"),
            "energy_drift_pct": metrics.get("energy_drift_pct"),
            "max_angle_deg": metrics.get("max_angle_deg"),
            "max_relative_angle_deg": metrics.get("max_relative_angle_deg"),
            "impact_speed": metrics.get("impact_speed"),
            "post_impact_speed": metrics.get("post_impact_speed"),
            "contact_events": metrics.get("contact_events"),
            "settling_time": metrics.get("settling_time"),
            "stack_toppled": metrics.get("stack_toppled"),
            "roll_capture_time": metrics.get("roll_capture_time"),
            "slip_distance": metrics.get("slip_distance"),
            "roll_distance": metrics.get("roll_distance"),
            "lane_offset_final": metrics.get("lane_offset_final"),
            "response_amplitude_m": metrics.get("response_amplitude_m"),
            "precession_turns": metrics.get("precession_turns"),
        },
        "quality": {
            "tier": metadata_row.get("quality_tier", "silver"),
            "reason": metadata_row.get("quality_reason", ["auto"]),
        },
    }


def _case_tokens(case: dict[str, Any]) -> dict[str, str]:
    scene = case.get("scene_spec") if isinstance(case.get("scene_spec"), dict) else {}
    mech = scene.get("mechanism") if isinstance(scene.get("mechanism"), dict) else {}
    obs = case.get("observables") if isinstance(case.get("observables"), dict) else {}
    params = case.get("params") if isinstance(case.get("params"), dict) else {}
    gravity = (scene.get("environment") or {}).get("gravity")
    gravity_token = (
        json.dumps(gravity, ensure_ascii=False, separators=(",", ":"))
        if gravity is not None
        else "n/a"
    )
    ground_enabled = params.get("ground_enabled")
    if isinstance(ground_enabled, bool):
        ground_token = "enabled" if ground_enabled else "disabled"
    else:
        ground_token = "unknown"
    stack_toppled = obs.get("stack_toppled")
    if isinstance(stack_toppled, bool):
        toppled_token = "toppled" if stack_toppled else "stable"
    else:
        toppled_token = str(stack_toppled if stack_toppled is not None else "n/a")
    return {
        "family": str(case.get("family", "unknown")),
        "dim": str(params.get("dim", "2D")),
        "mechanism_type": str(params.get("mechanism_type", "unknown")),
        "rendered_field": str(params.get("rendered_field", "not recorded")),
        "visual_signature": _visual_signature_text(params.get("visual_signature")),
        "duration": _fmt(params.get("duration_sec"), 3),
        "backend": str(params.get("backend", "unknown")),
        "gravity": gravity_token,
        "ground_enabled": ground_token,
        "primary_length": _fmt(
            mech.get("length")
            or mech.get("length_1")
            or mech.get("radius")
            or mech.get("mass_spacing")
            or mech.get("shaft_length")
            or mech.get("box_size", [None])[0],
            4,
        ),
        "max_speed": _fmt(obs.get("max_speed"), 4),
        "max_height": _fmt(obs.get("max_height"), 4),
        "max_angle_deg": _fmt(obs.get("max_angle_deg"), 4),
        "max_relative_angle_deg": _fmt(obs.get("max_relative_angle_deg"), 4),
        "energy_drift_pct": _fmt(obs.get("energy_drift_pct"), 4),
        "impact_speed": _fmt(obs.get("impact_speed"), 4),
        "post_impact_speed": _fmt(obs.get("post_impact_speed"), 4),
        "contact_events": str(obs.get("contact_events", "n/a")),
        "settling_time": _fmt(obs.get("settling_time"), 4),
        "stack_toppled": toppled_token,
        "roll_capture_time": _fmt(obs.get("roll_capture_time"), 4),
        "slip_distance": _fmt(obs.get("slip_distance"), 4),
        "roll_distance": _fmt(obs.get("roll_distance"), 4),
        "lane_offset_final": _fmt(obs.get("lane_offset_final"), 4),
        "response_amplitude_m": _fmt(obs.get("response_amplitude_m"), 4),
        "precession_turns": _fmt(obs.get("precession_turns"), 4),
    }


def _field_phrase(field: str, vals: dict[str, str], style: int) -> str:
    v = vals.get(field, "n/a")
    if field == "family":
        return _pick(
            [
                f"the dynamics family is {v}",
                f"it belongs to the {v} family",
                f"the scene comes from the {v} benchmark family",
            ],
            style,
        )
    if field == "dim":
        return _pick(
            [
                f"it is a {v} dynamics case",
                f"the simulation is posed in {v}",
                f"the benchmark instance is {v}",
            ],
            style,
        )
    if field == "mechanism_type":
        return _pick(
            [
                f"the mechanism type is {v}",
                f"the setup uses a {v} mechanism",
                f"the main motion system is a {v} configuration",
            ],
            style,
        )
    if field == "rendered_field":
        return _pick(
            [
                f"the rendered color field represents {v}",
                f"the video is colored by {v}",
                f"the scalar visualization shows {v}",
            ],
            style,
        )
    if field == "visual_signature":
        return _pick(
            [
                f"the visible scale signature is {v}",
                f"the geometry and initial-condition signature is {v}",
                f"the video-distinguishing setup bins are {v}",
            ],
            style,
        )
    if field == "duration":
        return _pick(
            [
                f"the run lasts about {v} s",
                f"the simulated duration is near {v} s",
                f"the total time horizon is roughly {v} s",
            ],
            style,
        )
    if field == "backend":
        return "the numerical backend is intentionally hidden from retrieval text"
    if field == "gravity":
        return _pick(
            [
                f"the gravity vector is {v}",
                f"gravity is configured as {v}",
                f"the scene uses gravity {v}",
            ],
            style,
        )
    if field == "ground_enabled":
        return _pick(
            [
                f"ground contact is {v}",
                f"the ground boundary is {v}",
                f"the setup keeps the ground plane {v}",
            ],
            style,
        )
    if field == "primary_length":
        return _pick(
            [
                f"the characteristic size is about {v}",
                f"the primary length scale is near {v}",
                f"the representative mechanism size is roughly {v}",
            ],
            style,
        )
    if field == "max_speed":
        return _pick(
            [
                f"the peak speed is about {v}",
                f"the observed maximum speed is near {v}",
                f"the motion reaches roughly {v} in speed",
            ],
            style,
        )
    if field == "max_height":
        return _pick(
            [
                f"the maximum height is about {v}",
                f"the trajectory peaks near height {v}",
                f"the scene reaches a top height around {v}",
            ],
            style,
        )
    if field == "max_angle_deg":
        family = str(vals.get("family", ""))
        if family == "gearPair3D":
            return _pick(
                [
                    f"the gear rotation span is about {v} degrees",
                    f"the largest gear angular sweep is near {v} degrees",
                    f"the gear train rotates through roughly {v} degrees",
                ],
                style,
            )
        if family in {"flexibleCrankSlider", "gyroscopeTop3D"}:
            return _pick(
                [
                    f"the mechanism angular span is about {v} degrees",
                    f"the largest mechanism angle reaches roughly {v} degrees",
                    f"the angular motion range is near {v} degrees",
                ],
                style,
            )
        return _pick(
            [
                f"the maximum angle is about {v} degrees",
                f"the swing amplitude reaches roughly {v} degrees",
                f"the largest angular excursion is near {v} degrees",
            ],
            style,
        )
    if field == "max_relative_angle_deg":
        return _pick(
            [
                f"the maximum relative angle is about {v} degrees",
                f"the largest inter-body angle is near {v} degrees",
                f"the relative angular span reaches roughly {v} degrees",
            ],
            style,
        )
    if field == "energy_drift_pct":
        return _pick(
            [
                f"the energy drift is about {v} percent",
                f"the reported energy drift stays near {v} percent",
                f"the energy error level is roughly {v} percent",
            ],
            style,
        )
    if field == "impact_speed":
        return _pick(
            [
                f"the impact speed is about {v}",
                f"contact occurs at roughly speed {v}",
                f"the pre-impact speed is near {v}",
            ],
            style,
        )
    if field == "post_impact_speed":
        return _pick(
            [
                f"the post-impact speed is about {v}",
                f"after impact the speed settles near {v}",
                f"the rebound speed is roughly {v}",
            ],
            style,
        )
    if field == "contact_events":
        return _pick(
            [
                f"it shows about {v} contact events",
                f"the contact-event count is approximately {v}",
                f"the interaction count is near {v} contacts",
            ],
            style,
        )
    if field == "settling_time":
        return _pick(
            [
                f"the settling time is about {v} s",
                f"the motion settles after roughly {v} s",
                f"the system relaxes in about {v} s",
            ],
            style,
        )
    if field == "stack_toppled":
        return _pick(
            [
                f"the stack outcome is {v}",
                f"the final stack state is {v}",
                f"the tower response is classified as {v}",
            ],
            style,
        )
    if field == "roll_capture_time":
        return _pick(
            [
                f"the roll-capture time is about {v} s",
                f"rolling is captured after roughly {v} s",
                f"the transition to rolling completes near {v} s",
            ],
            style,
        )
    if field == "slip_distance":
        return _pick(
            [
                f"the slip distance is about {v}",
                f"the object slides for roughly {v}",
                f"the measured slip travel is near {v}",
            ],
            style,
        )
    if field == "roll_distance":
        return _pick(
            [
                f"the roll distance is about {v}",
                f"the rolling travel is roughly {v}",
                f"the measured rolling distance is near {v}",
            ],
            style,
        )
    if field == "lane_offset_final":
        return _pick(
            [
                f"the final lane offset is about {v}",
                f"the terminal lateral offset is near {v}",
                f"the ending lane displacement is roughly {v}",
            ],
            style,
        )
    return _pick([f"{field} is {v}", f"the recorded {field} is {v}"], style)


def _family_scene_sentence(vals: dict[str, str]) -> str:
    family = vals.get("family", "unknown")
    dim = vals.get("dim", "dynamics")
    label = _public_label(family)
    mapping = {
        "pendulum": f"a {dim} pendulum sequence with one or more linked bobs swinging under gravity",
        "rigidCollision": f"a {dim} rigid-body collision where bodies translate, rotate, and separate after contact",
        "sphereDrop3D": "a 3D sphere drop with free fall, ground impact, and rebound",
        "stackStability": "a 3D stacked-block stability scene responding to a disturbance",
        "blockTower3D": "a 3D block tower struck by a moving impactor",
        "dropWeightImpactSMC": "a 3D drop-weight impact scene with a sphere striking a deforming plate",
        "flexibleCrankSlider": "a 3D crank-slider mechanism with a visibly flexible connecting rod",
        "gearPair3D": "a 3D train of interlocking spur gears driven by an angular input",
        "cantileverBeamTransient": "a 3D cantilever beam that bends transiently after a tip load",
        "harmonicResponseFRF": "a 3D harmonic-response case with resonant structural motion",
        "reissnerShellModal": "a 3D shell vibration case showing a standing mode shape",
        "stressWavePropagation": "a 3D rod where a stress front travels and reflects along the length",
        "rubberBlockCompression": "a 3D rubber-like block undergoing large compression and bulging",
        "j2PlasticBeam": "a 3D beam with plastic bending and residual deformation",
        "lemaitreDamageBar": "a 3D notched bar where damage localizes under cyclic loading",
        "massSpringDamper": "a mass-spring-damper system where blocks oscillate, decay, or respond to a harmonic input",
        "rollingSlidingTransition": "a rigid roller on an inclined track transitioning from sliding slip to pure rolling",
        "gyroscopeRotor3D": "a 3D spinning rotor showing tilt, precession, and nutation around a fixed pivot",
        "unbalancedRotor3D": "a 3D unbalanced rotor on bearing supports whose whirl grows or decays as spin speed crosses critical speed",
        "articulatedChain": "a linked chain mechanism with articulated rods and joints swinging under gravity or a visible root drive",
    }
    return mapping.get(family, f"a {dim} {label} dynamics scene")


def _friendly_signature(value: str) -> str:
    if not _usable(value):
        return ""
    parts: list[str] = []
    for raw in str(value).split(";"):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            key, val = item.split("=", 1)
            if _is_impl_detail_key(key):
                continue
            parts.append(f"{_clean_query_fragment(key)} {str(val).strip()}")
        else:
            parts.append(_clean_query_fragment(item))
    return _join_phrases(parts[:4])


def _gravity_phrase(value: str) -> str:
    s = str(value or "").strip()
    if not _usable(s):
        return ""
    compact = s.replace(" ", "")
    if compact in {"[0.0,0.0,0.0]", "[0,0,0]"}:
        return "gravity is absent so the motion is driven by the mechanism rather than falling"
    if "-9.81" in compact:
        return "Earth-scale gravity acts downward"
    return f"gravity is configured as {s}"


def _natural_field_phrase(field: str, vals: dict[str, str], style: int) -> str:
    v = vals.get(field, "n/a")
    if not _usable(v):
        return ""
    if field == "family":
        return _family_scene_sentence(vals)
    if field == "dim":
        return f"the view and geometry are {v}"
    if field == "mechanism_type":
        return f"the motion system is {_public_label(v)}"
    if field == "rendered_field":
        return f"colors encode {_clean_query_fragment(v)}"
    if field == "visual_signature":
        friendly = _friendly_signature(v)
        return f"the visible setup is distinguished by {friendly}" if friendly else ""
    if field == "duration":
        return f"the recorded motion covers roughly {v} s"
    if field == "gravity":
        return _gravity_phrase(v)
    if field == "ground_enabled":
        if v == "enabled":
            return "ground contact or support is part of the visible boundary condition"
        if v == "disabled":
            return "no ground contact plane is part of the visible setup"
        return ""
    if field == "primary_length":
        return f"the characteristic geometric size is about {v} m"
    if field == "max_speed":
        return f"the fastest visible motion reaches about {v} m/s"
    if field == "max_height":
        return f"the trajectory or deformation reaches a height near {v} m"
    if field == "max_angle_deg":
        family = str(vals.get("family", ""))
        if family == "gearPair3D":
            return f"the gears rotate through roughly {v} degrees during the clip"
        if family == "pendulum":
            return f"the swing reaches an angular excursion near {v} degrees"
        return f"the mechanism rotates through about {v} degrees"
    if field == "max_relative_angle_deg":
        return f"linked bodies reach a relative angle near {v} degrees"
    if field == "energy_drift_pct":
        return f"the energy drift stays around {v} percent"
    if field == "impact_speed":
        return f"impact occurs at about {v} m/s"
    if field == "post_impact_speed":
        return f"post-impact motion remains near {v} m/s"
    if field == "contact_events":
        try:
            noun = "event" if abs(float(v) - 1.0) < 1.0e-9 else "events"
        except (TypeError, ValueError):
            noun = "events"
        return f"the scene contains about {v} contact {noun}"
    if field == "settling_time":
        return f"the motion settles after roughly {v} s"
    if field == "stack_toppled":
        return f"the final stack state is {v}"
    if field == "roll_capture_time":
        return f"rolling behavior is captured after roughly {v} s"
    if field == "slip_distance":
        return f"the sliding travel is about {v} m"
    if field == "roll_distance":
        return f"the rolling travel is about {v} m"
    if field == "response_amplitude_m":
        return f"the mass-spring response amplitude is about {v} m"
    if field == "precession_turns":
        return f"the rotor precesses through about {v} turns"
    if field == "lane_offset_final":
        return f"the final lateral offset is about {v} m"
    return _field_phrase(field, vals, style)


def _query_body(
    vals: dict[str, str], fields: tuple[str, ...], style: int, *, max_parts: int
) -> str:
    ordered = [
        "family",
        "dim",
        "mechanism_type",
        "visual_signature",
        "rendered_field",
        "gravity",
        "ground_enabled",
        "primary_length",
        "max_speed",
        "max_angle_deg",
        "impact_speed",
        "post_impact_speed",
        "contact_events",
        "settling_time",
        "max_height",
        "stack_toppled",
        "energy_drift_pct",
        "roll_capture_time",
        "slip_distance",
        "roll_distance",
        "response_amplitude_m",
        "precession_turns",
        "lane_offset_final",
        "duration",
    ]
    seen: set[str] = set()
    parts: list[str] = []
    for field in ordered:
        if field not in fields or field in seen:
            continue
        seen.add(field)
        phrase = _natural_field_phrase(field, vals, style + len(parts))
        if phrase:
            parts.append(phrase)
        if len(parts) >= max_parts:
            break
    for field in fields:
        if len(parts) >= max_parts:
            break
        if field in seen:
            continue
        phrase = _natural_field_phrase(field, vals, style + len(parts))
        if phrase:
            parts.append(phrase)
    return _join_phrases(parts)


def _full_query_text(
    vals: dict[str, str], fields: tuple[str, ...], case_id: str
) -> str:
    style = _style_index(f"{case_id}|full", 97)
    body = _query_body(vals, fields, style, max_parts=max(8, len(fields)))
    text = f"Identify the dynamics benchmark video showing {body}."
    _assert_query_text_allowed(text)
    return text
