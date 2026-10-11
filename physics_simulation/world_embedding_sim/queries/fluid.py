"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

import re

from typing import Any


def _scenario_from_case(case: dict[str, Any]) -> str:
    params = case.get("params") if isinstance(case.get("params"), dict) else {}
    prefix = params.get("scenario_prefix")
    if prefix:
        return str(prefix)
    case_id = str(case["case_id"])
    return case_id.split("_Re", 1)[0]


_EVAL_FORBIDDEN_TERMS = (
    "solver",
    "mesh",
    "backend",
    "openfoam",
    "icofoam",
    "pimplefoam",
    "simplefoam",
    "rans",
    "kepsilon",
    "komega",
    "case_id",
    "boundary condition",
    "diagnostic signature",
)

_SCENE_LABEL = {
    "backwardFacingStep2d": "channel flow over a backward-facing step",
    "suddenContraction2d": "channel flow through a sudden contraction",
    "suddenExpansion2d": "channel flow through a sudden expansion",
    "suddenExpansion2d_nn_hb": "Herschel-Bulkley channel flow through a sudden expansion",
    "suddenExpansion2d_nn_powerlaw": "power-law channel flow through a sudden expansion",
    "buoyancy2d": "buoyancy-driven convection above a heated patch",
    "rayleighBenard2d": "Rayleigh-Benard convection between heated and cooled plates",
    "cavity2d_turb": "a lid-driven cavity flow",
    "lidDrivenCavity3d": "a lid-driven cavity flow",
    "dye2d": "passive dye transport in a Newtonian carrier flow",
    "flowpastcylinder2d": "flow past a circular cylinder",
    "flowpastsquare2d": "flow past a square cylinder",
    "jetaxi": "an axisymmetric round jet entering quiescent fluid",
    "jetaxi_turb": "a turbulent axisymmetric round jet",
    "naca0012airfoil2d": "flow over a NACA airfoil",
}

_SCENE_SIGNATURE = {
    "backwardFacingStep2d": "a separated shear layer and a recirculation zone forming behind the step",
    "suddenContraction2d": "the stream accelerating as it squeezes into the narrower channel",
    "suddenExpansion2d": "the incoming jet spreading and recirculating after the expansion",
    "suddenExpansion2d_nn_hb": "a yield-stress fluid spreading after the expansion with blunted velocity profiles",
    "suddenExpansion2d_nn_powerlaw": "a shear-thinning fluid spreading after the expansion",
    "buoyancy2d": "warm fluid rising from the heated region in buoyant plumes",
    "rayleighBenard2d": "convection cells circulating between the bottom and top plates",
    "cavity2d_turb": "the moving lid dragging fluid into a large rotating vortex",
    "lidDrivenCavity3d": "the moving lid dragging fluid into a three-dimensional cavity vortex",
    "dye2d": "a finite dye pulse advecting and diffusing along the channel",
    "flowpastcylinder2d": "an oscillating vortex street shedding behind the cylinder",
    "flowpastsquare2d": "an oscillating vortex street shedding behind the square obstacle",
    "jetaxi": "a jet column penetrating and mixing into the surrounding fluid",
    "jetaxi_turb": "a jet column spreading and mixing strongly downstream",
    "naca0012airfoil2d": "flow following the airfoil surface and leaving a downstream wake",
}

_PHYSICS_MECHANISM = {
    "backwardFacingStep2d": "the flow separates at the step edge and reattaches downstream, trapping a slowly rotating recirculation bubble",
    "suddenContraction2d": "mass conservation forces the fluid to speed up through the reduced cross-section",
    "suddenExpansion2d": "the abrupt area increase decelerates the jet, which detaches and feeds corner recirculation zones",
    "suddenExpansion2d_nn_hb": "below the yield stress the material moves as a plug, so the expanding jet keeps a blunted core while the corners recirculate weakly",
    "suddenExpansion2d_nn_powerlaw": "shear-thinning viscosity lowers resistance where shear is high, flattening the velocity profile as the jet expands",
    "buoyancy2d": "heated fluid becomes lighter and rises, entraining cooler fluid and forming unsteady thermal plumes",
    "rayleighBenard2d": "the unstable temperature stratification drives organized convection rolls between the plates",
    "cavity2d_turb": "the shear imposed by the sliding lid drives a primary vortex with weaker counter-rotating corner eddies",
    "lidDrivenCavity3d": "the shear imposed by the sliding lid drives a primary vortex whose core bends in three dimensions",
    "dye2d": "the carrier flow advects the injected pulse downstream while diffusion smears its edges at a rate set by the Peclet number",
    "flowpastcylinder2d": "above the critical Reynolds number the cylinder wake becomes unstable and sheds vortices alternately from each side",
    "flowpastsquare2d": "the sharp corners fix the separation points, so the square body sheds a stronger, wider vortex street",
    "jetaxi": "the momentum of the jet entrains ambient fluid, so the column widens and slows with downstream distance",
    "jetaxi_turb": "turbulent mixing entrains ambient fluid rapidly, so the jet spreads faster and decays sooner than a laminar one",
    "naca0012airfoil2d": "the airfoil turns the oncoming flow, accelerating it over the suction side and shedding a momentum-deficit wake",
}

_COLOR_CUE = {
    "backwardFacingStep2d": "fast colors mark the inlet stream while the recirculation bubble behind the step stays dark",
    "suddenContraction2d": "colors brighten sharply where the channel narrows and the flow accelerates",
    "suddenExpansion2d": "the bright core jet fades as it spreads into the darker expanded section",
    "suddenExpansion2d_nn_hb": "a bright blunt core persists after the expansion while near-wall regions stay dark",
    "suddenExpansion2d_nn_powerlaw": "the bright core spreads with a flattened front after the expansion",
    "buoyancy2d": "rising plumes appear as bright moving filaments over the heated region",
    "rayleighBenard2d": "counter-rotating cells appear as alternating bright updrafts and downdrafts",
    "cavity2d_turb": "the brightest band follows the lid and wraps around the main vortex",
    "lidDrivenCavity3d": "the brightest band follows the lid while the vortex core stays dimmer",
    "dye2d": "the dye pulse appears as a bright patch that stretches and fades as it travels",
    "flowpastcylinder2d": "alternating bright vortices peel off the cylinder and drift downstream",
    "flowpastsquare2d": "alternating bright vortices peel off the square body and drift downstream",
    "jetaxi": "the bright jet core gradually dims and widens with distance from the nozzle",
    "jetaxi_turb": "the bright jet core breaks up quickly into a broad mixing region",
    "naca0012airfoil2d": "bright accelerated flow hugs the upper surface while the wake trails as a darker streak",
}

_AXISYM_SCENARIOS = {"jetaxi", "jetaxi_turb"}

_PULSE_LABELS = {"t1": "short", "t2": "medium", "t3": "long"}


def _efmt(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}g}"
    except Exception:
        return "n/a"


def _efmt_re(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return "n/a"
    if x == int(x) and abs(x) < 1e7:
        return str(int(x))
    return f"{x:.4g}"


def _eval_case_values(case: dict[str, Any]) -> dict[str, Any]:
    params = case.get("params") if isinstance(case.get("params"), dict) else {}
    scene = case.get("scene_spec") if isinstance(case.get("scene_spec"), dict) else {}
    geom = (
        scene.get("geometry_params")
        if isinstance(scene.get("geometry_params"), dict)
        else {}
    )
    obs = case.get("observables") if isinstance(case.get("observables"), dict) else {}
    diag = case.get("diagnostics") if isinstance(case.get("diagnostics"), dict) else {}
    scenario = str(params.get("scenario_prefix") or _scenario_from_case(case))
    u_max = obs.get("U_max")
    if (
        u_max is None
        and diag.get("u_max_over_u_ref") is not None
        and diag.get("u_ref") is not None
    ):
        u_max = float(diag["u_max_over_u_ref"]) * float(diag["u_ref"])
    return {
        "scenario": scenario,
        "qual": "axisymmetric"
        if scenario in _AXISYM_SCENARIOS
        else str(scene.get("dim", "2D")),
        "flow_regime": "turbulent"
        if str(scene.get("model", "")).upper() == "RANS"
        else "laminar",
        "Re": (scene.get("nondim_numbers") or {}).get("Re"),
        "lx": geom.get("lx"),
        "ly": geom.get("ly"),
        "chord": geom.get("chord"),
        "obstacle_radius": geom.get("obstacle_radius"),
        "obstacle_half_width": geom.get("obstacle_half_width"),
        "r_jet": geom.get("r_jet"),
        "U_ref": params.get("U_ref"),
        "nu": params.get("nu"),
        "U_max": u_max,
        "Pe": params.get("Pe"),
        "pulse_band": params.get("pulse_band"),
        "pulse_duration": params.get("pulse_duration"),
        "pulse_label": _PULSE_LABELS.get(str(params.get("pulse_duration_label")), None),
        "rheology_model": params.get("rheology_model"),
        "rheology_n": params.get("rheology_n"),
        "rendered_field": "the transported dye concentration field"
        if scenario == "dye2d"
        else "velocity magnitude",
    }


def _eval_extras_phrases(v: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if v["chord"] is not None:
        out.append(f"the airfoil chord is about {_efmt(v['chord'])}")
    if v["obstacle_radius"] is not None:
        out.append(f"the cylinder radius is about {_efmt(v['obstacle_radius'])}")
    if v["obstacle_half_width"] is not None:
        out.append(
            f"the obstacle half-width is about {_efmt(v['obstacle_half_width'])}"
        )
    if v["r_jet"] is not None:
        out.append(f"the jet radius is about {_efmt(v['r_jet'])}")
    if v["pulse_band"] is not None:
        dur = (
            f" lasting about {_efmt(v['pulse_duration'], 3)}"
            if v["pulse_duration"] is not None
            else ""
        )
        label = f"{v['pulse_label']} " if v["pulse_label"] else ""
        out.append(
            f"the dye enters through the {v['pulse_band']} inlet band as a {label}pulse{dur}"
        )
    if v["rheology_model"] is not None:
        name = (
            "Herschel-Bulkley yield-stress"
            if "hersch" in str(v["rheology_model"]).lower()
            else "power-law shear-thinning"
        )
        idx = (
            f" with flow index about {_efmt(v['rheology_n'], 3)}"
            if v["rheology_n"] is not None
            else ""
        )
        out.append(f"the working fluid is a {name} fluid{idx}")
    return out


def _assert_eval_text_allowed(text: str) -> None:
    lowered = text.lower()
    for term in _EVAL_FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            raise ValueError(f"Forbidden term {term!r} in query text: {text}")


def _eval_raw_query_text(case: dict[str, Any]) -> str:
    v = _eval_case_values(case)
    s = v["scenario"]
    parts = [
        _SCENE_LABEL.get(s, s),
        f"the view is {v['qual']}",
        f"the flow is {v['flow_regime']}",
    ]
    if v["lx"] is not None and v["ly"] is not None:
        parts.append(f"the domain spans about {_efmt(v['lx'])} by {_efmt(v['ly'])}")
    parts.append(f"the Reynolds number is about {_efmt_re(v['Re'])}")
    parts.extend(_eval_extras_phrases(v))
    if s == "dye2d" and v["Pe"] is not None:
        parts.append(f"the Peclet number is about {_efmt_re(v['Pe'])}")
    parts.extend(
        [
            f"the reference velocity is about {_efmt(v['U_ref'])}",
            f"the kinematic viscosity is about {_efmt(v['nu'])}",
            f"colors encode {v['rendered_field']}",
            f"the peak speed reaches about {_efmt(v['U_max'])}",
        ]
    )
    body = ", ".join(p for p in parts if p and "n/a" not in p)
    text = f"Identify the fluid mechanics benchmark video showing {body}."
    _assert_eval_text_allowed(text)
    return text


def _eval_parsed_query_text(case: dict[str, Any]) -> str:
    v = _eval_case_values(case)
    s = v["scenario"]
    extras = _eval_extras_phrases(v)
    variant_bits = [f"Reynolds number about {_efmt_re(v['Re'])}"]
    variant_bits.extend(extras[:2])
    if s == "dye2d" and v["Pe"] is not None:
        variant_bits.append(f"Peclet number about {_efmt_re(v['Pe'])}")
    if v["lx"] is not None and v["ly"] is not None:
        third_detail = f"the domain spans about {_efmt(v['lx'])} by {_efmt(v['ly'])}"
    else:
        third_detail = f"the kinematic viscosity is about {_efmt(v['nu'])}"
    detail_bits = [
        f"the reference velocity is about {_efmt(v['U_ref'])}",
        f"the peak speed reaches about {_efmt(v['U_max'])}",
        third_detail,
    ]
    sentences = [
        f"Find the {v['qual']} fluid mechanics video of {_SCENE_LABEL.get(s, s)}, showing {_SCENE_SIGNATURE.get(s, 'the flow evolving visibly')}.",
        f"Physically, {_PHYSICS_MECHANISM.get(s, 'the flow develops according to the imposed conditions')}.",
        f"The color map represents {v['rendered_field']}, so {_COLOR_CUE.get(s, 'the brightest colors mark the fastest visible motion')}.",
        f"Use the variant with {', '.join(variant_bits)}; the flow is {v['flow_regime']}.",
        f"To identify the exact video, look for {', '.join(detail_bits[:-1])}, and {detail_bits[-1]}.",
    ]
    text = " ".join(sentences)
    _assert_eval_text_allowed(text)
    return text
