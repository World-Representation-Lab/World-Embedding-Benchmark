"""Example configuration validation."""
import math

BACKENDS = {
    "dynamics": {"chrono", "surrogate"},
    "fluid": {None},
    "solid": {"fenicsx"},
    "optics": {"hcipy", "mitsuba3", "meep"},
}


def validate_example(config):
    """Check the branch, single-case structure, and finite input values."""
    domain = config.get("domain")
    if domain not in BACKENDS:
        raise ValueError(f"Unsupported domain: {domain}")
    if config.get("backend") not in BACKENDS[domain]:
        raise ValueError(f"Unsupported backend for {domain}")
    if not isinstance(config.get("family"), str) or not config["family"]:
        raise ValueError("An example needs a family name")
    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be an object")
    if "cases" in simulation:
        if not isinstance(simulation["cases"], list) or len(simulation["cases"]) != 1:
            raise ValueError("Each example must contain exactly one case")
        case = simulation["cases"][0]
    else:
        case = simulation
    if not isinstance(case, dict):
        raise ValueError("Each case must be an object")
    if domain != "fluid" and case.get("family", case.get("scene")) != config["family"]:
        raise ValueError("Example family differs from simulation family")
    if domain == "fluid":
        for key in ["Re_list", "pulse_variants", "pulse_bands"]:
            if key in case and (not isinstance(case[key], list) or len(case[key]) != 1):
                raise ValueError(f"{key} must select exactly one value")
    material = case.get("material", {})
    if "E" in material and material["E"] <= 0:
        raise ValueError("Young modulus must be positive")
    if "nu" in material and not -1 < material["nu"] < 0.5:
        raise ValueError("Poisson ratio must be between -1 and 0.5")

    def finite(value):
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Parameters must be finite")

    finite(config)
    render = config.get("render", {})
    if not isinstance(render, dict):
        raise ValueError("render must be an object")
    for key in ("width", "height", "fps"):
        value = render.get(key, {"width": 1280, "height": 720, "fps": 24}[key])
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        if key in ("width", "height") and value % 2:
            raise ValueError(f"{key} must be even for H.264 video")
    seconds = render.get("seconds", 15)
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or seconds <= 0
    ):
        raise ValueError("seconds must be positive")
