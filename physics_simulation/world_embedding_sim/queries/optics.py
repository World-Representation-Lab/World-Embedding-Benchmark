"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

import re

from pathlib import Path

from typing import Any, Optional

RENDERED_FIELD_LABEL = "normalized irradiance field"

FORBIDDEN_QUERY_TERMS = {
    "backend",
    "solver",
    "mitsuba",
    "hcipy",
    "meep",
    "surrogate",
    "case_id",
}

PHENOMENON_LABELS = {
    "planeMirrorReflection": "plane mirror reflection",
    "concaveMirrorFocus": "concave mirror focusing",
    "dielectricBlockRefraction": "dielectric block refraction",
    "totalInternalReflection": "total internal reflection",
    "fresnelBrewster": "Fresnel Brewster-angle reflection",
    "beamSplitterPlate": "beam-splitter plate branching",
    "singleSlitFraunhofer": "single-slit Fraunhofer diffraction",
    "doubleSlitInterference": "double-slit interference",
    "circularApertureAiry": "circular-aperture Airy diffraction",
    "knifeEdgeFresnel": "knife-edge Fresnel diffraction",
    "diffractionGrating": "multi-slit diffraction grating",
    "thinFilmInterference": "thin-film interference",
    "gaussianBeamPropagation": "Gaussian beam propagation",
    "prismDispersion": "prism dispersion",
    "thinLensFocus": "thin-lens focusing",
    "glassSphereCaustic": "glass-sphere caustic formation",
    "cornerCubeRetroreflector": "corner-cube retroreflection",
    "polarizerMalus": "Malus-law polarizer transmission",
    "waveplatePolarization": "waveplate polarization modulation",
    "fdtdCylinderScattering": "full-wave cylinder scattering",
    "waveguideModePropagation": "waveguide mode propagation",
    "photonicCrystalDefectMode": "photonic-crystal defect localization",
    "plasmonicNanoDimerHotspot": "plasmonic dimer near-field hotspot",
    "evanescentNearFieldTIR": "evanescent near field under total internal reflection",
    "meepCylinderScattering": "full-wave cylinder scattering",
    "meepWaveguideMode": "full-wave waveguide mode propagation",
    "meepPhotonicCrystalDefect": "photonic-crystal defect localization",
    "meepPhotonicCrystalWaveguide": "photonic-crystal line-defect guiding",
    "meepGratingDiffraction": "full-wave diffraction grating",
    "meepPlasmonicDimer": "plasmonic dimer near-field hotspot",
    "meepDipoleRadiation": "dipole radiation pattern",
    "meepEvanescentTIR": "evanescent near field under total internal reflection",
}

PHENOMENON_SIGNATURES = {
    "planeMirrorReflection": "a specular reflected beam leaving a flat mirror",
    "concaveMirrorFocus": "a curved mirror concentrating reflected light near a focus",
    "dielectricBlockRefraction": "a transmitted beam bending through a dielectric block",
    "totalInternalReflection": "a high-index interface transitioning into total internal reflection",
    "fresnelBrewster": "a reflectance minimum around the Brewster condition",
    "beamSplitterPlate": "one incident beam dividing into reflected and transmitted branches",
    "singleSlitFraunhofer": "a broad diffraction envelope from a narrow slit",
    "doubleSlitInterference": "regular bright and dark interference fringes",
    "circularApertureAiry": "a central Airy spot with surrounding diffraction rings",
    "knifeEdgeFresnel": "a shadow boundary with Fresnel ripples",
    "diffractionGrating": "multiple narrow diffraction orders separating as wavelength changes",
    "thinFilmInterference": "spectral bright and dark bands from a thin dielectric film",
    "gaussianBeamPropagation": "a Gaussian laser spot expanding away from its beam waist",
    "prismDispersion": "wavelength-dependent angular separation through a prism",
    "thinLensFocus": "a lens bringing a beam to a compact focal spot",
    "glassSphereCaustic": "a transparent sphere producing a refractive caustic hotspot",
    "cornerCubeRetroreflector": "a beam returning toward its incoming direction after multiple reflections",
    "polarizerMalus": "a rotating analyzer cue with transmitted light fading and recovering",
    "waveplatePolarization": "polarized intensity modulation as a birefringent waveplate rotates",
    "fdtdCylinderScattering": "a plane electromagnetic wave scattering around a cylinder",
    "waveguideModePropagation": "a confined electromagnetic mode guided along a dielectric channel",
    "photonicCrystalDefectMode": "field energy localized around a defect inside a periodic lattice",
    "plasmonicNanoDimerHotspot": "a bright near-field hotspot concentrated in a narrow dimer gap",
    "evanescentNearFieldTIR": "an interface-bound evanescent field decaying away from total internal reflection",
    "meepCylinderScattering": "a plane electromagnetic wave scattering around a dielectric cylinder",
    "meepWaveguideMode": "a confined electromagnetic mode guided along a dielectric channel",
    "meepPhotonicCrystalDefect": "field energy localized around a defect inside a periodic lattice",
    "meepPhotonicCrystalWaveguide": "a wave packet guided through a missing-row channel in a photonic crystal",
    "meepGratingDiffraction": "a full-wave diffraction fan forming behind a periodic grating",
    "meepPlasmonicDimer": "a bright near-field hotspot concentrated in a narrow dimer gap",
    "meepDipoleRadiation": "concentric electromagnetic radiation expanding from a point dipole",
    "meepEvanescentTIR": "an interface-bound evanescent field decaying away from total internal reflection",
}

VISUAL_DYNAMICS = {
    "planeMirrorReflection": "the incident ray bundle sweeps while the reflected branch stays mirror-symmetric",
    "concaveMirrorFocus": "the reflected light tightens into and then relaxes away from a focal region",
    "dielectricBlockRefraction": "the transmitted beam bends inside the block and exits at a shifted position",
    "totalInternalReflection": "the refracted branch fades while the reflected branch brightens near the critical condition",
    "fresnelBrewster": "the reflected branch dims strongly near the Brewster-angle frame",
    "beamSplitterPlate": "two outgoing branches remain visible while their relative brightness changes",
    "singleSlitFraunhofer": "the central diffraction lobe widens or narrows with side lobes around it",
    "doubleSlitInterference": "bright and dark fringes shift while retaining a regular spacing pattern",
    "circularApertureAiry": "a central Airy spot is surrounded by circular diffraction rings",
    "knifeEdgeFresnel": "a moving shadow boundary carries rippled Fresnel fringes",
    "diffractionGrating": "separated diffraction orders move apart or together across the screen",
    "thinFilmInterference": "thin-film bands brighten, darken, and slide as the optical thickness changes",
    "gaussianBeamPropagation": "the beam waist appears as the spot contracts and expands along propagation",
    "prismDispersion": "colored ray branches spread apart after crossing the prism",
    "thinLensFocus": "the beam contracts into a compact focal spot near the lens focus",
    "glassSphereCaustic": "a refractive hotspot forms behind the transparent sphere",
    "cornerCubeRetroreflector": "the return ray bundle points back toward the incoming direction",
    "polarizerMalus": "the analyzer direction changes while the transmitted ray bundle visibly fades and recovers",
    "waveplatePolarization": "a rotating waveplate changes the transmitted polarized brightness",
    "fdtdCylinderScattering": "wavefronts wrap around the scatterer and form changing side lobes",
    "waveguideModePropagation": "the guided field remains concentrated in the core while weak leakage surrounds it",
    "photonicCrystalDefectMode": "the periodic structure suppresses propagation while the defect traps energy",
    "plasmonicNanoDimerHotspot": "the near field concentrates in the inter-particle gap with strong localized contrast",
    "evanescentNearFieldTIR": "the transmitted near field fades exponentially away from the interface",
    "meepCylinderScattering": "the incident wavefront reaches the cylinder and then bends into scattered side lobes",
    "meepWaveguideMode": "the pulse enters the guide and remains concentrated around the dielectric core",
    "meepPhotonicCrystalDefect": "the pulse interacts with the periodic lattice and leaves energy trapped around the defect",
    "meepPhotonicCrystalWaveguide": "the wavefront follows the line defect while the surrounding lattice suppresses leakage",
    "meepGratingDiffraction": "the incident wavefront crosses the grating and separates into visible diffraction orders",
    "meepPlasmonicDimer": "the wavefront drives the particles and creates a localized gap hotspot",
    "meepDipoleRadiation": "radiation expands outward from the dipole with a directional field pattern",
    "meepEvanescentTIR": "the field follows the interface while its upper-side intensity fades away from the boundary",
}

PHYSICS_MECHANISM = {
    "planeMirrorReflection": "the reflected beam obeys the law of reflection, so its direction mirrors the incident direction about the surface normal",
    "concaveMirrorFocus": "the curved mirror converges the reflected rays, concentrating light near the focal region",
    "dielectricBlockRefraction": "Snell's law bends the transmitted beam at each face, so the exit beam leaves laterally shifted",
    "totalInternalReflection": "beyond the critical angle the transmitted branch vanishes and the interface reflects nearly all of the light",
    "fresnelBrewster": "p-polarized reflection weakens toward the Brewster angle, so the reflected branch dims near that condition",
    "beamSplitterPlate": "the plate divides the incident energy between a reflected and a transmitted branch whose balance depends on the plate geometry",
    "singleSlitFraunhofer": "a narrower slit diffracts light more strongly, widening the central lobe of the far-field pattern",
    "doubleSlitInterference": "path-length differences between the two slits create alternating constructive and destructive fringes",
    "circularApertureAiry": "diffraction at the circular aperture forms an Airy disk whose ring scale follows wavelength over aperture size",
    "knifeEdgeFresnel": "near-field diffraction at the straight edge produces intensity ripples beside a softened shadow boundary",
    "diffractionGrating": "the periodic slits send light into discrete diffraction orders whose angles depend on wavelength and period",
    "thinFilmInterference": "reflections from the two film surfaces interfere, so the bands shift with optical thickness and wavelength",
    "gaussianBeamPropagation": "the beam is narrowest at its waist and spreads with a divergence set by the waist size",
    "prismDispersion": "the refractive index varies with wavelength, so different colors leave the prism at different angles",
    "thinLensFocus": "the lens converges the beam toward the image point, and the spot is smallest at best focus",
    "glassSphereCaustic": "refraction through the sphere folds the rays into a bright caustic concentrated behind it",
    "cornerCubeRetroreflector": "three mutually perpendicular reflections send the beam back parallel to its arrival direction",
    "polarizerMalus": "the transmitted intensity follows Malus's law, falling toward zero as the analyzer approaches the crossed orientation",
    "waveplatePolarization": "the waveplate retardance reshapes the polarization state, modulating the intensity passed by the analyzer",
    "meepCylinderScattering": "the incident wave excites the cylinder, which re-radiates and interferes with the incoming wavefront to form side lobes",
    "meepWaveguideMode": "index contrast confines the field, so the mode travels along the core with little sideways leakage",
    "meepPhotonicCrystalDefect": "the lattice band gap blocks propagation while the defect supports a localized resonant field",
    "meepPhotonicCrystalWaveguide": "the missing-row channel guides the wave while the surrounding band-gap lattice suppresses sideways leakage",
    "meepGratingDiffraction": "the periodic structure splits the transmitted wave into discrete diffraction orders behind the grating",
    "meepPlasmonicDimer": "the driven particles couple across the gap, strongly enhancing the near field between them",
    "meepDipoleRadiation": "the oscillating dipole emits expanding wavefronts with an angular pattern set by its orientation",
    "meepEvanescentTIR": "beyond the critical angle the transmitted field becomes evanescent and decays exponentially away from the interface",
}

COLOR_MAP_CUE = {
    "planeMirrorReflection": "bright ray paths trace the incident and mirrored branches against the darker background",
    "concaveMirrorFocus": "the brightest colors gather where the reflected rays crowd toward the focal region",
    "dielectricBlockRefraction": "bright colors follow the beam as it bends inside the block and exits displaced",
    "totalInternalReflection": "brightness shifts between the transmitted and reflected branches as the interface approaches total reflection",
    "fresnelBrewster": "the reflected branch visibly dims near the Brewster frames while the transmitted branch stays lit",
    "beamSplitterPlate": "the two outgoing branches carry complementary brightness levels",
    "singleSlitFraunhofer": "a bright central lobe stands over weaker side lobes on the screen",
    "doubleSlitInterference": "evenly spaced bright and dark bands appear across the observation screen",
    "circularApertureAiry": "a bright central disk is ringed by progressively fainter circles",
    "knifeEdgeFresnel": "the lit side shows ripples that fade into the darker geometric shadow",
    "diffractionGrating": "each diffraction order appears as a separate bright streak on the screen",
    "thinFilmInterference": "the film bands brighten and darken as the interference condition changes",
    "gaussianBeamPropagation": "the brightest, tightest colors mark the beam waist along the propagation path",
    "prismDispersion": "separated bright branches fan out behind the prism at wavelength-dependent angles",
    "thinLensFocus": "brightness peaks sharply where the converging beam reaches its smallest spot",
    "glassSphereCaustic": "an intense hotspot appears behind the sphere where the refracted rays pile up",
    "cornerCubeRetroreflector": "the returning bundle stays bright and heads back toward the source direction",
    "polarizerMalus": "the transmitted path visibly fades toward the crossed setting and recovers afterwards",
    "waveplatePolarization": "the transmitted brightness cycles as the waveplate orientation advances",
    "meepCylinderScattering": "wavefront stripes wrap around the cylinder and scattered lobes stand out around it",
    "meepWaveguideMode": "the strongest colors stay concentrated inside the guiding core",
    "meepPhotonicCrystalDefect": "intensity remains trapped as a bright pocket at the lattice defect",
    "meepPhotonicCrystalWaveguide": "the bright trace follows the line-defect channel through the lattice",
    "meepGratingDiffraction": "distinct bright fans emerge behind the grating where the orders separate",
    "meepPlasmonicDimer": "the most intense colors condense inside the narrow gap between the particles",
    "meepDipoleRadiation": "concentric bright shells expand outward from the source point",
    "meepEvanescentTIR": "brightness clings to the interface and fades quickly with distance above it",
}

VISUAL_FIRST_FIELD_PHRASES = {
    "scan_range": [
        "the animation covers a visible sweep of the controlled optical parameter",
        "the sequence shows the response across the scan interval",
        "the frames progress through a parameter sweep rather than a static setup",
    ],
    "scan_count": [
        "the response is sampled as a smooth multi-frame progression",
        "the video contains enough states to show the transition continuously",
        "the scan is rendered as a temporal sequence of optical states",
    ],
    "peak_irradiance_max": [
        "one state contains a clearly bright irradiance crest",
        "the brightest frame has a strong localized intensity peak",
        "the rendered field includes a high-intensity highlight",
    ],
    "peak_irradiance_mean": [
        "the typical frame keeps a visible irradiance core",
        "the sequence maintains a recognizable bright response over time",
        "the average frame brightness remains visually measurable",
    ],
    "integrated_power_mean": [
        "the receiver collects a visibly sustained amount of light",
        "the overall brightness level remains apparent across the sequence",
        "the video shows a stable integrated light response",
    ],
    "spot_shift_span": [
        "the beam footprint visibly shifts across the detector",
        "the illuminated spot moves over the observation plane",
        "the detector pattern changes position during the scan",
    ],
    "reflection_efficiency_mean": [
        "the reflected branch carries a visible share of the light",
        "the mirror-side response remains apparent in the animation",
        "the reflected ray bundle has a noticeable brightness level",
    ],
    "transmission_efficiency_mean": [
        "the transmitted branch remains visible after the optical element",
        "the outgoing beam carries a noticeable fraction of the light",
        "the transmitted pattern is strong enough to follow visually",
    ],
    "snell_residual_deg": [
        "the refracted ray direction remains consistent with Snell-style bending",
        "the entrance and exit angles line up with a physically plausible refraction path",
        "the block-bending geometry appears coherent across the scan",
    ],
    "split_balance_error": [
        "the two splitter branches differ by a visible but controlled brightness balance",
        "the reflected and transmitted branches form a recognizable power split",
        "the beam splitter produces two simultaneously visible paths",
    ],
    "branch_separation": [
        "the outgoing branches are spatially separated on the detector",
        "the split rays are visually distinguishable as different paths",
        "the branch spacing is large enough to separate the two beams",
    ],
    "central_lobe_width": [
        "the central diffraction lobe has a visible width",
        "the main coherent lobe dominates the diffraction pattern",
        "the diffraction core is broad enough to compare against side structure",
    ],
    "fringe_visibility": [
        "the bright and dark fringes have clear contrast",
        "the interference bands remain visually resolvable",
        "the video shows a high-contrast fringe pattern",
    ],
    "fringe_spacing": [
        "adjacent fringes are visibly separated",
        "the interference bands form a regular spacing pattern",
        "the fringe pitch can be read from the rendered field",
    ],
    "fringe_shift_span": [
        "the interference pattern visibly slides during the scan",
        "the fringe maxima drift across the observation plane",
        "the banded pattern shifts position over time",
    ],
    "airy_ring_radius": [
        "the first Airy ring is visible around the central spot",
        "the circular diffraction pattern has a readable ring scale",
        "the Airy disk is accompanied by a surrounding ring",
    ],
    "edge_transition_width": [
        "the knife-edge shadow boundary spreads across a visible transition zone",
        "the blocked-to-lit boundary is not an abrupt flat cutoff",
        "the edge transition is wide enough to reveal diffraction structure",
    ],
    "edge_overshoot": [
        "a Fresnel overshoot appears near the shadow boundary",
        "the edge ripple produces a bright band near the cutoff",
        "the diffraction edge shows a visible overshoot feature",
    ],
    "grating_order_spacing": [
        "multiple grating orders are separated into distinct bright spots",
        "the diffracted orders appear as separated peaks",
        "the multi-slit pattern forms discrete spatial orders",
    ],
    "grating_order_count": [
        "several diffraction orders are visible at once",
        "the grating pattern resolves multiple bright orders",
        "the detector shows more than a single central order",
    ],
    "spectral_spread": [
        "wavelength variation visibly changes the order spacing",
        "the spectral scan moves the diffraction orders across the screen",
        "the order pattern spreads or contracts with wavelength",
    ],
    "thin_film_reflectance_mean": [
        "the film retains a visible reflected intensity level",
        "the reflected thin-film pattern remains bright enough to follow",
        "the coating produces a sustained reflected response",
    ],
    "thin_film_contrast": [
        "the thin-film bands have clear bright-dark contrast",
        "the reflected interference colors or bands are visually separated",
        "the film pattern contains visible alternating intensity bands",
    ],
    "thin_film_fringe_period_nm": [
        "the film bands change over a visible optical-thickness period",
        "the reflected bands cycle as the film thickness changes",
        "the thin-film pattern visibly repeats over the scan",
    ],
    "beam_waist_min": [
        "the Gaussian beam reaches a tight visible waist",
        "the laser spot contracts to a narrow core before expanding",
        "the propagation sequence includes a compact beam waist",
    ],
    "rayleigh_range": [
        "the beam stays narrow over a visible propagation length before spreading",
        "the waist region extends across part of the propagation scan",
        "the spot-size evolution has a recognizable Rayleigh-scale transition",
    ],
    "divergence_angle_deg": [
        "the far-field beam visibly spreads after the waist",
        "the propagated spot expands with a readable divergence",
        "the laser beam opens gradually over the sequence",
    ],
    "critical_angle_est_deg": [
        "the interface passes through a visible critical-angle transition",
        "the refracted branch disappears around the TIR threshold",
        "the reflected branch dominates after the critical transition",
    ],
    "critical_angle_error_deg": [
        "the critical-angle transition is visually aligned with the expected threshold",
        "the TIR onset appears at a physically consistent angle",
        "the reflection-dominated regime begins where the geometry predicts it",
    ],
    "brewster_angle_est_deg": [
        "the reflected branch reaches a visible minimum near Brewster alignment",
        "the scan includes a dark reflected frame near the Brewster condition",
        "the Brewster dip appears as a clear brightness minimum",
    ],
    "brewster_angle_error_deg": [
        "the Brewster minimum is visually aligned with the expected angle",
        "the reflectance dip occurs where the interface geometry predicts it",
        "the low-reflection frame is physically consistent with Brewster behavior",
    ],
    "dispersion_span": [
        "the red and blue ray paths separate visibly after the prism",
        "the prism produces a readable chromatic fan",
        "the wavelength-dependent branches spread apart on the detector",
    ],
    "focus_distance_best": [
        "one scan state gives the sharpest focal spot",
        "the detector passes through a best-focus position",
        "the sequence contains a visually compact focus frame",
    ],
    "focus_gain": [
        "the focused state is visibly brighter than defocused states",
        "the optical element produces a clear concentration of intensity",
        "the best-focus frame stands out from the blurrier frames",
    ],
    "min_spot_radius": [
        "the focal spot becomes visibly compact",
        "the smallest blur radius appears as a tight bright core",
        "the focused beam footprint is narrow in one part of the scan",
    ],
    "caustic_gain": [
        "the caustic appears as a concentrated refractive hotspot",
        "the sphere forms a visibly amplified bright region",
        "the refractive focus produces a sharp caustic highlight",
    ],
    "retroreflection_error_deg": [
        "the returning ray bundle is nearly antiparallel to the incoming path",
        "the corner cube sends light back toward the source direction",
        "the return path visually tracks the incident direction in reverse",
    ],
    "return_efficiency_mean": [
        "the returning beam remains visibly bright",
        "the retroreflected branch carries a noticeable amount of light",
        "the return path is strong enough to follow across the sequence",
    ],
    "return_spot_offset": [
        "the returned footprint stays close to the incoming path",
        "the retroreflected spot lands near the expected return position",
        "the return spot offset is small enough to preserve retroreflection",
    ],
    "malus_r2": [
        "the brightness follows the expected Malus-law fade-and-recover trend",
        "the transmitted beam dims and brightens in a cosine-squared-like rhythm",
        "the analyzer sweep produces a physically smooth polarization intensity curve",
    ],
    "extinction_ratio": [
        "crossed-polarizer states become visibly darker than aligned states",
        "the dark frames reveal the analyzer's extinction behavior",
        "the transmitted ray bundle strongly fades near crossed alignment",
    ],
    "waveplate_modulation_depth": [
        "the waveplate causes a visible transmitted-intensity modulation",
        "the polarized brightness rises and falls as the fast axis rotates",
        "the birefringent element creates a clear brightness oscillation",
    ],
    "polarization_contrast": [
        "the polarization state produces high-contrast analyzer frames",
        "the analyzer output alternates between bright and dim states",
        "the transmitted polarized light has a visible contrast cycle",
    ],
    "retardance_error_deg": [
        "the waveplate behavior remains close to a recognizable quarter- or half-wave response",
        "the birefringent phase delay produces the expected modulation family",
        "the polarization modulation stays physically consistent with the chosen waveplate type",
    ],
    "scattering_contrast": [
        "the cylinder produces visible scattered side-lobe contrast",
        "the field around the particle is not a flat transmitted wave",
        "the wavefront shows a readable scattering disturbance around the cylinder",
    ],
    "forward_power_proxy": [
        "a transmitted wave remains visible downstream of the scatterer",
        "the forward side of the scene still carries outgoing field energy",
        "the downstream field stays bright enough to track after scattering",
    ],
    "backward_power_proxy": [
        "a weaker back-scattered field is visible on the incident side",
        "the incident side contains a readable reflected-scattering component",
        "the backward lobe remains visible near the incoming wave",
    ],
    "mode_confinement": [
        "the guided mode remains visibly concentrated near the channel core",
        "the waveguide core contains most of the bright field structure",
        "the channel shows clear modal confinement",
    ],
    "waveguide_leakage": [
        "weak leakage is visible outside the guided channel",
        "the surrounding region shows the residual field escaping the core",
        "the leakage halo around the waveguide is visually distinguishable",
    ],
    "localization_ratio": [
        "the defect region contains a localized bright field pocket",
        "energy visibly collects around the defect inside the periodic lattice",
        "the defect stands out from the surrounding photonic-crystal pattern",
    ],
    "line_defect_confinement": [
        "the line defect channels the wave through the photonic crystal",
        "the bright field follows the missing-row guide",
        "the lattice confines the wave to the line-defect path",
    ],
    "right_transmission_proxy": [
        "the guided wave reaches the downstream side of the structure",
        "a transmitted field appears beyond the photonic-crystal channel",
        "the right side shows the outgoing guided response",
    ],
    "hotspot_enhancement": [
        "the dimer gap forms a high-contrast near-field hotspot",
        "the field between the particles is visibly enhanced",
        "a localized bright spot appears in the narrow particle gap",
    ],
    "gap_localization_ratio": [
        "the hotspot remains spatially concentrated in the dimer gap",
        "the strongest field is visually tied to the inter-particle gap",
        "the dimer response localizes the near field between the particles",
    ],
    "radiation_shell_contrast": [
        "the dipole radiation pattern has directional angular contrast",
        "the expanding wavefront is not azimuthally featureless",
        "the radiation shell shows a visible dipole-like pattern",
    ],
    "radiated_power_proxy": [
        "the outgoing dipole wave carries visible field energy",
        "the radiation front remains bright enough to track outward",
        "the expanding field has a sustained visible amplitude",
    ],
    "evanescent_decay_length": [
        "the upper-side evanescent field decays away from the interface",
        "the interface-bound field fades over a visible decay length",
        "the near field weakens as distance from the boundary increases",
    ],
    "upper_field_fraction": [
        "a measurable evanescent tail appears above the interface",
        "the low-index side contains a visible but decaying near field",
        "the upper region shows the interface-bound evanescent response",
    ],
}

SCAN_LABELS = {
    "incident_angle_deg": "incident angle",
    "incident_trim_deg": "incident-angle trim",
    "sensor_distance": "sensor distance",
    "source_offset": "source offset",
    "aperture_scale": "aperture scale",
    "slit_width": "slit width",
    "slit_separation": "slit separation",
    "phase_delay_deg": "relative phase delay",
    "edge_position": "knife-edge position",
    "edge_offset": "knife-edge offset",
    "incident_tilt_deg": "incident tilt",
    "wavelength_nm": "wavelength",
    "film_thickness_nm": "film thickness",
    "propagation_distance": "propagation distance",
    "waveplate_angle_deg": "waveplate fast-axis angle",
    "analyzer_angle_deg": "analyzer angle",
    "sphere_radius": "sphere radius",
    "lens_focal_length": "lens focal length",
    "detector_distance": "detector distance",
    "field_time": "field evolution time",
}


def _case_dim(meta: dict[str, Any]) -> str:
    dim = str(meta.get("dim", "")).strip().upper()
    return dim if dim in {"2D", "3D"} else "2D"


def _video_for_case(case_dir: Path) -> Optional[str]:
    case_name = case_dir.name
    candidates = [
        case_dir.parents[1] / "videos" / f"{case_name}_opt.mp4",
        case_dir.parents[2] / "videos" / "flat" / f"{case_name}_opt.mp4",
        case_dir.parents[2] / "videos" / f"{case_name}_opt.mp4",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _fmt(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}g}"
    except Exception:
        return "n/a"


def _float_or_none(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _about_deg(v: Any, step: int = 1) -> str:
    val = _float_or_none(v)
    if val is None:
        return "an unspecified angle"
    if step <= 1:
        return f"about {val:.1f} deg"
    rounded = int(round(val / max(1, step)) * max(1, step))
    return f"about {rounded} deg"


def _scan_path_visual(parameter: str, start: Any, stop: Any, units: Any) -> str:
    label = _scan_label(parameter)
    unit = str(units or "").strip().lower()
    if unit in {"deg", "degree", "degrees"}:
        return f"{label} moves from {_about_deg(start)} to {_about_deg(stop)}"
    a = _fmt(start, 3)
    b = _fmt(stop, 3)
    return f"{label} moves from about {a} to about {b}".strip()


def _temporal_sampling_class(count: Any) -> str:
    n = _float_or_none(count)
    if n is None:
        return "multi-frame temporal sampling"
    if n <= 38:
        return "coarser temporal sampling"
    if n <= 44:
        return "moderately smooth temporal sampling"
    if n <= 50:
        return "smooth temporal sampling"
    return "very smooth temporal sampling"


def _bucket_label(bucket: Any) -> str:
    labels = {
        "high_extinction": "near-crossed frames become very dark",
        "low_extinction": "the dimmest frames retain visible leakage",
        "near_crossed": "the analyzer path clearly passes through a dark crossing",
        "near_parallel": "the analyzer path emphasizes a bright near-parallel interval",
        "small_particle": "a smaller scatterer perturbs the incident wavefront",
        "resonant_size": "a wavelength-scale scatterer creates stronger side lobes",
        "high_index": "higher material contrast strengthens the field disturbance",
        "low_index": "lower material contrast leaves a softer field disturbance",
        "narrow_core": "a narrow guide concentrates the guided mode",
        "wide_core": "a wider guide supports a broader modal stripe",
        "high_contrast": "large index contrast confines the mode tightly",
        "leaky_mode": "a lower-contrast guide shows a visible leakage halo",
        "compact_defect": "the defect hotspot is compact inside the lattice",
        "wide_defect": "the defect pocket spreads across a wider central region",
        "small_holes": "smaller lattice rods leave weaker periodic modulation",
        "large_holes": "larger lattice rods create stronger periodic blocking",
        "straight_line_defect": "the wave follows a clean straight line defect",
        "tight_lattice": "a tighter lattice surrounds the guided channel",
        "wide_lattice": "a wider lattice spacing changes the channel texture",
        "strong_scatterers": "stronger lattice scatterers sharpen the channel contrast",
        "few_orders": "the grating produces a small number of separated orders",
        "many_orders": "the grating produces several separated diffraction orders",
        "wide_period": "a wider period spreads the diffraction geometry",
        "narrow_period": "a narrow period compresses the diffraction geometry",
        "narrow_gap": "a narrow particle gap creates a compact near-field hotspot",
        "wide_gap": "a wider dimer gap weakens and broadens the hotspot",
        "small_particles": "smaller particles create a tighter near-field response",
        "large_particles": "larger particles create a broader dimer response",
        "short_wavelength": "shorter-wavelength radiation forms tighter wavefront spacing",
        "long_wavelength": "longer-wavelength radiation forms wider wavefront spacing",
        "x_oriented": "the dipole orientation changes the angular radiation pattern",
        "z_polarized": "the out-of-plane dipole component gives a symmetric expanding pattern",
        "near_critical": "the evanescent tail is longer near the critical condition",
        "deep_tir": "deeper total internal reflection gives a shorter near-field tail",
        "shallow_decay": "the evanescent field remains visible farther from the interface",
    }
    raw = str(bucket or "").strip()
    return labels.get(
        raw, _public_label(raw) if raw else "a visually distinct scale regime"
    )


def _angle_pose_label(angle: Any) -> str:
    val = _float_or_none(angle)
    if val is None:
        return "unspecified input polarization tilt"
    if val <= -25:
        band = "strong negative"
    elif val <= -10:
        band = "moderate negative"
    elif val < 10:
        band = "near-horizontal"
    elif val < 25:
        band = "moderate positive"
    else:
        band = "strong positive"
    return f"{band} input polarization tilt ({_about_deg(val, step=5)})"


def _extinction_label(value: Any) -> str:
    val = _float_or_none(value)
    if val is None:
        return "unspecified extinction contrast"
    if val < 80:
        return "soft extinction contrast"
    if val < 180:
        return "moderate extinction contrast"
    if val < 300:
        return "deep extinction contrast"
    return "very deep extinction contrast"


def _brightness_label(value: Any) -> str:
    val = _float_or_none(value)
    if val is None:
        return "visible brightness level"
    if val < 0.35:
        return "lower average brightness"
    if val < 0.55:
        return "middle average brightness"
    return "higher average brightness"


def _power_label(value: Any) -> str:
    val = _float_or_none(value)
    if val is None:
        return "visible collected light level"
    if val < 60:
        return "lower collected-light level"
    if val < 95:
        return "middle collected-light level"
    return "higher collected-light level"


def _usable(v: Any) -> bool:
    s = str(v).strip().lower()
    return s not in {"", "n/a", "none", "null", "unknown", "[]", "{}"}


def _public_label(value: Any) -> str:
    raw = str(value).strip()
    if not raw:
        return "unknown"
    if raw == "meep_fdtd":
        return "full-wave electromagnetic FDTD model"
    if raw in PHENOMENON_LABELS:
        return PHENOMENON_LABELS[raw]
    if raw in SCAN_LABELS:
        return SCAN_LABELS[raw]
    spaced = raw.replace("_", " ").replace("-", " ")
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    return " ".join(spaced.split()).lower()


def _phenomenon_label(family: Any) -> str:
    return PHENOMENON_LABELS.get(str(family), _public_label(family))


def _phenomenon_signature(family: Any) -> str:
    return PHENOMENON_SIGNATURES.get(
        str(family), f"a {_public_label(family)} optical response"
    )


def _scan_label(value: Any) -> str:
    return SCAN_LABELS.get(str(value), _public_label(value))


def _assert_query_text_allowed(text: str) -> None:
    lowered = text.lower()
    hits = sorted(
        term
        for term in FORBIDDEN_QUERY_TERMS
        if re.search(rf"\b{re.escape(term)}\b", lowered)
    )
    if hits:
        raise ValueError(
            f"Query text contains backend/internal tokens: {', '.join(hits)}"
        )


def _pick(options: list[str], style: int) -> str:
    return options[int(style) % len(options)]


def _join_phrases(parts: list[str]) -> str:
    vals = [p.strip() for p in parts if p and p.strip()]
    if not vals:
        return "the recorded optics attributes"
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return f"{', '.join(vals[:-1])}, and {vals[-1]}"


def _quality_from_self_check(
    self_check: dict[str, Any], fallback: dict[str, Any]
) -> dict[str, Any]:
    status = str(self_check.get("status", "")).upper()
    flags = list(self_check.get("hard_flags", [])) + list(
        self_check.get("soft_flags", [])
    )
    if status == "PASS":
        return {"tier": "gold", "reason": []}
    if status == "SUSPECT":
        return {"tier": "silver", "reason": flags or fallback.get("reason", [])}
    if status == "FAIL":
        return {"tier": "bronze", "reason": flags or fallback.get("reason", [])}
    return fallback


def _case_record(
    *,
    case_dir: Path,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    metadata_row: dict[str, Any],
    self_check: dict[str, Any],
) -> dict[str, Any]:
    family = str(
        meta.get("family") or meta.get("case_prefix") or case_dir.name.split("_", 1)[0]
    )
    optics = meta.get("optics") if isinstance(meta.get("optics"), dict) else {}
    scan = meta.get("scan") if isinstance(meta.get("scan"), dict) else {}
    scale = meta.get("scale") if isinstance(meta.get("scale"), dict) else {}
    solver = meta.get("solver") if isinstance(meta.get("solver"), dict) else {}
    illumination = (
        meta.get("illumination") if isinstance(meta.get("illumination"), dict) else {}
    )
    receiver = meta.get("receiver") if isinstance(meta.get("receiver"), dict) else {}
    backend = str(metrics.get("backend", metadata_row.get("backend", "unknown")))
    fallback_quality = {
        "tier": metadata_row.get("quality_tier", "silver"),
        "reason": metadata_row.get("quality_reason", ["auto"]),
    }
    self_check_flags = list(self_check.get("hard_flags", [])) + list(
        self_check.get("soft_flags", [])
    )

    return {
        "case_id": case_dir.name,
        "case_path": str(case_dir.resolve()),
        "family": family,
        "params": {
            "dim": _case_dim(meta),
            "backend": backend,
            "optics_model": optics.get("model", family),
            "scan_parameter": scan.get("parameter"),
            "scan_units": scan.get("units"),
            "scan_count": scan.get("count"),
            "scale_bucket": scale.get("bucket"),
            "samples_per_pixel": solver.get("samples_per_pixel"),
            "variant_preferences": solver.get("variant_preferences"),
        },
        "scene_spec": {
            "family": family,
            "dim": _case_dim(meta),
            "optics": optics,
            "illumination": illumination,
            "receiver": receiver,
            "scan": scan,
            "scale": scale,
            "solver": f"{backend}_optics",
        },
        "assets": {
            "video_path": _video_for_case(case_dir),
            "field_pvd_path": str(case_dir / "results" / "sensor_fields.pvd"),
            "ray_pvd_path": str(case_dir / "results" / "ray_paths.pvd"),
            "metrics_path": str(case_dir / "postProcessing" / "metrics.json"),
            "self_check_path": str(case_dir / "postProcessing" / "self_check.json"),
        },
        "observables": {
            "peak_irradiance_max": metrics.get("peak_irradiance_max"),
            "peak_irradiance_mean": metrics.get("peak_irradiance_mean"),
            "integrated_power_mean": metrics.get("integrated_power_mean"),
            "spot_shift_span": metrics.get("spot_shift_span"),
            "reflection_efficiency_mean": metrics.get("reflection_efficiency_mean"),
            "transmission_efficiency_mean": metrics.get("transmission_efficiency_mean"),
            "snell_residual_deg": metrics.get("snell_residual_deg"),
            "split_balance_error": metrics.get("split_balance_error"),
            "branch_separation": metrics.get("branch_separation"),
            "central_lobe_width": metrics.get("central_lobe_width"),
            "fringe_visibility": metrics.get("fringe_visibility"),
            "fringe_spacing": metrics.get("fringe_spacing"),
            "fringe_shift_span": metrics.get("fringe_shift_span"),
            "airy_ring_radius": metrics.get("airy_ring_radius"),
            "edge_transition_width": metrics.get("edge_transition_width"),
            "edge_overshoot": metrics.get("edge_overshoot"),
            "grating_order_spacing": metrics.get("grating_order_spacing"),
            "grating_order_count": metrics.get("grating_order_count"),
            "spectral_spread": metrics.get("spectral_spread"),
            "thin_film_reflectance_mean": metrics.get("thin_film_reflectance_mean"),
            "thin_film_contrast": metrics.get("thin_film_contrast"),
            "thin_film_fringe_period_nm": metrics.get("thin_film_fringe_period_nm"),
            "beam_waist_min": metrics.get("beam_waist_min"),
            "rayleigh_range": metrics.get("rayleigh_range"),
            "divergence_angle_deg": metrics.get("divergence_angle_deg"),
            "critical_angle_est_deg": metrics.get("critical_angle_est_deg"),
            "critical_angle_error_deg": metrics.get("critical_angle_error_deg"),
            "brewster_angle_est_deg": metrics.get("brewster_angle_est_deg"),
            "brewster_angle_error_deg": metrics.get("brewster_angle_error_deg"),
            "dispersion_span": metrics.get("dispersion_span"),
            "focus_distance_best": metrics.get("focus_distance_best"),
            "focus_gain": metrics.get("focus_gain"),
            "min_spot_radius": metrics.get("min_spot_radius"),
            "caustic_gain": metrics.get("caustic_gain"),
            "retroreflection_error_deg": metrics.get("retroreflection_error_deg"),
            "return_efficiency_mean": metrics.get("return_efficiency_mean"),
            "return_spot_offset": metrics.get("return_spot_offset"),
            "malus_r2": metrics.get("malus_r2"),
            "extinction_ratio": metrics.get("extinction_ratio"),
            "waveplate_modulation_depth": metrics.get("waveplate_modulation_depth"),
            "polarization_contrast": metrics.get("polarization_contrast"),
            "retardance_error_deg": metrics.get("retardance_error_deg"),
            "scattering_contrast": metrics.get("scattering_contrast"),
            "forward_power_proxy": metrics.get("forward_power_proxy"),
            "backward_power_proxy": metrics.get("backward_power_proxy"),
            "mode_confinement": metrics.get("mode_confinement"),
            "waveguide_leakage": metrics.get("waveguide_leakage"),
            "localization_ratio": metrics.get("localization_ratio"),
            "lattice_material_fill": metrics.get("lattice_material_fill"),
            "line_defect_confinement": metrics.get("line_defect_confinement"),
            "right_transmission_proxy": metrics.get("right_transmission_proxy"),
            "hotspot_enhancement": metrics.get("hotspot_enhancement"),
            "gap_localization_ratio": metrics.get("gap_localization_ratio"),
            "radiation_shell_contrast": metrics.get("radiation_shell_contrast"),
            "radiated_power_proxy": metrics.get("radiated_power_proxy"),
            "evanescent_decay_length": metrics.get("evanescent_decay_length"),
            "upper_field_fraction": metrics.get("upper_field_fraction"),
        },
        "simulation_status": {
            "status": "COMPLETED"
            if self_check.get("has_end")
            and self_check.get("has_metrics")
            and self_check.get("has_fields")
            else "INCOMPLETE",
            "log_has_end": bool(self_check.get("has_end")),
            "frames": int(self_check.get("n_frames", metrics.get("n_frames", 0)) or 0),
            "expected_frames": self_check.get("expected_frames"),
        },
        "self_check": self_check,
        "diagnostics": {
            "status": str(
                self_check.get("diagnostic_status", self_check.get("status", "UNKNOWN"))
            ),
            "flags": self_check_flags,
        },
        "quality": _quality_from_self_check(self_check, fallback_quality),
    }


def _case_tokens(case: dict[str, Any]) -> dict[str, str]:
    params = case.get("params") if isinstance(case.get("params"), dict) else {}
    scene = case.get("scene_spec") if isinstance(case.get("scene_spec"), dict) else {}
    optics = scene.get("optics") if isinstance(scene.get("optics"), dict) else {}
    scan = scene.get("scan") if isinstance(scene.get("scan"), dict) else {}
    scale = scene.get("scale") if isinstance(scene.get("scale"), dict) else {}
    polarizers = (
        optics.get("polarizers") if isinstance(optics.get("polarizers"), dict) else {}
    )
    obs = case.get("observables") if isinstance(case.get("observables"), dict) else {}
    family = str(case.get("family", "unknown"))
    scan_parameter = str(params.get("scan_parameter", "unknown"))
    extinction_value = polarizers.get("extinction_ratio", obs.get("extinction_ratio"))
    return {
        "family": family,
        "phenomenon": _phenomenon_label(family),
        "visible_signature": _phenomenon_signature(family),
        "visual_dynamics": VISUAL_DYNAMICS.get(family, _phenomenon_signature(family)),
        "scale_bucket": (
            _bucket_label(params.get("scale_bucket", scale.get("bucket")))
            if _usable(params.get("scale_bucket", scale.get("bucket")))
            else ""
        ),
        "dim": str(params.get("dim", "2D")),
        "rendered_field": RENDERED_FIELD_LABEL,
        "optics_model": _public_label(params.get("optics_model", family)),
        "backend": str(params.get("backend", "unknown")),
        "scan_parameter": _scan_label(scan_parameter),
        "scan_count": str(params.get("scan_count", "n/a")),
        "scan_range": f"{_fmt(scan.get('start'), 4)} to {_fmt(scan.get('stop'), 4)} {str(scan.get('units', '')).strip()}".strip(),
        "scan_path_visual": _scan_path_visual(
            scan_parameter, scan.get("start"), scan.get("stop"), scan.get("units")
        ),
        "temporal_sampling_class": _temporal_sampling_class(params.get("scan_count")),
        "input_polarization_pose": _angle_pose_label(
            polarizers.get("initial_polarization_deg")
        ),
        "extinction_class": _extinction_label(extinction_value),
        "brightness_class": _brightness_label(obs.get("peak_irradiance_mean")),
        "power_class": _power_label(obs.get("integrated_power_mean")),
        "variant_preferences": ",".join(
            str(x) for x in (params.get("variant_preferences") or [])
        )
        or "n/a",
        "peak_irradiance_max": _fmt(obs.get("peak_irradiance_max"), 5),
        "peak_irradiance_mean": _fmt(obs.get("peak_irradiance_mean"), 5),
        "integrated_power_mean": _fmt(obs.get("integrated_power_mean"), 5),
        "spot_shift_span": _fmt(obs.get("spot_shift_span"), 5),
        "reflection_efficiency_mean": _fmt(obs.get("reflection_efficiency_mean"), 4),
        "transmission_efficiency_mean": _fmt(
            obs.get("transmission_efficiency_mean"), 4
        ),
        "snell_residual_deg": _fmt(obs.get("snell_residual_deg"), 5),
        "split_balance_error": _fmt(obs.get("split_balance_error"), 5),
        "branch_separation": _fmt(obs.get("branch_separation"), 5),
        "central_lobe_width": _fmt(obs.get("central_lobe_width"), 5),
        "fringe_visibility": _fmt(obs.get("fringe_visibility"), 5),
        "fringe_spacing": _fmt(obs.get("fringe_spacing"), 5),
        "fringe_shift_span": _fmt(obs.get("fringe_shift_span"), 5),
        "airy_ring_radius": _fmt(obs.get("airy_ring_radius"), 5),
        "edge_transition_width": _fmt(obs.get("edge_transition_width"), 5),
        "edge_overshoot": _fmt(obs.get("edge_overshoot"), 5),
        "grating_order_spacing": _fmt(obs.get("grating_order_spacing"), 5),
        "grating_order_count": _fmt(obs.get("grating_order_count"), 5),
        "spectral_spread": _fmt(obs.get("spectral_spread"), 5),
        "thin_film_reflectance_mean": _fmt(obs.get("thin_film_reflectance_mean"), 5),
        "thin_film_contrast": _fmt(obs.get("thin_film_contrast"), 5),
        "thin_film_fringe_period_nm": _fmt(obs.get("thin_film_fringe_period_nm"), 5),
        "beam_waist_min": _fmt(obs.get("beam_waist_min"), 5),
        "rayleigh_range": _fmt(obs.get("rayleigh_range"), 5),
        "divergence_angle_deg": _fmt(obs.get("divergence_angle_deg"), 5),
        "critical_angle_est_deg": _fmt(obs.get("critical_angle_est_deg"), 4),
        "critical_angle_error_deg": _fmt(obs.get("critical_angle_error_deg"), 4),
        "brewster_angle_est_deg": _fmt(obs.get("brewster_angle_est_deg"), 4),
        "brewster_angle_error_deg": _fmt(obs.get("brewster_angle_error_deg"), 4),
        "dispersion_span": _fmt(obs.get("dispersion_span"), 5),
        "focus_distance_best": _fmt(obs.get("focus_distance_best"), 5),
        "focus_gain": _fmt(obs.get("focus_gain"), 4),
        "min_spot_radius": _fmt(obs.get("min_spot_radius"), 5),
        "caustic_gain": _fmt(obs.get("caustic_gain"), 4),
        "retroreflection_error_deg": _fmt(obs.get("retroreflection_error_deg"), 5),
        "return_efficiency_mean": _fmt(obs.get("return_efficiency_mean"), 4),
        "return_spot_offset": _fmt(obs.get("return_spot_offset"), 5),
        "malus_r2": _fmt(obs.get("malus_r2"), 5),
        "extinction_ratio": _fmt(obs.get("extinction_ratio"), 4),
        "waveplate_modulation_depth": _fmt(obs.get("waveplate_modulation_depth"), 5),
        "polarization_contrast": _fmt(obs.get("polarization_contrast"), 5),
        "retardance_error_deg": _fmt(obs.get("retardance_error_deg"), 5),
        "scattering_contrast": _fmt(obs.get("scattering_contrast"), 5),
        "forward_power_proxy": _fmt(obs.get("forward_power_proxy"), 5),
        "backward_power_proxy": _fmt(obs.get("backward_power_proxy"), 5),
        "mode_confinement": _fmt(obs.get("mode_confinement"), 5),
        "waveguide_leakage": _fmt(obs.get("waveguide_leakage"), 5),
        "localization_ratio": _fmt(obs.get("localization_ratio"), 5),
        "lattice_material_fill": _fmt(obs.get("lattice_material_fill"), 5),
        "line_defect_confinement": _fmt(obs.get("line_defect_confinement"), 5),
        "right_transmission_proxy": _fmt(obs.get("right_transmission_proxy"), 5),
        "hotspot_enhancement": _fmt(obs.get("hotspot_enhancement"), 5),
        "gap_localization_ratio": _fmt(obs.get("gap_localization_ratio"), 5),
        "radiation_shell_contrast": _fmt(obs.get("radiation_shell_contrast"), 5),
        "radiated_power_proxy": _fmt(obs.get("radiated_power_proxy"), 5),
        "evanescent_decay_length": _fmt(obs.get("evanescent_decay_length"), 5),
        "upper_field_fraction": _fmt(obs.get("upper_field_fraction"), 5),
        "polarization_mode": str(optics.get("polarization_mode", "n/a")),
    }


def _field_pool(vals: dict[str, str]) -> tuple[list[str], list[str]]:
    family = vals.get("family", "")
    base_fields = [
        "phenomenon",
        "visible_signature",
        "visual_dynamics",
        "scale_bucket",
        "rendered_field",
        "dim",
        "optics_model",
        "scan_parameter",
        "scan_path_visual",
        "temporal_sampling_class",
        "scan_range",
        "scan_count",
    ]
    family_fields: dict[str, list[str]] = {
        "planeMirrorReflection": [
            "reflection_efficiency_mean",
            "spot_shift_span",
            "peak_irradiance_max",
        ],
        "concaveMirrorFocus": ["focus_distance_best", "focus_gain", "min_spot_radius"],
        "dielectricBlockRefraction": [
            "snell_residual_deg",
            "transmission_efficiency_mean",
            "spot_shift_span",
        ],
        "totalInternalReflection": [
            "critical_angle_est_deg",
            "critical_angle_error_deg",
            "reflection_efficiency_mean",
        ],
        "fresnelBrewster": [
            "brewster_angle_est_deg",
            "brewster_angle_error_deg",
            "reflection_efficiency_mean",
        ],
        "beamSplitterPlate": [
            "split_balance_error",
            "branch_separation",
            "transmission_efficiency_mean",
        ],
        "singleSlitFraunhofer": [
            "central_lobe_width",
            "spot_shift_span",
            "peak_irradiance_max",
        ],
        "doubleSlitInterference": [
            "fringe_visibility",
            "fringe_spacing",
            "fringe_shift_span",
        ],
        "circularApertureAiry": [
            "airy_ring_radius",
            "central_lobe_width",
            "peak_irradiance_max",
        ],
        "knifeEdgeFresnel": [
            "edge_transition_width",
            "edge_overshoot",
            "peak_irradiance_mean",
        ],
        "diffractionGrating": [
            "grating_order_spacing",
            "grating_order_count",
            "spectral_spread",
        ],
        "thinFilmInterference": [
            "thin_film_reflectance_mean",
            "thin_film_contrast",
            "thin_film_fringe_period_nm",
        ],
        "gaussianBeamPropagation": [
            "beam_waist_min",
            "rayleigh_range",
            "divergence_angle_deg",
        ],
        "prismDispersion": [
            "dispersion_span",
            "peak_irradiance_max",
            "spot_shift_span",
        ],
        "thinLensFocus": ["focus_distance_best", "focus_gain", "min_spot_radius"],
        "glassSphereCaustic": [
            "caustic_gain",
            "focus_distance_best",
            "peak_irradiance_max",
        ],
        "cornerCubeRetroreflector": [
            "retroreflection_error_deg",
            "return_efficiency_mean",
            "return_spot_offset",
        ],
        "polarizerMalus": [
            "input_polarization_pose",
            "extinction_class",
            "brightness_class",
            "power_class",
            "malus_r2",
            "extinction_ratio",
            "polarization_mode",
        ],
        "waveplatePolarization": [
            "waveplate_modulation_depth",
            "polarization_contrast",
            "retardance_error_deg",
        ],
        "meepCylinderScattering": [
            "scattering_contrast",
            "forward_power_proxy",
            "backward_power_proxy",
        ],
        "meepWaveguideMode": [
            "mode_confinement",
            "waveguide_leakage",
            "peak_irradiance_max",
        ],
        "meepPhotonicCrystalDefect": [
            "localization_ratio",
            "lattice_material_fill",
            "peak_irradiance_max",
        ],
        "meepPhotonicCrystalWaveguide": [
            "line_defect_confinement",
            "right_transmission_proxy",
            "peak_irradiance_max",
        ],
        "meepGratingDiffraction": [
            "grating_order_count",
            "grating_order_spacing",
            "peak_irradiance_max",
        ],
        "meepPlasmonicDimer": [
            "hotspot_enhancement",
            "gap_localization_ratio",
            "peak_irradiance_max",
        ],
        "meepDipoleRadiation": [
            "radiation_shell_contrast",
            "radiated_power_proxy",
            "peak_irradiance_max",
        ],
        "meepEvanescentTIR": [
            "evanescent_decay_length",
            "upper_field_fraction",
            "peak_irradiance_max",
        ],
    }
    extras = family_fields.get(
        family, ["peak_irradiance_max", "integrated_power_mean", "spot_shift_span"]
    )
    base = [f for f in base_fields if _usable(vals.get(f, ""))]
    extras_clean = [f for f in extras if _usable(vals.get(f, ""))]
    return base, extras_clean


def _field_phrase(
    field: str, vals: dict[str, str], style: int, *, prefer_numeric: bool = False
) -> str:
    v = vals.get(field, "n/a")
    visual_options = VISUAL_FIRST_FIELD_PHRASES.get(field)
    if visual_options and not prefer_numeric:
        return _pick(visual_options, style)
    numeric_templates = {
        "scattering_contrast": "the scattering contrast is {v}",
        "forward_power_proxy": "the forward scattered power level is {v}",
        "backward_power_proxy": "the backward scattered power level is {v}",
        "mode_confinement": "the mode confinement fraction is {v}",
        "waveguide_leakage": "the guide leakage level is {v}",
        "localization_ratio": "the defect localization ratio is {v}",
        "lattice_material_fill": "the lattice material fill fraction is {v}",
        "line_defect_confinement": "the line-defect confinement is {v}",
        "right_transmission_proxy": "the downstream transmission level is {v}",
        "hotspot_enhancement": "the gap hotspot enhancement reaches {v}",
        "gap_localization_ratio": "the gap localization ratio is {v}",
        "radiation_shell_contrast": "the radiation shell contrast is {v}",
        "radiated_power_proxy": "the radiated power level is {v}",
        "evanescent_decay_length": "the evanescent decay length is {v}",
        "upper_field_fraction": "the upper-side field fraction is {v}",
    }
    if field in numeric_templates:
        return numeric_templates[field].format(v=v)
    if field == "phenomenon":
        return _pick(
            [
                f"the visible phenomenon is {v}",
                f"the optical setup shows {v}",
                f"the scene is a {v} experiment",
            ],
            style,
        )
    if field == "visible_signature":
        return _pick(
            [
                f"the visible signature is {v}",
                f"the image sequence shows {v}",
                f"the optical response appears as {v}",
            ],
            style,
        )
    if field == "visual_dynamics":
        return _pick(
            [
                f"visually, {v}",
                f"over time, {v}",
                f"the rendered motion shows that {v}",
            ],
            style,
        )
    if field == "scale_bucket":
        return _pick(
            [
                f"the visual scale regime is that {v}",
                f"the sequence belongs to the regime where {v}",
                f"the scale variant is visually marked by {v}",
            ],
            style,
        )
    if field == "family":
        return _pick(
            [
                f"the physical family is {_phenomenon_label(v)}",
                f"the retrieval family corresponds to {_phenomenon_label(v)}",
                f"the optical family describes {_phenomenon_label(v)}",
            ],
            style,
        )
    if field == "dim":
        return _pick(
            [
                f"the view is {v}",
                f"the benchmark instance is represented as {v}",
                f"the optical field is sampled in a {v} view",
            ],
            style,
        )
    if field == "rendered_field":
        return _pick(
            [
                f"the video renders the {v}",
                f"the color map represents the {v}",
                f"the displayed scalar quantity is the {v}",
            ],
            style,
        )
    if field == "optics_model":
        return _pick(
            [
                f"the optical element model is {v}",
                f"the physical optics model is {v}",
                f"the setup is represented as {v}",
            ],
            style,
        )
    if field == "scan_parameter":
        return _pick(
            [
                f"the sweep parameter is {v}",
                f"the animated scan varies {v}",
                f"the scan axis is {v}",
            ],
            style,
        )
    if field == "scan_path_visual":
        return _pick(
            [
                f"the visible analyzer path is that {v}",
                f"the optical control path shows {v}",
                f"the animated orientation cue shows {v}",
            ],
            style,
        )
    if field == "temporal_sampling_class":
        return _pick(
            [
                f"the animation has {v}",
                f"the scan is shown with {v}",
                f"the temporal progression uses {v}",
            ],
            style,
        )
    if field == "scan_range":
        return _pick(
            [
                f"the scan range is {v}",
                f"the parameter sweep spans {v}",
                f"the recorded scan interval is {v}",
            ],
            style,
        )
    if field == "scan_count":
        return _pick(
            [
                f"the animation uses {v} sampled optical states",
                f"the parameter sweep has {v} samples",
                f"the sequence records {v} scan positions",
            ],
            style,
        )
    if field == "peak_irradiance_max":
        return _pick(
            [
                f"the peak irradiance reaches {v}",
                f"the maximum irradiance is {v}",
                f"the irradiance crest is {v}",
            ],
            style,
        )
    if field == "peak_irradiance_mean":
        return _pick(
            [
                f"the mean frame peak irradiance is {v}",
                f"the average peak irradiance is {v}",
                f"the framewise peak average is {v}",
            ],
            style,
        )
    if field == "integrated_power_mean":
        return _pick(
            [
                f"the mean integrated power is {v}",
                f"the average collected power is {v}",
                f"the receiver power average is {v}",
            ],
            style,
        )
    if field == "spot_shift_span":
        return _pick(
            [
                f"the spot shift span is {v}",
                f"the detector spot sweeps across {v}",
                f"the beam footprint moves by {v}",
            ],
            style,
        )
    if field == "reflection_efficiency_mean":
        return _pick(
            [
                f"the mean reflection efficiency is {v}",
                f"the average reflected-energy fraction is {v}",
                f"the reflection efficiency averages {v}",
            ],
            style,
        )
    if field == "transmission_efficiency_mean":
        return _pick(
            [
                f"the mean transmission efficiency is {v}",
                f"the average transmitted-energy fraction is {v}",
                f"the transmission efficiency averages {v}",
            ],
            style,
        )
    if field == "snell_residual_deg":
        return _pick(
            [
                f"the Snell-law residual is {v} deg",
                f"the refraction-angle mismatch is {v} deg",
                f"the fitted Snell residual stays at {v} deg",
            ],
            style,
        )
    if field == "split_balance_error":
        return _pick(
            [
                f"the splitter balance error is {v}",
                f"the reflected/transmitted imbalance is {v}",
                f"the branch power mismatch is {v}",
            ],
            style,
        )
    if field == "branch_separation":
        return _pick(
            [
                f"the branch separation is {v}",
                f"the split beams separate by {v}",
                f"the detector branch spacing is {v}",
            ],
            style,
        )
    if field == "central_lobe_width":
        return _pick(
            [
                f"the central lobe width is {v}",
                f"the diffraction core width is {v}",
                f"the main coherent lobe spans {v}",
            ],
            style,
        )
    if field == "fringe_visibility":
        return _pick(
            [
                f"the fringe visibility is {v}",
                f"the interference visibility reaches {v}",
                f"the fringe contrast is {v}",
            ],
            style,
        )
    if field == "fringe_spacing":
        return _pick(
            [
                f"the fringe spacing is {v}",
                f"the adjacent interference fringes are spaced by {v}",
                f"the mean fringe pitch is {v}",
            ],
            style,
        )
    if field == "fringe_shift_span":
        return _pick(
            [
                f"the fringe shift span is {v}",
                f"the interference pattern shifts by {v}",
                f"the peak-fringe drift spans {v}",
            ],
            style,
        )
    if field == "airy_ring_radius":
        return _pick(
            [
                f"the Airy ring radius is {v}",
                f"the first diffraction ring occurs near radius {v}",
                f"the fitted Airy-ring radius is {v}",
            ],
            style,
        )
    if field == "edge_transition_width":
        return _pick(
            [
                f"the knife-edge transition width is {v}",
                f"the Fresnel edge transition spans {v}",
                f"the diffracted edge spreads across {v}",
            ],
            style,
        )
    if field == "edge_overshoot":
        return _pick(
            [
                f"the edge overshoot is {v}",
                f"the Fresnel overshoot level is {v}",
                f"the edge diffraction ripple overshoots by {v}",
            ],
            style,
        )
    if field == "grating_order_spacing":
        return _pick(
            [
                f"the grating-order spacing is {v}",
                f"the diffraction orders are separated by {v}",
                f"the mean multi-slit order pitch is {v}",
            ],
            style,
        )
    if field == "grating_order_count":
        return _pick(
            [
                f"the visible grating-order count is {v}",
                f"the multi-slit pattern resolves about {v} orders",
                f"the detected diffraction-order count averages {v}",
            ],
            style,
        )
    if field == "spectral_spread":
        return _pick(
            [
                f"the wavelength-driven order spread is {v}",
                f"the spectral spread across the scan is {v}",
                f"the grating order spacing changes by {v}",
            ],
            style,
        )
    if field == "thin_film_reflectance_mean":
        return _pick(
            [
                f"the mean thin-film reflectance is {v}",
                f"the average film-reflection level is {v}",
                f"the film reflectance averages {v}",
            ],
            style,
        )
    if field == "thin_film_contrast":
        return _pick(
            [
                f"the thin-film fringe contrast is {v}",
                f"the spectral-band contrast is {v}",
                f"the multiple-beam interference contrast reaches {v}",
            ],
            style,
        )
    if field == "thin_film_fringe_period_nm":
        return _pick(
            [
                f"the thin-film fringe period is {v} nm",
                f"the spectral fringe pitch is {v} nm",
                f"the reflected-band period is {v} nm",
            ],
            style,
        )
    if field == "beam_waist_min":
        return _pick(
            [
                f"the minimum Gaussian beam waist is {v}",
                f"the narrowest laser spot radius is {v}",
                f"the smallest propagated beam waist is {v}",
            ],
            style,
        )
    if field == "rayleigh_range":
        return _pick(
            [
                f"the Rayleigh range is {v}",
                f"the Gaussian propagation length scale is {v}",
                f"the beam's Rayleigh distance is {v}",
            ],
            style,
        )
    if field == "divergence_angle_deg":
        return _pick(
            [
                f"the Gaussian divergence angle is {v} deg",
                f"the far-field beam divergence is {v} deg",
                f"the laser spread angle is {v} deg",
            ],
            style,
        )
    if field == "critical_angle_est_deg":
        return _pick(
            [
                f"the estimated critical angle is {v} deg",
                f"the TIR threshold is near {v} deg",
                f"the fitted critical-angle estimate is {v} deg",
            ],
            style,
        )
    if field == "critical_angle_error_deg":
        return _pick(
            [
                f"the critical-angle error is {v} deg",
                f"the TIR fit error is {v} deg",
                f"the critical-angle residual is {v} deg",
            ],
            style,
        )
    if field == "brewster_angle_est_deg":
        return _pick(
            [
                f"the estimated Brewster angle is {v} deg",
                f"the reflectance minimum occurs near {v} deg",
                f"the fitted Brewster-angle estimate is {v} deg",
            ],
            style,
        )
    if field == "brewster_angle_error_deg":
        return _pick(
            [
                f"the Brewster-angle error is {v} deg",
                f"the fitted Brewster residual is {v} deg",
                f"the Brewster mismatch is {v} deg",
            ],
            style,
        )
    if field == "dispersion_span":
        return _pick(
            [
                f"the red-blue dispersion span is {v}",
                f"the chromatic separation reaches {v}",
                f"the detector dispersion span is {v}",
            ],
            style,
        )
    if field == "focus_distance_best":
        return _pick(
            [
                f"the best-focus distance is {v}",
                f"the optimal sensor distance is {v}",
                f"the sharpest focus occurs at {v}",
            ],
            style,
        )
    if field == "focus_gain":
        return _pick(
            [
                f"the focus gain is {v}",
                f"the best-to-worst focus contrast is {v}",
                f"the focusing gain reaches {v}",
            ],
            style,
        )
    if field == "min_spot_radius":
        return _pick(
            [
                f"the minimum spot radius is {v}",
                f"the tightest focal spot is {v}",
                f"the smallest blur radius is {v}",
            ],
            style,
        )
    if field == "caustic_gain":
        return _pick(
            [
                f"the caustic gain is {v}",
                f"the refractive hotspot gain is {v}",
                f"the caustic amplification reaches {v}",
            ],
            style,
        )
    if field == "retroreflection_error_deg":
        return _pick(
            [
                f"the retroreflection error is {v} deg",
                f"the return-angle mismatch is {v} deg",
                f"the retroreflected beam deviates by {v} deg",
            ],
            style,
        )
    if field == "return_efficiency_mean":
        return _pick(
            [
                f"the mean return efficiency is {v}",
                f"the average returned-energy fraction is {v}",
                f"the retroreflected power fraction averages {v}",
            ],
            style,
        )
    if field == "return_spot_offset":
        return _pick(
            [
                f"the return spot offset is {v}",
                f"the retroreflected spot shifts by {v}",
                f"the return-beam footprint offset is {v}",
            ],
            style,
        )
    if field == "malus_r2":
        return _pick(
            [
                f"the Malus-law fit R^2 is {v}",
                f"the analyzer sweep fits Malus law with R^2={v}",
                f"the polarization fit quality is {v}",
            ],
            style,
        )
    if field == "extinction_ratio":
        return _pick(
            [
                f"the crossed-polarizer extinction is visually in the {vals.get('extinction_class', v)} band",
                f"the dimmest frames show {vals.get('extinction_class', v)}",
                f"the analyzer extinction appears as {vals.get('extinction_class', v)}",
            ],
            style,
        )
    if field == "input_polarization_pose":
        return _pick(
            [
                f"the incoming polarization cue has {v}",
                f"the input polarization marker shows {v}",
                f"the polarizer setup starts with {v}",
            ],
            style,
        )
    if field == "extinction_class":
        return _pick(
            [
                f"the dark-frame contrast is {v}",
                f"the analyzer minimum is described by {v}",
                f"the extinction behavior is {v}",
            ],
            style,
        )
    if field == "brightness_class":
        return _pick(
            [
                f"the sequence has {v}",
                f"the transmitted beam maintains {v}",
                f"the typical frame shows {v}",
            ],
            style,
        )
    if field == "power_class":
        return _pick(
            [
                f"the detector response has {v}",
                f"the collected light appears in the {v}",
                f"the overall transmitted-light level is {v}",
            ],
            style,
        )
    if field == "waveplate_modulation_depth":
        return _pick(
            [
                f"the waveplate modulation depth is {v}",
                f"the analyzer intensity modulation reaches {v}",
                f"the birefringent polarization modulation is {v}",
            ],
            style,
        )
    if field == "polarization_contrast":
        return _pick(
            [
                f"the polarization contrast is {v}",
                f"the analyzer max/min contrast is {v}",
                f"the waveplate transmission contrast is {v}",
            ],
            style,
        )
    if field == "retardance_error_deg":
        return _pick(
            [
                f"the retardance error is {v} deg",
                f"the waveplate retardance offset is {v} deg",
                f"the birefringent phase error is {v} deg",
            ],
            style,
        )
    if field == "polarization_mode":
        return _pick(
            [
                f"the polarization mode is {v}",
                f"the case uses {v} polarization",
                f"the polarization state is tagged as {v}",
            ],
            style,
        )
    return f"{field} = {v}"


def _query_field_order(vals: dict[str, str]) -> list[str]:
    base_fields, extra_fields = _field_pool(vals)
    general_observables = [
        "peak_irradiance_max",
        "peak_irradiance_mean",
        "integrated_power_mean",
        "spot_shift_span",
    ]
    return [
        f
        for f in dict.fromkeys([*base_fields, *extra_fields, *general_observables])
        if _usable(vals.get(f, ""))
    ]


def _query_body(
    vals: dict[str, str],
    fields: tuple[str, ...] | list[str],
    style: int,
    *,
    max_parts: int,
    prefer_numeric: bool = False,
) -> str:
    priority = [
        "phenomenon",
        "visible_signature",
        "visual_dynamics",
        "scale_bucket",
        "rendered_field",
        "dim",
        "scan_parameter",
        "scan_path_visual",
        "temporal_sampling_class",
        "scan_range",
        "scan_count",
        "input_polarization_pose",
        "extinction_class",
        "brightness_class",
        "power_class",
        "reflection_efficiency_mean",
        "transmission_efficiency_mean",
        "snell_residual_deg",
        "split_balance_error",
        "branch_separation",
        "central_lobe_width",
        "fringe_visibility",
        "fringe_spacing",
        "fringe_shift_span",
        "airy_ring_radius",
        "edge_transition_width",
        "edge_overshoot",
        "grating_order_spacing",
        "grating_order_count",
        "spectral_spread",
        "thin_film_reflectance_mean",
        "thin_film_contrast",
        "thin_film_fringe_period_nm",
        "beam_waist_min",
        "rayleigh_range",
        "divergence_angle_deg",
        "critical_angle_est_deg",
        "critical_angle_error_deg",
        "brewster_angle_est_deg",
        "brewster_angle_error_deg",
        "dispersion_span",
        "focus_distance_best",
        "focus_gain",
        "min_spot_radius",
        "caustic_gain",
        "retroreflection_error_deg",
        "return_efficiency_mean",
        "return_spot_offset",
        "malus_r2",
        "extinction_ratio",
        "waveplate_modulation_depth",
        "polarization_contrast",
        "retardance_error_deg",
        "polarization_mode",
        "scattering_contrast",
        "forward_power_proxy",
        "backward_power_proxy",
        "mode_confinement",
        "waveguide_leakage",
        "localization_ratio",
        "lattice_material_fill",
        "line_defect_confinement",
        "right_transmission_proxy",
        "hotspot_enhancement",
        "gap_localization_ratio",
        "radiation_shell_contrast",
        "radiated_power_proxy",
        "evanescent_decay_length",
        "upper_field_fraction",
        "peak_irradiance_max",
        "peak_irradiance_mean",
        "integrated_power_mean",
        "spot_shift_span",
    ]
    field_set = set(fields)
    seen: set[str] = set()
    parts: list[str] = []
    for field in priority:
        if field not in field_set or field in seen:
            continue
        seen.add(field)
        variant = 0 if prefer_numeric else style + len(parts)
        phrase = _field_phrase(field, vals, variant, prefer_numeric=prefer_numeric)
        if phrase:
            parts.append(phrase)
        if len(parts) >= max_parts:
            break
    for field in fields:
        if len(parts) >= max_parts:
            break
        if field in seen or field == "backend":
            continue
        variant = 0 if prefer_numeric else style + len(parts)
        phrase = _field_phrase(field, vals, variant, prefer_numeric=prefer_numeric)
        if phrase:
            parts.append(phrase)
    return _join_phrases(parts)


def _full_query_text(
    vals: dict[str, str], fields: tuple[str, ...], case_id: str
) -> str:
    body = _query_body(vals, fields, 0, max_parts=16, prefer_numeric=True)
    text = f"Identify the optics benchmark video showing {body}."
    _assert_query_text_allowed(text)
    return text


def _parsed_query_text(case: dict[str, Any], vals: dict[str, str]) -> str:
    family = str(case.get("family", "unknown"))
    dim = vals.get("dim", "2D")
    phenomenon = vals.get("phenomenon", _phenomenon_label(family))
    signature = vals.get("visible_signature", _phenomenon_signature(family))
    mechanism = PHYSICS_MECHANISM.get(family, vals.get("visual_dynamics", signature))
    color_cue = COLOR_MAP_CUE.get(
        family, "the brightest colors mark where the optical energy concentrates"
    )
    rendered_field = vals.get("rendered_field", RENDERED_FIELD_LABEL)

    variant_parts: list[str] = []
    if _usable(vals.get("scale_bucket", "")):
        variant_parts.append(vals["scale_bucket"])
    variant_parts.append(
        vals.get("scan_path_visual", "the controlled parameter sweeps through the clip")
    )
    scan_count = vals.get("scan_count", "")
    variant_tail = f" across {scan_count} sampled frames" if _usable(scan_count) else ""

    _, extras = _field_pool(vals)
    detail_phrases = [
        _field_phrase(field, vals, 0, prefer_numeric=True) for field in extras[:3]
    ]
    if not detail_phrases:
        detail_phrases = [
            f"the scan covers {vals.get('scan_range', 'the recorded interval')}"
        ]

    sentences = [
        f"Find the {dim} optics video of {phenomenon}, showing {signature}.",
        f"Physically, {mechanism}.",
        f"The color map represents the {rendered_field}, so {color_cue}.",
        f"Use the variant where {'; '.join(variant_parts)}{variant_tail}.",
        f"To identify the exact video, look for {_join_phrases(detail_phrases)}.",
    ]
    text = " ".join(sentences)
    _assert_query_text_allowed(text)
    return text
