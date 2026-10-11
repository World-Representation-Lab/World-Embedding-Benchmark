from __future__ import annotations
from run_case import (Any, _family, _simulate_surrogate_cantilever_beam_transient, _simulate_surrogate_drop_weight_impact, _simulate_surrogate_flexible_crank_slider, _simulate_surrogate_harmonic_frf, _simulate_surrogate_j2_plastic_beam, _simulate_surrogate_lemaitre_damage, _simulate_surrogate_modal_block, _simulate_surrogate_reissner_shell_modal, _simulate_surrogate_rubber_block, _simulate_surrogate_stress_wave)

def _simulate_surrogate_structural_demo(case: dict[str, Any]) -> dict[str, Any]:
    family = _family(case)
    if family == "cantileverBeamTransient":
        return _simulate_surrogate_cantilever_beam_transient(case)
    if family == "modalBlockAnalysis":
        return _simulate_surrogate_modal_block(case)
    if family == "harmonicResponseFRF":
        return _simulate_surrogate_harmonic_frf(case)
    if family == "stressWavePropagation":
        return _simulate_surrogate_stress_wave(case)
    if family == "dropWeightImpactSMC":
        return _simulate_surrogate_drop_weight_impact(case)
    if family == "j2PlasticBeam":
        return _simulate_surrogate_j2_plastic_beam(case)
    if family == "lemaitreDamageBar":
        return _simulate_surrogate_lemaitre_damage(case)
    if family == "rubberBlockCompression":
        return _simulate_surrogate_rubber_block(case)
    if family == "flexibleCrankSlider":
        return _simulate_surrogate_flexible_crank_slider(case)
    if family == "reissnerShellModal":
        return _simulate_surrogate_reissner_shell_modal(case)
    raise ValueError(f"Unsupported structural family: {family}")
