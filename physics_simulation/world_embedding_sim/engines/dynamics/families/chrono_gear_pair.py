from __future__ import annotations
from run_case import (Any, _simulate_surrogate_gear_pair)

def _simulate_chrono_gear_pair(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    return _simulate_surrogate_gear_pair(case)
