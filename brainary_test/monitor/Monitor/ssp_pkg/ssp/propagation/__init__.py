"""Module: Propagation subpackage | Paper section: §3 | Status: wip"""

from ssp.propagation.admissibility import AdmissibilityReport, check_admissibility
from ssp.propagation.fixed_point import NonConvergenceError, PropagationResult
from ssp.propagation.operator import propagate
from ssp.propagation.params import PropagationParams

__all__ = [
    "AdmissibilityReport",
    "NonConvergenceError",
    "PropagationParams",
    "PropagationResult",
    "check_admissibility",
    "propagate",
]
