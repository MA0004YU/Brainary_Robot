"""Module: Fixed-point convergence and stopping criteria | Paper section: §3.2 | Status: wip"""

from __future__ import annotations

from pydantic import BaseModel

from ssp.ontology.schema import RiskVector
from ssp.propagation.admissibility import AdmissibilityReport


class NonConvergenceError(Exception):
    """Raised when propagation does not converge within max_iter."""


class PropagationResult(BaseModel):
    """Result of the bounded risk propagation operator."""

    final_risks: dict[str, RiskVector]
    iterations: int
    converged: bool
    bounded: bool = True
    theory_guaranteed: bool = False
    max_change: float
    admissibility_report: AdmissibilityReport
    warnings: list[str] = []
    trajectory: list[dict[str, list[float]]] | None = None
