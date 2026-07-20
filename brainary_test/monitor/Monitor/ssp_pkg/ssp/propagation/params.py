"""Module: Propagation parameters configuration | Paper section: §3.3 | Status: wip"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ssp.ontology.risk_events import RiskEventType


class PropagationParams(BaseModel):
    """Configuration for the bounded risk propagation operator."""

    max_iter: int = 50
    tol: float = 1e-4
    r_max: float = 1.0
    damping: Annotated[float, Field(gt=0.0, le=1.0)] = 0.7
    record_trajectory: bool = False

    aggregation_severity: Literal["max"] = "max"
    aggregation_likelihood: Literal["noisy_or", "max"] = "noisy_or"
    suppression_mode: Literal["multiplicative_gate"] = "multiplicative_gate"

    max_incoming_positive: float = 0.95
    max_incoming_suppression: float = 0.95
    enforce_admissibility: bool = True
    raise_on_nonconvergence: bool = False

    num_risk_dims: int = len(RiskEventType)  # K = number of RE types (v0.2: 14)

    # ADR-015: floating-point zero tolerance for the "residual ~ 0" sanity check
    # used when routing a hard-mitigated factor into the `suppressed` bucket.
    # This is a numerical guard, NOT a tunable semantic threshold.
    epsilon_risk: float = 1e-3
