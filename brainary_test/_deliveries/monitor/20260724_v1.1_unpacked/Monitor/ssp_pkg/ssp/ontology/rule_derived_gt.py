"""Module: Rule-derived GT schema (#002 C-layer) | Paper section: §4.3 | Status: wip

Ground truth derived MECHANICALLY by running the live SSP parser
(lift + propagate + activate, ADR-015 semantics) over a scene. Stored as a
sidecar `<scene>_gt.json` next to each BenchSample, NEVER mutating the scene
file or its legacy `ground_truth`.

CIRCULARITY (read this before trusting any metric built on this GT):
  This GT is produced by the same code path the parser uses. Therefore a
  parser-vs-rule-GT comparison on the SAME parser is identically 1.0 and
  carries ZERO information about external correctness. Rule-derived GT measures
  rule self-consistency / template coverage, not whether the rules match the
  world. External-correctness anchors live elsewhere:
    (a) human review of disagreement cases (C4),
    (b) rule-GT vs legacy-GT diff on the simple/ slice (C3),
    (c) ABLATED parsers (E2/E3/E4) scored against this GT — ablation breaks the
        shared-code identity, so the gap is meaningful.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ssp.ontology.risk_events import RiskEventType


class FactorGT(BaseModel):
    """A factor node that the lift step instantiated in the scene (existence)."""

    model_config = ConfigDict(extra="forbid")

    factor_id: str
    re_type: RiskEventType
    template_id: str
    hazard_id: str
    target_id: str


class FactorRef(BaseModel):
    """Reference to a factor in an action-level bucket.

    `residual_peak_severity` carries the propagation operator's r* peak for
    diagnostic transparency ONLY. Per ADR-015 it is NOT a membership gate:
    bucket membership is decided by mitigation mode (hard -> hard_suppressed)
    and activation, never by a residual threshold.
    """

    model_config = ConfigDict(extra="forbid")

    factor_id: str
    re_type: RiskEventType
    residual_peak_severity: float


class ActionGT(BaseModel):
    """Per-action three-set classification mirroring ADR-015 / QueryResult.

    - active:          ActivatedRiskEvent with NO soft mitigation applied.
    - soft_mitigated:  ActivatedRiskEvent WITH a soft mitigation (guarded_by);
                       risk is discounted but the event stays activated.
    - hard_suppressed: SuppressedRiskEvent (isolated_by / neutralized_by, a fact
                       that removes the hazard).
    Factors that exist in the scene but are not activated by THIS action appear
    only in scene_level_existing_factors (they are orphans w.r.t. this action).
    """

    model_config = ConfigDict(extra="forbid")

    active: list[FactorRef] = []
    soft_mitigated: list[FactorRef] = []
    hard_suppressed: list[FactorRef] = []


class RuleDerivedGT(BaseModel):
    """Rule-derived ground truth for one scene (sidecar to a BenchSample).

    Two levels (Open Issue #002 §5c):
      - scene_level_existing_factors: every factor lift instantiates, including
        orphans that no candidate action activates (existence is scene-intrinsic).
      - action_level: per-action three-set classification (active /
        soft_mitigated / hard_suppressed).
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    pair_id: str
    variant: Literal["risky", "mitigated"]
    source: str
    scene_level_existing_factors: list[FactorGT]
    action_level: dict[str, ActionGT]
    provenance: dict
