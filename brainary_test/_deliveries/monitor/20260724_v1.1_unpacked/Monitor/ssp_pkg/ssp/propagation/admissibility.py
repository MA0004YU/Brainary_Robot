"""Module: Admissibility check for propagation convergence | Paper section: §3.2 | Status: wip"""

from __future__ import annotations

from pydantic import BaseModel

from ssp.graph.g_reason import ReasoningGraph
from ssp.ontology.relations import L1PropagationRelation, L1SuppressionRelation
from ssp.propagation.params import PropagationParams


class AdmissibilityReport(BaseModel):
    """Report on whether the graph satisfies contraction conditions."""

    is_admissible: bool
    violating_nodes: list[str] = []
    max_positive_per_node: dict[str, float] = {}
    max_suppression_per_node: dict[str, float] = {}


def check_admissibility(
    graph: ReasoningGraph,
    params: PropagationParams,
) -> AdmissibilityReport:
    """Check if incoming propagation/suppression weights satisfy contraction condition.

    For each node j, checks:
      sum_i(psi+_ij) <= max_incoming_positive
      sum_k(psi-_kj) <= max_incoming_suppression
    """
    valid_ids = graph.all_node_ids()
    max_pos: dict[str, float] = dict.fromkeys(valid_ids, 0.0)
    max_sup: dict[str, float] = dict.fromkeys(valid_ids, 0.0)

    for edge in graph.edges:
        if isinstance(edge.relation, L1PropagationRelation):
            max_pos[edge.dst] = max_pos.get(edge.dst, 0.0) + edge.weight
        elif isinstance(edge.relation, L1SuppressionRelation):
            # ADR-015: edge.weight is now the residual_multiplier (fraction of
            # risk that REMAINS). The "suppression applied" magnitude is
            # (1 - weight), preserving the pre-ADR-015 semantic where a higher
            # value means stronger suppression. For hard mitigations this is
            # numerically identical to the old weight (old 0.95 == 1 - 0.05).
            max_sup[edge.dst] = max_sup.get(edge.dst, 0.0) + (1.0 - edge.weight)

    violating: list[str] = []
    for nid in valid_ids:
        if max_pos.get(nid, 0.0) > params.max_incoming_positive:
            violating.append(nid)
        if max_sup.get(nid, 0.0) > params.max_incoming_suppression:
            violating.append(nid)

    return AdmissibilityReport(
        is_admissible=len(violating) == 0,
        violating_nodes=sorted(set(violating)),
        max_positive_per_node={k: v for k, v in max_pos.items() if v > 0},
        max_suppression_per_node={k: v for k, v in max_sup.items() if v > 0},
    )
