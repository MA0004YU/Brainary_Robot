"""Module: Bounded risk propagation operator | Paper section: §3.2 | Status: wip"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssp.graph.g_reason import ReasoningGraph
from ssp.ontology.relations import L1PropagationRelation, L1SuppressionRelation
from ssp.ontology.risk_events import RiskEventType
from ssp.ontology.schema import FactorNode, RiskVector
from ssp.propagation.admissibility import check_admissibility
from ssp.propagation.aggregation import max_aggregation, noisy_or
from ssp.propagation.fixed_point import NonConvergenceError, PropagationResult
from ssp.propagation.params import PropagationParams


def _get_intrinsic_risk(
    fn: FactorNode, k: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Extract intrinsic severity and likelihood as K-dim arrays."""
    sev = np.zeros(k, dtype=np.float64)
    lik = np.zeros(k, dtype=np.float64)
    for i, v in enumerate(fn.severity[:k]):
        sev[i] = v
    for i, v in enumerate(fn.likelihood[:k]):
        lik[i] = v
    return sev, lik


def propagate(
    graph: ReasoningGraph,
    params: PropagationParams | None = None,
) -> PropagationResult:
    """Bounded risk propagation with multiplicative suppression gate.

    Algorithm (per iteration, per factor node j, per dimension k):
      1. s_tilde = max(phi_s, max_i(psi+_s * s_i))
      2. p_tilde = noisy_or(phi_p, {psi+_p * p_i})
      3. m_s, m_p = suppression gates (noisy-OR of effective suppression weights)
      4. s_new = clip(s_tilde * (1 - m_s), 0, r_max)
      5. p_new = clip(p_tilde * (1 - m_p), 0, r_max)
      6. Apply damping: s = (1-eta)*s_old + eta*s_new
    """
    if params is None:
        params = PropagationParams()

    K = params.num_risk_dims  # noqa: N806
    factor_nodes = graph.factor_nodes
    warnings: list[str] = []

    admissibility_report = check_admissibility(graph, params)
    if not admissibility_report.is_admissible and params.enforce_admissibility:
        warnings.append(
            f"Graph not admissible: {len(admissibility_report.violating_nodes)} violating nodes"
        )

    if not factor_nodes:
        return PropagationResult(
            final_risks={},
            iterations=0,
            converged=True,
            bounded=True,
            theory_guaranteed=True,
            max_change=0.0,
            admissibility_report=admissibility_report,
            warnings=warnings,
            trajectory=[] if params.record_trajectory else None,
        )

    # Build adjacency: for each factor node, find incoming propagation and suppression edges
    pos_edges: dict[str, list[tuple[str, NDArray[np.floating]]]] = {
        fn_id: [] for fn_id in factor_nodes
    }
    sup_edges: dict[str, list[tuple[float, NDArray[np.floating]]]] = {
        fn_id: [] for fn_id in factor_nodes
    }

    for edge in graph.edges:
        if edge.dst in factor_nodes:
            fn = factor_nodes[edge.dst]
            re_idx = list(RiskEventType).index(fn.re_type)
            if isinstance(edge.relation, L1PropagationRelation):
                weight_vec = np.zeros(K, dtype=np.float64)
                weight_vec[re_idx] = edge.weight * edge.confidence
                pos_edges[edge.dst].append((edge.src, weight_vec))
            elif isinstance(edge.relation, L1SuppressionRelation):
                src_node = graph.nodes.get(edge.src)
                suppressor_conf = src_node.confidence if src_node else 1.0
                # ADR-015: edge.weight is the residual_multiplier (fraction of
                # risk that REMAINS). Confidence in the suppression evidence
                # interpolates the multiplier toward 1.0 (no effect) when the
                # edge / suppressor is uncertain:
                #   effective_residual = 1 - conf * (1 - weight)
                edge_conf = edge.confidence * suppressor_conf
                effective_residual = 1.0 - edge_conf * (1.0 - edge.weight)
                # residual vector defaults to 1.0 (no suppression) on every dim,
                # set to effective_residual on the suppressed dims only.
                residual_vec = np.ones(K, dtype=np.float64)
                if edge.suppression_dims:
                    re_type_list = list(RiskEventType)
                    for dim_re_type in edge.suppression_dims:
                        dim_idx = re_type_list.index(dim_re_type)
                        residual_vec[dim_idx] = effective_residual
                else:
                    residual_vec[re_idx] = effective_residual
                sup_edges[edge.dst].append((effective_residual, residual_vec))

    # Initialize risk arrays: severity and likelihood per factor node
    sev: dict[str, NDArray[np.floating]] = {}
    lik: dict[str, NDArray[np.floating]] = {}
    phi_sev: dict[str, NDArray[np.floating]] = {}
    phi_lik: dict[str, NDArray[np.floating]] = {}

    for fn_id, fn in factor_nodes.items():
        s, p = _get_intrinsic_risk(fn, K)
        phi_sev[fn_id] = s
        phi_lik[fn_id] = p
        sev[fn_id] = s.copy()
        lik[fn_id] = p.copy()

    trajectory: list[dict[str, list[float]]] | None = [] if params.record_trajectory else None
    converged = False
    max_change = float("inf")
    iterations = 0

    for t in range(params.max_iter):
        if trajectory is not None:
            snapshot: dict[str, list[float]] = {}
            for fn_id in factor_nodes:
                snapshot[fn_id] = list(np.concatenate([sev[fn_id], lik[fn_id]]))
            trajectory.append(snapshot)

        max_change = 0.0
        for fn_id in factor_nodes:
            # --- Severity: max aggregation ---
            sev_contributions: list[NDArray[np.floating]] = [phi_sev[fn_id]]
            for src_id, w_vec in pos_edges[fn_id]:
                if src_id in sev:
                    # Source is another factor node: propagate its risk
                    sev_contributions.append(w_vec * sev[src_id])
                else:
                    # Source is entity node: edge weight IS the contribution
                    sev_contributions.append(w_vec)
            s_tilde = max_aggregation(sev_contributions)

            # --- Likelihood: noisy-OR aggregation ---
            lik_contributions: list[NDArray[np.floating]] = []
            for src_id, w_vec in pos_edges[fn_id]:
                if src_id in lik:
                    # Source is another factor node
                    lik_contributions.append(w_vec * lik[src_id])
                else:
                    # Source is entity node: edge weight as activation probability
                    lik_contributions.append(w_vec)
            if lik_contributions:
                p_agg = noisy_or(lik_contributions)
                p_tilde = noisy_or([phi_lik[fn_id], p_agg])
            else:
                p_tilde = phi_lik[fn_id].copy()

            # --- Suppression gate (ADR-015: multiplicative residual) ---
            # Each suppression edge contributes a residual vector (fraction of
            # risk that remains, 1.0 = no effect). Independent gates compose
            # multiplicatively: combined_residual = product over edges.
            residual_s = np.ones(K, dtype=np.float64)
            residual_p = np.ones(K, dtype=np.float64)
            for _eff, residual_vec in sup_edges[fn_id]:
                residual_s *= residual_vec
                residual_p *= residual_vec

            # Apply residual + clip
            s_new = np.clip(s_tilde * residual_s, 0.0, params.r_max)
            p_new = np.clip(p_tilde * residual_p, 0.0, params.r_max)

            # Damping
            s_damped = (1.0 - params.damping) * sev[fn_id] + params.damping * s_new
            p_damped = (1.0 - params.damping) * lik[fn_id] + params.damping * p_new

            # Track max change
            change = max(
                float(np.max(np.abs(s_damped - sev[fn_id]))),
                float(np.max(np.abs(p_damped - lik[fn_id]))),
            )
            max_change = max(max_change, change)

            sev[fn_id] = s_damped
            lik[fn_id] = p_damped

        iterations = t + 1
        if max_change < params.tol:
            converged = True
            break

    # Build result
    if not converged and params.raise_on_nonconvergence:
        msg = f"Propagation did not converge in {params.max_iter} iterations (max_change={max_change:.6f})"
        raise NonConvergenceError(msg)

    if not converged:
        warnings.append(f"Did not converge in {iterations} iterations (max_change={max_change:.6f})")

    theory_guaranteed = admissibility_report.is_admissible and converged

    final_risks: dict[str, RiskVector] = {}
    for fn_id in factor_nodes:
        final_risks[fn_id] = RiskVector(
            severity={},
            likelihood={},
            confidence=factor_nodes[fn_id].confidence,
        )
        re_types = list(RiskEventType)
        for i in range(min(K, len(re_types))):
            if sev[fn_id][i] > 0:
                final_risks[fn_id].severity[re_types[i]] = float(sev[fn_id][i])
            if lik[fn_id][i] > 0:
                final_risks[fn_id].likelihood[re_types[i]] = float(lik[fn_id][i])

    return PropagationResult(
        final_risks=final_risks,
        iterations=iterations,
        converged=converged,
        bounded=True,
        theory_guaranteed=theory_guaranteed,
        max_change=max_change,
        admissibility_report=admissibility_report,
        warnings=warnings,
        trajectory=trajectory,
    )
