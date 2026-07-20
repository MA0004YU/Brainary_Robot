"""Module: Action activation logic | Paper section: §5 | Status: wip"""

from __future__ import annotations

from pydantic import BaseModel

from ssp.graph.g_percept import PerceptualGraph
from ssp.graph.g_reason import ReasoningGraph
from ssp.ontology.risk_events import RiskEventTemplate
from ssp.ontology.schema import CandidateAction, FactorNode
from ssp.ontology.template_registry import TemplateRegistry
from ssp.query.predicates import evaluate_predicate


class FactorActivation(BaseModel):
    """Result of activating a single factor node by an action."""

    factor_id: str
    re_type: str
    activation_rule_id: str
    activation_strength: float
    evidence: list[str]


class ActivationResult(BaseModel):
    """Result of action activation across all factor nodes."""

    activated: list[FactorActivation]
    inactive_factor_ids: list[str]


def _try_activate_factor(
    factor: FactorNode,
    action: CandidateAction,
    template: RiskEventTemplate,
    g_p: PerceptualGraph,
    relation_blind: bool = False,
) -> FactorActivation | None:
    """Try to activate a factor node with the given action using template rules.

    Returns FactorActivation if any rule matches, None otherwise.
    Uses the FIRST matching rule (highest priority = first in template).
    relation_blind (E3): forwarded to the predicate evaluator.
    """
    for rule_idx, rule in enumerate(template.activation_rules):
        if rule.action_type != action.type:
            continue

        # Evaluate all conditions in this rule
        all_conditions_met = True
        evidence: list[str] = [f"action.type={action.type} matches rule[{rule_idx}]"]

        for condition in rule.conditions:
            if evaluate_predicate(condition, action, factor, g_p, relation_blind):
                evidence.append(f"{condition.op}({condition.args}) = True")
            else:
                all_conditions_met = False
                break

        if all_conditions_met:
            return FactorActivation(
                factor_id=factor.id,
                re_type=str(factor.re_type),
                activation_rule_id=f"{template.id}_rule_{rule_idx}",
                activation_strength=rule.activation_strength,
                evidence=evidence,
            )

    return None


def activate_by_action(
    g_r: ReasoningGraph,
    g_p: PerceptualGraph,
    action: CandidateAction,
    registry: TemplateRegistry,
    relation_blind: bool = False,
) -> ActivationResult:
    """Determine which factor nodes are activated by a candidate action.

    For each factor node in G_R, looks up its template's activation_rules
    and evaluates them against the action using the typed predicate evaluator.

    Args:
        g_r: Reasoning graph with factor nodes (post-lift).
        g_p: Perceptual graph (needed for relation_exists predicates).
        action: Candidate action to evaluate.
        registry: Template registry for looking up activation rules.
        relation_blind: E3 ablation (ADR-020). Forwarded to predicates so
            relation_exists conditions are ignored. Default False = unchanged.

    Returns:
        ActivationResult with activated factors and inactive factor IDs.
    """
    activated: list[FactorActivation] = []
    inactive_ids: list[str] = []

    for fn_id, factor in g_r.factor_nodes.items():
        template = registry.get(factor.re_type)
        activation = _try_activate_factor(factor, action, template, g_p, relation_blind)
        if activation is not None:
            activated.append(activation)
        else:
            inactive_ids.append(fn_id)

    return ActivationResult(activated=activated, inactive_factor_ids=inactive_ids)
