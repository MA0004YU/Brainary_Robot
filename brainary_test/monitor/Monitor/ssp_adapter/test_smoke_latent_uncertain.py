"""Downstream smoke test: vendored SSP (>= 278415c) + adapter B1/B2 integration.

Runs the REAL vendored ssp_pkg (not the live SSP repo) through the adapter's
attribute map + constraint_writer, covering the four things the B patches depend
on at RUNTIME (patch --dry-run only proved text hunks apply):

  1. multi-subtype: a cup resolves to subtypes=[fragile_object, container] and
     the vendored Node accepts it -> spill + fragile_breakage co-instantiate.
  2. latent serialization: constraint_writer.build_diagnostics emits
     latent_risk_events (ADR-027).
  3. instantiation uncertain: an undecidable-evidence factor serializes as an
     uncertain event with risk_computed=False (ADR-029).
  4. constraint-status preservation: collect_candidate_constraints tags each
     item with diagnostic_status (activated|latent|uncertain), reason, etc.

Run:  cd <pipeline_root> && PYTHONPATH=Monitor/ssp_pkg:. \
          .venv/bin/python -m pytest Monitor/ssp_adapter/test_smoke_latent_uncertain.py
(The venv is the SSP repo's; only deps are needed. sys.path bootstrap below
mirrors __main__.py so the test is runnable directly too.)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ADAPTER_DIR = Path(__file__).resolve().parent
_MONITOR_DIR = _ADAPTER_DIR.parent
_REPO_ROOT = _MONITOR_DIR.parent
_SSP_PKG = _MONITOR_DIR / "ssp_pkg"
for _p in (_REPO_ROOT, _SSP_PKG):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest

from ssp.graph.g_percept import PerceptualGraph
from ssp.ontology.schema import Edge, Node, StateSchema
from ssp.parser import SceneSafetyParser

from Monitor.ssp_adapter import constraint_writer as cw
from Monitor.ssp_adapter.object_attribute_map import resolve_spec

TEMPLATES = _SSP_PKG / "configs" / "re_templates"


@pytest.fixture(scope="module")
def parser() -> SceneSafetyParser:
    return SceneSafetyParser(TEMPLATES)


def test_cup_resolves_to_multi_subtype() -> None:
    # B1: a cup must now map to BOTH fragile_object and container.
    spec = resolve_spec("黄色杯子", "杯子")
    vals = [s.value for s in spec.subtypes]
    assert "fragile_object" in vals
    assert "container" in vals
    assert spec.has_liquid is True


def test_cup_coinstantiates_fragile_and_spill(parser: SceneSafetyParser) -> None:
    # B1 runtime: multi-subtype cup of water near a laptop -> fragile_breakage
    # AND spill co-instantiate (both latent when stable+upright).
    spec = resolve_spec("黄色杯子", "杯子")
    nodes = [
        Node(id="table", type="surface"),
        Node(id="cup", type="physical_object",
             subtypes=[s.value for s in spec.subtypes],
             attributes=StateSchema(stability="stable", orientation="upright",
                                    containment="open", motion="static", energy="none")),
        Node(id="water", type="substance"),
        Node(id="laptop", type="physical_object", subtypes=["electronic"],
             attributes=StateSchema(stability="stable")),
    ]
    edges = [
        Edge(src="table", dst="cup", relation="supports", sign="+"),
        Edge(src="cup", dst="water", relation="contains", sign="+"),
        Edge(src="cup", dst="laptop", relation="near", sign="+"),
    ]
    res = parser.query_risk(PerceptualGraph(nodes=nodes, edges=edges))
    latent = {e.re_type.value for e in res.latent_risk_events}
    assert "fragile_breakage" in latent
    assert "spill_damage" in latent


def _demo_like_graph() -> PerceptualGraph:
    # 3 stable objects + a container with unknown containment (uncertain spill).
    nodes = [
        Node(id="table", type="surface"),
        Node(id="cup", type="physical_object", subtypes=["fragile_object", "container"],
             attributes=StateSchema(stability="stable", orientation="upright",
                                    containment="open", motion="static")),
        Node(id="water", type="substance"),
        Node(id="teabox", type="physical_object", subtypes=["container"],
             attributes=StateSchema(stability="stable", motion="static")),  # containment unknown
        Node(id="fruit", type="physical_object", subtypes=["food_item"],
             attributes=StateSchema(stability="stable")),
    ]
    edges = [
        Edge(src="table", dst="cup", relation="supports", sign="+"),
        Edge(src="cup", dst="water", relation="contains", sign="+"),
        Edge(src="table", dst="teabox", relation="supports", sign="+"),
        Edge(src="teabox", dst="fruit", relation="near", sign="+"),
    ]
    return PerceptualGraph(nodes=nodes, edges=edges)


def test_build_diagnostics_serializes_latent_and_uncertain(
    parser: SceneSafetyParser,
) -> None:
    res = parser.query_risk(_demo_like_graph())
    diag = cw.build_diagnostics(res, id_to_name={})
    # 2/3: latent + uncertain keys present and populated
    assert "latent_risk_events" in diag
    assert "uncertain_risk_events" in diag
    assert diag["summary"]["num_latent"] >= 1
    assert diag["summary"]["num_uncertain"] >= 1
    # uncertain instantiation-stage event has risk_computed False + no risk vector
    unc = diag["uncertain_risk_events"]
    inst = [u for u in unc if u["uncertainty_stage"] == "instantiation"]
    assert inst
    assert inst[0]["risk_computed"] is False
    assert inst[0]["risk"] == {}


def test_candidate_constraints_preserve_diagnostic_status(
    parser: SceneSafetyParser,
) -> None:
    res = parser.query_risk(_demo_like_graph())
    diag = cw.build_diagnostics(res, id_to_name={})
    cands = cw.collect_candidate_constraints(diag)
    # 4: every candidate carries a diagnostic_status; latent + uncertain present
    assert cands
    statuses = {c["diagnostic_status"] for c in cands}
    assert "latent" in statuses
    assert "uncertain" in statuses
    # uncertain candidates carry uncertainty_stage + reason
    for c in cands:
        if c["diagnostic_status"] == "uncertain":
            assert "uncertainty_stage" in c
            assert "reason" in c
