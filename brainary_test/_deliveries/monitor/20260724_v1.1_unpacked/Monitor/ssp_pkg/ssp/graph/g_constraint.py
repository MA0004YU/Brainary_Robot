"""Module: Constraint Graph (G_C) | Paper section: §2.1 L3 | Status: wip"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ssp.ontology.risk_events import RiskEventType


class ConstraintBinding(BaseModel):
    """A constraint template instantiated with entity bindings."""

    template_id: str
    risk_event_type: RiskEventType
    entity_bindings: dict[str, str]
    parameters: dict[str, float] = {}


class ConstraintGraph:
    """G_C: activated constraint templates with entity bindings."""

    def __init__(self, bindings: list[ConstraintBinding]) -> None:
        self._bindings = list(bindings)

    @property
    def bindings(self) -> list[ConstraintBinding]:
        return self._bindings

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "layer": "G_C",
            "bindings": [b.model_dump(mode="json") for b in self._bindings],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> ConstraintGraph:
        """Deserialize from JSON string."""
        parsed: dict[str, Any] = json.loads(data)
        bindings = [ConstraintBinding.model_validate(b) for b in parsed["bindings"]]
        return cls(bindings=bindings)
