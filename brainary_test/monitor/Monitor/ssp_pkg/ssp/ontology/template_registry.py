"""Module: Template registry for RE template loading | Paper section: §4 | Status: wip"""

from __future__ import annotations

from pathlib import Path

import yaml

from ssp.ontology.risk_events import RiskEventTemplate, RiskEventType


class TemplateNotFoundError(Exception):
    """Raised when a requested template is not in the registry."""


class TemplateRegistry:
    """Loads and provides access to all risk event templates."""

    def __init__(self, templates_dir: Path) -> None:
        self._templates_dir = templates_dir
        self._templates: dict[RiskEventType, RiskEventTemplate] = {}

    def load_all(self) -> None:
        """Load all YAML templates from the configured directory."""
        if not self._templates_dir.exists():
            msg = f"Templates directory not found: {self._templates_dir}"
            raise FileNotFoundError(msg)

        for yaml_path in sorted(self._templates_dir.glob("*.yaml")):
            with yaml_path.open() as f:
                data = yaml.safe_load(f)
            template = RiskEventTemplate.model_validate(data)
            self._templates[template.name] = template

    def get(self, re_type: RiskEventType) -> RiskEventTemplate:
        """Get a template by risk event type."""
        if re_type not in self._templates:
            raise TemplateNotFoundError(f"No template for {re_type}")
        return self._templates[re_type]

    def all_templates(self) -> list[RiskEventTemplate]:
        """Return all loaded templates."""
        return list(self._templates.values())
