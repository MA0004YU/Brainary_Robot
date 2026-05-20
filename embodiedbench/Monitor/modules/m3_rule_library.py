"""M3: Safety Constitution & Rule Library — 安全规则库。"""
import json
import os

from ..core.base_module import BaseModule
from ..core.context import SafetyContext


class RuleLibrary(BaseModule):

    def __init__(self, rules_path: str = None):
        if rules_path is None:
            rules_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "rules", "default_rules.json"
            )
        self.rules_path = rules_path
        self.rules = self._load_rules()

    @property
    def name(self) -> str:
        return "M3-RuleLibrary"

    def _load_rules(self):
        if os.path.exists(self.rules_path):
            with open(self.rules_path) as f:
                return json.load(f)
        return []

    def process(self, ctx: SafetyContext) -> SafetyContext:
        ctx.retrieved_rules = self.rules
        return ctx
