"""SafetyBrainMonitor: implements MonitorInterface for Brainary_Robot integration."""
from typing import Any, Dict, List, Optional, Tuple

from .core.llm_client import LLMClient
from .core.enums import GateDecision
from .modules.m2_scene_parser import SceneSafetyParser
from .modules.m3_rule_library import RuleLibrary
from .modules.m5_safety_critic import SafetyCritic
from .modules.m7_confidence import ConfidenceController
from .pipeline import SafetyPipeline
from .format_adapter import to_safety_context


class SafetyBrainMonitor:
    """
    Stateful safety monitor for Brainary_Robot.

    Implements the MonitorInterface contract:
        check(observation, memory_snapshot) -> (ok: bool, alert: str | None)

    Tracks action history within an episode. Resets when episode_id changes.
    """

    def __init__(self, model: str = "qwen-plus", rules_path: str = None):
        self._llm = LLMClient(model=model)
        self._pipeline = SafetyPipeline(modules=[
            SceneSafetyParser(),
            RuleLibrary(rules_path=rules_path),
            SafetyCritic(self._llm),
            ConfidenceController(),
        ])
        self._episode_id: Optional[str] = None
        self._past_actions: List[str] = []

    def check(
        self, observation: Any, memory_snapshot: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate the last executed action for safety.

        Args:
            observation: Raw env observation (unused in v1, reserved for future VLM).
            memory_snapshot: Full memory system snapshot from arch_memory.snapshot().

        Returns:
            (True, None) if safe; (False, alert_message) if dangerous.
        """
        episode_id = memory_snapshot.get("episode_id", "")
        if episode_id != self._episode_id:
            self._reset(episode_id)

        ctx = to_safety_context(memory_snapshot, self._past_actions)

        if not ctx.plan_steps:
            return (True, None)

        ctx = self._pipeline.run_once(ctx)

        current_action = ctx.plan_steps[0]
        if not ctx.execution_halted:
            self._past_actions.append(current_action)

        if ctx.execution_halted or ctx.gate_decision == GateDecision.BLOCK:
            msg = ctx.halt_reason or ctx.critic_reason or "Action blocked by safety monitor"
            return (False, msg)

        return (True, None)

    def _reset(self, new_episode_id: str):
        self._episode_id = new_episode_id
        self._past_actions = []

    @property
    def total_tokens_used(self) -> int:
        return self._llm.total_tokens
