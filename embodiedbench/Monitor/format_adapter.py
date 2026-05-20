"""Format adapter: converts Brainary memory_snapshot -> SafetyContext."""
from typing import Dict, Any, List, Optional

from .core.context import SafetyContext


def to_safety_context(
    memory_snapshot: Dict[str, Any],
    past_actions: List[str],
) -> SafetyContext:
    """
    Convert a Brainary memory_snapshot into a SafetyContext.

    The most recent action from action_buffer becomes the current action to evaluate.
    past_actions is maintained externally by the stateful monitor.
    """
    working = memory_snapshot.get("working", {})
    goal = working.get("goal", {})
    obs = working.get("observation", {})
    action_buffer = working.get("action_buffer", [])

    current_action = _extract_current_action(action_buffer)

    ctx = SafetyContext(
        input_mode="natural_language",
        task_id=memory_snapshot.get("episode_id", ""),
        user_instruction=goal.get("instruction", ""),
        scene_objects_list=obs.get("visible_objects", []),
        scene_text=obs.get("text", ""),
        current_location=obs.get("current_location", ""),
        held_object=obs.get("held_object"),
        memory_objects=obs.get("memory_objects", {}),
        plan_steps=[current_action] if current_action else [],
        current_step_index=0,
        past_actions=list(past_actions),
    )
    return ctx


def _extract_current_action(action_buffer) -> Optional[str]:
    """Extract the most recent action string from the action buffer."""
    if not action_buffer:
        return None
    last = action_buffer[-1] if isinstance(action_buffer, list) else None
    if last is None:
        return None
    return last.get("action", "") or None
