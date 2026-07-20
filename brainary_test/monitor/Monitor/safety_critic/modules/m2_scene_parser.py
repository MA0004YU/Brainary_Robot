"""M2: Scene Safety Parser — 场景解析 + 对象过滤 + 风险标注。"""
import re
import itertools

from ..core.base_module import BaseModule
from ..core.context import SafetyContext

try:
    from ..utils.parse_graph import parse_graph_with_id, parsed_graph_text_to_dict
    _HAS_PROTEA_UTILS = True
except ImportError:
    _HAS_PROTEA_UTILS = False


class SceneSafetyParser(BaseModule):

    @property
    def name(self) -> str:
        return "M2-SceneParser"

    def process(self, ctx: SafetyContext) -> SafetyContext:
        if ctx.input_mode == "natural_language":
            return self._process_natural_language(ctx)
        return self._process_protea(ctx)

    def _process_protea(self, ctx: SafetyContext) -> SafetyContext:
        if not ctx.scene_graph_raw or not _HAS_PROTEA_UTILS:
            return ctx
        parsed_text = parse_graph_with_id(ctx.scene_graph_raw)
        filtered = self._filter_for_plan(parsed_text, ctx.plan_steps)
        ctx.scene_graph_dict = parsed_graph_text_to_dict("\n".join(filtered))
        return ctx

    def _process_natural_language(self, ctx: SafetyContext) -> SafetyContext:
        env_dict = {}
        for obj_name in ctx.scene_objects_list:
            mem_info = ctx.memory_objects.get(obj_name, {})
            parts = []
            loc = mem_info.get("location", "")
            if loc:
                parts.append(f"location: {loc}")
            if obj_name == ctx.held_object:
                parts.append("held by robot")
            env_dict[obj_name] = ", ".join(parts) if parts else "visible"

        if ctx.current_location:
            env_dict["_robot_location"] = ctx.current_location
        if ctx.scene_text:
            env_dict["_scene_description"] = ctx.scene_text

        ctx.scene_graph_dict = env_dict
        return ctx

    def _filter_for_plan(self, parsed_text: str, plan_steps: list) -> list:
        plan_items = set(
            (name, int(obj_id.split('.')[-1]))
            for name, obj_id in itertools.chain.from_iterable(
                re.findall(r'<([A-Za-z_]+)>\s*\(([\d.]+)\)', step)
                for step in plan_steps
            )
        )

        filtered = ["The environment contains the following objects:\n"]
        for line in parsed_text.splitlines():
            match = re.match(r'^([A-Za-z_]+)\s+\(id:\s*(\d+)\)', line)
            if match:
                name = match.group(1)
                obj_id = int(match.group(2))
                if (name, obj_id) in plan_items or name == "character":
                    filtered.append(line + "\n")
        return filtered
