"""Safety Brain Pipeline — 编排所有模块的顺序执行。"""
from typing import List

from .core.context import SafetyContext
from .core.base_module import BaseModule


class SafetyPipeline:
    """顺序执行所有 WP4 模块，支持逐步评估。"""

    def __init__(self, modules: List[BaseModule]):
        self.modules = modules

    def run_once(self, ctx: SafetyContext) -> SafetyContext:
        """对当前 step 运行一次完整 pipeline。"""
        for module in self.modules:
            if module.should_skip(ctx):
                continue
            ctx = module.process(ctx)
            if ctx.execution_halted:
                break
        return ctx

    def run_stepwise(self, ctx: SafetyContext) -> SafetyContext:
        """PROTEA 风格：逐步执行，每步运行 critic pipeline。"""
        for i in range(len(ctx.plan_steps)):
            ctx.current_step_index = i
            # 重置单步输出
            ctx.critic_decision = ""
            ctx.critic_reason = ""
            ctx.hazards_identified = []

            ctx = self.run_once(ctx)
            if ctx.execution_halted:
                break
            ctx.past_actions.append(ctx.plan_steps[i])
        return ctx
