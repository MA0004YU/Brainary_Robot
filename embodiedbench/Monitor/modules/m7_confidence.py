"""M7: Confidence Calibration — 置信度校准与门控决策。"""
from ..core.base_module import BaseModule
from ..core.context import SafetyContext
from ..core.enums import RiskLevel, GateDecision


class ConfidenceController(BaseModule):

    @property
    def name(self) -> str:
        return "M7-Confidence"

    def process(self, ctx: SafetyContext) -> SafetyContext:
        risk = ctx.critic_risk_level
        ood = ctx.ood_score

        if risk == RiskLevel.CRITICAL:
            ctx.gate_decision = GateDecision.BLOCK
            ctx.final_risk_score = 1.0
        elif risk == RiskLevel.HIGH:
            ctx.gate_decision = GateDecision.BLOCK
            ctx.final_risk_score = 0.8
        elif risk == RiskLevel.MEDIUM:
            if ood > 0.5:
                ctx.gate_decision = GateDecision.GATHER_MORE_INFORMATION
            else:
                ctx.gate_decision = GateDecision.ALLOW_WITH_CONSTRAINTS
            ctx.final_risk_score = 0.5
        else:
            ctx.gate_decision = GateDecision.ALLOW
            ctx.final_risk_score = 0.1

        ctx.confidence_score = 1.0 - ood
        return ctx
