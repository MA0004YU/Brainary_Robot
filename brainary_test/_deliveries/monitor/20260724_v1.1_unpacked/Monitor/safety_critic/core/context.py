from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .enums import RiskLevel, EpistemicState, GateDecision


@dataclass
class Hazard:
    description: str
    severity: RiskLevel = RiskLevel.LOW
    likelihood: float = 0.5
    evidence: str = ""


@dataclass
class SafetyContext:
    """所有 WP4 模块间流转的核心数据结构。"""

    # --- 运行模式 ---
    input_mode: str = "protea"  # "protea" | "natural_language"

    # --- 输入数据（M1: Working Memory Read 填充）---
    task_id: str = ""
    user_instruction: str = ""
    scene_graph_raw: Dict[str, Any] = field(default_factory=dict)
    plan_steps: List[str] = field(default_factory=list)
    current_step_index: int = 0

    # --- 自然语言模式输入（integration 层填充）---
    scene_objects_list: List[str] = field(default_factory=list)
    scene_text: str = ""
    current_location: str = ""
    held_object: Optional[str] = None
    memory_objects: Dict[str, Any] = field(default_factory=dict)

    # --- M2: Scene Safety Parser ---
    scene_graph_dict: Dict[str, str] = field(default_factory=dict)
    scene_risks: List[Hazard] = field(default_factory=list)
    risk_propagation_paths: List[str] = field(default_factory=list)

    # --- M3: Rule Library ---
    retrieved_rules: List[Dict[str, Any]] = field(default_factory=list)

    # --- M4: Constraint Generator ---
    constraints: Dict[str, List[str]] = field(default_factory=dict)

    # --- M5: Safety Critic ---
    critic_decision: str = ""
    critic_reason: str = ""
    critic_risk_level: RiskLevel = RiskLevel.LOW
    hazards_identified: List[Hazard] = field(default_factory=list)
    violated_rules: List[str] = field(default_factory=list)
    safe_alternative: str = ""

    # --- M6: OOD Detection ---
    ood_score: float = 0.0
    epistemic_state: EpistemicState = EpistemicState.NKK
    uncertainty_sources: List[str] = field(default_factory=list)

    # --- M7: Confidence Calibration ---
    confidence_score: float = 1.0
    final_risk_score: float = 0.0
    gate_decision: GateDecision = GateDecision.ALLOW

    # --- M8: Gating ---
    gating_approved: bool = True
    gating_constraints: List[str] = field(default_factory=list)

    # --- M9: Fallback ---
    fallback_level: int = 0
    fallback_action: str = ""

    # --- M10: Safety Memory ---
    incident_logged: bool = False

    # --- 运行时状态 ---
    past_actions: List[str] = field(default_factory=list)
    execution_halted: bool = False
    halt_reason: str = ""

    @property
    def current_action(self) -> Optional[str]:
        if 0 <= self.current_step_index < len(self.plan_steps):
            return self.plan_steps[self.current_step_index]
        return None
