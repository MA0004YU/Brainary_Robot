from typing import Any, Dict, List

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    raw_input: str
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    expected_output: str = "workflow_plan"


class TaskModel(BaseModel):
    task_type: str
    domain: str
    user_goal: str
    root_problem: str
    success_criteria: List[str] = Field(default_factory=list)

    # 从 List[str] 改成 Dict[str, Any]
    constraints: Dict[str, Any] = Field(default_factory=dict)

    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    task_decomposition_hint: List[str] = Field(default_factory=list)
    risk_points: List[str] = Field(default_factory=list)


class CapabilityModel(BaseModel):
    llm_strengths: List[str] = Field(default_factory=list)
    llm_limitations: List[str] = Field(default_factory=list)

    available_skills: List[Dict[str, Any]] = Field(default_factory=list)
    available_mcp_services: List[Dict[str, Any]] = Field(default_factory=list)
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)

    unavailable_capabilities: List[str] = Field(default_factory=list)
    resource_budget: Dict[str, Any] = Field(default_factory=dict)
    execution_feasibility: Dict[str, Any] = Field(default_factory=dict)
    recommended_solution_modes: List[Dict[str, Any]] = Field(default_factory=list)


class ConfidenceReport(BaseModel):
    understanding_confidence: float
    capability_confidence: float
    information_sufficiency: float
    planning_feasibility: float
    overall_confidence: float
    blocking_issues: List[str] = Field(default_factory=list)
    recommended_action: str


class ArchitecturePlan(BaseModel):
    workflow_graph: Dict[str, Any] = Field(default_factory=dict)
    problem_formalization: Dict[str, Any] = Field(default_factory=dict)
    node_capability_mapping: List[Dict[str, Any]] = Field(default_factory=list)
    intermediate_artifacts: Dict[str, Any] = Field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = Field(default_factory=list)
    fallback_strategies: List[Dict[str, Any]] = Field(default_factory=list)

class SolutionBundle(BaseModel):
    selected_capabilities: List[Dict[str, Any]] = Field(default_factory=list)
    capability_trace: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_solutions: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_summary: str = ""
    expected_performance: Dict[str, Any] = Field(default_factory=dict)
    unsolved_parts: List[str] = Field(default_factory=list)
    uncertainty_points: List[str] = Field(default_factory=list)


class FeedbackReport(BaseModel):
    evaluation_result: str
    criteria_satisfaction: Dict[str, Any] = Field(default_factory=dict)
    failure_analysis: Dict[str, Any] = Field(default_factory=dict)
    revision_suggestions: List[str] = Field(default_factory=list)
    next_action: Dict[str, Any] = Field(default_factory=dict)