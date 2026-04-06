import json

from schemas import (
    ArchitecturePlan,
    CapabilityModel,
    ConfidenceReport,
    FeedbackReport,
    SolutionBundle,
    TaskModel,
    TaskRequest,
)
from prompts import (
    ARCHITECTURE_PROMPT,
    CONFIDENCE_PROMPT,
    EVALUATOR_PROMPT,
    SELF_ASSESSMENT_PROMPT,
    SOLVER_PROMPT,
    TASK_UNDERSTANDING_PROMPT,
)


class TaskUnderstandingModule:
    def __init__(self, llm_client, memory):
        self.llm = llm_client
        self.memory = memory

    def run(self, request: TaskRequest) -> TaskModel:
        memory_context = self.memory.retrieve_long_term(request.raw_input)
        user_prompt = json.dumps(
            {
                "request": request.model_dump(),
                "memory_context": memory_context,
            },
            ensure_ascii=False,
            indent=2,
        )
        result = self.llm.generate_json(TASK_UNDERSTANDING_PROMPT, user_prompt)
        # print("task_understanding_result:", result)
        return TaskModel.model_validate(result)
    
import json

from schemas import CapabilityModel, TaskModel
from prompts import SELF_ASSESSMENT_PROMPT


class SelfAssessmentModule:
    def __init__(self, llm_client, memory, registry=None):
        self.llm = llm_client
        self.memory = memory
        self.registry = registry

    def run(self, task_model: TaskModel) -> CapabilityModel:
        registry_snapshot = self.registry.snapshot() if self.registry else {
            "skills": [],
            "mcps": [],
            "tools": [],
        }

        user_prompt = json.dumps(
            {
                "task_model": task_model.model_dump(),
                "system_capabilities": registry_snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )

        result = self.llm.generate_json(SELF_ASSESSMENT_PROMPT, user_prompt)
        # print("self_assessment_result:", result)
        return CapabilityModel.model_validate(result)


class ConfidenceEstimatorModule:
    def __init__(self, llm_client, memory):
        self.llm = llm_client
        self.memory = memory

    def run(self, task_model: TaskModel, capability_model: CapabilityModel) -> ConfidenceReport:
        user_prompt = json.dumps(
            {
                "task_model": task_model.model_dump(),
                "capability_model": capability_model.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
        result = self.llm.generate_json(CONFIDENCE_PROMPT, user_prompt)
        # print("confidence_result:", result)
        return ConfidenceReport.model_validate(result)   
    
class ArchitectureModelingModule:
    def __init__(self, llm_client, memory):
        self.llm = llm_client
        self.memory = memory

    def run(
        self,
        task_model: TaskModel,
        capability_model: CapabilityModel,
        confidence_report: ConfidenceReport,
    ) -> ArchitecturePlan:
        user_prompt = json.dumps(
            {
                "task_model": task_model.model_dump(),
                "capability_model": capability_model.model_dump(),
                "confidence_report": confidence_report.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
        result = self.llm.generate_json(ARCHITECTURE_PROMPT, user_prompt)
        # print("architecture_result:", result)
        return ArchitecturePlan.model_validate(result)


class SolverModule:
    def __init__(self, llm_client, memory, registry=None):
        self.llm = llm_client
        self.memory = memory
        self.registry = registry

    def run(self, architecture_plan: ArchitecturePlan) -> SolutionBundle:
        node_mapping = architecture_plan.node_capability_mapping

        user_prompt = json.dumps(
            {
                "architecture_plan": architecture_plan.model_dump(),
                "node_capability_mapping": node_mapping,
            },
            ensure_ascii=False,
            indent=2,
        )

        result = self.llm.generate_json(SOLVER_PROMPT, user_prompt)
        # print("solver_result:", result)
        return SolutionBundle.model_validate(result)

class EvaluatorModule:
    def __init__(self, llm_client, memory):
        self.llm = llm_client
        self.memory = memory

    def run(self, task_model: TaskModel, solution_bundle: SolutionBundle) -> FeedbackReport:
        user_prompt = json.dumps(
            {
                "task_model": task_model.model_dump(),
                "solution_bundle": solution_bundle.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
        result = self.llm.generate_json(EVALUATOR_PROMPT, user_prompt)
        # print("evaluator_result:", result)
        return FeedbackReport.model_validate(result)