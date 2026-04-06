from modules import (
    ArchitectureModelingModule,
    ConfidenceEstimatorModule,
    EvaluatorModule,
    SelfAssessmentModule,
    SolverModule,
    TaskUnderstandingModule,
)
from memory import NoopMemory
from schemas import TaskRequest


class Workflow:
    def __init__(self, llm_client, memory=None, registry=None):
        self.memory = memory or NoopMemory()
        self.registry = registry

        self.task_understanding = TaskUnderstandingModule(llm_client, self.memory)
        self.self_assessment = SelfAssessmentModule(llm_client, self.memory, self.registry)
        self.confidence_estimator = ConfidenceEstimatorModule(llm_client, self.memory)
        self.architecture_modeling = ArchitectureModelingModule(llm_client, self.memory)
        self.solver = SolverModule(llm_client, self.memory)
        self.evaluator = EvaluatorModule(llm_client, self.memory)

    def run(self, request_data: dict) -> dict:
        request = TaskRequest.model_validate(request_data)

        task_model = self.task_understanding.run(request)
        capability_model = self.self_assessment.run(task_model)
        confidence_report = self.confidence_estimator.run(task_model, capability_model)

        if confidence_report.recommended_action not in {"proceed", "continue", "done"}:
            return {
                "status": "stopped",
                "task_model": task_model.model_dump(),
                "capability_model": capability_model.model_dump(),
                "confidence_report": confidence_report.model_dump(),
            }

        architecture_plan = self.architecture_modeling.run(
            task_model,
            capability_model,
            confidence_report,
        )
        solution_bundle = self.solver.run(architecture_plan)
        feedback_report = self.evaluator.run(task_model, solution_bundle)

        next_action_name = feedback_report.next_action.get("recommended", "")

        return {
            "status": "done" if next_action_name == "done" else "needs_refinement",
            "task_model": task_model.model_dump(),
            "capability_model": capability_model.model_dump(),
            "confidence_report": confidence_report.model_dump(),
            "architecture_plan": architecture_plan.model_dump(),
            "solution_bundle": solution_bundle.model_dump(),
            "feedback_report": feedback_report.model_dump(),
        }