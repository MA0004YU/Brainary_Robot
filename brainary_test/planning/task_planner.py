import json
from pathlib import Path
from .llm_client import LLMClient
from .prompt_templates import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from .goal_reasoner import GoalReasoner
from .sdg_planner import SDGPlanner

ROOT = Path(__file__).resolve().parent.parent

class TaskPlanner:
    def __init__(self, llm_client: LLMClient, use_ltm: bool = False):
        self.llm = llm_client
        self.use_ltm = use_ltm
        self.goal_reasoner = GoalReasoner(self.llm)
        self.sdg_planner = SDGPlanner(self.llm, use_ltm=self.use_ltm)

    def generate_plan(self, planning_input: dict) -> list:
        """
        Generate a Task Dependency Graph (list of steps with 'depends_on')
        based on the provided memory planning input.
        This handles the 3-stage pipeline: Intent -> SDG -> Task Plan.
        """
        task_instruction = planning_input.get("task_instruction", "")
        constraints_raw = planning_input.get("constraints", {})
        constraints = json.dumps(constraints_raw, ensure_ascii=False)
        manipulable_objects = json.dumps(planning_input.get("manipulable_objects", {}), ensure_ascii=False)
        available_skills = json.dumps(planning_input.get("available_skills", []), ensure_ascii=False)

        # Stage 1: Goal Reasoner
        print("Stage 1: Extracting Deep Intent...")
        goal_intent = self.goal_reasoner.extract_intent(task_instruction)
        
        # Save intermediate intent
        intent_file = ROOT / "output" / "goal_intent.json"
        with open(intent_file, "w", encoding="utf-8") as f:
            json.dump(goal_intent, f, ensure_ascii=False, indent=2)

        # Stage 2: SDG Planner
        print("Stage 2: Generating State Dependency Graph (SDG)...")
        sdg_graph = self.sdg_planner.generate_sdg(goal_intent, constraints_raw)
        
        # Save intermediate SDG
        sdg_file = ROOT / "output" / "sdg_plan.json"
        with open(sdg_file, "w", encoding="utf-8") as f:
            json.dump(sdg_graph, f, ensure_ascii=False, indent=2)

        # Stage 3: Task Planner (Grounding)
        print("Stage 3: Grounding SDG into Task Plan...")
        user_prompt = PLANNER_USER_PROMPT.format(
            goal_intent=json.dumps(goal_intent, ensure_ascii=False),
            sdg_graph=json.dumps(sdg_graph, ensure_ascii=False),
            constraints=constraints,
            manipulable_objects=manipulable_objects,
            available_skills=available_skills
        )
        
        response = self.llm.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        # Print reasoning if available
        if isinstance(response, dict) and "reasoning" in response:
            print("\n--- LLM Reasoning ---")
            for line in response["reasoning"]:
                print(f"- {line}")
            print("---------------------\n")
            
            steps = response.get("steps", [])
            return steps if isinstance(steps, list) else []
            
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            for k, v in response.items():
                if isinstance(v, list) and k == "steps":
                    return v
            return [response]
        else:
            print("Failed to parse valid list from LLM response.")
            return []
