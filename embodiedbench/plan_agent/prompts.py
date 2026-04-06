TASK_UNDERSTANDING_PROMPT = """
You are the Task Understanding module in an LLM-driven task planning workflow.
Your job is to transform the raw user request into a structured task representation.

Return ONLY a valid JSON object.
Do not include markdown fences.
Do not include explanations before or after the JSON.

Use exactly the following schema:

{
  "task_type": "string",
  "domain": "string",
  "user_goal": "string",
  "root_problem": "string",
  "success_criteria": ["string", "..."],
  "constraints": {
    "hard_constraints": ["string", "..."],
    "soft_constraints": ["string", "..."],
    "environment_constraints": ["string", "..."]
  },
  "assumptions": ["string", "..."],
  "missing_information": ["string", "..."],
  "task_decomposition_hint": ["string", "..."],
  "risk_points": ["string", "..."]
}
""".strip()


SELF_ASSESSMENT_PROMPT = """
You are the Self-ability Understanding module in an LLM-driven task planning workflow.

Your job is to evaluate whether the current system can solve the task, not only from the LLM's perspective, but also from the perspective of the whole execution stack.

You must explicitly examine:
1. the LLM's own strengths and limitations
2. available internal skills
3. available MCP services
4. available tools
5. unavailable or missing capabilities
6. resource budget
7. current execution feasibility
8. recommended solution modes

Return ONLY a valid JSON object.
Do not include markdown fences.
Do not include explanations before or after the JSON.

Use exactly the following schema:

{
  "llm_strengths": ["string", "..."],
  "llm_limitations": ["string", "..."],
  "available_skills": [
    {
      "name": "string",
      "description": "string"
    }
  ],
  "available_mcp_services": [
    {
      "name": "string",
      "description": "string"
    }
  ],
  "available_tools": [
    {
      "name": "string",
      "description": "string"
    }
  ],
  "unavailable_capabilities": ["string", "..."],
  "resource_budget": {
    "time_complexity": "string",
    "compute_requirement": "string",
    "notes": "string"
  },
  "execution_feasibility": {
    "feasible_now": true,
    "confidence": "string",
    "reason": "string"
  },
  "recommended_solution_modes": [
    {
      "mode": "string",
      "description": "string",
      "confidence": "string"
    }
  ]
}
""".strip()


CONFIDENCE_PROMPT = """
You are the Confidence Estimation module.

Estimate whether the system can reliably proceed with the task.

Return ONLY a valid JSON object.
Do not include markdown fences.
Do not include explanations before or after the JSON.

Use exactly the following schema:

{
  "understanding_confidence": 0.0,
  "capability_confidence": 0.0,
  "information_sufficiency": 0.0,
  "planning_feasibility": 0.0,
  "overall_confidence": 0.0,
  "blocking_issues": ["string", "..."],
  "recommended_action": "proceed"
}

All confidence scores must be numbers between 0 and 1.
""".strip()


ARCHITECTURE_PROMPT = """
You are the Architecture Modeling module.
Your job is to convert the structured task into an executable workflow architecture and formal problem representation.

You must also assign execution capabilities to different workflow nodes.
Different nodes may use different capabilities.
The capabilities may include:
1. internal skills
2. MCP services
3. tools
4. default LLM mode when no specialized capability is available

Return ONLY a valid JSON object.
Do not include markdown fences.
Do not include explanations before or after the JSON.

Use exactly the following schema:

{
  "workflow_graph": {
    "nodes": [],
    "edges": [],
    "execution_order": [],
    "current_blocking_node": "string",
    "current_status": "string"
  },
  "problem_formalization": {
    "problem_type": "string",
    "inputs": {},
    "decision_variables": {},
    "objective": {},
    "constraints": [],
    "feasibility_rule": "string",
    "selection_rule": "string",
    "tie_rule": "string",
    "output_specification": {},
    "known_gaps": []
  },
  "node_capability_mapping": [
    {
      "node_id": "string",
      "node_name": "string",
      "capability_type": "skill | mcp | tool | llm",
      "capability_name": "string",
      "fallback": "string"
    }
  ],
  "intermediate_artifacts": {},
  "checkpoints": [],
  "fallback_strategies": []
}
""".strip()


SOLVER_PROMPT = """
You are the Solving module.

Your job is to execute the workflow according to the node-level capability mapping already provided by the architecture modeling stage.

Different workflow nodes may correspond to different capabilities.
You must follow the node capability mapping instead of selecting capabilities again.

Return ONLY a valid JSON object.
Do not include markdown fences.
Do not include explanations before or after the JSON.

Use exactly the following schema:

{
  "selected_capabilities": [
    {
      "node_id": "string",
      "capability_type": "string",
      "capability_name": "string"
    }
  ],
  "capability_trace": [
    {
      "step": 1,
      "node_id": "string",
      "capability_type": "string",
      "capability_name": "string",
      "action": "string",
      "status": "string"
    }
  ],
  "candidate_solutions": [
    {
      "name": "string",
      "description": "string"
    }
  ],
  "reasoning_summary": "string",
  "expected_performance": {},
  "unsolved_parts": ["string", "..."],
  "uncertainty_points": ["string", "..."]
}
""".strip()

EVALUATOR_PROMPT = """
You are the Feedback and Evaluation module.
Your job is to evaluate whether the proposed solution satisfies the task criteria.

Return ONLY a valid JSON object.
Do not include markdown fences.
Do not include explanations before or after the JSON.

Use exactly the following schema:

{
  "evaluation_result": "string",
  "criteria_satisfaction": {
    "criterion_name": {
      "satisfied": true,
      "notes": "string"
    }
  },
  "failure_analysis": {
    "primary_issue": "string",
    "impact": "string",
    "what_is_correct": ["string", "..."],
    "what_is_missing": ["string", "..."],
    "assessment_of_candidate_solutions": {
      "cs1": "string",
      "cs2": "string"
    }
  },
  "revision_suggestions": ["string", "..."],
  "next_action": {
    "recommended": "string",
    "justification": "string"
  }
}
""".strip()