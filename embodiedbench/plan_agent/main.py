import json

from config import Settings
from llm_clients import OpenAIHTTPClient, OpenAISDKClient
from workflow import Workflow
from registry import CapabilityRegistry


def load_input(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_llm_client(settings: Settings):
    if settings.llm_mode.lower() == "http":
        return OpenAIHTTPClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
        )

    return OpenAISDKClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.llm_model,
    )


def build_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()

    # registry.register_skill(
    #     name="task_decomposition",
    #     description="Breaks a complex task into structured subtasks",
    #     inputs=["task_description"],
    #     outputs=["subtask_list"],
    # )

    # registry.register_skill(
    #     name="constraint_analysis",
    #     description="Extracts and normalizes hard and soft constraints",
    #     inputs=["task_description", "context"],
    #     outputs=["constraint_set"],
    # )

    # registry.register_mcp(
    #     name="graph_planner_mcp",
    #     description="Provides graph-based planning and shortest path related services",
    #     endpoint="mcp://graph-planner",
    #     capabilities=["path_search", "graph_reasoning"],
    # )

    # registry.register_tool(
    #     name="cost_calculator",
    #     description="Computes objective values for candidate solutions",
    #     inputs=["candidate_options", "cost_function"],
    #     outputs=["scored_candidates"],
    # )

    return registry


def main():
    settings = Settings()
    request_data = load_input("demo_input.json")

    llm_client = build_llm_client(settings)
    registry = build_registry()

    workflow = Workflow(llm_client, registry=registry)
    result = workflow.run(request_data)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()