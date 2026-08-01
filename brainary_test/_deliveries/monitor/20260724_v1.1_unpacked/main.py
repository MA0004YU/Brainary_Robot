#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================
#  main.py —— 感知(GPT-5.5) + 记忆 一键流水线
# ======================================================================================
#  读 input/ 里的 5 张视角图 -> GPT-5.5 感知(枚举物体+关系) -> 喂记忆模块 ->
#  喂给规划模块(Intent -> SDG -> 带有CoT的动作落地) ->
#  把感知、记忆、规划结果 全部写成 output/ 下的 JSON。
#
#  一键运行:
#     export API_zhongzhuan=sk-...        # 或 OPENAI_API_KEY
#     python main.py                      # 用带 torch+numpy+requests 的 python
#
#  输出(output/):
#     perception.json               感知输出(GPT-5.5:objects[] + relations[] + scene_summary)
#     memory_planning_input.json    ★记忆给规划模块的文件(manipulable_objects/skills/constraints)
#     memory_planning_context.json  记忆产出的完整 PlanningContext
#     memory_snapshot.json          记忆三层快照(working/episodic/semantic)
#     goal_intent.json              规划内部产出：抽象深层意图
#     sdg_plan.json                 规划内部产出：环境状态依赖图
#     planned_actions.json          规划最终输出(含推理链和具体动作序列)
# ======================================================================================

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
sys.path.insert(0, str(ROOT / "memory_pkg"))            # vendored 记忆模块
sys.path.insert(0, str(ROOT / "Monitor" / "ssp_pkg"))   # vendored 场景安全解析器 SSP 引擎
# Monitor（安全监控层：ssp_adapter + safety_critic）作为包从仓库根 import，ROOT 已在 sys.path[0]

VIEWS = ["front", "wrist", "left", "right", "top"]

# ---------------- 配置(可用环境变量覆盖) ----------------
RELAY_BASE = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("VLM_MODEL", "gpt-5.5")
REASONING = os.environ.get("VLM_REASONING", "high")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_zhongzhuan")
TASK = os.environ.get("TASK", "把桌面物品按类别分拣进三个篮子")
# 分拣任务的分类->篮子规则(作为记忆 constraints 注入;可按需改)
CATEGORY_RULES = {"水果": "Prop_KLT_3", "工具": "Prop_KLT_1", "杯具": "Prop_KLT_2"}

# ---------------- 感知输出的 JSON Schema(GPT 结构化输出) ----------------
PERCEPTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "scene_summary": {"type": "string"},
        "objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "name": {"type": "string"}, "category": {"type": "string"},
                "appearance": {"type": "string"}, "location": {"type": "string"},
                "seen_in_views": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "category", "appearance", "location", "seen_in_views"]}},
        "relations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"subject": {"type": "string"}, "predicate": {"type": "string"},
                           "object": {"type": "string"}, "description": {"type": "string"}},
            "required": ["subject", "predicate", "object", "description"]}},
    },
    "required": ["scene_summary", "objects", "relations"],
}

PERCEPTION_SYSTEM = (
    "你是机器人桌面场景理解模型。你会收到【同一个 Franka 桌面场景】的 5 个相机视角"
    "(front/wrist/left/right/top)。请【逐个枚举场景里每一个不同的物体】(同一物体多视角出现算一个,去重),"
    "给出类别/外观/位置/在哪些视角可见,并列出物体之间的空间关系。忽略机器人手臂本身作为主体——重点是桌面物品"
    "(水果/盒/罐/杯/刀/剪刀/方块…)、篮子、柜子、抽屉、咖啡机。所有文字用中文。只输出要求的 JSON。")


# ======================================================================================
#  1) 感知模块:OpenAI API
# ======================================================================================
def run_perception(image_paths) -> dict:
    import requests
    import urllib3
    urllib3.disable_warnings()   # 裸 IP 自签证书 -> 关 TLS 校验警告
    if not API_KEY:
        raise RuntimeError("缺少 API key:请 export API_zhongzhuan=... 或 OPENAI_API_KEY=...")

    content = [{"type": "text", "text": "以下是同一场景的多个视角,请枚举所有物体并给出关系。"}]
    for view, p in image_paths:
        b64 = base64.b64encode(p.read_bytes()).decode()
        content.append({"type": "text", "text": f"[视角 {view}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PERCEPTION_SYSTEM},
            {"role": "user", "content": content}
        ],
        "max_tokens": 8000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "scene",
                "strict": True,
                "schema": PERCEPTION_SCHEMA
            }
        }
    }
    print(f"[1/2] 感知: 送 {len(image_paths)} 视角给 {MODEL} ...", flush=True)
    t = time.time()
    r = requests.post(RELAY_BASE.rstrip("/") + "/chat/completions",
                      headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=body, verify=False, timeout=600)
    r.raise_for_status()
    resp = r.json()
    txt = _output_text(resp)
    data = json.loads(txt)
    print(f"[1/2] 感知 done {time.time()-t:.1f}s -> objects={len(data['objects'])} relations={len(data['relations'])}",
          flush=True)
    return data


def _output_text(resp: dict) -> str:
    return resp["choices"][0]["message"]["content"]


# ======================================================================================
#  2) 记忆模块:把感知喂进三层记忆,产出给规划的文件
# ======================================================================================
def run_memory(perception: dict, task: str) -> dict:
    from perception_vlm import RecognitionResult
    from memory_module.perception_adapter import encode_recognition_results
    from memory_module import PerceptionMemoryPipeline
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    from embodiedbench.memory_manip.config import MemorySystemConfig

    objs = perception["objects"]
    summary = perception["scene_summary"]
    print(f"[2/2] 记忆: 喂入 {len(objs)} 物体 ...", flush=True)

    store = tempfile.mkdtemp(prefix="pmp_mem_")
    mem = EmbodiedManipulationMemorySystem(config=MemorySystemConfig(
        store_dir=store, embodiedltm_base_url=None,
        rig_metadata={"robot": "franka", "dof": 7, "env": "isaac_sim", "perception": MODEL}))
    pipe = PerceptionMemoryPipeline.create_from_existing(object(), mem)   # 直接喂,不用内置感知器
    pipe.begin_episode("pmp_ep", task)

    # 感知 -> RecognitionResult -> 记忆(特征 + 观测)
    rr = RecognitionResult(
        image_path="scene_5views", primary_label=(objs[0]["name"] if objs else "scene"), confidence=0.9,
        objects=[{"name": o["name"], "confidence": 0.9, "evidence": f"{o['category']}; {o['appearance']}"} for o in objs],
        attributes={}, scene=summary, reasoning=f"{MODEL} 多视角枚举", uncertainty="")
    mem.on_perception_features(encode_recognition_results([rr]))
    mem.update_observation(
        visible_objects=[o["name"] for o in objs],
        memory_objects={o["name"]: {"category": o["category"], "location": o["location"],
                                    "seen_in_views": o.get("seen_in_views", []), "source": MODEL} for o in objs},
        text=summary, current_location="table")

    # 任务规则约束 -> constraints
    pipe.set_task_constraints({"category_rules": CATEGORY_RULES,
                               "no_category_mixing": True, "collision_avoidance": True})

    ctx = pipe.get_planning_context()
    snap = mem.snapshot()
    enriched_objects = {}
    for obj_name, skills in ctx.manipulable_objects.items():
        cat = "未知"
        for o in objs:
            if o["name"] == obj_name:
                cat = o.get("category", "未知")
                break
        enriched_objects[obj_name] = {"category": cat, "skills": skills}

    planning_input = {"task_instruction": ctx.task_instruction,
                      "manipulable_objects": enriched_objects,
                      "available_skills": ctx.available_skills,
                      "constraints": ctx.constraints}
    print(f"[2/2] 记忆 done -> manipulable_objects={len(ctx.manipulable_objects)} "
          f"skills={len(ctx.available_skills)} constraints={list(ctx.constraints)}", flush=True)
    return {"planning_input": planning_input, "planning_context": ctx.to_dict(), "snapshot": snap}


# ======================================================================================
#  Monitor 安全监控层：
#    3) SSP 场景风险（planner 之前：读记忆出候选约束，回流给 planner）
#    5) Safety Critic 逐动作裁判（planner 之后：判 plan 是否危险）
# ======================================================================================
def run_ssp(planning_input: dict) -> dict:
    """第 3 阶段（Monitor/ssp_adapter）· 在 planner 之前跑：把记忆快照转成 SSP 的
    PerceptualGraph，用 scene_intrinsic 模式做场景风险解析，产出 ssp_perceptual_graph.json
    + ssp_safety_constraints.json，并把候选安全约束合并进 memory_planning_input.json 的
    constraints.safety_constraints —— **回流给 planner**。

    返回：合并了 safety_constraints 的 planning_input（内存 dict），供 planner 直接消费。
    这样 SSP 的风险约束真正影响本轮 planning，而不是写完文件就被忽略。

    ⚠️ 边界：SSP 只输出候选约束模板 id(CT-*) + 实体绑定 + 证据，不生成 LTL、不做
       accept/reject（那是下游 L3 的职责）。回写内容已如实标注该边界。

    容错：整段包 try/except，失败只打印、返回原 planning_input（planner 照常跑）。
    """
    import json as _json

    from Monitor.ssp_adapter.ssp_runner import run_ssp_pipeline

    templates_dir = ROOT / "Monitor" / "ssp_pkg" / "configs" / "re_templates"
    result = run_ssp_pipeline(
        snapshot_path=OUTPUT / "memory_snapshot.json",
        planning_input_path=OUTPUT / "memory_planning_input.json",
        output_dir=OUTPUT,
        templates_dir=templates_dir,
        mode="scene_intrinsic",   # planner 之前：不依赖 plan
    )
    gp = result.gp_build
    print(f"[3/5] SSP done -> G_P {len(gp.graph.nodes)} 节点/{len(gp.graph.edges)} 边, "
          f"场景 activated {result.num_activated}, 候选约束 {len(result.candidate_constraints)} 条 "
          f"-> 回流给 planner", flush=True)

    # 把 SSP 写回文件的 safety_constraints 读进内存 dict，回流给 planner
    merged = _json.loads((OUTPUT / "memory_planning_input.json").read_text(encoding="utf-8"))
    planning_input["constraints"] = merged.get("constraints", planning_input.get("constraints", {}))
    return planning_input


def run_safety_critic() -> None:
    """第 5 阶段（Monitor/safety_critic）：读规划出的 plan + 记忆快照，逐动作跑安全裁判，
    产出 output/safety_critic_review.json（逐步 malicious / not malicious 裁决 + 汇总）。

    评价语义：评完整条 plan（遇 malicious 记录 would_halt_here，但继续评完给全貌）。
    需要 LLM（复用本仓库 OPENAI_API_KEY / API_zhongzhuan + VLM_BASE_URL，默认 gpt-4o）。

    容错：整段包 try/except，失败只打印、不影响前 4 阶段产物。
    """
    from Monitor.safety_critic.critic_runner import run_safety_critic as _run
    from Monitor.safety_critic.pipeline_llm import PipelineLLMClient

    result = _run(
        snapshot_path=OUTPUT / "memory_snapshot.json",
        planned_actions_path=OUTPUT / "planned_actions.json",
        output_path=OUTPUT / "safety_critic_review.json",
        llm=PipelineLLMClient(),
    )
    s = result.summary()
    tail = f"，按原语义会在第 {s['first_halt_index']} 步中断" if s["first_halt_index"] is not None else ""
    print(f"[5/5] SafetyCritic done -> 评价 {s['num_steps']} 动作, 总体 {s['overall']}, "
          f"malicious {s['num_malicious']}{tail}", flush=True)


# ======================================================================================
#  主流程
# ======================================================================================
def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_paths = [(v, INPUT / f"{v}.png") for v in VIEWS if (INPUT / f"{v}.png").is_file()]
    if not image_paths:
        print(f"ERROR: input/ 里没有视角图({[v+'.png' for v in VIEWS]})", flush=True)
        return 2
    print(f"输入视角: {[v for v, _ in image_paths]}", flush=True)

    # 1) 感知
    perception = run_perception(image_paths)
    (OUTPUT / "perception.json").write_text(json.dumps(perception, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) 记忆
    mem = run_memory(perception, TASK)
    (OUTPUT / "memory_planning_input.json").write_text(
        json.dumps(mem["planning_input"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "memory_planning_context.json").write_text(
        json.dumps(mem["planning_context"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "memory_snapshot.json").write_text(
        json.dumps(mem["snapshot"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 3) Monitor · 场景安全解析 (SSP) —— planner 之前：读记忆出候选约束，回流给 planner。
    #    try/except 包裹，失败返回原 planning_input（planner 照常跑）。
    planning_input = mem["planning_input"]
    try:
        print("[3/5] SSP: 记忆快照 -> PerceptualGraph -> 场景风险 -> 回流约束给 planner...", flush=True)
        planning_input = run_ssp(planning_input)
    except Exception as e:
        print(f"SSP 阶段执行失败（不影响 planner）: {e}")

    # 4) 规划 (Planner) —— 吃已合并 SSP 安全约束的 planning_input
    from planning.llm_client import LLMClient
    from planning.task_planner import TaskPlanner
    try:
        llm = LLMClient()
        planner = TaskPlanner(llm)
        print("[4/5] 规划: 基于记忆输入(含 SSP 安全约束)生成动作序列...", flush=True)
        t = time.time()
        plan_graph = planner.generate_plan(planning_input)
        (OUTPUT / "planned_actions.json").write_text(
            json.dumps(plan_graph, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[4/5] 规划 done {time.time()-t:.1f}s -> 生成 {len(plan_graph)} 步动作", flush=True)
    except Exception as e:
        print(f"规划模块执行失败: {e}")

    # 5) Monitor · Safety Critic —— 规划之后。逐动作安全裁判，失败不影响前几阶段。
    try:
        print("[5/5] SafetyCritic: 逐动作安全裁判...", flush=True)
        run_safety_critic()
    except Exception as e:
        print(f"SafetyCritic 阶段执行失败（不影响前几阶段）: {e}")

    print("\n=== 完成,输出文件 ===", flush=True)
    for f in ("perception.json", "memory_planning_input.json",
              "memory_planning_context.json", "memory_snapshot.json", "planned_actions.json",
              "ssp_perceptual_graph.json", "ssp_safety_constraints.json",
              "safety_critic_review.json"):
        print(f"  output/{f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
