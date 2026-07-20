#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================
#  main.py —— 感知(GPT-5.5) + 记忆 一键流水线
# ======================================================================================
#  读 input/ 里的 5 张视角图 -> GPT-5.5 感知(枚举物体+关系) -> 喂记忆模块 ->
#  喂给规划模块(Intent -> SDG -> 带有CoT的动作落地) ->
#  把感知、记忆、规划结果 全部写成 output/ 下的 JSON。
#
#  一键运行(key 从环境变量 API_zhongzhuan 读取,交互终端会自动加载 ~/.bashrc,通常无需在命令里指定):
#     conda activate env_isaaclab         # 需带 torch+numpy+requests 的环境
#     python main.py
#  (如果 key 没自动加载,才临时: export API_zhongzhuan=<你的完整key>)
#
#  输出(output/):
#     perception.json               感知输出(GPT-5.5:objects[] + relations[] + scene_summary)
#     memory_planning_input.json    ★记忆给规划模块的文件(manipulable_objects/skills/constraints)
#     memory_planning_context.json  记忆产出的完整 PlanningContext
#     memory_snapshot.json          记忆三层快照(working/episodic/semantic)
#     goal_intent.json              规划内部产出：抽象深层意图
#     sdg_plan.json                 规划内部产出：环境状态依赖图
#     planned_actions.json          规划最终输出(含推理链和具体动作序列)
#     safety_critic_review.json     ★安全裁判(Monitor):逐动作 malicious/not malicious 裁决(可选阶段)
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
sys.path.insert(0, str(ROOT / "memory_pkg"))   # vendored 记忆模块

VIEWS = ["front", "wrist", "left", "right", "top"]

# ---------------- 配置(可用环境变量覆盖) ----------------
RELAY_BASE = os.environ.get("VLM_BASE_URL", "https://165.154.193.90")   # 中转 Responses API 端点根(/responses)
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

    # 用 Responses API(本中转 wire_api=responses;实测 chat/completions 不强制 json_schema,会返回纯文本)。
    content = [{"type": "input_text", "text": "以下是同一场景的多个视角,请枚举所有物体并给出关系。"}]
    for view, p in image_paths:
        b64 = base64.b64encode(p.read_bytes()).decode()
        content.append({"type": "input_text", "text": f"[视角 {view}]"})
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": "high"})

    body = {
        "model": MODEL, "instructions": PERCEPTION_SYSTEM,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 8000, "reasoning": {"effort": REASONING},
        "text": {"format": {"type": "json_schema", "name": "scene", "strict": True, "schema": PERCEPTION_SCHEMA}},
    }
    print(f"[1/2] 感知: 送 {len(image_paths)} 视角给 {MODEL} (Responses API) ...", flush=True)
    t = time.time()
    r = requests.post(RELAY_BASE.rstrip("/") + "/responses",
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
    # Responses API: 从 output[].content[] 里取 output_text
    for item in resp.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                return c["text"]
    raise RuntimeError(f"Responses 返回里没有 output_text: {json.dumps(resp, ensure_ascii=False)[:400]}")


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

    # 3) 规划 (Planner)
    from planning.llm_client import LLMClient
    from planning.task_planner import TaskPlanner
    try:
        llm = LLMClient()
        planner = TaskPlanner(llm)
        print("[3/4] 规划: 开始基于记忆输入生成动作序列...", flush=True)
        t = time.time()
        plan_graph = planner.generate_plan(mem["planning_input"])
        (OUTPUT / "planned_actions.json").write_text(
            json.dumps(plan_graph, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[3/4] 规划 done {time.time()-t:.1f}s -> 生成 {len(plan_graph)} 步动作", flush=True)
    except Exception as e:
        print(f"规划模块执行失败: {e}")

    # 4) 安全裁判 (Safety Critic) —— 读规划出的 plan + 记忆快照,逐动作判安全性
    #    可选阶段:Monitor 模块不在或无 key 时,只打印跳过,不影响前 3 阶段产物。
    #    Monitor 根目录可用环境变量 MONITOR_ROOT 覆盖(默认 brainary/Monitor/Perception_memory_planner_monitor_pipeline)。
    run_safety_critic_stage()

    print("\n=== 完成,输出文件 ===", flush=True)
    for f in ("perception.json", "memory_planning_input.json",
              "memory_planning_context.json", "memory_snapshot.json", "planned_actions.json",
              "safety_critic_review.json"):
        if (OUTPUT / f).exists():
            print(f"  output/{f}", flush=True)
    return 0


# ======================================================================================
#  4) 安全裁判 (Monitor / SafetyCritic):对规划出的 plan 逐动作判 malicious / not malicious
# ======================================================================================
def run_safety_critic_stage() -> None:
    plan_file = OUTPUT / "planned_actions.json"
    snap_file = OUTPUT / "memory_snapshot.json"
    if not plan_file.exists() or not snap_file.exists():
        print("[4/4] 安全裁判: 缺 plan / snapshot,跳过。", flush=True)
        return
    try:
        monitor_root = os.environ.get("MONITOR_ROOT") or str(
            ROOT.parents[1] / "Monitor" / "Perception_memory_planner_monitor_pipeline")
        if not Path(monitor_root).exists():
            print(f"[4/4] 安全裁判: 未找到 Monitor 模块({monitor_root}),跳过。", flush=True)
            return
        if monitor_root not in sys.path:
            sys.path.insert(0, monitor_root)
        # SafetyCritic 的 LLM 客户端读 VLM_BASE_URL;把感知/规划用的中转 base 传下去,口径一致
        os.environ["VLM_BASE_URL"] = RELAY_BASE
        from Monitor.safety_critic.critic_runner import run_safety_critic
        from Monitor.safety_critic.pipeline_llm import PipelineLLMClient

        print("[4/4] 安全裁判: 逐动作评价 plan 安全性 ...", flush=True)
        t = time.time()
        llm_sc = PipelineLLMClient(model=MODEL)   # 用与规划一致的模型(中转 gpt-5.5)
        sc = run_safety_critic(
            snapshot_path=snap_file,
            planned_actions_path=plan_file,
            output_path=OUTPUT / "safety_critic_review.json",
            llm=llm_sc,
        )
        s = sc.summary()
        halt = "" if s["first_halt_index"] is None else f" (首个中断步 #{s['first_halt_index']})"
        print(f"[4/4] 安全裁判 done {time.time()-t:.1f}s -> {s['num_steps']}步 overall={s['overall']} "
              f"malicious={s['num_malicious']}{halt} tokens={s['total_tokens']}", flush=True)
    except Exception as e:
        print(f"安全裁判阶段跳过/失败: {e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
