#!/usr/bin/env python3
"""把最新仿真场景的 GPT-5.5 感知(26物体)喂进记忆模块,dump 完整记忆输出。"""
import sys, json, tempfile, traceback
from pathlib import Path
sys.path.insert(0, "/home1/banghai/Documents/IsaacLab/projects/xiaoyu/brainary_memory_pkg_active")

GPT = json.load(open("logs/perceive_scene/gpt_perception.json", encoding="utf-8"))
OBJS = GPT["objects"]; SUMMARY = GPT["scene_summary"]
OUT = Path("logs/perceive_scene/记忆模块完整输出_GPT感知.md")

def main():
    from perception_vlm import RecognitionResult
    from memory_module.perception_adapter import encode_recognition_results
    from memory_module import PerceptionMemoryPipeline
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    from embodiedbench.memory_manip.config import MemorySystemConfig

    store = tempfile.mkdtemp(prefix="gpt_mem_")
    mem = EmbodiedManipulationMemorySystem(config=MemorySystemConfig(store_dir=store, embodiedltm_base_url=None,
                                          rig_metadata={"robot": "franka", "dof": 7, "env": "isaac_sim", "perception": "gpt-5.5"}))
    pipe = PerceptionMemoryPipeline.create_from_existing(object(), mem)   # 直接喂,不用内置感知
    task = "把桌面物品按类别分拣进三个篮子"
    pipe.begin_episode("gpt_scene_ep", task)

    # ---- GPT 感知 -> RecognitionResult -> 记忆 ----
    rr = RecognitionResult(
        image_path="scene_5views", primary_label=(OBJS[0]["name"] if OBJS else "scene"),
        confidence=0.9,
        objects=[{"name": o["name"], "confidence": 0.9, "evidence": f"{o['category']}; {o['appearance']}"} for o in OBJS],
        attributes={}, scene=SUMMARY, reasoning="GPT-5.5 5视角枚举", uncertainty="")
    mem.on_perception_features(encode_recognition_results([rr]))
    mem.update_observation(
        visible_objects=[o["name"] for o in OBJS],
        memory_objects={o["name"]: {"category": o["category"], "location": o["location"],
                                    "seen_in_views": o.get("seen_in_views", []), "source": "gpt-5.5"} for o in OBJS},
        text=SUMMARY, current_location="table")

    # ---- 任务规则约束(分拣) -> constraints ----
    pipe.set_task_constraints({
        "category_rules": {"水果": "Prop_KLT_3", "工具": "Prop_KLT_1", "杯具": "Prop_KLT_2"},
        "no_category_mixing": True, "collision_avoidance": True})

    # ---- 记几个动作(闭环感) ----
    pipe.record_action("grasp 香蕉", success=True, feedback="gripper closed")
    pipe.record_action("place -> Prop_KLT_3", success=True, feedback="dropped in basket")

    ctx = pipe.get_planning_context()
    snap = mem.snapshot()
    pipe.export_planning_input("logs/planning/planning_input.json")
    pi = json.load(open("logs/planning/planning_input.json", encoding="utf-8"))

    # ---- 写完整文档 ----
    L = ["# 记忆模块完整输出(输入=GPT-5.5 对最新仿真 5 视角的感知)", "",
         f"- 感知: gpt-5.5 枚举 {len(OBJS)} 物体 | 任务: {task}", "",
         "## 1. 工作记忆 Working(当前局)", "```json",
         json.dumps(snap["working"], ensure_ascii=False, indent=2), "```", "",
         "## 2. 情景记忆 Episodic", "```json",
         json.dumps(snap["episodic"], ensure_ascii=False, indent=2), "```", "",
         "## 3. 语义记忆 Semantic", "```json",
         json.dumps(snap["semantic"], ensure_ascii=False, indent=2), "```", "",
         "## 4. PlanningContext(交给规划器)", "", "### to_prompt_text():", "```",
         ctx.to_prompt_text(), "```", "", "### 完整字段:", "```json",
         json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2), "```", "",
         "## 5. planning_input.json(规划三样:物体/动作/约束)", "```json",
         json.dumps(pi, ensure_ascii=False, indent=2), "```"]
    OUT.write_text("\n".join(L), encoding="utf-8")

    print(f"[mem] 感知喂入: {len(OBJS)} 物体")
    print(f"[mem] visible_objects: {len(ctx.visible_objects)}")
    print(f"[mem] manipulable_objects: {len(ctx.manipulable_objects)}")
    print(f"[mem] available_skills: {ctx.available_skills}")
    print(f"[mem] constraints: {json.dumps(ctx.constraints, ensure_ascii=False)}")
    print(f"[mem] 文档 -> {OUT}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(1)
