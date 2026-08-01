#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# ======================================================================================
#  Brainary —— 大脑闭环一键运行:仿真 -> 感知(ChatGPT) -> 记忆 -> SSP(安全约束) -> 规划(LTM) -> 监控(SafetyCritic)
# ======================================================================================
#  一条指令跑通五个模块,每次运行在 output/<时间戳>/ 下建独立文件夹,分别存每个模块的输入/输出:
#     output/<ts>/sim/         仿真给感知的【照片(5视角RGB)+深度+场景状态】
#     output/<ts>/perception/  感知给记忆的【物体/关系识别结果 perception.json】
#     output/<ts>/memory/      记忆给规划的【planning_input.json + memory_snapshot.json + 快照报告】
#     output/<ts>/planning/    规划器产出【plan.json + planned_actions.json(+ goal_intent/sdg_plan)】
#     output/<ts>/monitor/     SafetyCritic 逐动作安全裁判【safety_critic_review.json】
#  各模块 canonical 代码在 brainary/{perception,memory,planning,monitor,simulation}/;
#  协作者原始交付归档在 brainary/_deliveries/<模块>/。另 output/latest 软链指向最近一次。
#
#  运行(在 IsaacLab 根目录):
#     conda activate env_isaaclab
#     ./isaaclab.sh -p brainary/run_brainary.py                 # 默认 headless,感知 auto(有GPT服务器就用,否则GT兜底)
#     ./isaaclab.sh -p brainary/run_brainary.py --perception gpt # 强制走 ChatGPT(需 :5599 服务器 + API key)
#     ./isaaclab.sh -p brainary/run_brainary.py --perception mock # 只用仿真GT感知(不联网,便于验证管线)
#
#  感知默认 = ChatGPT(scene_describer :5599 服务器,自带 venv+openai)。见 README「感知模块」。
# ======================================================================================
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_BRAINARY = Path(__file__).resolve().parent           # IsaacLab/brainary
_SIM = _BRAINARY / "sim"
_MEMORY = _BRAINARY / "memory"
_PERCEPTION = _BRAINARY / "perception"
_MONITOR = _BRAINARY / "monitor"                       # canonical monitor(内含 Monitor 包)
# _BRAINARY 上 path -> `import planning.xxx`; _MONITOR 上 path -> `import Monitor.xxx`;
# _MONITOR/Monitor/ssp_pkg -> ssp_adapter 内部 `from ssp.xxx`(绝对 import)。
# 注意:_BRAINARY 放最后插入 => 最高优先级,确保 canonical planning/ 永远赢过任何模块里捆绑的旧副本。
_SSP_PKG = _MONITOR / "Monitor" / "ssp_pkg"
for _p in (_SSP_PKG, _MONITOR, _SIM, _MEMORY, _PERCEPTION, _PERCEPTION / "scene_describer", _BRAINARY):
    sys.path.insert(0, str(_p))

_TASK_DEFAULT = "把桌面物品按类别分拣进三个篮子"
# 简单类别映射(GT-mock 感知 + 分拣约束用)。真实感知(GPT)会自己给 category。
_CATEGORY = {
    # 子串匹配(name / appearance 里含该关键词即命中)。中英并存,兼容 GPT 的中文外观描述。
    "banana": "水果", "orange": "水果", "lemon": "水果", "pomegranate": "水果", "apple": "水果",
    "香蕉": "水果", "橙": "水果", "橘": "水果", "柠檬": "水果", "苹果": "水果", "水果": "水果",
    "scissors": "工具", "clamp": "工具", "large_clamp": "工具", "knife": "工具", "tool": "工具",
    "剪": "工具", "刀": "工具", "钳": "工具", "工具": "工具",
    "mug": "杯具", "cup": "杯具", "杯": "杯具", "马克": "杯具", "杯具": "杯具",
    "cracker": "食品", "box": "食品", "meat": "食品", "can": "食品",
    "package": "食品", "snack": "食品", "food": "食品",
    "包装": "食品", "盒": "食品", "零食": "食品", "罐": "食品", "食品": "食品",
}
_SORT_RULES = {"水果": "Prop_KLT_3", "工具": "Prop_KLT_1", "杯具": "Prop_KLT_2", "食品": "Prop_KLT_1"}
_VALID_CATS = {"水果", "工具", "杯具", "食品"}
# planning / monitor 直连中转 LLM(感知走 scene_describer:5599 另算)
_RELAY_BASE = "https://165.154.193.90"
_LLM_MODEL = "gpt-5.5"


def _cat_of(name: str) -> str:
    low = name.lower()
    for k, v in _CATEGORY.items():
        if k in low:
            return v
    return "其他"


def _normalize_category(obj: dict) -> str:
    """把感知给的宽松 category(如 GPT 的 'ball'/'cup'/'package')归一到分拣分类法(水果/工具/杯具/食品)。
    否则规划器会把 orange_ball 这类当'球'直接丢弃(它不在 category_rules 里)。
    顺序:name 关键词 -> appearance 关键词 -> 感知已给的合法 category -> '其他'。"""
    cat = _cat_of(str(obj.get("name", "")))
    if cat != "其他":
        return cat
    cat = _cat_of(str(obj.get("appearance", "")))
    if cat != "其他":
        return cat
    gpt = str(obj.get("category", "")).strip()
    return gpt if gpt in _VALID_CATS else "其他"


# ============================================================ 1) 仿真 SIM
def stage_sim(out_sim: Path, headless: bool, device: str, seed: int, app_launcher=None):
    """启动 brainary_test 仿真,抓 5 视角 RGB+深度 + 场景状态,存进 out_sim/。返回 (sim, state_dict)。"""
    import numpy as np
    from brainary_api import BrainaryAPI

    print("[brainary] === 阶段1 仿真:启动场景 + 抓 5 视角 ===", flush=True)
    sim = BrainaryAPI.launch(headless=headless, device=device, seed=seed, _app_launcher=app_launcher)
    (out_sim / "rgb").mkdir(parents=True, exist_ok=True)
    (out_sim / "depth").mkdir(parents=True, exist_ok=True)
    from PIL import Image
    cams = sim.get_all_cameras() if hasattr(sim, "get_all_cameras") else {}
    view_files = {}
    for name, fr in (cams or {}).items():
        rgb = fr.get("rgb"); depth = fr.get("depth")
        if rgb is not None:
            p = out_sim / "rgb" / f"{name}.png"
            Image.fromarray(np.asarray(rgb)[..., :3].astype("uint8")).save(p)
            view_files[name] = str(p)
        if depth is not None:
            np.save(out_sim / "depth" / f"{name}.npy", np.asarray(depth))
    # 场景状态(物体位姿 + 机器人本体)
    state = {"task": _TASK_DEFAULT, "graspable": sim.list_graspable(), "baskets": sim.list_baskets(),
             "objects": {}, "robot": {}}
    for o in sim.list_graspable() + sim.list_baskets():
        op = sim.get_object_pose(o)
        if op is not None:
            state["objects"][o] = op
    try:
        state["robot"] = sim.get_robot_state()
    except Exception:
        pass
    (out_sim / "scene_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[brainary] 仿真产物: {len(view_files)} 张RGB + 深度 + scene_state.json -> {out_sim}", flush=True)
    return sim, state, view_files


# ============================================================ 2) 感知 PERCEPTION(默认 ChatGPT)
def _gpt_server_up(addr: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(addr.rstrip("/") + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _ensure_gpt_server(addr: str) -> bool:
    """GPT 感知服务器(scene_describer)没起就自动起它:用 perception/scene_describer/.venv_vlm(自带openai)
    + 环境里的 API_zhongzhuan/OPENAI_API_KEY(交互 shell 里已 export,isaaclab.sh 会继承)。起不来返回 False。"""
    if _gpt_server_up(addr):
        return True
    import os
    import subprocess
    sd = _PERCEPTION / "scene_describer"
    venv_py = sd / ".venv_vlm" / "bin" / "python"
    if not venv_py.exists():
        print(f"[brainary] 未找到 GPT venv({venv_py}),无法自动起服务器(见 README §3)。", flush=True)
        return False
    if not (os.environ.get("API_zhongzhuan") or os.environ.get("OPENAI_API_KEY")):
        print("[brainary] 环境无 API_zhongzhuan/OPENAI_API_KEY,GPT 服务器起不来。", flush=True)
        return False
    port = addr.rstrip("/").rsplit(":", 1)[-1]
    env = dict(os.environ); env["VLM_PORT"] = str(port)
    print(f"[brainary] 自动启动 ChatGPT 感知服务器 {addr} ...", flush=True)
    subprocess.Popen([str(venv_py), "vlm_describe_server.py"], cwd=str(sd), env=env,
                     stdout=open("/tmp/brainary_vlm_server.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(20):
        time.sleep(2)
        if _gpt_server_up(addr):
            print("[brainary] ChatGPT 感知服务器已就绪。", flush=True)
            return True
    print("[brainary] GPT 服务器启动超时。", flush=True)
    return False


def stage_perception(mode: str, out_perc: Path, sim, state: dict, view_files: dict, gpt_addr: str):
    """感知:默认 ChatGPT(scene_describer :5599)。服务器不可达则退回仿真GT(mock)。产 perception.json。"""
    result = None
    used = None
    if mode in ("gpt", "auto"):
        if _ensure_gpt_server(gpt_addr):
            try:
                from describer_client import describe   # stdlib urllib,无需 openai
                import base64, io
                import numpy as np
                from PIL import Image

                def _b64(name):
                    p = view_files.get(name)
                    if not p:
                        return None
                    buf = io.BytesIO(); Image.open(p).convert("RGB").save(buf, format="PNG")
                    return base64.b64encode(buf.getvalue()).decode("ascii")

                payload = {"rgb_front_b64": _b64("front"), "rgb_wrist_b64": _b64("wrist"), "state": state.get("robot", {})}
                resp = describe(payload, addr=gpt_addr)
                if resp.get("ok"):
                    r = resp["result"]
                    result = {"scene_summary": r.get("scene_summary", ""),
                              "objects": r.get("objects", []), "relations": r.get("relations", []),
                              "model": resp.get("model", "gpt")}
                    used = "gpt (chatgpt/scene_describer:5599)"
            except Exception as exc:
                print(f"[brainary] GPT 感知失败,退回 GT-mock: {exc}", flush=True)
        else:
            print(f"[brainary] GPT 感知服务器 {gpt_addr} 未就绪 (启动见 README)。", flush=True)
    if result is None:
        # GT-mock:用仿真真值当感知(不联网,验证管线/无 key 时可跑)
        objs = []
        for name in state.get("graspable", []):
            op = state["objects"].get(name, {})
            objs.append({"name": name, "category": _cat_of(name),
                         "appearance": "sim-GT", "location": "table",
                         "position": op.get("position")})
        result = {"scene_summary": f"仿真GT感知:桌面 {len(objs)} 个可抓物 + {len(state.get('baskets', []))} 个篮子",
                  "objects": objs, "relations": [], "model": "sim-gt-mock"}
        used = "mock (仿真GT,未用ChatGPT)"
    result["perception_backend"] = used
    # 归一 category 到分拣分类法:GPT 常给 'ball'/'package' 这类宽松标签,规划器会因其不在
    # category_rules 里而丢弃物体(如橙子 orange_ball 被当'球'漏排)。用 name/appearance 关键词纠正。
    for _o in result.get("objects", []):
        _fix = _normalize_category(_o)
        if _fix != "其他":
            _o["category"] = _fix
    (out_perc / "perception.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[brainary] === 阶段2 感知[{used}]: {len(result['objects'])} 物体 -> {out_perc/'perception.json'} ===", flush=True)
    return result


# ============================================================ 3) 记忆 MEMORY
def stage_memory(out_mem: Path, perception: dict, task: str):
    """感知结果喂进记忆(三层记忆),导出 planning_input.json(规划三样) + 记忆快照报告。"""
    import tempfile
    print("[brainary] === 阶段3 记忆:感知入记忆 + 导出规划输入 ===", flush=True)
    from perception_vlm import RecognitionResult
    from memory_module.perception_adapter import encode_recognition_results
    from memory_module import PerceptionMemoryPipeline
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    from embodiedbench.memory_manip.config import MemorySystemConfig

    objs = perception.get("objects", [])
    summary = perception.get("scene_summary", "")
    store = str(out_mem / "memory_store")
    mem = EmbodiedManipulationMemorySystem(config=MemorySystemConfig(
        store_dir=store, embodiedltm_base_url=None,
        rig_metadata={"robot": "franka", "dof": 7, "env": "isaac_sim",
                      "perception": perception.get("model", "?")}))
    pipe = PerceptionMemoryPipeline.create_from_existing(object(), mem)
    pipe.begin_episode("brainary_ep", task)
    rr = RecognitionResult(
        image_path="scene_5views", primary_label=(objs[0]["name"] if objs else "scene"), confidence=0.9,
        objects=[{"name": o["name"], "confidence": 0.9,
                  "evidence": f"{o.get('category','?')}; {o.get('appearance','')}"} for o in objs],
        attributes={}, scene=summary, reasoning=perception.get("perception_backend", ""), uncertainty="")
    mem.on_perception_features(encode_recognition_results([rr]))
    mem.update_observation(
        visible_objects=[o["name"] for o in objs],
        memory_objects={o["name"]: {"category": o.get("category", "?"), "location": o.get("location", "table"),
                                    "source": perception.get("model", "?")} for o in objs},
        text=summary, current_location="table")
    pipe.set_task_constraints({"category_rules": _SORT_RULES, "no_category_mixing": True,
                               "collision_avoidance": True})
    ctx = pipe.get_planning_context()
    snap = mem.snapshot()
    # 记忆快照落盘(供 Monitor/SafetyCritic 读 working.observation.* 判安全)
    (out_mem / "memory_snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    planning_input_path = out_mem / "planning_input.json"
    pipe.export_planning_input(str(planning_input_path))
    pi = json.loads(planning_input_path.read_text(encoding="utf-8"))
    # 记忆快照报告
    report = ["# 记忆模块输出(输入=感知结果)", "",
              f"- 感知后端: {perception.get('perception_backend')} | 物体: {len(objs)} | 任务: {task}", "",
              "## PlanningContext.to_prompt_text()", "```", ctx.to_prompt_text(), "```", "",
              "## planning_input.json(交给规划器)", "```json",
              json.dumps(pi, ensure_ascii=False, indent=2), "```"]
    (out_mem / "memory_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[brainary] 记忆产物: planning_input.json ({len(pi.get('manipulable_objects', []))} 可操作物) + report -> {out_mem}", flush=True)
    return pi


# ============================================================ 4) 规划 PLANNING
# ============================================================ 3.5) 场景安全解析 SSP(planner 之前)
def stage_ssp(out_ssp: Path, snapshot_path: Path, planning_input_path: Path, planning_input: dict) -> dict:
    """SSP 场景安全解析(记忆之后、规划之前):记忆快照 -> 候选安全约束 -> 回灌 planning_input。
    V1.1 已修风险爆炸(三态 latent/uncertain + 消 N² 笛卡尔积),约束条数已可控。
    可选阶段:缺 Monitor/依赖/快照时打印跳过,返回原 planning_input 不变,不影响规划。"""
    print("[brainary] === 阶段3.5 SSP:场景风险解析 -> 安全约束回流 planner ===", flush=True)
    if not snapshot_path.exists() or not planning_input_path.exists():
        print("[brainary] 缺 snapshot / planning_input,SSP 跳过。", flush=True)
        return planning_input
    try:
        from Monitor.ssp_adapter.ssp_runner import run_ssp_pipeline
        templates = _SSP_PKG / "configs" / "re_templates"
        res = run_ssp_pipeline(
            snapshot_path=snapshot_path,
            planning_input_path=planning_input_path,   # 我们的 planning_input.json(SSP 会把约束合并进来)
            output_dir=out_ssp,
            templates_dir=templates,
            mode="scene_intrinsic",                    # planner 之前,不依赖 plan
        )
        enriched = json.loads(Path(planning_input_path).read_text(encoding="utf-8"))
        s = res.diagnostics.get("summary", {})
        print(f"[brainary] SSP: activated={s.get('num_activated')} latent={s.get('num_latent')} "
              f"uncertain={s.get('num_uncertain')} -> {len(res.candidate_constraints)} 候选约束回灌 planning_input", flush=True)
        return enriched
    except Exception as exc:
        print(f"[brainary] SSP 阶段跳过/失败(不影响规划): {exc}", flush=True)
        return planning_input


def _rule_based_steps(planning_input: dict) -> list:
    """兜底规划:按类别规则 grasp->place,统一输出 {id,action,target,depends_on}。"""
    objs = planning_input.get("manipulable_objects", [])
    names = list(objs.keys()) if isinstance(objs, dict) else [
        (o if isinstance(o, str) else o.get("name", str(o))) for o in objs]
    rules = planning_input.get("constraints", {}).get("category_rules", _SORT_RULES)
    steps, i, prev = [], 0, None
    for name in names:
        cat = _cat_of(name)
        if cat == "其他":
            continue
        basket = rules.get(cat, "Prop_KLT_1")
        i += 1; gid = f"T{i}"
        steps.append({"id": gid, "action": "grasp", "target": name, "depends_on": [prev] if prev else []})
        i += 1; pid = f"T{i}"
        steps.append({"id": pid, "action": "place", "target": basket, "depends_on": [gid]})
        prev = pid
    return steps


def stage_planning(out_plan: Path, planning_input: dict, task: str):
    """规划:优先协作者 LTM 规划器(gpt-5.5 中转,Intent->SDG->落地);失败退回类别规则分拣。
    统一输出 plan.json(富信息) + planned_actions.json([{id,action,target,depends_on}],供 Monitor)。"""
    print("[brainary] === 阶段4 规划:读 planning_input -> 生成动作序列 ===", flush=True)
    steps, backend = [], None
    try:
        from planning.llm_client import LLMClient
        from planning.task_planner import TaskPlanner
        llm = LLMClient()                       # 默认走中转 gpt-5.5(见 planning/llm_client.py)
        planner = TaskPlanner(llm)              # use_ltm=False:不依赖 EmbodiedLTM 服务
        t = time.time()
        steps = planner.generate_plan(planning_input) or []
        # 过滤退化/无效步:LLM 失败(如中转 401)时 task_planner 可能返回 [{}],要当作没规划出来
        steps = [s for s in steps if isinstance(s, dict) and s.get("action") and s.get("target")]
        # planner 把中间件写到 <planning包父目录>/output/ 即 brainary/output/;搬进本次 run 目录
        for f in ("goal_intent.json", "sdg_plan.json"):
            src = _BRAINARY / "output" / f
            if src.exists():
                src.replace(out_plan / f)
        if steps:
            backend = f"ltm_planner (gpt-5.5, {time.time()-t:.1f}s)"
    except Exception as exc:
        print(f"[brainary] LTM 规划失败,退回规则分拣: {exc}", flush=True)
    if not steps:
        steps = _rule_based_steps(planning_input)
        backend = backend or "rule_based (兜底,未用LLM)"
    plan = {"task": task, "num_steps": len(steps), "planning_backend": backend,
            "constraints": planning_input.get("constraints", {}), "plan": steps}
    (out_plan / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    # Monitor 口径:纯 steps 列表
    (out_plan / "planned_actions.json").write_text(
        json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[brainary] 规划产物[{backend}]: {len(steps)} 步 -> {out_plan/'plan.json'}", flush=True)
    return plan


# ============================================================ 4.2) 物理沙盒校验(可选 --verify)
def stage_verify(out_verify: Path, plan_dag: list, sim) -> dict | None:
    """物理沙盒校验(可选):把 plan + 相机数据 POST 给 simulation 的【独立 HTTP 服务】(brainary_sim 进程)。
    这里【不 import sapien】,只发网络请求 -> 和 env_isaaclab 零冲突、不动驱动、不碰其它模块。
    服务没起(默认 :5600)-> 打印提示并跳过(返回 None,按放行处理)。起服务见 simulation/INTEGRATION.md。
    返回服务的结果 dict(含 success / llm_reflection_prompt)或 None。"""
    print("[brainary] === (可选)阶段4.2 物理沙盒校验(HTTP 服务)===", flush=True)
    import os
    addr = os.environ.get("SIM_VERIFY_ADDR", "http://127.0.0.1:5600")
    try:
        sys.path.insert(0, str(_BRAINARY / "simulation"))
        from verify_client import service_up, verify_via_service
        if not service_up(addr):
            print(f"[brainary] simulation 服务未在 {addr} 运行 -> 跳过物理校验"
                  f"(在 brainary_sim 环境起 `python simulation/serve.py`,见 simulation/INTEGRATION.md)", flush=True)
            return None
        res = verify_via_service(sim, plan_dag, addr=addr)
        (out_verify / "physics_verify.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[brainary] 物理校验: success={res.get('success')} | {res.get('message', '')}", flush=True)
        return res
    except Exception as exc:
        print(f"[brainary] 物理校验跳过/失败: {exc}", flush=True)
        return None


# ============================================================ 4.5) 项目管理 PM(调度+执行)
def stage_project_management(out_pm: Path, planned_actions_path: Path, planning_input_path: Path, sim=None):
    """PM 项目管理:别名解析(中文名→仿真ID)+ 依赖分波并行调度 + 执行。
    sim=None -> dry_run(离线/不动机器人,只调度不执行);传入 BrainaryAPI -> 真实执行 grasp/place。
    产出 pm_execution_result.json + pm_planned_actions.json(优化后的动作序列,交给监控/执行)。
    可选阶段:缺 plan/模块时打印跳过,返回 None。"""
    print("[brainary] === 阶段4.5 PM:别名解析 + 并行调度 + 执行 ===", flush=True)
    if not planned_actions_path.exists():
        print("[brainary] 缺 planned_actions,PM 跳过。", flush=True)
        return None
    try:
        from project_management import PMConfig, ProjectManager
        dry = sim is None
        pm = ProjectManager(sim=sim, config=PMConfig(dry_run=dry),
                            object_alias_path=_BRAINARY / "project_management" / "object_aliases.json")
        report = pm.execute_plan_file(
            planned_actions_path, planning_input_path=planning_input_path,
            output_path=out_pm / "pm_execution_result.json")
        final_actions = [{"id": it.get("id"), "action": it.get("action"),
                          "target": it.get("target"), "depends_on": it.get("depends_on", [])}
                         for it in report.get("scheduled_plan", [])]
        (out_pm / "pm_planned_actions.json").write_text(
            json.dumps(final_actions, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[brainary] PM[{'dry_run' if dry else 'execute'}]: "
              f"scheduled={len(report.get('scheduled_plan', []))} "
              f"executed={len(report.get('executed', []))} failed={len(report.get('failed', []))} "
              f"-> {out_pm/'pm_execution_result.json'}", flush=True)
        return report
    except Exception as exc:
        print(f"[brainary] PM 阶段跳过/失败: {exc}", flush=True)
        return None


# ============================================================ 5) 监控 MONITOR / SafetyCritic
def stage_monitor(out_mon: Path, snapshot_path: Path, planned_actions_path: Path, model: str):
    """Monitor/SafetyCritic:逐动作判 malicious/not malicious,写 safety_critic_review.json。
    可选阶段:缺 key/缺模块/缺输入时打印跳过,不影响前 4 阶段产物。"""
    print("[brainary] === 阶段5 监控:SafetyCritic 逐动作安全裁判 ===", flush=True)
    if not snapshot_path.exists() or not planned_actions_path.exists():
        print("[brainary] 缺 memory_snapshot / planned_actions,监控跳过。", flush=True)
        return None
    try:
        import os
        os.environ.setdefault("VLM_BASE_URL", _RELAY_BASE)   # 归一化后自动补 /v1
        from Monitor.safety_critic.critic_runner import run_safety_critic
        from Monitor.safety_critic.pipeline_llm import PipelineLLMClient
        llm = PipelineLLMClient(model=model)
        res = run_safety_critic(
            snapshot_path=snapshot_path,
            planned_actions_path=planned_actions_path,
            output_path=out_mon / "safety_critic_review.json",
            llm=llm,
        )
        s = res.summary()
        print(f"[brainary] 监控产物: {s['num_steps']}步 overall={s['overall']} "
              f"malicious={s['num_malicious']} tokens={s['total_tokens']} "
              f"-> {out_mon/'safety_critic_review.json'}", flush=True)
        return s
    except Exception as exc:
        print(f"[brainary] 监控阶段跳过/失败: {exc}", flush=True)
        return None


# ============================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="Brainary 大脑闭环一键运行: 仿真->感知->记忆->规划")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--perception", choices=["auto", "gpt", "mock"], default="auto",
                    help="感知后端: auto=有GPT服务器就用ChatGPT否则GT兜底(默认) / gpt=强制ChatGPT / mock=仅仿真GT")
    ap.add_argument("--gpt_addr", default="http://127.0.0.1:5599", help="ChatGPT感知服务器(scene_describer)地址")
    ap.add_argument("--task", default=_TASK_DEFAULT, help="任务指令")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--execute", action="store_true",
                    help="PM 阶段把 plan 真实下发到机器人执行(grasp/place);默认 dry_run 只调度不动机器人")
    ap.add_argument("--verify", action="store_true",
                    help="规划后用物理沙盒 PhysicalValidator 预演校验 plan,失败则反思 replan(需 sim CUDA 环境+权重;缺则跳过)")
    args, _ = ap.parse_known_args()
    args.enable_cameras = True                      # 5 相机渲染
    args.headless = True                            # 批处理闭环:无窗口(更快;要看画面就跑 sim/brainary_test_ui.py)
    app_launcher = AppLauncher(args)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _BRAINARY / "output" / ts
    dirs = {k: run_dir / k for k in ("sim", "perception", "memory", "planning", "project_management", "monitor")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[brainary] 本次运行输出目录: {run_dir}", flush=True)

    sim = None
    try:
        sim, state, view_files = stage_sim(dirs["sim"], headless=bool(args.headless),
                                           device=args.device, seed=args.seed, app_launcher=app_launcher)
        perception = stage_perception(args.perception, dirs["perception"], sim, state, view_files, args.gpt_addr)
        planning_input = stage_memory(dirs["memory"], perception, args.task)
        planning_input = stage_ssp(dirs["monitor"], dirs["memory"] / "memory_snapshot.json",
                                   dirs["memory"] / "planning_input.json", planning_input)
        plan = stage_planning(dirs["planning"], planning_input, args.task)
        # (可选)物理沙盒校验:success 放行;False 把 llm_reflection_prompt 注入后单次 replan
        if getattr(args, "verify", False):
            vres = stage_verify(dirs["planning"], plan.get("plan", []), sim)
            if vres and not vres.get("success") and vres.get("llm_reflection_prompt"):
                planning_input = {**planning_input, "physics_reflection": vres["llm_reflection_prompt"]}
                plan = stage_planning(dirs["planning"], planning_input, args.task)
        # PM:--execute 时把真实 sim 传进去执行 grasp/place(闭环到机器人);否则 dry_run 只调度
        pm = stage_project_management(dirs["project_management"], dirs["planning"] / "planned_actions.json",
                                      dirs["memory"] / "planning_input.json",
                                      sim=(sim if getattr(args, "execute", False) else None))
        # 监控读 PM 优化后的动作序列(缺则退回原 planned_actions)
        pm_actions = dirs["project_management"] / "pm_planned_actions.json"
        actions_for_monitor = pm_actions if pm_actions.exists() else dirs["planning"] / "planned_actions.json"
        safety = stage_monitor(dirs["monitor"], dirs["memory"] / "memory_snapshot.json",
                               actions_for_monitor, _LLM_MODEL)
        # 汇总 + latest 软链
        summary = {"timestamp": ts, "task": args.task,
                   "perception_backend": perception.get("perception_backend"),
                   "num_objects": len(perception.get("objects", [])),
                   "planning_backend": plan.get("planning_backend"),
                   "num_plan_steps": plan.get("num_steps"),
                   "pm": ({"mode": pm.get("mode"), "ok": pm.get("ok"),
                           "executed": len(pm.get("executed", [])),
                           "failed": len(pm.get("failed", []))} if pm else None),
                   "safety": safety,
                   "outputs": {k: str(v) for k, v in dirs.items()}}
        (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        latest = _BRAINARY / "output" / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(ts)
        except Exception:
            pass
        print(f"[brainary] ✅ 闭环完成。汇总: {run_dir/'run_summary.json'} | output/latest -> {ts}", flush=True)
        return 0
    finally:
        if sim is not None:
            sim.close()
        try:
            app_launcher.app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
