#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================
#  Brainary 离线一键运行 —— 【无需 Isaac Sim】
# ======================================================================================
#  给没有仿真器的协作者:直接读【静态的仿真数据(5视角RGB + 深度 + scene_state.json)】当输入,
#  一键跑完仿真之后的所有模块:感知 -> 记忆 -> 规划 -> 监控。用于各自测试自己的模块。
#
#  与 run_brainary.py 的唯一区别:第1阶段"仿真"被"读静态数据"替换,其余 4 阶段完全复用同一份代码。
#
#  运行(在 IsaacLab 根目录,或 brainary 所在任意目录):
#     conda activate <带 torch/numpy/pillow/requests/pydantic/networkx/structlog 的环境>
#     python brainary/run_offline.py                       # 默认读 sample_data/sim,感知走 mock(GT,不联网)
#     python brainary/run_offline.py --perception gpt      # 感知走 ChatGPT(需 API_zhongzhuan + scene_describer:5599)
#     python brainary/run_offline.py --sim-data <某次运行的 sim 目录>   # 换成别的静态数据
#
#  依赖见 brainary/requirements-offline.txt;部署见 brainary/DEPLOY_OFFLINE.md。
#  输出与 run_brainary 一致:output/<时间戳>/{perception,memory,planning,monitor}/ + output/latest。
# ======================================================================================
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_BRAINARY = Path(__file__).resolve().parent

# 复用 run_brainary.py 里的阶段函数(它在 import 时会把 planning/monitor/memory/perception 挂上 sys.path;
# 且不会 import isaacsim —— AppLauncher 只在 run_brainary.main() 内部 import,这里不触发)。
sys.path.insert(0, str(_BRAINARY))
import run_brainary as rb   # noqa: E402


def load_static_sim(sim_data_dir: Path):
    """读静态仿真数据 -> (state, view_files),对齐 run_brainary.stage_sim 的返回(去掉 sim 句柄)。

    期望目录结构(即 run_brainary 每次跑出来的 output/<ts>/sim/):
        <sim_data_dir>/rgb/<view>.png      5 视角 RGB
        <sim_data_dir>/depth/<view>.npy    深度(可选,当前下游不强依赖)
        <sim_data_dir>/scene_state.json    场景状态(task/graspable/baskets/objects/robot)
    """
    sim_data_dir = Path(sim_data_dir)
    state_file = sim_data_dir / "scene_state.json"
    rgb_dir = sim_data_dir / "rgb"
    if not state_file.exists():
        raise FileNotFoundError(f"缺 scene_state.json: {state_file}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"缺 rgb/ 目录: {rgb_dir}")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    view_files = {p.stem: str(p) for p in sorted(rgb_dir.glob("*.png"))}
    if not view_files:
        raise FileNotFoundError(f"{rgb_dir} 下没有 *.png 视角图")
    return state, view_files


def main() -> int:
    ap = argparse.ArgumentParser(description="Brainary 离线一键运行(无需 Isaac Sim):感知->记忆->规划->监控")
    ap.add_argument("--sim-data", default=str(_BRAINARY / "sample_data" / "sim"),
                    help="静态仿真数据目录(含 rgb/ + scene_state.json)。默认 brainary/sample_data/sim")
    ap.add_argument("--perception", choices=["auto", "gpt", "mock"], default="mock",
                    help="感知后端: mock=仿真GT不联网(默认,零配置) / gpt=ChatGPT(需key+服务器) / auto=有GPT就用否则GT")
    ap.add_argument("--gpt_addr", default="http://127.0.0.1:5599", help="ChatGPT感知服务器地址")
    ap.add_argument("--task", default=rb._TASK_DEFAULT, help="任务指令")
    args = ap.parse_args()

    state, view_files = load_static_sim(Path(args.sim_data))
    task = state.get("task") or args.task

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _BRAINARY / "output" / ts
    dirs = {k: run_dir / k for k in ("sim", "perception", "memory", "planning", "monitor")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[offline] 本次运行输出目录: {run_dir}", flush=True)
    print(f"[offline] === 阶段1 读静态仿真数据: {len(view_files)} 视角 <- {args.sim_data} ===", flush=True)
    # 把静态数据也镜像进本次 run 的 sim/(留痕,与 run_brainary 输出结构一致)
    (dirs["sim"] / "scene_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    perception = rb.stage_perception(args.perception, dirs["perception"], None, state, view_files, args.gpt_addr)
    planning_input = rb.stage_memory(dirs["memory"], perception, task)
    plan = rb.stage_planning(dirs["planning"], planning_input, task)
    safety = rb.stage_monitor(dirs["monitor"], dirs["memory"] / "memory_snapshot.json",
                              dirs["planning"] / "planned_actions.json", rb._LLM_MODEL)

    summary = {"timestamp": ts, "mode": "offline (no-isaac)", "task": task,
               "sim_data": str(args.sim_data),
               "perception_backend": perception.get("perception_backend"),
               "num_objects": len(perception.get("objects", [])),
               "planning_backend": plan.get("planning_backend"),
               "num_plan_steps": plan.get("num_steps"),
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
    print(f"[offline] ✅ 完成。汇总: {run_dir/'run_summary.json'} | output/latest -> {ts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
