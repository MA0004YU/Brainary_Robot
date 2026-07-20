"""SSP 适配层独立入口：python -m Monitor.ssp_adapter

吃现成的 output/*.json（memory_snapshot + planned_actions + memory_planning_input），
建 G_P -> 跑 SSP（action 模式）-> 写约束回写文件。
不重跑感知/记忆、不碰 torch、不需要 API key —— 用于单独验证 SSP 这条路径。

用法（在仓库根目录）：
    conda activate biea_ssp
    python -m Monitor.ssp_adapter
    # 或指定目录：
    python -m Monitor.ssp_adapter --output-dir output --templates Monitor/ssp_pkg/configs/re_templates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- 路径 bootstrap（必须在 import ssp_runner 之前，它内部 from ssp... 需要 ssp_pkg 在 path）---
_ADAPTER_DIR = Path(__file__).resolve().parent          # Monitor/ssp_adapter
_MONITOR_DIR = _ADAPTER_DIR.parent                      # Monitor
_REPO_ROOT = _MONITOR_DIR.parent                        # 仓库根
_SSP_PKG = _MONITOR_DIR / "ssp_pkg"                     # vendored SSP 引擎
for _p in (_REPO_ROOT, _SSP_PKG):                       # 保证 Monitor.* 与 ssp.* 都能 import
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Monitor.ssp_adapter.ssp_runner import run_ssp_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SSP 适配层独立运行")
    ap.add_argument("--output-dir", default=str(_REPO_ROOT / "output"),
                    help="output 目录（读 snapshot/planned_actions，写 ssp_*.json）")
    ap.add_argument("--templates", default=str(_SSP_PKG / "configs" / "re_templates"),
                    help="SSP 风险模板目录（默认 Monitor/ssp_pkg/configs/re_templates）")
    ap.add_argument("--mode", choices=["scene_intrinsic", "action_conditioned"],
                    default="scene_intrinsic",
                    help="scene_intrinsic(默认,planner 之前,不需 plan) | action_conditioned(需 planned_actions.json)")
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    snapshot = out / "memory_snapshot.json"
    planned = out / "planned_actions.json"
    planning_input = out / "memory_planning_input.json"

    if not snapshot.exists():
        print(f"[SSP] 缺少 {snapshot}，无法独立运行。请先跑感知/记忆产出快照。", file=sys.stderr)
        return 2
    if args.mode == "action_conditioned" and not planned.exists():
        print(f"[SSP] action_conditioned 模式缺少 {planned}。", file=sys.stderr)
        return 2

    print(f"[SSP] 读取快照: {snapshot}（模式: {args.mode}）")
    result = run_ssp_pipeline(
        snapshot_path=snapshot,
        planning_input_path=planning_input,
        output_dir=out,
        templates_dir=args.templates,
        planned_actions_path=planned if args.mode == "action_conditioned" else None,
        mode=args.mode,
    )

    gp = result.gp_build
    print(f"[SSP] G_P: {len(gp.graph.nodes)} 节点, {len(gp.graph.edges)} 边")
    print(f"[SSP] activated 风险: {result.num_activated}")
    print(f"[SSP] 候选约束(去重后): {len(result.candidate_constraints)} 条")
    print(f"[SSP] 已写: {out / 'ssp_perceptual_graph.json'}")
    print(f"[SSP] 已写: {out / 'ssp_safety_constraints.json'}")
    print(f"[SSP] 已更新: {planning_input}（constraints.safety_constraints）")

    # 终端人身风险应为 0（无 human/animal victim）—— 校验预期
    terminal_types = {
        "cut_injury", "pinch_crush_injury", "fall_injury", "burn_injury",
        "electrical_shock", "hazardous_material_exposure", "bio_food_contamination",
        "choking_ingestion", "dangerous_object_transfer", "collision_impact",
    }
    terminal_hits = [
        c for c in result.candidate_constraints if c["re_type"] in terminal_types
    ]
    if terminal_hits:
        print(f"[SSP] ⚠️ 意外：出现终端人身风险 {len(terminal_hits)} 条（预期 0，因无 human/animal）")
    else:
        print("[SSP] ✅ 终端人身风险 = 0（符合预期：场景无 human/animal victim）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
