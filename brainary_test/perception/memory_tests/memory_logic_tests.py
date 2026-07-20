#!/usr/bin/env python3
"""记忆模块逻辑有效性测试(A-E,离线、确定性,不需要 Qwen/仿真)。

用受控/GT 输入隔离记忆机制本身,避免感知噪声混入。每个测试返回 {name, passed, detail}。
Run:  <python-with-numpy> projects/lianyu/memory_tests/memory_logic_tests.py
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

_MEM_PKG = Path(__file__).resolve().parents[2] / "xiaoyu" / "brainary_memory_pkg_active"
sys.path.insert(0, str(_MEM_PKG))

from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
from embodiedbench.memory_manip.config import MemorySystemConfig
from embodiedbench.memory_manip.episodic_memory import compute_activation
from memory_module.planning_interface import PlanningMemoryInterface

RESULTS = []


def _mem(store):
    return EmbodiedManipulationMemorySystem(
        config=MemorySystemConfig(store_dir=str(store), embodiedltm_base_url=None))


def _run_episode(mem, scene_id, instruction, visible, location, action, success, skills):
    mem.reset_episode(scene_id=scene_id)
    mem.begin_task(instruction=instruction)
    mem.update_observation(visible_objects=visible, current_location=location)
    mem.record_action(action=action, action_id=action, success=success)
    mem.end_episode(success=success, blueprint_skills=skills)


def test_A_task_learning(store):
    """跨回合积累成功率 + 推荐技能(只成功才更新,失败不覆盖)。"""
    mem = _mem(store)
    good = ["move_above", "descend", "grasp", "lift", "place", "retreat"]
    # 正好 5 回合(=归纳周期 episodic_generalize_every_n),[成成败成成]:4 成功 1 失败 -> 全部被归纳。
    # 失败回合用错误序列,验证它不会覆盖推荐(record_blueprint_skills 只在 success 时写)。
    plan = [True, True, False, True, True]
    for i, ok in enumerate(plan):
        _run_episode(mem, f"A{i}", "pick up the cube and place it on the target",
                     ["cube", "target"], "table", "pick cube",
                     ok, good if ok else ["wrong", "bad"])
    mem.decay_and_prune()
    schema = mem.query_task_schema("pick up the cube and place it on the target")
    sr = schema["success_rate"]
    rec = schema["blueprint_skills"]
    passed = (schema["task_type"] == "pick_and_place"
              and sr is not None and abs(sr - 4 / 5) < 0.02
              and rec == good)
    RESULTS.append({"name": "A_task_learning", "passed": passed,
                    "detail": {"task_type": schema["task_type"], "success_rate": round(sr, 3) if sr else None,
                               "recommended_skills": rec, "expected_sr": 0.8, "episodes": 5,
                               "note": "5 回合(=归纳周期)全被归纳;success_rate=4/5;失败序列未覆盖推荐"}})


def test_B_similar_retrieval(store):
    """相似回合检索:cube 查询应召回 cube 回合,drawer 回合排后/排除。"""
    mem = _mem(store)
    for i in range(3):
        _run_episode(mem, f"c{i}", "pick up the cube", ["cube"], "table", "pick cube", True, ["grasp"])
    for i in range(2):
        _run_episode(mem, f"d{i}", "open the drawer", ["drawer"], "cabinet", "pull drawer", True, ["pull"])
    sim = mem.query_similar_episodes("pick up the red cube", top_k=5)
    top3 = sim[:3]
    all_cube_top = all("cube" in s["task_instruction"] for s in top3)
    drawer_after = all(s["task_instruction"] == "open the drawer" for s in sim[3:]) if len(sim) > 3 else True
    RESULTS.append({"name": "B_similar_retrieval", "passed": bool(all_cube_top and drawer_after),
                    "detail": {"query": "pick up the red cube",
                               "ranking": [(s["task_instruction"], s["success"]) for s in sim]}})


def test_C_object_priors(store):
    """物体位置先验:cube 多在 table、少在 shelf -> 先验首位应为 table + 可 pick。"""
    mem = _mem(store)
    for i in range(4):
        _run_episode(mem, f"t{i}", "pick up the cube", ["cube"], "table", "pick cube", True, ["grasp"])
    _run_episode(mem, "s0", "pick up the cube", ["cube"], "shelf", "pick cube", True, ["grasp"])  # 第5个触发归纳
    kb = mem.query_object("cube")
    locs = kb["likely_locations"]  # [(loc, freq), ...] desc
    passed = (bool(locs) and locs[0][0] == "table" and locs[0][1] > (locs[1][1] if len(locs) > 1 else 0)
              and "pick" in kb["affordances"])
    RESULTS.append({"name": "C_object_priors", "passed": passed,
                    "detail": {"likely_locations": locs, "affordances": kb["affordances"]}})


def test_D_activation_decay(store):
    """ACT-R 激活公式:age 衰减 / access 回升;并验证检索会 reactivate。"""
    a0 = compute_activation(1.0, 0, creation_episode=0, current_episode=0)     # age0
    a_age = compute_activation(1.0, 0, creation_episode=0, current_episode=10)  # 老
    a_acc = compute_activation(1.0, 5, creation_episode=0, current_episode=0)   # 被用过5次
    formula_ok = (abs(a0 - 1.0) < 1e-9
                  and abs(a_age - math.exp(-0.5 * 10)) < 1e-6
                  and abs(a_acc - (1.0 + math.log1p(5) * 0.3)) < 1e-6)
    # 检索 reactivate:同一回合查两次,access_count 递增 -> 激活升高
    mem = _mem(store)
    for i in range(3):
        _run_episode(mem, f"e{i}", "pick up the cube", ["cube"], "table", "pick cube", True, ["grasp"])
    before = mem.episodic.get_activation_stats()["mean"]
    mem.query_similar_episodes("pick up the cube", top_k=3)
    mem.query_similar_episodes("pick up the cube", top_k=3)
    after = mem.episodic.get_activation_stats()["mean"]
    reactivate_ok = after > before
    RESULTS.append({"name": "D_activation_decay", "passed": bool(formula_ok and reactivate_ok),
                    "detail": {"a_age0": round(a0, 4), "a_age10": round(a_age, 6),
                               "a_access5": round(a_acc, 4), "formula_ok": formula_ok,
                               "mean_before_retrieval": before, "mean_after_retrieval": after,
                               "reactivate_ok": reactivate_ok}})


def test_E_persistence(store):
    """持久化:写入知识 -> 销毁对象 -> 同 store_dir 重建 -> 知识一致。"""
    mem = _mem(store)
    good = ["move_above", "grasp", "lift", "place"]
    for i in range(5):
        _run_episode(mem, f"p{i}", "pick up the cube and place it", ["cube"], "table", "pick cube", True, good)
    sr1 = mem.query_task_schema("pick up the cube and place it")["success_rate"]
    rec1 = mem.query_task_schema("pick up the cube and place it")["blueprint_skills"]
    n1 = mem.episodic.total_episodes
    del mem
    mem2 = _mem(store)  # 重开进程/对象,读同一磁盘
    sr2 = mem2.query_task_schema("pick up the cube and place it")["success_rate"]
    rec2 = mem2.query_task_schema("pick up the cube and place it")["blueprint_skills"]
    n2 = mem2.episodic.total_episodes
    passed = (sr1 == sr2 and rec1 == rec2 and n1 == n2 and n1 == 5)
    RESULTS.append({"name": "E_persistence", "passed": passed,
                    "detail": {"before": {"success_rate": sr1, "recommended": rec1, "episodes": n1},
                               "after_reload": {"success_rate": sr2, "recommended": rec2, "episodes": n2}}})


def main():
    base = Path(tempfile.mkdtemp(prefix="mem_logic_"))
    try:
        for fn in (test_A_task_learning, test_B_similar_retrieval, test_C_object_priors,
                   test_D_activation_decay, test_E_persistence):
            store = base / fn.__name__
            try:
                fn(store)
            except Exception as exc:
                RESULTS.append({"name": fn.__name__, "passed": False, "detail": {"error": repr(exc)}})
        n_pass = sum(1 for r in RESULTS if r["passed"])
        out = {"summary": f"{n_pass}/{len(RESULTS)} passed", "results": RESULTS}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        report = Path(__file__).resolve().parent / "memory_logic_results.json"
        report.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[saved] {report}")
        return 0 if n_pass == len(RESULTS) else 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
