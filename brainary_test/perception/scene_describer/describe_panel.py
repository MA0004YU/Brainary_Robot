# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""omni.ui 面板:点一下把当前帧发给所选感知后端,拿回场景理解结果。

两个后端,UI 下拉切换(单卡共用):
  - GPT-5.5 (API)   -> scene_describer 的 /describe:两图+状态 -> {scene_summary, objects[], relations[]}
  - Qwen2.5-VL(local)-> perception_qwen 的 /recognize:两图 -> 每图 {primary_label, confidence, objects...}

时序(关键):相机读取必须在仿真主线程,HTTP(数秒~数十秒)绝不能卡主循环。所以:
  - 按钮 clicked_fn 只置 flag。
  - 主循环每帧 update():有 flag 且不忙 -> 主线程抓帧(快)-> 起后台线程发 HTTP -> 结果回写(加锁)。
  - omni.ui 渲染不了中文 -> 面板只显示 ASCII 骨架;完整结果(含中文)打到终端 stdout。
"""

from __future__ import annotations

import json
import threading

from .describer_client import build_payload, describe

_BACKENDS = ["GPT-5.5 (API)", "Qwen2.5-VL (local)"]


class DescribePanel:
    def __init__(self, session, gpt_addr: str = "http://127.0.0.1:5599",
                 qwen_addr: str = "http://127.0.0.1:5601"):
        import omni.ui as ui

        self.session = session
        self.gpt_addr = gpt_addr
        self.qwen_addr = qwen_addr
        self._req = False
        self._busy = False
        self._lock = threading.Lock()
        self._result: tuple | None = None       # (backend, dict) 后台线程写,update() 读后清空

        self.window = ui.Window("Scene Perception (GPT / Qwen)", width=440, height=360)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Backend")
                self._backend_model = ui.ComboBox(0, *_BACKENDS).model
                ui.Label(f"GPT: {gpt_addr}   Qwen: {qwen_addr}")
                ui.Button("Run perception now", clicked_fn=self._on_click)
                self._status = ui.Label("status: ready")
                ui.Label("Result (full text -> terminal):")
                self._body = ui.Label("(none yet)", word_wrap=True)

    # ----------------------------------------------------------------- callbacks
    def _on_click(self):
        if not self._busy:
            self._req = True

    def _backend_index(self) -> int:
        try:
            return int(self._backend_model.get_item_value_model().get_value_as_int())
        except Exception:
            return 0

    def _set_status(self, s: str):
        self._status.text = "status: " + s

    # ----------------------------------------------------------------- worker
    def _worker(self, backend: str, payload: dict):
        if backend == "qwen":
            from franka_v1_skill_lab.perception_qwen.qwen_client import recognize

            res = recognize(payload, addr=self.qwen_addr)
        else:
            res = describe(payload, addr=self.gpt_addr)
        with self._lock:
            self._result = (backend, res)
        self._busy = False

    # ----------------------------------------------------------------- per-frame
    def update(self):
        """主循环每帧调一次。"""
        if self._req and not self._busy:
            self._req = False
            backend = "qwen" if self._backend_index() == 1 else "gpt"
            try:
                if backend == "qwen":
                    from franka_v1_skill_lab.perception_qwen.qwen_client import build_qwen_payload

                    payload = build_qwen_payload(self.session)
                else:
                    payload = build_payload(self.session)
            except Exception as exc:
                self._set_status(f"capture failed: {exc}")
                return
            self._busy = True
            self._set_status(f"calling {backend} ...")
            threading.Thread(target=self._worker, args=(backend, payload), daemon=True).start()
            return

        res = None
        with self._lock:
            if self._result is not None:
                res = self._result
                self._result = None
        if res is None:
            if self._busy:
                self._set_status("calling ...")
            return

        backend, data = res
        if not data.get("ok"):
            self._set_status(f"ERROR: {data.get('error', 'unknown')}")
            self._body.text = json.dumps(data, ensure_ascii=False, indent=2)
            print(f"[{backend}] ERROR: {data.get('error')}", flush=True)
            return

        lat = data.get("latency_s", "?")
        if backend == "qwen":
            results = data.get("results", [])
            self._set_status(f"qwen ok ({len(results)} views, {lat}s) -- full text in terminal")
            self._body.text = self._format_qwen(results)
        else:
            result = data.get("result", {})
            n_obj = len(result.get("objects", []))
            n_rel = len(result.get("relations", []))
            self._set_status(f"gpt ok ({n_obj} objects, {n_rel} relations, {lat}s) -- full text in terminal")
            self._body.text = self._format_gpt(result)
        print(f"[{backend}] " + json.dumps(data.get("results", data.get("result")),
                                           ensure_ascii=False, indent=2), flush=True)

    # ----------------------------------------------------------------- formatting (ASCII-only)
    @staticmethod
    def _format_gpt(result: dict) -> str:
        lines = ["[summary -> terminal]", "", "Objects:"]
        for o in result.get("objects", []):
            lines.append(f"  - {o.get('name')}")
        lines.append("")
        lines.append("Relations:")
        for r in result.get("relations", []):
            lines.append(f"  - {r.get('subject')} --{r.get('predicate')}--> {r.get('object')}")
        return "\n".join(lines)

    @staticmethod
    def _format_qwen(results: list) -> str:
        lines = []
        for r in results:
            lines.append(f"[{r.get('name')}] label={r.get('primary_label')} "
                         f"conf={r.get('confidence')}")
            for o in r.get("objects", []):
                lines.append(f"    - {o.get('name')} ({o.get('confidence')})")
        return "\n".join(lines) if lines else "(no results)"
