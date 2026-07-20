# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen2.5-VL 识别 HTTP server(独立 venv,常驻加载本地 3B 模型,占 GPU)。

收 POST /recognize  body = {"images":[{"name":"front","b64":"..."},...], "candidate_labels":[...],
                            "save_memory": true}
把每张 b64 写成临时 PNG -> QwenVLMPerception.recognize_batch(paths, labels) -> 返回
{"ok": true, "results":[{"name","primary_label","confidence","objects","attributes","scene",
"reasoning","uncertainty"}...], "latency_s": ...}。出错 {"ok": false, "error": "..."}。

启动时就 load 一次模型(首次会从 HuggingFace 下载 ~7GB 到 ~/.cache/huggingface;慢可设
HF_ENDPOINT=https://hf-mirror.com)。GET /health 返回是否就绪 + 配置。

环境变量:
  QWEN_MODEL_ID     默认 Qwen/Qwen2.5-VL-3B-Instruct(可指向本地已下好的路径)
  QWEN_AUX_VIEWS    默认 1   (轮廓/纹理辅助视图;关掉省显存/提速 -> 0)
  QWEN_DEPTH        默认 0   (深度辅助视图,会再下 Depth-Anything-V2-Small)
  QWEN_MAX_NEW_TOKENS 默认 512
  QWEN_MEMORY       默认 perception_vlm_memory.json(外部长期记忆)
  QWEN_PORT         默认 5601

启动:
  cd projects/franka_v1_skill_lab/perception_qwen
  bash setup_venv.sh
  .venv_qwen/bin/python qwen_perception_server.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _cfg() -> dict:
    def _flag(name, default):
        return os.environ.get(name, default) not in ("0", "false", "False", "")
    return {
        "model_id": os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "use_aux": _flag("QWEN_AUX_VIEWS", "1"),
        "enable_depth": _flag("QWEN_DEPTH", "0"),
        "max_new_tokens": int(os.environ.get("QWEN_MAX_NEW_TOKENS", "512")),
        "memory_path": os.environ.get("QWEN_MEMORY", "perception_vlm_memory.json"),
        "port": int(os.environ.get("QWEN_PORT", "5601")),
    }


def _build_recognizer(cfg: dict):
    """实例化识别器(此处触发 transformers 加载 + 首次下载)。延迟 import 让缺依赖时报清晰错。"""
    from perception_vlm import QwenVLMPerception

    return QwenVLMPerception(
        model_id=cfg["model_id"],
        memory_path=cfg["memory_path"],
        max_new_tokens=cfg["max_new_tokens"],
        use_auxiliary_views=cfg["use_aux"],
        enable_depth=cfg["enable_depth"],
    )


def _recognize(recognizer, images: list, labels: list, save_memory: bool) -> list:
    """把 b64 图写临时 PNG,按顺序识别,结果贴回各自的 name。"""
    tmp_paths, names = [], []
    try:
        for i, im in enumerate(images):
            name = im.get("name", f"img{i}")
            raw = base64.b64decode(im["b64"])
            fd, p = tempfile.mkstemp(suffix=".png", prefix=f"qwen_{name}_")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            tmp_paths.append(p)
            names.append(name)
        results = recognizer.recognize_batch(tmp_paths, candidate_labels=labels or None,
                                             save_memory=save_memory)
        out = []
        for name, res in zip(names, results):
            d = asdict(res)
            d["name"] = name
            d.pop("raw_response", None)   # 原始长文本不外发(需要可加回)
            out.append(d)
        return out
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/health":
            c = self.server.cfg
            self._send(200, {"ok": True, "ready": self.server.recognizer is not None,
                             "model_id": c["model_id"], "use_aux": c["use_aux"],
                             "enable_depth": c["enable_depth"]})
        else:
            self._send(404, {"ok": False, "error": "use POST /recognize or GET /health"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/recognize":
            self._send(404, {"ok": False, "error": "unknown path; use POST /recognize"})
            return
        if self.server.recognizer is None:
            self._send(503, {"ok": False, "error": "model not loaded"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as exc:
            self._send(400, {"ok": False, "error": f"bad request body: {exc}"})
            return
        images = payload.get("images") or []
        if not images:
            self._send(400, {"ok": False, "error": "images[] required"})
            return
        labels = payload.get("candidate_labels") or []
        save_memory = bool(payload.get("save_memory", True))
        t0 = time.time()
        try:
            results = _recognize(self.server.recognizer, images, labels, save_memory)
            self._send(200, {"ok": True, "results": results,
                             "latency_s": round(time.time() - t0, 2)})
        except Exception as exc:
            print(f"[qwen-server] recognize error: {exc}", flush=True)
            self._send(502, {"ok": False, "error": str(exc),
                             "latency_s": round(time.time() - t0, 2)})

    def log_message(self, fmt, *args):
        return


def main() -> int:
    cfg = _cfg()
    print(f"[qwen-server] loading {cfg['model_id']} (aux={cfg['use_aux']} depth={cfg['enable_depth']}) ... "
          "first run downloads weights from HuggingFace", flush=True)
    try:
        recognizer = _build_recognizer(cfg)
    except Exception as exc:
        print(f"[qwen-server] FATAL: failed to load model: {exc}", flush=True)
        return 2
    print("[qwen-server] model ready", flush=True)
    httpd = ThreadingHTTPServer(("0.0.0.0", cfg["port"]), _Handler)
    httpd.cfg = cfg
    httpd.recognizer = recognizer
    print(f"[qwen-server] listening on :{cfg['port']}  POST /recognize  |  GET /health", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[qwen-server] bye", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
