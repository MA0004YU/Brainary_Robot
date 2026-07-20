# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""VLM 场景描述 HTTP server(独立 venv,持有 openai SDK + API key)。

收 POST /describe  body = {"rgb_front_b64": "...", "rgb_wrist_b64": "...", "state": {...}}
调 OpenAI **Responses API**(中转,wire_api=responses)gpt-5.5 视觉 + JSON Schema 结构化输出,
返回 {"ok": true, "result": {scene_summary, objects[], relations[]}, "model": "...", "latency_s": ...}
出错返回 {"ok": false, "error": "..."}。

环境变量(对齐你的中转 config):
  OPENAI_API_KEY     必填(requires_openai_auth=true)
  VLM_BASE_URL       默认 https://165.154.193.90   (Responses API 端点根;SDK 会 POST {base}/responses)
  VLM_MODEL          默认 gpt-5.5
  VLM_REASONING      默认 xhigh   (置空字符串可关闭 reasoning 传参)
  VLM_INSECURE_TLS   默认 1       (裸 IP 自签证书 -> 不校验 TLS;有正规证书设 0)
  VLM_PORT           默认 5599
  VLM_MAX_TOKENS     默认 4000    (max_output_tokens)
  GET /health 返回当前配置(不含 key)。

启动:
  cd projects/franka_v1_skill_lab/scene_describer
  bash setup_venv.sh            # 一次性建 .venv_vlm + 装 openai
  OPENAI_API_KEY=sk-... .venv_vlm/bin/python vlm_describe_server.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from schema import SCENE_SCHEMA, build_input_text, system_prompt


def _cfg() -> dict:
    return {
        "base_url": os.environ.get("VLM_BASE_URL", "https://165.154.193.90"),
        "model": os.environ.get("VLM_MODEL", "gpt-5.5"),
        "reasoning": os.environ.get("VLM_REASONING", "medium").strip(),
        "insecure_tls": os.environ.get("VLM_INSECURE_TLS", "1") not in ("0", "false", "False", ""),
        "max_output_tokens": int(os.environ.get("VLM_MAX_TOKENS", "4000")),
        "timeout": float(os.environ.get("VLM_TIMEOUT", "300")),
        "lang": os.environ.get("VLM_LANG", "zh").strip().lower(),
        "port": int(os.environ.get("VLM_PORT", "5599")),
    }


def _make_client(cfg: dict):
    """构造 openai 客户端(裸 IP 自签证书时关 TLS 校验)。延迟 import,让无 SDK 时报清晰错误。"""
    from openai import OpenAI

    # key 来源:VLM_API_KEY_ENV 指定的变量名优先,否则按序找 OPENAI_API_KEY / API_zhongzhuan
    # (用户中转 key 写在 ~/.bashrc 的 API_zhongzhuan;交互终端起 server 时该变量已 export)。
    candidates = [os.environ["VLM_API_KEY_ENV"]] if os.environ.get("VLM_API_KEY_ENV") else \
        ["OPENAI_API_KEY", "API_zhongzhuan"]
    api_key = next((os.environ[k] for k in candidates if os.environ.get(k)), None)
    if not api_key:
        raise RuntimeError(
            f"no API key in env (looked for {candidates}). "
            "export OPENAI_API_KEY or API_zhongzhuan, or set VLM_API_KEY_ENV to the var name."
        )
    timeout = float(cfg["timeout"])
    if cfg["insecure_tls"]:
        import httpx

        http_client = httpx.Client(verify=False, timeout=httpx.Timeout(timeout))
        return OpenAI(base_url=cfg["base_url"], api_key=api_key, http_client=http_client)
    return OpenAI(base_url=cfg["base_url"], api_key=api_key, timeout=timeout)


def _describe(client, cfg: dict, front_b64: str, wrist_b64: str, state: dict) -> dict:
    """调一次 Responses API:两图(input_image) + 文本状态,JSON Schema 强约束输出。"""
    lang = cfg.get("lang", "zh")
    content = [
        {"type": "input_text", "text": build_input_text(state, lang)},
        {"type": "input_image", "image_url": f"data:image/png;base64,{front_b64}", "detail": "auto"},
        {"type": "input_image", "image_url": f"data:image/png;base64,{wrist_b64}", "detail": "auto"},
    ]
    kwargs = dict(
        model=cfg["model"],
        instructions=system_prompt(lang),
        input=[{"role": "user", "content": content}],
        max_output_tokens=cfg["max_output_tokens"],
        text={
            "format": {
                "type": "json_schema",
                "name": "scene_description",
                "strict": True,
                "schema": SCENE_SCHEMA,
            }
        },
    )
    if cfg["reasoning"]:
        kwargs["reasoning"] = {"effort": cfg["reasoning"]}
    resp = client.responses.create(**kwargs)
    text = _output_text(resp)
    return json.loads(text)


def _output_text(resp) -> str:
    """从 Responses 返回里取出文本(优先 SDK 便捷属性 output_text)。"""
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    # 兜底:遍历 output -> content -> text
    chunks = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
    if not chunks:
        raise RuntimeError(f"no text in response: {resp!r}")
    return "".join(chunks)


class _Handler(BaseHTTPRequestHandler):
    # 注入(server 实例上挂 client/cfg)
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/health":
            c = self.server.cfg
            self._send(200, {"ok": True, "model": c["model"], "base_url": c["base_url"],
                             "reasoning": c["reasoning"], "insecure_tls": c["insecure_tls"]})
        else:
            self._send(404, {"ok": False, "error": "use POST /describe or GET /health"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/describe":
            self._send(404, {"ok": False, "error": "unknown path; use POST /describe"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as exc:
            self._send(400, {"ok": False, "error": f"bad request body: {exc}"})
            return
        front = payload.get("rgb_front_b64")
        wrist = payload.get("rgb_wrist_b64")
        state = payload.get("state", {})
        if not front or not wrist:
            self._send(400, {"ok": False, "error": "rgb_front_b64 and rgb_wrist_b64 are required"})
            return
        t0 = time.time()
        try:
            result = _describe(self.server.client, self.server.cfg, front, wrist, state)
            self._send(200, {"ok": True, "result": result, "model": self.server.cfg["model"],
                             "latency_s": round(time.time() - t0, 2)})
        except Exception as exc:
            print(f"[vlm-server] describe error: {exc}", flush=True)
            self._send(502, {"ok": False, "error": str(exc), "latency_s": round(time.time() - t0, 2)})

    def log_message(self, fmt, *args):  # 静默默认每请求日志
        return


def main() -> int:
    cfg = _cfg()
    try:
        client = _make_client(cfg)
    except Exception as exc:
        print(f"[vlm-server] FATAL: {exc}", flush=True)
        return 2
    httpd = ThreadingHTTPServer(("0.0.0.0", cfg["port"]), _Handler)
    httpd.cfg = cfg
    httpd.client = client
    print(f"[vlm-server] listening on :{cfg['port']}  model={cfg['model']}  base_url={cfg['base_url']}  "
          f"reasoning={cfg['reasoning'] or '(off)'}  insecure_tls={cfg['insecure_tls']}", flush=True)
    print("[vlm-server] POST /describe  |  GET /health", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[vlm-server] bye", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
