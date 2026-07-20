#!/usr/bin/env python3
"""感知后端=GPT-5.5:读 5 视角图,逐个枚举所有物体+关系,写完整输出文档。"""
import os, base64, json, sys, time
from pathlib import Path

VIEWS = ["front", "wrist", "left", "right", "top"]
IMG_DIR = Path("logs/perceive_scene")
OUT_MD = Path("logs/perceive_scene/gpt_perception.md")
OUT_JSON = Path("logs/perceive_scene/gpt_perception.json")

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "scene_summary": {"type": "string"},
        "objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "category": {"type": "string"},
                "appearance": {"type": "string"},
                "location": {"type": "string"},
                "seen_in_views": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "category", "appearance", "location", "seen_in_views"],
        }},
        "relations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"subject": {"type": "string"}, "predicate": {"type": "string"},
                           "object": {"type": "string"}, "description": {"type": "string"}},
            "required": ["subject", "predicate", "object", "description"],
        }},
    },
    "required": ["scene_summary", "objects", "relations"],
}

SYSTEM = ("你是机器人桌面场景理解模型。你会收到【同一个 Franka 桌面场景】的 5 个相机视角"
          "(front/wrist/left/right/top)。请【逐个枚举场景里每一个不同的物体】(同一物体在多个视角里出现算"
          "一个,去重),给出类别/外观/位置/在哪些视角可见,并列出物体之间的空间关系。忽略机器人手臂本身作为主体"
          "——重点是桌面上的物品(水果/盒子/罐/杯/刀/剪刀/方块…)、篮子、柜子、抽屉、咖啡机。所有文字用中文。只输出要求的 JSON。")

def main():
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_zhongzhuan")
    if not key:
        print("ERROR: 没有 API key(OPENAI_API_KEY / API_zhongzhuan)"); return 2
    from openai import OpenAI
    import httpx
    client = OpenAI(base_url=os.environ.get("VLM_BASE_URL", "https://165.154.193.90"),
                    api_key=key, http_client=httpx.Client(verify=False, timeout=httpx.Timeout(300.0)))
    content = [{"type": "input_text", "text": "以下是同一场景的 5 个视角。请枚举所有物体并给出关系。"}]
    used = []
    for v in VIEWS:
        p = IMG_DIR / f"{v}.png"
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        content.append({"type": "input_text", "text": f"[视角 {v}]"})
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": "high"})
        used.append(v)
    print(f"[gpt] 送 {len(used)} 视角给 gpt-5.5 ...", flush=True)
    t = time.time()
    resp = client.responses.create(
        model=os.environ.get("VLM_MODEL", "gpt-5.5"),
        instructions=SYSTEM,
        input=[{"role": "user", "content": content}],
        max_output_tokens=8000,
        reasoning={"effort": os.environ.get("VLM_REASONING", "high")},
        text={"format": {"type": "json_schema", "name": "scene", "strict": True, "schema": SCHEMA}},
    )
    txt = resp.output_text
    data = json.loads(txt)
    dt = time.time() - t
    print(f"[gpt] done {dt:.1f}s  objects={len(data['objects'])} relations={len(data['relations'])}", flush=True)

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # 写文档
    lines = ["# 感知模块完整输出(GPT-5.5,5 视角)", "",
             f"- 模型: gpt-5.5(中转 Responses API) | reasoning=high | 视角: {', '.join(used)} | 耗时 {dt:.1f}s",
             f"- 图像: logs/perceive_scene/{{front,wrist,left,right,top}}.png", "",
             "## 场景总述", data["scene_summary"], "",
             f"## 识别到的物体({len(data['objects'])} 个)", "",
             "| 名称 | 类别 | 外观 | 位置 | 可见视角 |", "|---|---|---|---|---|"]
    for o in data["objects"]:
        lines.append(f"| {o['name']} | {o['category']} | {o['appearance']} | {o['location']} | {', '.join(o['seen_in_views'])} |")
    lines += ["", f"## 物体间关系({len(data['relations'])} 条)", "",
              "| 主体 | 关系 | 客体 | 说明 |", "|---|---|---|---|"]
    for r in data["relations"]:
        lines.append(f"| {r['subject']} | {r['predicate']} | {r['object']} | {r['description']} |")
    lines += ["", "## 原始 JSON", "```json", json.dumps(data, ensure_ascii=False, indent=2), "```"]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[gpt] 文档 -> {OUT_MD}", flush=True)
    print("\n=== 物体清单 ===")
    for o in data["objects"]:
        print(f"  - {o['name']} [{o['category']}] @ {o['location']}  ({','.join(o['seen_in_views'])})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
