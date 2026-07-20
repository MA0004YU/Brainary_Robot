# scene_describer — 两路 RGB + 场景状态 → GPT → 结构化场景描述

把当前帧的 **front + wrist 两路 RGB** 和 **仿真数值状态**(物体位姿 / 把手 / 家电关节 / 夹爪)喂给
GPT(走你中转的 **Responses API**,`gpt-5.5`),拿回:

```json
{
  "scene_summary": "...",
  "objects":   [{ "name", "category", "appearance", "location", "state" }],
  "relations": [{ "subject", "predicate", "object", "description" }]
}
```

## 架构(为什么分两块)

```
Isaac 进程(isaaclab conda)                         独立 venv(.venv_vlm)
┌───────────────────────────────┐  HTTP POST   ┌──────────────────────────────┐
│ test_mode_ui --describe        │  /describe   │ vlm_describe_server.py        │
│  └ DescribePanel(按钮)         │ ───────────► │  └ openai SDK -> Responses API │
│     └ describer_client         │ ◄─────────── │     (gpt-5.5, JSON Schema)     │
│        (stdlib urllib + PIL)   │   JSON       │  持有 OPENAI_API_KEY           │
└───────────────────────────────┘              └──────────────────────────────┘
```

- **isaac 端零新依赖**(只 stdlib + 已装的 PIL)。`openai`/`httpx`/API key 全关在 `.venv_vlm`,
  不污染重型 isaac 环境。
- 相机/物理读取在**仿真主线程**抓帧;HTTP(数秒)在**后台线程**发,不卡仿真循环。
- 纯只读,不创建 SkillExecutor,可与正在跑的 `SkillTestController` / `BrainInterface` 共存。

## 一次性准备

```bash
cd projects/franka_v1_skill_lab/scene_describer
bash setup_venv.sh          # 建 .venv_vlm + 装 openai/httpx
```

## 跑(两个终端)

```bash
# 终端 1：VLM server(独立 venv)。在交互终端里跑,key 自动从 API_zhongzhuan 读(无需改名);
# base_url/model/reasoning/TLS 都有默认值(已对齐你的中转 config),通常直接起即可。
cd projects/franka_v1_skill_lab/scene_describer
.venv_vlm/bin/python vlm_describe_server.py
# -> [vlm-server] listening on :5599  model=gpt-5.5  base_url=https://165.154.193.90 ...

# 想覆盖默认时再显式给(等价于上面的内置默认):
# OPENAI_API_KEY=sk-... VLM_MODEL=gpt-5.5 VLM_REASONING=xhigh \
#   .venv_vlm/bin/python vlm_describe_server.py

# 健康检查（可选）
curl -s http://127.0.0.1:5599/health

# 终端 2：场景，加 --describe(其余参数照旧)
cd /home1/banghai/Documents/IsaacLab
./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
    --no_stream --pengzhuang \
    --controller state_machine.skill_test_controller:SkillTestController \
    --describe
```

场景里会多出一个 **"Scene Describe (GPT)"** 窗口:点 **Describe scene now** → 状态变
`calling GPT ...` → 几秒后窗口显示物品列表 + 关系,完整 JSON 同时打到 stdout。

## server 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` / `API_zhongzhuan` | (必填,二者其一) | 认证 key;按序自动查找。或用 `VLM_API_KEY_ENV` 指定变量名 |
| `VLM_BASE_URL` | `https://165.154.193.90` | Responses API 端点根;SDK POST `{base}/responses` |
| `VLM_MODEL` | `gpt-5.5` | 模型名 |
| `VLM_REASONING` | `medium` | reasoning effort(`low`/`medium`/`high`/`xhigh`);medium≈30s,xhigh≈50-120s+;置空关闭 |
| `VLM_TIMEOUT` | `300` | 请求超时秒(client urllib + server httpx 同值) |
| `VLM_LANG` | `zh` | 输出语言:`zh`=自然语言字段用中文(name 保留 sim 名、predicate 保留英文);`en`=全英文 |
| `VLM_INSECURE_TLS` | `1` | 裸 IP 自签证书→不校验 TLS;有正规证书设 `0` |
| `VLM_MAX_TOKENS` | `4000` | `max_output_tokens` |
| `VLM_PORT` | `5599` | 监听端口(与 `--describe_addr` 对齐) |

## 不接真 API 先验证管线

server 没起 / key 没填时,面板会显示 `ERROR: <连接/认证错误>`,管线本身(抓帧→打包→POST→
显示)照常验证。要纯离线 mock,可临时把 `vlm_describe_server.py` 的 `_describe()` 换成返回写死的
样例 dict。

## 文件

- `vlm_describe_server.py` — Responses API server(独立 venv)
- `describer_client.py` — isaac 端抓帧/打包/POST(stdlib + PIL)
- `describe_panel.py` — omni.ui 按钮面板
- `schema.py` — 输出 JSON Schema + 提示拼装
- `setup_venv.sh` / `requirements.txt`
