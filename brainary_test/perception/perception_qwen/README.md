# perception_qwen — 本地 Qwen2.5-VL-3B 识别器接入仿真(第二个感知后端)

把外部开发的 `perception_vlm.py`(基于 **Qwen2.5-VL-3B-Instruct** 的本地识别器:辅助视图 + 复核 +
外部记忆 harness)包成 HTTP server,接进 test_mode 的感知面板,与 GPT-5.5(scene_describer)在 UI 里**切换**。

## 和 scene_describer(GPT)的关系

| | perception_qwen | scene_describer |
|---|---|---|
| 模型 | 本地 Qwen2.5-VL-3B(占 GPU) | GPT-5.5 中转 API(不占 GPU) |
| 任务 | 单图物体识别(primary_label + 属性) | 双图场景理解 + 物体关系 |
| 端口 | `:5601` `/recognize` | `:5599` `/describe` |

UI 面板 **"Scene Perception (GPT / Qwen)"** 下拉切后端;单卡共用(选 Qwen 时本地模型占显存,选 GPT 不占)。

## 架构

```
Isaac 进程(isaaclab conda)                       .venv_qwen(--system-site-packages,继承 torch)
┌──────────────────────────────┐  POST /recognize ┌──────────────────────────────────┐
│ test_mode_ui --describe       │ ───────────────► │ qwen_perception_server.py          │
│  └ DescribePanel(后端=Qwen)   │ ◄─────────────── │  └ perception_vlm.QwenVLMPerception │
│     └ qwen_client(stdlib+PIL) │   JSON           │     常驻加载 3B(首次下载 ~7GB)    │
└──────────────────────────────┘                  └──────────────────────────────────┘
```

- **server 端**:独立 venv,启动时 load 一次模型常驻;收两张 b64 图 -> 写临时 PNG -> `recognize_batch`。
- **isaac 端**:零新依赖(复用 scene_describer 的相机抓帧),抓 front/wrist 两图发过去。

## 一次性准备

**不用建 venv** —— 现成的 `qwen3vl` conda 环境依赖已齐(torch 2.11+cu128 / transformers 5.9 /
qwen-vl-utils 0.0.14)。直接用它跑即可。Qwen2.5-VL-3B 权重(~7GB)已下载缓存到 `~/.cache/huggingface`,
以后秒级加载、不再下。

> (`setup_venv.sh` 仍保留:若以后换机器、想要隔离 venv,`PYTHON=/path/to/torch-python bash setup_venv.sh`。)

## 跑(两/三个终端)

```bash
# 终端 1：Qwen server(用 qwen3vl 环境;权重已缓存,秒级 ready)
cd projects/franka_v1_skill_lab/perception_qwen
/home1/banghai/miniconda3/envs/qwen3vl/bin/python qwen_perception_server.py
# -> [qwen-server] model ready / listening on :5601
curl -s http://127.0.0.1:5601/health      # {"ok":true,"ready":true,...}

# 终端 2(可选)：GPT server,想在 UI 里切到 GPT 才需要
cd ../scene_describer && .venv_vlm/bin/python vlm_describe_server.py

# 终端 3：场景 + 感知面板
cd /home1/banghai/Documents/IsaacLab
./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
    --no_stream --pengzhuang \
    --controller state_machine.skill_test_controller:SkillTestController \
    --describe
```

面板里 **Backend** 选 `Qwen2.5-VL (local)` → **Run perception now** → 终端打印每路视图的
`primary_label / confidence / objects`(完整含中文在 stdout;面板因 omni.ui 不渲染 CJK 只显示 ASCII 骨架)。

## server 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | 可指向本地已下好的路径,免运行时下载 |
| `QWEN_AUX_VIEWS` | `1` | 轮廓/纹理辅助视图;关掉(`0`)省显存/提速 |
| `QWEN_DEPTH` | `0` | 深度辅助视图(再下 Depth-Anything-V2-Small) |
| `QWEN_MAX_NEW_TOKENS` | `512` | 生成上限 |
| `QWEN_MEMORY` | `perception_vlm_memory.json` | 外部长期记忆文件 |
| `QWEN_PORT` | `5601` | 监听端口(与 `--qwen_addr` 对齐) |

## 显存吃紧时(单卡和仿真共用)

- 先关辅助视图:`QWEN_AUX_VIEWS=0 .venv_qwen/bin/python qwen_perception_server.py`
- 别开深度(默认就关)。3B bf16 约 7-8GB;加仿真注意总量。

## 评测(对照仿真 GT,后续)

`eval_vs_gt.py`(待加):抓帧 + 读 `SceneState` 真值物体清单 -> 调 `/recognize` -> 算识别 precision/recall;
可做 候选标签 on/off、辅助视图 on/off 的消融。先把识别跑通再加。
