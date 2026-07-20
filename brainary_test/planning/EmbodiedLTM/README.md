# MemVerse（Embodied LTM）

MemVerse 是一个面向 **Brainary_Robot** 的长期记忆系统，支持多模态输入（文本 / 图像 / 视频 / 音频）写入，并通过记忆检索增强问答能力。

当前服务基于 FastAPI，核心能力由 LightRAG + OpenAI 接口驱动。

## 功能特性

- 多模态记忆写入（`POST /insert`）
- 长期记忆检索问答（`POST /query`）
- 一键清空记忆库（`POST /reset`）
- 情景记忆（episodic）与语义记忆（semantic）并行检索
- **记忆对齐机制**：具备“物体恒存性”追踪能力，能够在由于环境刷新等原因导致物品 ID 发生变化时，依然通过比对物品的物理属性与动态状态完成历史实体对齐。

## 运行环境

- Python 3.10

## 快速开始

### 1) 安装依赖

```bash
conda create -n embodiedltm python=3.10 -y
conda activate embodiedltm
pip install -r requirements.txt
```

## 启动服务

```bash
export OPENAI_API_BASE="<your_openai_base>"
export OPENAI_API_KEY="<your_openai_api_key>"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，默认地址为：`http://127.0.0.1:8000`

## API 快速使用

### 1) 写入记忆：`POST /insert`

```bash
curl -X POST "http://127.0.0.1:8000/insert" \
  -F "query=There are vegetables and fruits on the dinner table." \
  -F "image=@/path/to/image.png"\
  -F "video=@/path/to/video.mp4" \
  -F "audio=@/path/to/audio.wav"
```

### 2) 查询记忆：`POST /query`

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -F "query=What's on the table?" 
```

### 3) 清空记忆：`POST /reset`

```bash
curl -X POST "http://127.0.0.1:8000/reset"
```

## 目录结构（核心）

```text
.
├── app.py                         # FastAPI 服务入口
├── memory_coordinator.py                # 记忆写入、检索与生成调度逻辑
├── requirements.txt               # Python 依赖
├── MemoryKB/
│   ├── build_memory.py            # 记忆构建流程
│   ├── User_Conversation/         # 多模态处理与会话数据
│   └── Long_Term_Memory/
│       ├── Graph_Construction/    # LightRAG 图构建与检索
│       ├── memory_chunks/         # 记忆切片结果
│       └── system/                # 系统级记忆规则文件
└── README.md
```

## 接入 `memory_manip`

下面是当前仓库里 `memory_manip` 已落地的接入方式（`agent_memory.py` + `longterm_memory.py`）。

### 1) 配置地址与超时

在 `memory_manip/config.py` 中配置：

- `embodiedltm_base_url`：EmbodiedLTM 服务地址（默认 `http://127.0.0.1:8000`）
- `embodiedltm_timeout_sec`：请求超时时间

### 2) 初始化时绑定 EmbodiedLTM

`EmbodiedManipulationMemorySystem` 初始化时会自动执行：

- 创建 `ManipulationLongTermMemory`
- 调用 `configure_embodiedltm(base_url, timeout_sec)`

### 3) Episode 结束自动写入长期记忆

在 `end_episode(...)` 中会把 episode summary 写入 EmbodiedLTM：

- 通过 `insert_longterm_memory(summary)` 调用 EmbodiedLTM `/insert`

### 4) 查询长期记忆

通过 `query_longterm_memory(query)` 调用 EmbodiedLTM `/query`。

### 5) 多模态写入

`longterm_memory.py` 的 `insert_longterm_memory(...)` 已支持可选文件参数：

- `image_path`
- `video_path`
- `audio_path`

### 最小示例（`memory_manip` 侧）

```python
from embodiedbench.memory_manip.config import MemorySystemConfig
from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem

cfg = MemorySystemConfig(
    embodiedltm_base_url="http://127.0.0.1:8000",
    embodiedltm_timeout_sec=90.0,
)

mem = EmbodiedManipulationMemorySystem(config=cfg)
mem.begin_task("Put the apple on the plate", "task_001")
mem.end_episode(success=True, extra={"note": "demo"})

res = mem.query_longterm_memory("Where is the apple?")
print(res)

# 直接多模态写入
mem.long_term.insert_longterm_memory(
    {"instruction": "multi-modal example", "success": True},
    image_path="/path/to/image.png",
    video_path="/path/to/video.mp4",
    audio_path="/path/to/audio.wav",
)
```


