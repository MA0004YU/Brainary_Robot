# Brainary_Robot Memory Module — 快速开始

## 解压后你会看到

```
brainary_memory_pkg/
├── README_FIRST.md          ← 本文件
├── memory_module/           ← 主体：感知→记忆→规划 桥接代码
│   ├── INTEGRATION_GUIDE.md ← 详细集成指南（Isaac Sim / Franka）
│   ├── pipeline.py
│   ├── perception_adapter.py
│   ├── planning_interface.py
│   └── scene_state_builder.py
├── perception_vlm.py        ← 感知模块（Qwen VLM）
└── embodiedbench/
    └── memory_manip/        ← 三层记忆系统核心
```

## 三步上手

**第一步：设置 Python 路径**

把解压目录加入 `PYTHONPATH`（只需要这一个目录）：

```bash
# Linux / Mac
export PYTHONPATH="/path/to/brainary_memory_pkg:$PYTHONPATH"

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\brainary_memory_pkg;" + $env:PYTHONPATH
```

**第二步：安装依赖**

```bash
pip install torch transformers accelerate pillow numpy qwen-vl-utils
```

**第三步：运行最小示例**

```python
from memory_module import PerceptionMemoryPipeline

pipeline = PerceptionMemoryPipeline.create(store_dir="memory_store/")
pipeline.session_start()
pipeline.begin_episode("ep001", "pick up the cube and place it on the target")

results = pipeline.process_perception(["your_image.png"])
ctx = pipeline.get_planning_context()
print(ctx.to_prompt_text())

pipeline.end_episode(success=True)
pipeline.session_end()
```

## 详细集成指南

见 `memory_module/INTEGRATION_GUIDE.md`，包含：
- Isaac Sim 文件协议与完整代码示例
- 真实 Franka 机械臂 ROS 集成示例
- 所有 API 速查表
