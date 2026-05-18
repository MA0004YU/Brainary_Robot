# EmbodiedBench Intent Agent

针对 [EmbodiedBench](https://github.com/scuama/EmbodiedBench) 的**意图推理智能体 (Intent Reasoning Agent)**。区别于传统指令执行 Agent，本模块通过「五层 Why」挖掘深层意图，在目标缺失时主动挂起并与用户对话，并结合 **EmbodiedLTM**（GraphRAG 长期记忆服务）跨任务积累场景知识与用户偏好。

> 此目录 (`agent/`) 是在原始仓库 `origin/master` 基础上的增量实现，可放入已配置好 EmbodiedBench 环境的机器中独立运行。

---

## 核心特性

1. **五层 “Why” 意图解析** (`goal_reasoner.py`)：对指令进行反思，提取深层意图与可接受的物理替代物。
2. **解空间建模** (`solution_space.py`)：从观测与技能集构建合法动作组合，并通过 LTM 同步物品/地点状态。
3. **方案排序与决策** (`solution_ranker.py` + `task_validator.py`)：多方案打分排序，选取 Top-1 动作；支持 Action 9999 触发用户对话。
4. **长期记忆** (`memory_manager.py` + `ltm_client.py`)：通过 EmbodiedLTM HTTP API（默认 `http://127.0.0.1:8000`）写入情景记忆、状态更新与 checkpoint。
5. **运行日志** (`logger_utils.py`)：每步生成双视角截图、`agent_decision.log` 与 `run_summary.md` 战报。

---

## 架构与模块依赖

```text
interactive_test.py          # 入口：连接 EBHabEnv，驱动主循环
        │
        ▼
core_agent.py                # 调度：错误恢复 → 意图 → 解空间 → 排序决策
   ├── goal_reasoner.py      # 意图提取（LLM + prompt_templates）
   ├── solution_space.py     # 解空间更新（LLM + LTMClient）
   ├── task_validator.py     # 调用 SolutionRanker 选动作
   │     └── solution_ranker.py
   ├── memory_manager.py     # LTM checkpoint
   ├── llm_client.py         # OpenAI 兼容接口
   └── logger_utils.py       # 日志与 Markdown 战报
```

**外部依赖**：运行前需启动 `agent/EmbodiedLTM` 服务。

---

## 目录结构

```text
agent/
├── core_agent.py            # Agent 核心调度与状态机
├── goal_reasoner.py         # 五层 Why 意图推理
├── solution_space.py        # 解空间 / 世界模型 / LTM 同步
├── solution_ranker.py       # 多方案生成与排序
├── task_validator.py        # 决策执行器（封装 Ranker）
├── memory_manager.py        # LTM 记忆读写与 checkpoint
├── ltm_client.py            # EmbodiedLTM HTTP 客户端
├── llm_client.py            # LLM JSON 生成封装
├── logger_utils.py          # 日志、截图、run_summary.md
├── prompt_templates.py      # 各模块 System/User Prompt
├── interactive_test.py      # [入口] 交互式测试框架
├── quick_start.sh           # [入口] 一键启动测试
├── config_patch/
│   └── visual.yaml          # 需覆盖到 EB 源码的观测配置
├── EmbodiedLTM/             # GraphRAG 长期记忆服务
│   ├── app.py               # FastAPI 入口
│   ├── reset_ltm.sh         # 清空图谱并重启服务
│   └── MemoryKB/            # 图谱与记忆数据
└── logs/                    # 运行产物（.gitignore，可定期清理）
```

---

## 快速上手

### 步骤 0：启动 / 重置 EmbodiedLTM

```bash
cd agent/EmbodiedLTM
conda activate embodiedltm
uvicorn app:app --host 0.0.0.0 --port 8000
```

如需清空跨任务污染的记忆图谱：

```bash
cd agent/EmbodiedLTM
bash reset_ltm.sh
```

### 步骤 1：覆盖底层环境配置

```bash
cp agent/config_patch/visual.yaml embodiedbench/envs/eb_habitat/config/task/task_obs/visual.yaml
```

### 步骤 2：启动测试

```bash
cd agent
conda activate embench
xvfb-run -a python interactive_test.py --instruction "Get me a can of coke."
```

或使用快捷脚本（需已启动 LTM）：

```bash
cd agent
bash quick_start.sh
# 自定义指令: bash quick_start.sh "Bring me an apple"
```

运行日志写入 `agent/logs/run_<episode_id>_<timestamp>/`。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBODIED_LTM_URL` | `http://127.0.0.1:8000` | EmbodiedLTM 服务地址 |
| `OPENAI_API_KEY` | — | LLM 调用密钥（由 `llm_client.py` 使用） |
