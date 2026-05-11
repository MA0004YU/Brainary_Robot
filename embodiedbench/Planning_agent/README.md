# EmbodiedBench Intent Agent

这是一个针对 [EmbodiedBench](https://github.com/scuama/EmbodiedBench) 开发的**意图推理智能体 (Intent Reasoning Agent)**。
区别于传统的指令执行 Agent，此 Agent 具备**跨任务的长期记忆**与**断点对话**能力，能根据用户的深层意图和环境进行自发探索及意图推断，并在找不到原定目标时，通过与用户对话提供替代方案。

> 此目录 (`agent/`) 是在原始仓库 `origin/master` 基础上的**完整增量实现**，您可以直接将此目录放入任何配置好 EmbodiedBench 环境的机器中独立运行。

---

## 核心增量特性

1. **五层 “Why” 意图解析** (`goal_reasoner.py`): 大模型对指令进行反思，挖掘真正的目标（如：指令是拿可乐，真实意图是解渴）。
2. **断点交互与状态机** (`core_agent.py`): 当目标缺失时，Agent 主动挂起当前探索进度并与用户对话，等待下一步指令。
3. **长期记忆与场景绑定** (`memory_manager.py`): 
   - 跨任务记忆用户偏好（如：用户饿了喜欢吃香蕉）。
   - 将静态场景（如 `FloorPlan22`）中探索过的地点与物品写入硬盘 (`persistent_memory.md`)，下次在该场景中自动跳过重复探索。
4. **日志继承** (`logger_utils.py`): 即便挂起脚本，下次唤醒时也能在同一个 log 文件夹下完美续写对话与决策。

---

## 核心文件结构与说明

```text
agent/
├── core_agent.py          # Agent 核心调度逻辑，管理状态机与主循环
├── memory_manager.py      # 记忆管理器，负责解析/读写双MD文档
├── persistent_memory.md   # 长期记忆库（用户偏好、场景知识）
├── session_context.md     # 临时会话库（当前探索进度、断点日志路径）
├── interactive_test.py    # [入口点] 交互式运行框架
├── quick_start.sh         # [入口点] 一键启动测试用例
└── config_patch/          # 环境补丁配置
    └── visual.yaml        # 需要覆盖进 EmbodiedBench 源码中的配置
```

---

## 快速上手指南

### 步骤 1：覆盖底层环境配置
为了使 Agent 能够获得正确的观测格式，请将本目录下的 `config_patch/visual.yaml` 覆盖到您的 EmbodiedBench 安装路径中：
```bash
cp agent/config_patch/visual.yaml embodiedbench/envs/eb_habitat/config/task/task_obs/visual.yaml
```

### 步骤 2：启动全新会话
执行全新任务，Agent 会清空临时会话，但**会继承并利用长期记忆中的场景知识**。
```bash
cd agent
conda activate embench
# 使用 Headless 模式运行仿真
xvfb-run -a python interactive_test.py --new_session --scene_id "FloorPlan22" --instruction "给我拿一杯可乐"
```
运行后，Agent 会在场景中搜寻，一旦找不到可乐并发现替代品（如易拉罐），它会保存进度并自动挂起。

### 步骤 3：多轮交互与继续执行
在 Agent 挂起并等待回复后，您可以用反馈继续上一次的进程。日志将自动追加到上一次的运行目录中。
```bash
xvfb-run -a python interactive_test.py --resume --scene_id "FloorPlan22" --feedback "我不渴，我想吃点水果"
```
此时，Agent 会结合您的长期偏好（比如喜欢香蕉），直接前往之前记录的存放位置拾取香蕉！
