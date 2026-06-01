# Ntu_jinao — Franka 状态机小脑与技能执行性能预测

基于 **Isaac Lab / Isaac Sim 4.5** 的 Franka 操作流水线：在仿真中执行 VLM 或人工编写的技能蓝图，采集每个小技能的执行性能标签，并训练 **技能执行性能预测器（Skill Performance Predictor）**。

本仓库**不包含** Isaac Lab 官方源码、仿真数据集或预训练权重。仿真部分需在已安装 Isaac Lab 的环境中通过 `isaaclab.sh` 运行；预测器训练为独立 PyTorch 项目，**不需要启动 Isaac Sim**。

> **迁移 / AI 接手入口**：如果要把项目上传到 GitHub 并转移到大显存机器，请先阅读 [`docs/AI_HANDOFF.md`](docs/AI_HANDOFF.md)。该文档记录了当前完成状态、小显存机器限制、GitHub 上传注意事项、真实 Qwen3-VL 接入步骤和后续任务优先级。
>
> **本机（/home1/banghai/IsaacLab）部署说明**：见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)（Isaac Sim 5.1 / Isaac Lab 2.3.2 / `env_isaaclab`）。
>
> **大脑模块替换接口（重点）**：见 [`docs/VLM_BRAIN_INTERFACE.md`](docs/VLM_BRAIN_INTERFACE.md)（详细说明 VLM 输入/输出、每个技能参数语义、蓝图模板与对接约束）。

---

## 项目目标

整条链路解决一个问题：

> **在当前场景下，若执行某个小技能并使用某组参数，预期会产生怎样的执行性能？**

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage I / II  仿真执行（Isaac Lab）                                      │
│  JSON 技能计划 / 技能蓝图 → 状态机小脑 → Franka IK-Rel 控制 → 数据采集    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ predictor_dataset.jsonl
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Skill Performance Predictor  离线训练（PyTorch）                         │
│  scene + skill + params → 预测 success / timeout / failure / 连续性能指标 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 功能概览

| 模块 | 阶段 | 说明 |
|------|------|------|
| **状态机小脑** | Stage I | JSON `skill_plan` → 固定 pick/place 流程 → episode / trajectory 日志 |
| **蓝图执行器** | Stage II | JSON `skill_blueprint` → 状态图（skill / condition / parallel）→ primitive skills |
| **Parallel 控制** | Stage II+ | 同一 IK 控制循环内并行收敛 `position_goal` 与 `orientation_goal` |
| **性能采集** | Stage II | 每个 skill / parallel 节点一条 `predictor_dataset.jsonl` 样本 |
| **性能预测器** | 离线 | 多任务 MLP：分类（success / timeout / failure_reason）+ v0 13 维 / v1 29 维回归 |
| **VLM 反馈桥接** | Stage IV/V | node-level performance feedback → VLM 修正 `skill_blueprint` 参数 |

---

## 支持的小技能（Primitive Skills）

`move_above` · `reach` · `descend` · `grasp` · `lift` · `place` · `retreat` · `wait` · `align_orientation`

## 支持的蓝图逻辑

| 类型 | 说明 |
|------|------|
| `sequence` | 顺序执行 skill 节点 |
| `condition` | 条件分支（如 `object_in_gripper`） |
| `parallel` | 位置与姿态目标在同一控制步内联合求解（`parallel_mode`: `all_success`） |

---

## 环境要求

### 仿真执行（Stage I / II）

- **Isaac Sim 4.5**
- **Isaac Lab v2.0.x / v2.1.x**
- Conda 环境：`isaaclab45`（或官方等价环境）
- 推荐任务：`Isaac-Lift-Cube-Franka-IK-Rel-v0`

### 预测器训练（Stage III）

- Python 3.10+
- **PyTorch**（可使用与 `isaaclab45` 相同的 Conda 环境）
- 可选：`sklearn`（用于评估时计算 `success_auc`）

---

## 目录结构

```text
ntu_jinao_repo/
├── README.md
├── docs/
│   └── AI_HANDOFF.md                                  # 大显存机器迁移与 AI 接手说明
├── source/standalone/franka_state_machine_cerebellum/   # 仿真执行模块
│   ├── configs/
│   │   ├── example_skill_plan.json                      # Stage I 示例
│   │   ├── example_skill_blueprint.json                 # Stage II baseline
│   │   └── example_skill_blueprint_parallel.json        # Stage II + parallel
│   ├── run_state_machine_cerebellum.py                  # Stage I 入口
│   ├── run_skill_blueprint_executor.py                  # Stage II 入口
│   ├── state_machine_cerebellum.py                      # 状态机 / IK-Rel 控制
│   ├── skill_graph_executor.py                          # 蓝图状态图解释器
│   ├── primitive_skills.py                              # 小技能执行
│   ├── performance_collector.py                       # predictor 样本采集
│   └── …
├── skill_performance_predictor/                       # 离线预测器训练
    ├── README.md
    ├── configs/default_config.json
    ├── build_dataset.py                               # JSONL → train/val/test.pt
    ├── train_predictor.py
    ├── evaluate_predictor.py
    ├── infer_predictor.py
    ├── model.py                                       # 多任务 MLP
    ├── feature_extractor.py                           # 65 维特征提取
    ├── docs/
    │   ├── predictor_io_reference.md                  # 输入输出详细说明
    │   ├── numeric_features_65.csv                    # 65 维特征表
    │   └── skill_performance_matrix.csv               # 各 skill 训练标签矩阵
    ├── data/.gitkeep                                  # 处理后数据集（本地）
│   └── outputs/.gitkeep                               # 模型 checkpoint（本地）
└── vlm_brain/                                         # VLM 蓝图修正骨架
    ├── predictor_bridge.py                            # predictor → VLM feedback
    ├── brain_intervention_logger.py                   # Stage V intervention JSONL
    └── prompts/blueprint_revision_prompt.md           # 蓝图修正提示词
```

---

## 快速开始：端到端流程

### 1. 克隆仓库并放入 Isaac Lab

```bash
git clone https://github.com/ysdjy/Ntu_jinao.git
# 将 franka_state_machine_cerebellum 链接或复制到 Isaac Lab 的 source/standalone/ 下
```

### 2. Stage II 仿真采集数据

在 **Isaac Lab 根目录**执行：

```bash
cd /path/to/IsaacLab
conda activate isaaclab45

# 推荐：parallel 蓝图（pick + place，位置/姿态并行收敛）
TERM=xterm ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 50 \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --blueprint_path source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint_parallel.json \
  --output_dir ./datasets/state_machine_stage2_parallel \
  --target_mode custom_tabletop \
  --seed 0 \
  --save_trajectory true
```

可视化调试时可去掉 `--headless`；批量采集建议加 `--headless`。

### 3. 构建预测器训练集

在 **本仓库根目录**（或任意有 PyTorch 的环境）：

```bash
cd /path/to/ntu_jinao_repo
conda activate isaaclab45

python skill_performance_predictor/build_dataset.py \
  --input_jsonl /path/to/IsaacLab/datasets/state_machine_stage2_parallel/predictor_dataset.jsonl \
  --output_dir skill_performance_predictor/data/predictor_v0 \
  --seed 0 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

### 4. 训练预测器

```bash
python skill_performance_predictor/train_predictor.py \
  --data_dir skill_performance_predictor/data/predictor_v0 \
  --output_dir skill_performance_predictor/outputs/predictor_v0 \
  --config skill_performance_predictor/configs/default_config.json \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --device auto
```

### 5. 评估与单样本推理

```bash
python skill_performance_predictor/evaluate_predictor.py \
  --data_dir skill_performance_predictor/data/predictor_v0 \
  --checkpoint skill_performance_predictor/outputs/predictor_v0/best_model.pt \
  --split test

python skill_performance_predictor/infer_predictor.py \
  --checkpoint skill_performance_predictor/outputs/predictor_v0/best_model.pt \
  --sample_json /path/to/one_predictor_sample.json
```

---

## Stage I：Skill Plan 执行

适用于简单的 pick/place JSON 技能计划（非状态图蓝图）。

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_state_machine_cerebellum.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 5 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --output_dir ./datasets/state_machine_stage1 \
  --save_trajectory true \
  --target_mode custom_tabletop \
  --seed 0
```

输出：`episodes.jsonl`、`summary.json`、可选 `trajectories/`。**不生成** `predictor_dataset.jsonl`。

---

## Stage II：Skill Blueprint 执行

蓝图 JSON 定义有向图：`skill` / `condition` / `parallel` / `terminal` 节点。每个被执行的 skill 或 parallel 节点由 `PerformanceCollector` 写一条 predictor 训练样本。

### Baseline 蓝图（顺序 + 条件）

```bash
TERM=xterm ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 5 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --blueprint_path source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint.json \
  --output_dir ./datasets/state_machine_stage2_blueprint \
  --target_mode custom_tabletop \
  --seed 0 \
  --save_trajectory true
```

### Parallel 蓝图（位置 + 姿态联合 IK）

`example_skill_blueprint_parallel.json` 典型流程：

```text
p1 (parallel: move_above + align_orientation @ cube)
 → s2 descend → s3 grasp → c1 → s4 lift
 → p2 (parallel @ target) → s6 place → c2 → s7 retreat
```

Parallel 节点在每个控制步发送统一的 6D IK-Rel 指令（位置增量 + 姿态增量），避免“先到位再转姿态”导致的抓取偏差。

---

## 仿真数据输出

运行 Stage II 后，在 `--output_dir` 下生成：

| 文件 / 目录 | 说明 |
|-------------|------|
| `episodes.jsonl` | 每个 episode 一条：蓝图执行轨迹、条件分支、最终结果 |
| `predictor_dataset.jsonl` | **预测器训练源数据**：每个 skill/parallel 节点一条样本 |
| `summary.json` | 成功率、各 skill 统计、predictor 样本数 |
| `trajectories/` | 逐步状态与 action（`--save_trajectory true` 时） |

### predictor_dataset.jsonl 单条样本结构

```json
{
  "sample_id": "episode_000000_p1",
  "episode_id": "episode_000000",
  "skill": "parallel",
  "target": "cube+cube",
  "scene_state_before": { "ee_pose": [...], "cube_pose": [...], "target_pose": [...], "gripper_width": 0.08 },
  "skill_params": { "timeout_steps": 400, "goals": { ... } },
  "performance_query": ["success", "execution_steps", "trajectory_length", ...],
  "measured_performance": { "success": true, "execution_steps": 202, ... }
}
```

---

## Skill Performance Predictor 简介

独立 PyTorch 项目，详见 [`skill_performance_predictor/README.md`](skill_performance_predictor/README.md)。

### 模型输入

| 输入 | 维度 | 来源 |
|------|------|------|
| 数值特征 | **65** | 场景位姿、相对距离、skill_params（含 mask）、parallel 特征 |
| skill_id | Embedding 8 | move_above / descend / parallel / … |
| target_id | Embedding 6 | cube / target / current / … |

65 维特征逐维定义见 [`skill_performance_predictor/docs/numeric_features_65.csv`](skill_performance_predictor/docs/numeric_features_65.csv)。

### 模型输出（所有 skill 共享同一组输出头）

| 类型 | 输出 | 说明 |
|------|------|------|
| 分类 | success, timeout | 二分类 |
| 分类 | failure_reason | 10 类（none / timeout / object_not_in_gripper / …） |
| 回归 | predictor_v0: 13 维连续指标 | steps、time、trajectory_length、各类误差与 gripper 宽度等 |
| 回归 | predictor_v1: 29 维连续指标 | 新增 final_ee_position、target_position、final_object_position、final_ee_linear_speed、object_target_xy_error 等 |

> **注意**：模型对每条样本都输出完整的分类头和固定回归头，但 **每个 skill 实际参与训练的回归维度不同**，由 blueprint 中该节点的 `performance_query` 决定；缺失标签通过 `regression_mask` 从 loss 中排除。  
> 各 skill 训练标签对照表：[`skill_performance_predictor/docs/skill_performance_matrix.csv`](skill_performance_predictor/docs/skill_performance_matrix.csv)  
> 完整说明：[`skill_performance_predictor/docs/predictor_io_reference.md`](skill_performance_predictor/docs/predictor_io_reference.md)

`predictor_v1` 的输出可由 `vlm_brain/predictor_bridge.py` 整理为 `predictor_feedback.json`：每个节点包含成功概率、超时概率、错误类型、末端最终位置、目标位置、到达误差、到达速度、物体最终位置、物体到目标区域误差、风险等级和参数修正建议。VLM 仍然只输出 `skill_blueprint` JSON，不直接输出连续控制动作；高频控制仍由状态机小脑执行。

### 模型结构

```text
numeric[65] + skill_emb[8] + target_emb[6]
    → MLP backbone [128, 128, 64]
    → success_head / timeout_head / failure_reason_head / regression_head[13 or 29]
```

Loss = BCE(success) + BCE(timeout) + CE(failure_reason) + masked SmoothL1(regression)

### 数据切分

按 **`episode_id`** 切分 train / val / test（默认 70% / 15% / 15%），避免同一 episode 内多个 skill sample 泄漏到测试集。

---

## Stage IV：Qwen3-VL Brain Integration

`vlm_brain/` 提供 Qwen3-VL 大脑接入骨架：

```text
scene image + task + scene_state + schema
    → Qwen3-VL
    → skill_blueprint JSON
    → parser / validator
    → predictor_v1 node-level feedback
    → Qwen3-VL revised skill_blueprint JSON
```

关键点：

- Qwen3-VL 只生成或修正 `skill_blueprint` JSON。
- validator 会在执行前检查节点类型、skill、condition、parallel goals、performance_query 和边连接。
- `predictor_v1` 在执行前预测每个 skill / parallel 节点的 success、timeout、failure_reason、末端位置、目标位置、速度、物体目标误差和风险建议。
- predictor feedback 会返回给 VLM，用于修正 `height_offset`、`speed`、`timeout_steps`、`place_height`、`target_tolerance`、`orientation_tolerance` 等参数。
- LoRA / SFT 训练脚本目前是 Stage IV 骨架，支持 dry-run 和数据格式检查；正式训练需要满足依赖、显存和数据规模。

Dry-run 闭环：

```bash
python vlm_brain/vlm_predictor_loop.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_dir vlm_brain/outputs/loop_dry_run \
  --dry_run true \
  --mock_predictor true \
  --max_refine_iters 1
```

详见 [`vlm_brain/README.md`](vlm_brain/README.md)。

### 小数据集提示

样本数 < 50 时仍可跑通 pipeline，但会提示：

```text
Dataset is too small for reliable training; this run is for pipeline validation only.
```

建议采集 **500+ episodes**，并包含成功、超时、抓取失败、放置失败等多种情况，以训练可靠预测器。

---

## 可视化与调试

Stage II 执行时（不加 `--headless`）可看到：

- **较大坐标系**：episode 级 tabletop 放置目标
- **较小坐标系**：当前子技能的目标位姿（随 p1 → s2 → … 更新）

低显存或远程服务器建议：

```bash
--headless --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

---

## Git 与数据管理

以下内容**不会**提交到 GitHub（见 `.gitignore`）：

- `datasets/`、`*.jsonl`、仿真 trajectory
- `skill_performance_predictor/data/`（除 `.gitkeep`）
- `skill_performance_predictor/outputs/`（除 `.gitkeep`）
- `*.pt`、`*.pth` 模型权重

请在本机或对象存储中管理大规模采集数据与训练产物。

---

## 注意事项

1. 仿真模块以 **standalone 脚本** 形式调用 Isaac Lab，不修改官方核心代码。
2. 本仓库**不包含** VLM 训练、RL、BC 或 Robomimic 相关代码。
3. 蓝图 JSON 位于 `configs/`，可直接编辑；Stage II 支持 `--watch_blueprint` 在 episode 边界热加载。
4. 若仓库克隆在 Isaac Lab 之外，请将脚本路径改为你本地的 `franka_state_machine_cerebellum` 绝对路径，或在 Isaac Lab 中创建符号链接。

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [skill_performance_predictor/README.md](skill_performance_predictor/README.md) | 预测器训练命令与配置 |
| [skill_performance_predictor/docs/predictor_io_reference.md](skill_performance_predictor/docs/predictor_io_reference.md) | 65 维特征 + 各 skill 输入输出 |
| [skill_performance_predictor/docs/numeric_features_65.csv](skill_performance_predictor/docs/numeric_features_65.csv) | 特征维表（CSV） |
| [skill_performance_predictor/docs/regression_targets_v1.csv](skill_performance_predictor/docs/regression_targets_v1.csv) | predictor_v1 29 维回归目标 |
| [skill_performance_predictor/docs/skill_performance_matrix.csv](skill_performance_predictor/docs/skill_performance_matrix.csv) | skill × 性能标签矩阵 |

---

## License

Research / academic use. Isaac Lab and Isaac Sim are subject to their respective NVIDIA licenses.
