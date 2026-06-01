# 实验数据统一管理（`data/`）

每次实验使用同一个 **`run_id`**（精确到分钟，例如 `2026-05-26_1549`），在四个目录下各有一个同名子文件夹，并用 `manifests/<run_id>.json` 串联整条链路。

## 目录结构

```text
data/
├── README.md
├── run_layout.py
├── manifests/                    # 每次运行的总清单（路径 + 产物索引）
│   └── 2026-05-26_1549.json
├── 01_vlm_inputs/                # VLM 推理输入
│   └── 2026-05-26_1549/
├── 02_vlm_outputs/               # VLM 推理输出
│   └── 2026-05-26_1549/
├── 03_predictor_outputs/         # 性能预测器输出
│   └── 2026-05-26_1549/
└── 04_execution_data/            # 小脑 / 仿真执行采集
    └── 2026-05-26_1549/
```

## 各目录应存放的文件

### 01_vlm_inputs（VLM 输入）

| 文件 | 说明 |
|------|------|
| `scene_state.json` | 场景数值状态（位姿、夹爪等） |
| `task.txt` | 任务自然语言指令 |
| `image.jpg` / `image.png` | 场景图像（从 `--image` 复制） |
| `vlm_config.json` | 本次使用的 VLM 配置快照 |
| `input_meta.json` | 可选：dry_run、命令行备注等 |

### 02_vlm_outputs（VLM 输出）

| 文件 | 说明 |
|------|------|
| `raw_response.txt` | 模型原始文本 |
| `parsed_blueprint.json` | 解析出的 JSON |
| `validation_report.json` | 校验报告 |
| `generated_blueprint.json` | 通过校验的最终蓝图 |
| `dry_run_prompt.txt` | 仅 dry-run 时 |
| `initial_blueprint.json` | 闭环：首轮蓝图 |
| `initial_raw_response.txt` | 闭环：首轮原始输出 |
| `revised_blueprint_iter1.json` | 闭环：修正后蓝图 |
| `revision_raw_response_iter1.txt` | 闭环：修正轮原始输出 |
| `loop_summary.json` | 闭环：摘要 |

### 03_predictor_outputs（预测器输出）

| 文件 | 说明 |
|------|------|
| `predictor_feedback.json` | 节点级执行前反馈 |
| `predictor_meta.json` | mock / checkpoint 路径、时间等 |

### 04_execution_data（蓝图执行采集）

| 文件 | 说明 |
|------|------|
| `blueprint_used.json` | 本次仿真使用的蓝图副本 |
| `run_args.json` | 执行参数（task、num_episodes、seed 等） |
| `episodes.jsonl` | 每 episode 一条 |
| `predictor_dataset.jsonl` | 每 skill/parallel 节点一条（Stage II） |
| `summary.json` | 汇总统计 |
| `trajectories/` | 逐步轨迹（可选） |

## 原先容易散落、现已纳入的数据

- 任务文字（以前只在命令行）→ `task.txt`
- VLM 配置 → `vlm_config.json`
- 闭环修正蓝图 / loop 摘要 → `02_vlm_outputs/`
- 预测器 mock/真实模式说明 → `predictor_meta.json`
- 执行时实际用的蓝图 → `blueprint_used.json`
- 执行命令行参数 → `run_args.json`

## 仍可能缺少、需你按需补充的数据

| 数据 | 建议位置 | 说明 |
|------|----------|------|
| 仿真相机原图 | `01_vlm_inputs/` 或 `04_execution_data/camera/` | 需自己在仿真里保存 RGB |
| Stage I 专用日志 | `04_execution_data/` | 与 Stage II 共用目录，看 `run_args.json` 区分 |
| 训练好的 predictor 权重 | `skill_performance_predictor/outputs/` | 大文件，不放入 `data/` |
| LoRA / SFT 数据 | `vlm_brain/data/` | 训练管线，非单次推理 |

## 使用方法

### 1. 新建一次实验 run

```bash
cd /home1/banghai/IsaacLab/ntu_jinao_repo
python scripts/new_experiment_run.py --note "第一次统一目录测试"
```

会打印 `run_id` 和四个目录路径。

### 2. VLM 推理（自动归档输入 + 写入 02）

```bash
conda activate qwen3vl

python vlm_brain/run_vlm_inference.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --image /path/to/scene.png \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_json vlm_brain/outputs/placeholder.json \
  --validate true \
  --experiment_run_id 2026-05-26_1549
```

使用 `--experiment_run_new` 可自动创建新的 `run_id`（无需先调 `new_experiment_run.py`）。

### 3. 预测器反馈（写入 03）

```bash
python vlm_brain/predictor_bridge.py \
  --blueprint data/02_vlm_outputs/2026-05-26_1549/generated_blueprint.json \
  --scene_state data/01_vlm_inputs/2026-05-26_1549/scene_state.json \
  --mock true \
  --experiment_run_id 2026-05-26_1549
```

### 4. 仿真执行（写入 04）

```bash
cd /home1/banghai/IsaacLab
conda activate env_isaaclab

./isaaclab.sh -p ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --blueprint_path ntu_jinao_repo/data/02_vlm_outputs/2026-05-26_1549/generated_blueprint.json \
  --experiment_run_id 2026-05-26_1549 \
  --num_episodes 1 \
  --headless \
  --output_dir ntu_jinao_repo/data/04_execution_data/placeholder
```

`--output_dir` 在指定 `--experiment_run_id` 时会被自动覆盖为 `04_execution_data/<run_id>/`。

## Git

`data/` 下各 run 子目录内容默认不提交（见仓库 `.gitignore`），仅保留目录说明与 `run_layout.py`。
