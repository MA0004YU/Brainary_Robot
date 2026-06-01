# AI Handoff：Franka 状态机小脑、执行性能预测器与 Qwen3-VL 大脑

本文档用于把当前项目迁移到一台显存更大的机器，并让新的 AI/开发者快速接手。

当前小显存电脑已经完成 Stage I-IV 的代码骨架和 dry-run 验证，但 **不适合继续下载/加载 Qwen3-VL-8B-Instruct**。下一台机器的重点是跑通真实 Qwen3-VL 推理、采集更多 predictor 数据、训练更可靠的 predictor_v1，并进入 VLM 修正闭环实验。

---

## 1. 项目一句话说明

本项目把 Franka pick/place 操作拆成可执行的 `skill_blueprint` JSON：

```text
Qwen3-VL / 人工蓝图
    → skill_blueprint JSON
    → 状态机小脑 / Isaac Lab 执行
    → predictor_dataset.jsonl
    → skill_performance_predictor_v1
    → node-level performance feedback
    → Qwen3-VL 修正 skill_blueprint 参数
```

核心原则：

- VLM 只生成和修正 `skill_blueprint` JSON。
- VLM 不输出连续控制动作、轨迹或关节命令。
- 高频控制由状态机小脑和 Isaac Lab IK-Rel 执行。
- predictor 只做执行前性能预测和风险反馈。

---

## 2. 当前仓库状态

仓库路径：

```bash
/home1/banghai/IsaacLab/ntu_jinao_repo
```

> **本机部署说明**：见 [`docs/DEPLOYMENT.md`](DEPLOYMENT.md)。原开发环境为 Isaac Sim 4.5 + `isaaclab45`；本机为 **Isaac Lab 2.3.2 + Isaac Sim 5.1 + `env_isaaclab`**。

当前已经完成：

- Stage I：简单 `skill_plan` 状态机执行。
- Stage II：`skill_blueprint` 图执行器，支持 `skill` / `condition` / `parallel` / `terminal`。
- Stage II+：parallel 位置目标和姿态目标联合 IK。
- Stage II 数据采集：`PerformanceCollector` 生成 `predictor_dataset.jsonl`。
- Stage III：`skill_performance_predictor`，包含 predictor_v1。
- Stage IV：`vlm_brain` dry-run 骨架，包含 parser、validator、predictor_bridge、VLM-predictor loop、SFT/LoRA 脚本骨架。

当前没有完成：

- 没有在本机下载 Qwen3-VL 权重。
- 没有真实 Qwen3-VL image + text 推理结果。
- 没有正式 LoRA 训练。
- predictor_v1 目前只做了小数据 smoke test，不代表预测质量。

---

## 3. 目录导航

```text
ntu_jinao_repo/
├── README.md
├── docs/
│   └── AI_HANDOFF.md
├── source/standalone/franka_state_machine_cerebellum/
│   ├── run_state_machine_cerebellum.py
│   ├── run_skill_blueprint_executor.py
│   ├── state_machine_cerebellum.py
│   ├── skill_graph_executor.py
│   ├── primitive_skills.py
│   ├── performance_collector.py
│   ├── skill_blueprint_loader.py
│   ├── skill_blueprint_schema.py
│   └── configs/
│       ├── example_skill_plan.json
│       ├── example_skill_blueprint.json
│       └── example_skill_blueprint_parallel.json
├── skill_performance_predictor/
│   ├── build_dataset.py
│   ├── train_predictor.py
│   ├── evaluate_predictor.py
│   ├── infer_predictor.py
│   ├── dataset_schema.py
│   ├── feature_extractor.py
│   ├── model.py
│   └── docs/
│       ├── predictor_io_reference.md
│       ├── numeric_features_65.csv
│       ├── regression_targets_v1.csv
│       └── skill_performance_matrix.csv
└── vlm_brain/
    ├── README.md
    ├── requirements_vlm.txt
    ├── configs/
    ├── prompts/
    ├── schemas/
    ├── examples/
    ├── qwen3vl_loader.py
    ├── run_vlm_inference.py
    ├── blueprint_parser.py
    ├── blueprint_validator.py
    ├── predictor_bridge.py
    ├── vlm_predictor_loop.py
    ├── build_vlm_sft_dataset.py
    ├── train_qwen3vl_lora.py
    └── infer_qwen3vl_lora.py
```

---

## 4. GitHub 上传注意事项

不要提交：

- `datasets/`
- `outputs/`
- `vlm_brain/data/*`
- `vlm_brain/outputs/*`
- `skill_performance_predictor/data/*`
- `skill_performance_predictor/outputs/*`
- `*.jsonl`
- `*.pt`, `*.pth`, `*.ckpt`
- `*.safetensors`, `*.bin`, `*.gguf`
- `*.png`, `*.jpg`, `*.jpeg`
- Isaac Lab 官方源码或本地安装目录

`.gitignore` 已经覆盖上述内容，并保留 `.gitkeep`。

上传前建议检查：

```bash
git status --short --untracked-files=all
git diff --stat
```

如果看到模型权重、图片、大型数据集或 `vlm_brain/outputs/` 下的生成文件出现在待提交列表，先不要提交。

---

## 5. 已验证的测试

### Stage II 新指标采集

已在 `isaaclab45` 环境中跑通过 1 episode：

```bash
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -p ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 1 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --blueprint_path ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint_parallel.json \
  --output_dir ./datasets/stage_next_perf_metrics_test \
  --target_mode custom_tabletop \
  --seed 0 \
  --save_trajectory true
```

`predictor_dataset.jsonl` 中已看到：

- `final_ee_position`
- `target_position`
- `final_ee_linear_speed`
- `average_ee_linear_speed`
- `final_object_position`
- `object_target_xy_error`
- `object_target_position_error`

### predictor_v1 smoke test

已验证：

- `build_dataset.py` 可以生成 29 维回归目标。
- `train_predictor.py` 3 epoch smoke test 可跑通。
- `evaluate_predictor.py` 显示 `predictor_schema_version: predictor_v1` 和 `regression_target_count: 29`。

### Stage IV dry-run

已验证：

- `blueprint_parser.py` 能从 ```json 代码块中提取 JSON。
- `blueprint_validator.py` 能校验 sample blueprint 和 Stage II parallel blueprint。
- `run_vlm_inference.py --dry_run true` 可生成 prompt。
- `predictor_bridge.py --mock true` 可生成 node-level feedback。
- `vlm_predictor_loop.py --dry_run true --mock_predictor true` 可完成 `initial_blueprint → predictor_feedback → revised_blueprint` 文件流。
- `train_qwen3vl_lora.py --dry_run true` 可在无 SFT 数据时给出 warning 而不失败。

---

## 6. 当前小显存机器环境记录

当前机器 GPU：

- NVIDIA GeForce RTX 3070 Ti Laptop GPU
- 显存约 8GB

base 环境：

- Python 3.13.12
- 缺 `torch`
- 缺 `transformers`
- 缺 `qwen_vl_utils`

`isaaclab45` 环境：

- Python 3.10.20
- `torch 2.5.1+cu124`
- CUDA 可用
- GPU 可识别
- `transformers 4.41.2`
- 缺 `qwen_vl_utils`
- 不支持 `Qwen3VLForConditionalGeneration`

结论：

- 不建议在 `isaaclab45` 里升级 Transformers，因为可能影响 Isaac Lab。
- 已尝试创建独立 `qwen3vl` 环境，但真实依赖安装尚未完成，也没有下载模型。
- 建议在大显存机器上重新创建独立 VLM 环境。

---

## 7. 大显存机器推荐环境

建议独立环境：

```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r vlm_brain/requirements_vlm.txt
pip install -U git+https://github.com/huggingface/transformers
```

验证：

```bash
which python
python --version
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import transformers; print(transformers.__version__)"
python -c "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3VL ok')"
python -c "import qwen_vl_utils; print('qwen-vl-utils ok')"
```

如果显存不足：

- 先把 `vlm_brain/configs/qwen3vl_8b_infer.json` 中 `load_in_4bit` 改为 `true`。
- 安装 `bitsandbytes`。
- 降低 `max_new_tokens`，例如 2048。
- 先做 text-only dry-run。
- 临时使用更小 VLM 只验证 pipeline。

---

## 8. 在大显存机器继续 Stage IV-B

### 8.1 下载模型

```bash
python vlm_brain/download_model.py \
  --model_name Qwen/Qwen3-VL-8B-Instruct
```

要求：

- 权重进入 Hugging Face cache。
- 不要把模型下载到仓库。
- 不要提交任何权重文件。

### 8.2 找图像

优先使用真实 camera image：

```bash
find /path/to/IsaacLab/datasets -type f \( -name "*.png" -o -name "*.jpg" \) | head
```

如果没有真实机器人场景图，先用任意本地图像验证 pipeline，但要在记录中注明“不是真实机器人场景”。

### 8.3 真实 Qwen3-VL 推理

```bash
python vlm_brain/run_vlm_inference.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --image /path/to/scene.png \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_json vlm_brain/outputs/generated_blueprint.json \
  --validate true
```

期望输出：

- `vlm_brain/outputs/raw_response.txt`
- `vlm_brain/outputs/parsed_blueprint.json`
- `vlm_brain/outputs/validation_report.json`
- `vlm_brain/outputs/generated_blueprint.json`

如果 JSON 不合法：

1. 保存 `raw_response.txt`。
2. 查看 `validation_report.json`。
3. 用 `blueprint_repair.py` 尝试修复。
4. 加强 prompt，要求只输出 JSON，使用 sample blueprint 作为模板。
5. 至少重试一次真实推理。

### 8.4 predictor feedback

mock 模式：

```bash
python vlm_brain/predictor_bridge.py \
  --blueprint vlm_brain/outputs/generated_blueprint.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --output_feedback vlm_brain/outputs/generated_predictor_feedback.json \
  --mock true
```

真实 predictor checkpoint 模式：

```bash
python vlm_brain/predictor_bridge.py \
  --blueprint vlm_brain/outputs/generated_blueprint.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --predictor_checkpoint skill_performance_predictor/outputs/predictor_v1/best_model.pt \
  --predictor_data_dir skill_performance_predictor/data/predictor_v1 \
  --output_feedback vlm_brain/outputs/generated_predictor_feedback_real.json
```

当前仓库通常没有真正可靠的 `predictor_v1/best_model.pt`，需要先采集更多数据并训练。

### 8.5 VLM + predictor loop

```bash
python vlm_brain/vlm_predictor_loop.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --image /path/to/scene.png \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_dir vlm_brain/outputs/real_loop_test \
  --mock_predictor true \
  --max_refine_iters 1
```

如果模型加载太慢或显存不足，先只做：

- initial real VLM generation
- predictor mock feedback
- revision dry-run/mock

---

## 9. 后续优先任务

### P0：真实 Qwen3-VL 推理接入

目标：

- 真实 image + scene_state + task → `skill_blueprint` JSON。
- parser 成功。
- validator 通过。
- predictor_bridge mock 成功。

交付物：

- `raw_response.txt`
- `generated_blueprint.json`
- `validation_report.json`
- `generated_predictor_feedback.json`
- 一份简短实验记录：模型、GPU、显存、耗时、是否 4bit。

### P1：稳定 VLM 输出 JSON

如果 Qwen3-VL 经常输出非法 JSON：

- 强化 `prompts/system_prompt.md`。
- 强化 `prompts/blueprint_generation_prompt.md`。
- 在 prompt 中加入完整 sample blueprint 模板。
- 扩展 `blueprint_repair.py` 的修复规则。
- 记录常见 validation errors。

### P2：采集 predictor_v1 训练数据

需要在 Isaac Lab 机器上批量运行 Stage II：

```bash
TERM=xterm ./isaaclab.sh -p ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 500 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --blueprint_path ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint_parallel.json \
  --output_dir ./datasets/predictor_v1_500eps \
  --target_mode custom_tabletop \
  --seed 0 \
  --save_trajectory true
```

需要覆盖：

- 成功样本
- timeout
- `object_not_in_gripper`
- `object_not_near_target`
- orientation 不收敛
- place 偏差较大

### P3：训练 predictor_v1

```bash
python skill_performance_predictor/build_dataset.py \
  --input_jsonl /path/to/predictor_dataset.jsonl \
  --output_dir skill_performance_predictor/data/predictor_v1 \
  --seed 0

python skill_performance_predictor/train_predictor.py \
  --data_dir skill_performance_predictor/data/predictor_v1 \
  --output_dir skill_performance_predictor/outputs/predictor_v1 \
  --config skill_performance_predictor/configs/default_config.json \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --device auto
```

不要提交 `data/` 或 `outputs/` 内容。

### P4：闭环实验

目标：

```text
Qwen3-VL initial blueprint
    → validator
    → predictor_v1 feedback
    → Qwen3-VL revised blueprint
    → Stage II execution
    → compare initial vs revised
```

需要记录：

- VLM 初始蓝图是否可执行
- predictor 风险节点
- VLM 修正了哪些参数
- 修正后 Stage II 是否更稳定
- 是否减少 timeout / grasp fail / place fail

### P5：SFT / LoRA 数据

数据类型：

- `blueprint_generation`
- `blueprint_revision`

脚本：

```bash
python vlm_brain/build_vlm_sft_dataset.py \
  --episodes_jsonl /path/to/episodes.jsonl \
  --vlm_inputs_jsonl /path/to/vlm_inputs.jsonl \
  --predictor_feedback_dir /path/to/feedbacks \
  --output_train vlm_brain/data/sft_train.jsonl \
  --output_val vlm_brain/data/sft_val.jsonl

python vlm_brain/train_qwen3vl_lora.py \
  --config vlm_brain/configs/qwen3vl_lora_train.json \
  --train_jsonl vlm_brain/data/sft_train.jsonl \
  --val_jsonl vlm_brain/data/sft_val.jsonl \
  --dry_run true
```

正式 LoRA 训练是后续步骤，当前只需要保证数据格式和 dry-run 可用。

---

## 10. 新 AI 接手建议

建议新 AI 开始时按顺序做：

1. 读 `README.md`。
2. 读本文件 `docs/AI_HANDOFF.md`。
3. 读 `vlm_brain/README.md`。
4. 运行 Stage IV dry-run，确认迁移后文件路径正常。
5. 检查 GPU 和依赖。
6. 下载 Qwen3-VL 到 cache。
7. 做一次真实 `run_vlm_inference.py`。
8. 如果 validation 不通过，先改 prompt/repair，不要动 Stage II 控制代码。
9. predictor checkpoint 不可靠时，先用 `--mock true` 做 VLM loop。
10. 采集足够数据后再训练 predictor_v1。

最重要的边界：

- 不要让 VLM 输出连续动作。
- 不要改 Isaac Lab 官方核心代码。
- 不要把模型权重和数据提交到 GitHub。
- 不要为了兼容 VLM 输出而放宽执行器的安全校验。

