# Skill Performance Predictor

该目录是一个独立的 PyTorch 训练项目，用于从第二阶段生成的
`predictor_dataset.jsonl` 中学习“小技能执行性能预测器”。它不启动 Isaac Sim，
也不依赖 Isaac Lab 仿真运行时。

## 目标

预测器学习如下映射：

```text
scene_state_before + skill + target + skill_params + performance_query
    -> measured_performance
```

当前保留两个版本语义：

- `predictor_v0`：分类 `success`、`timeout`、`failure_reason`，以及 13 维基础回归：
  `execution_steps`、`execution_time`、`trajectory_length`、
  `final_ee_position_error`、`final_ee_orientation_error`、`object_lift_delta`、
  `ee_object_distance`、`min_ee_object_distance`、`object_target_xy_distance`、
  `final_position_error`、`object_displacement`、`gripper_width_start`、
  `gripper_width_end`。
- `predictor_v1`：在 v0 基础上扩展到 29 维回归，新增
  `final_ee_position`、`target_position`、`final_ee_rpy`、
  `final_ee_linear_speed`、`average_ee_linear_speed`、`final_object_position`、
  `object_target_position_error`、`object_target_xy_error` 等可供 VLM 读取的连续执行性能。

回归标签允许缺失。缺失或 `null` 标签会通过 `regression_mask` 从 loss 中排除。

## 数据格式

输入文件为 Stage II 生成的 JSONL，每行一条 predictor sample，包含：

- `episode_id`
- `skill`
- `target`
- `scene_state_before`
- `skill_params`
- `performance_query`
- `measured_performance`

数据切分按 `episode_id` 完成，避免同一 episode 的多个 skill sample 同时进入
train/test 造成数据泄漏。

## 构建数据集

```bash
python skill_performance_predictor/build_dataset.py \
  --input_jsonl /path/to/predictor_dataset.jsonl \
  --output_dir skill_performance_predictor/data/predictor_v1 \
  --seed 0 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

输出包括 `train.pt`、`val.pt`、`test.pt`、`feature_stats.json`、
`label_stats.json`、`vocab.json` 和 `dataset_info.json`。

## 训练

```bash
python skill_performance_predictor/train_predictor.py \
  --data_dir skill_performance_predictor/data/predictor_v1 \
  --output_dir skill_performance_predictor/outputs/predictor_v1 \
  --config skill_performance_predictor/configs/default_config.json \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --device auto
```

训练输出：

- `best_model.pt`
- `last_model.pt`
- `train_log.jsonl`
- `eval_report.json`
- `config_used.json`
- `feature_stats.json`
- `vocab.json`

## 评估

```bash
python skill_performance_predictor/evaluate_predictor.py \
  --data_dir skill_performance_predictor/data/predictor_v1 \
  --checkpoint skill_performance_predictor/outputs/predictor_v1/best_model.pt \
  --split test
```

如果安装了 `sklearn`，评估会额外输出 `success_auc`。否则会跳过 AUC，
不会导致脚本失败。

## 单样本推理

```bash
python skill_performance_predictor/infer_predictor.py \
  --checkpoint skill_performance_predictor/outputs/predictor_v1/best_model.pt \
  --sample_json /path/to/one_predictor_sample.json
```

`infer_predictor.py` 会同时输出原始 `regression` 字典和结构化字段，例如
`final_ee_position`、`target_position`、`final_object_position`、
`final_ee_linear_speed`、`object_target_xy_error`，便于 `vlm_brain/predictor_bridge.py`
生成 node-level predictor feedback。

详细字段见：

- `docs/predictor_io_reference.md`
- `docs/numeric_features_65.csv`
- `docs/regression_targets_v1.csv`
- `docs/skill_performance_matrix.csv`

## 小数据集说明

如果样本少于 50 条，脚本仍允许构建数据集和训练，但会打印：

```text
Dataset is too small for reliable training; this run is for pipeline validation only.
```

这种运行只适合验证数据处理、模型 forward、loss 和评估流程是否打通。
