# cerebellum_rl

本目录实现 Stage 1.1 的本地强化学习任务，**不修改 Isaac Lab 原始 task 源码**。

## 任务名称

- `Isaac-ConstrainedReach-Position-Franka-v0`

## 当前阶段

- Stage 1.1 Position Reach

## 当前目标

- 训练 Franka 末端执行器在随机目标位置附近收敛（仅位置，不含姿态）。

## 当前不包含

- 姿态控制
- 终点速度控制
- 避障
- 接触控制
- 夹爪控制
- JSON 输出
- 大脑/VLM 逻辑

## 训练命令（headless）

```bash
cd /home1/banghai/IsaacLab
./isaaclab.sh -p ntu_jinao_repo/cerebellum_rl/train_position_reach.py \
  --task Isaac-ConstrainedReach-Position-Franka-v0 \
  --num_envs 4096 \
  --headless
```

## GUI 调试命令

```bash
cd /home1/banghai/IsaacLab
./isaaclab.sh -p ntu_jinao_repo/cerebellum_rl/train_position_reach.py \
  --task Isaac-ConstrainedReach-Position-Franka-v0 \
  --num_envs 16
```

## Observation（严格 126 维）

按以下顺序拼接，代码位置：`constrained_reach/observations.py`

- A 机器人状态 28 维：`q_norm(7) + dq_scaled(7) + previous_action(7) + joint_limit_margin(7)`
- B 末端当前状态 13 维：`ee_position(3) + ee_orientation(4, 0填充) + ee_linear_velocity(3, 0填充) + ee_angular_velocity(3, 0填充)`
- C 目标与误差 25 维：`target_position(3) + target_orientation(4, 0填充) + target_linear_velocity(3, 0填充) + target_angular_velocity(3, 0填充) + position_error(3) + orientation_error(3, 0填充) + linear_velocity_error(3, 0填充) + angular_velocity_error(3, 0填充)`
- D 容差 12 维：`position_tolerance_xyz(3=0.03) + orientation_tolerance_rpy(3,0) + linear_velocity_tolerance_xyz(3,0) + angular_velocity_tolerance_xyz(3,0)`
- E 障碍物约束 30 维：占位全 0
- F 时间信息 1 维：`episode_progress`
- G 力/力矩/接触预留 7 维：占位全 0
- H 额外预留 10 维：占位全 0

代码内含强校验：`assert obs.shape[-1] == 126`

## Action（严格 7 维）

- 使用 `RelativeJointPositionActionCfg`，`joint_names=["panda_joint.*"]`
- 动作范围由策略输出约束在 `[-1, 1]`
- 语义：`q_cmd = q_current + 0.05 * action`
- 仅控制 7 个机械臂关节，不含夹爪

## Reward（Stage 1.1）

设：

- `pos_error_vec = target_position - ee_position`
- `pos_error = ||pos_error_vec||`
- `r_pos = exp(-10.0 * pos_error)`
- `r_dist = -pos_error`
- `r_success = 5.0 if pos_error < 0.03 else 0.0`
- `r_action = -0.01 * mean(action^2)`
- `r_smooth = -0.01 * mean((action - previous_action)^2)`
- `r_dq = -0.001 * mean(dq^2)`
- `r_limit = -0.05 * mean(relu(0.08 - joint_limit_margin)^2)`

最终：

`reward = 2.0*r_pos + 1.0*r_dist + r_success + r_action + r_smooth + r_dq + r_limit`

实现位置：`constrained_reach/rewards.py`

## Termination

- success: `pos_error < 0.03`
- timeout: episode length 到期
- safety（可选）: 关节状态出现 NaN/Inf

实现位置：`constrained_reach/terminations.py`

## 目标采样与可视化

- 每个 episode reset 重采样目标位置：
  - `x in [0.35, 0.65]`
  - `y in [-0.30, 0.30]`
  - `z in [0.20, 0.55]`
- 与 EE 使用同一世界坐标系
- 启用 target marker 小球（GUI可见）

## 后续预留（Stage 1.2 / Stage 2）

- Stage 1.2：加入姿态误差与姿态奖励
- Stage 2：引入速度约束、障碍物约束、接触和更复杂任务逻辑

