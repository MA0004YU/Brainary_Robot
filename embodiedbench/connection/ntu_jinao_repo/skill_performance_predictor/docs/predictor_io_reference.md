# 技能执行性能预测器：输入输出参考表

本文档说明预测模型 **65 维数值特征** 的含义，以及 **每个子技能** 实际参与训练的性能标签。

> **说明**
> - 除 65 维数值特征外，模型还有 2 个类别输入：`skill_id`（Embedding 8 维）、`target_id`（Embedding 6 维），拼接后 backbone 输入总维度为 **79**。
> - `predictor_v0` 输出 3 个分类头 + 13 维基础回归头。
> - `predictor_v1` 输出 3 个分类头 + 29 维扩展回归头；但 **loss 只对标签非 null 的维度生效**（回归用 `regression_mask`）。
> - `performance_query` 第一版 **不作为模型输入**，仅保存在 metadata。
> - `train` = 该 skill 的 blueprint 会采集且通常有有效标签；`usually_missing` = 常 null 或不参与 regression loss。

---

## 一、模型总输入输出

| 类型 | 名称 | 维度 | 说明 |
|------|------|------|------|
| 输入 | numeric_features | 65 | 场景 + 相对关系 + skill_params + parallel 特征，train 集归一化 |
| 输入 | skill_id | 1 → Emb 8 | move_above / descend / parallel / … |
| 输入 | target_id | 1 → Emb 6 | cube / target / current / … |
| 输出 | success | 1 | 二分类 logit → sigmoid |
| 输出 | timeout | 1 | 二分类 logit → sigmoid |
| 输出 | failure_reason | 10 | 多分类 logits → softmax |
| 输出 | regression | 29 | predictor_v1 连续性能指标 |

---

## 二、65 维数值特征完整表

| Index | 特征名 | 分组 | 含义 |
|------:|--------|------|------|
| 0 | ee_x | 场景绝对位姿 | 末端执行器世界坐标 X (m) |
| 1 | ee_y | 场景绝对位姿 | 末端执行器世界坐标 Y (m) |
| 2 | ee_z | 场景绝对位姿 | 末端执行器世界坐标 Z (m) |
| 3 | cube_x | 场景绝对位姿 | cube 世界坐标 X (m) |
| 4 | cube_y | 场景绝对位姿 | cube 世界坐标 Y (m) |
| 5 | cube_z | 场景绝对位姿 | cube 世界坐标 Z (m) |
| 6 | target_x | 场景绝对位姿 | 放置目标世界坐标 X (m) |
| 7 | target_y | 场景绝对位姿 | 放置目标世界坐标 Y (m) |
| 8 | target_z | 场景绝对位姿 | 放置目标世界坐标 Z (m) |
| 9 | gripper_width | 场景绝对位姿 | 夹爪开口宽度 (m) |
| 10 | ee_cube_dx | 场景相对关系 | cube.x − ee.x (m) |
| 11 | ee_cube_dy | 场景相对关系 | cube.y − ee.y (m) |
| 12 | ee_cube_dz | 场景相对关系 | cube.z − ee.z (m) |
| 13 | ee_target_dx | 场景相对关系 | target.x − ee.x (m) |
| 14 | ee_target_dy | 场景相对关系 | target.y − ee.y (m) |
| 15 | ee_target_dz | 场景相对关系 | target.z − ee.z (m) |
| 16 | cube_target_dx | 场景相对关系 | target.x − cube.x (m) |
| 17 | cube_target_dy | 场景相对关系 | target.y − cube.y (m) |
| 18 | cube_target_dz | 场景相对关系 | target.z − cube.z (m) |
| 19 | ee_cube_dist | 场景距离 | ‖ee − cube‖ (m) |
| 20 | ee_target_dist | 场景距离 | ‖ee − target‖ (m) |
| 21 | cube_target_dist | 场景距离 | ‖cube − target‖ (m) |
| 22 | cube_target_xy_dist | 场景距离 | XY 平面 ‖cube − target‖ (m) |
| 23 | height_offset | 技能参数值 | 目标上方高度偏移 (m) |
| 24 | has_height_offset | 参数 mask | height_offset 是否有效 (0/1) |
| 25 | xy_offset_x | 技能参数值 | XY 偏移 X (m) |
| 26 | has_xy_offset_x | 参数 mask | xy_offset_x 是否有效 |
| 27 | xy_offset_y | 技能参数值 | XY 偏移 Y (m) |
| 28 | has_xy_offset_y | 参数 mask | xy_offset_y 是否有效 |
| 29 | speed | 技能参数值 | 位置移动速度 (m/step) |
| 30 | has_speed | 参数 mask | speed 是否有效 |
| 31 | position_tolerance | 技能参数值 | 位置收敛容差 (m) |
| 32 | has_position_tolerance | 参数 mask | position_tolerance 是否有效 |
| 33 | timeout_steps | 技能参数值 | 最大执行步数 |
| 34 | has_timeout_steps | 参数 mask | timeout_steps 是否有效 |
| 35 | target_height | 技能参数值 | 目标绝对高度 Z (m) |
| 36 | has_target_height | 参数 mask | target_height 是否有效 |
| 37 | relative_z | 技能参数值 | 相对 Z 下降量 (m) |
| 38 | has_relative_z | 参数 mask | relative_z 是否有效 |
| 39 | close_wait_steps | 技能参数值 | 夹爪闭合等待步数 |
| 40 | has_close_wait_steps | 参数 mask | close_wait_steps 是否有效 |
| 41 | lift_height | 技能参数值 | 抬升高度 (m) |
| 42 | has_lift_height | 参数 mask | lift_height 是否有效 |
| 43 | place_height | 技能参数值 | 放置高度 (m) |
| 44 | has_place_height | 参数 mask | place_height 是否有效 |
| 45 | release_height | 技能参数值 | 释放后抬升高度 (m) |
| 46 | has_release_height | 参数 mask | release_height 是否有效 |
| 47 | open_wait_steps | 技能参数值 | 夹爪张开等待步数 |
| 48 | has_open_wait_steps | 参数 mask | open_wait_steps 是否有效 |
| 49 | target_tolerance | 技能参数值 | 放置目标容差 (m) |
| 50 | has_target_tolerance | 参数 mask | target_tolerance 是否有效 |
| 51 | retreat_height | 技能参数值 | 撤退抬升高度 (m) |
| 52 | has_retreat_height | 参数 mask | retreat_height 是否有效 |
| 53 | wait_steps | 技能参数值 | 等待步数 |
| 54 | has_wait_steps | 参数 mask | wait_steps 是否有效 |
| 55 | orientation_tolerance | 技能参数值 | 姿态收敛容差 (rad) |
| 56 | has_orientation_tolerance | 参数 mask | orientation_tolerance 是否有效 |
| 57 | angular_speed | 技能参数值 | 姿态角速度 (rad/step) |
| 58 | has_angular_speed | 参数 mask | angular_speed 是否有效 |
| 59 | has_position_goal | parallel | 是否含 position_goal (0/1) |
| 60 | has_orientation_goal | parallel | 是否含 orientation_goal (0/1) |
| 61 | position_goal_speed | parallel | 位置子目标速度 |
| 62 | orientation_goal_angular_speed | parallel | 姿态子目标角速度 |
| 63 | position_goal_tolerance | parallel | 位置子目标容差 |
| 64 | orientation_goal_tolerance | parallel | 姿态子目标容差 |

CSV 版本：`numeric_features_65.csv`

---

## 三、predictor_v1 29 维回归输出定义

| Index | 回归目标 | 含义 | 单位 |
|------:|----------|------|------|
| 0 | execution_steps | 技能执行步数 | steps |
| 1 | execution_time | 技能执行时间 | s |
| 2 | trajectory_length | 末端轨迹长度 | m |
| 3 | final_ee_x | 末端最终位置 X | m |
| 4 | final_ee_y | 末端最终位置 Y | m |
| 5 | final_ee_z | 末端最终位置 Z | m |
| 6 | target_x | 当前 skill 目标位置 X | m |
| 7 | target_y | 当前 skill 目标位置 Y | m |
| 8 | target_z | 当前 skill 目标位置 Z | m |
| 9 | final_ee_position_error | 末端与显式 target_pose 的位置误差 | m |
| 10 | final_ee_roll | 末端最终 roll | rad |
| 11 | final_ee_pitch | 末端最终 pitch | rad |
| 12 | final_ee_yaw | 末端最终 yaw | rad |
| 13 | final_ee_orientation_error | 末端最终姿态误差 | rad |
| 14 | final_ee_linear_speed | 末端结束线速度模长 | m/s |
| 15 | average_ee_linear_speed | 末端平均线速度 | m/s |
| 16 | final_object_x | cube 最终位置 X | m |
| 17 | final_object_y | cube 最终位置 Y | m |
| 18 | final_object_z | cube 最终位置 Z | m |
| 19 | object_target_position_error | cube 到 target 的三维距离 | m |
| 20 | object_target_xy_error | cube 到 target 的 XY 平面误差 | m |
| 21 | object_lift_delta | cube 高度变化 Δz | m |
| 22 | ee_object_distance | 结束时 ee 到 cube 距离 | m |
| 23 | min_ee_object_distance | 执行过程中 ee-cube 最小距离 | m |
| 24 | object_target_xy_distance | cube 到 target 的 XY 距离（v0 兼容别名） | m |
| 25 | final_position_error | cube 到 target 的三维距离（v0 兼容别名） | m |
| 26 | object_displacement | cube 执行前后位移 | m |
| 27 | gripper_width_start | 开始时夹爪宽度 | m |
| 28 | gripper_width_end | 结束时夹爪宽度 | m |

3 个分类输出：

| 输出 | 类别 |
|------|------|
| success | true / false |
| timeout | true / false |
| failure_reason | none, skill_failed, object_not_in_gripper, object_not_near_target, timeout, parallel_timeout, orientation_not_converged, reach_failed, place_failed, unknown |

---

## 三.1、Stage II 新增原始 measured_performance 字段

`performance_collector.py` 现在可采集以下 VLM 反馈字段；其中向量字段会在 `feature_extractor.py`
中展开为 predictor_v1 的标量回归目标。无法稳定计算的字段写 `null`，并在
`measured_performance_missing` 中记录原因。

| 字段 | 类型 | 说明 |
|------|------|------|
| final_ee_position | list[3] | skill 结束时末端位置 |
| final_ee_orientation | list[4] | skill 结束时末端四元数，反馈格式为 `[x, y, z, w]` |
| final_ee_rpy | list[3] | skill 结束时末端 roll/pitch/yaw |
| target_position | list[3] / null | 当前 skill 显式目标位置 |
| target_orientation | list[4] / null | 当前 skill 显式目标姿态 |
| final_ee_linear_velocity | list[3] / null | 使用最后最多 5 帧估计的末端线速度 |
| final_ee_linear_speed | float / null | `final_ee_linear_velocity` 模长 |
| average_ee_linear_speed | float / null | `trajectory_length / execution_time` |
| final_ee_angular_velocity | null | 预留，当前不做不可靠估计 |
| final_ee_angular_speed | null | 预留，当前不做不可靠估计 |
| final_object_position | list[3] | cube 最终位置 |
| final_object_orientation | list[4] | cube 最终姿态四元数 `[x, y, z, w]` |
| final_object_rpy | list[3] | cube 最终 roll/pitch/yaw |
| object_target_position_error | float | cube 到 target 的三维误差 |
| object_target_xy_error | float | cube 到 target 的 XY 误差 |
| reached_target_within_tolerance | bool / null | 末端是否在目标容差内 |
| performance_risk_level | string | low / medium / high |
| performance_risk_reason | string / null | VLM 可读风险原因 |

---

## 四、各子技能输入特征激活情况（has_* = 1）

所有技能 **0–22 维场景特征始终有效**。下表仅列出各 skill 通常激活的参数维（其余 param 维 value=0, has=0）。

| 技能 | target 典型值 | 通常激活的参数维 (index) |
|------|---------------|--------------------------|
| **parallel** | cube / target | 23–34, 55–58, 59–64 (height_offset, xy_offset, speed, position_tolerance, timeout, orientation_*, parallel_*) |
| **move_above** | cube / target | 23–34 (height_offset, xy_offset, speed, position_tolerance, timeout_steps) |
| **reach** | cube / target | 29–34 (speed, position_tolerance, timeout_steps) |
| **descend** | cube | 29–34, 35–36 (speed, position_tolerance, timeout_steps, target_height) |
| **grasp** | cube | 33–34, 39–40 (timeout_steps, close_wait_steps) |
| **lift** | cube | 29–34, 41–42 (speed, position_tolerance, timeout_steps, lift_height) |
| **place** | target | 31–34, 43–50 (position_tolerance, timeout_steps, place/release/open_wait/target_tolerance) |
| **retreat** | current | 29–30, 33–34, 51–52 (speed, timeout_steps, retreat_height) |
| **wait** | current | 33–34, 53–54 (timeout_steps, wait_steps) |
| **align_orientation** | cube / target | 33–34, 55–58 (timeout_steps, orientation_tolerance, angular_speed) |

---

## 五、各子技能训练的性能标签（example_skill_blueprint_parallel）

基于 `example_skill_blueprint_parallel.json` 的 `performance_query` 与预测器 v1 回归头对照。旧 JSONL 不含新增字段时，对应 regression mask 自动为 0。

图例：**✓** = blueprint 采集且通常参与 regression loss；**○** = 模型有输出头但 blueprint 未 query 或常 null；**—** = 不适用。

### 5.1 parallel（p1 / p2）

| 性能指标 | p1 (cube) | p2 (target) | 模型回归头 |
|----------|:---------:|:-----------:|:----------:|
| success | ✓ | ✓ | 分类 |
| timeout | ✓ | ✓ | 分类 |
| failure_reason | ✓ | ✓ | 分类 |
| execution_steps | ✓ | ✓ | ✓ |
| execution_time | ○ | ○ | ✓ (常 missing) |
| trajectory_length | ✓ | ✓ | ✓ |
| final_ee_position_error | ✓ | ✓ | ✓ |
| final_ee_orientation_error | ✓ | ✓ | ✓ |
| object_lift_delta | ○ | ○ | ✓ (collector 有值但未 query) |
| ee_object_distance | ○ | ○ | ✓ |
| min_ee_object_distance | ○ | ○ | ✓ |
| object_target_xy_distance | ○ | ✓ | ✓ |
| final_position_error | ○ | ○ | ✓ |
| object_displacement | ○ | ○ | ✓ |
| gripper_width_start/end | ○ | ○ | ✓ |
| position_converged 等 | query 但不进模型 | query 但不进模型 | — |

### 5.2 descend（s2）

| 性能指标 | 参与训练 |
|----------|:--------:|
| success / timeout / failure_reason | ✓ |
| execution_steps | ✓ |
| trajectory_length | ✓ |
| final_ee_position_error | ✓ |
| ee_object_distance | ✓ |
| min_ee_object_distance | ✓ |

### 5.3 grasp（s3）

| 性能指标 | 参与训练 |
|----------|:--------:|
| success / timeout / failure_reason | ✓ |
| execution_steps | ✓ |
| execution_time | ✓ |
| object_lift_delta | ✓ |
| ee_object_distance | ✓ |
| gripper_width_start / end | ✓ |

### 5.4 lift（s4）

| 性能指标 | 参与训练 |
|----------|:--------:|
| success / timeout / failure_reason | ✓ |
| execution_steps | ✓ |
| trajectory_length | ✓ |
| final_ee_position_error | ✓ |
| object_lift_delta | ✓ |
| ee_object_distance | ✓ |
| min_ee_object_distance | ✓ |

### 5.5 place（s6）

| 性能指标 | 参与训练 |
|----------|:--------:|
| success / timeout / failure_reason | ✓ |
| execution_steps | ✓ |
| execution_time | ✓ |
| trajectory_length | ✓ |
| final_ee_position_error | ✓ |
| object_target_xy_distance | ✓ |
| final_position_error | ✓ |
| object_displacement | ✓ |
| gripper_width_start / end | ✓ |

### 5.6 retreat（s7）

| 性能指标 | 参与训练 |
|----------|:--------:|
| success / timeout / failure_reason | ✓ |
| execution_steps | ✓ |
| trajectory_length | ✓ |
| final_ee_position_error | ✓ |
| gripper_width_start / end | ✓ |

---

## 六、汇总矩阵（CSV）

完整技能 × 性能矩阵见：`skill_performance_matrix.csv`。
predictor_v1 回归目标 CSV 见：`regression_targets_v1.csv`。

列含义：
- `train`：该 skill 的 blueprint 会 query 且标签通常非 null，参与 loss
- `usually_missing`：模型有回归头，但该 skill 的 blueprint 未 query 或 collector 常返回 null

---

## 七、第一版未纳入训练的指标

以下指标在 blueprint `performance_query` 中可能出现，但 **不在模型 13 维回归头中**：

- position_converged, orientation_converged
- parallel_mode, parallel_goal_count
- position_goal_success, orientation_goal_success
- object_stability, gripper_command_final
- max_contact_force, collision_count, collision_risk, object_drop_risk

这些保留为未来扩展。
