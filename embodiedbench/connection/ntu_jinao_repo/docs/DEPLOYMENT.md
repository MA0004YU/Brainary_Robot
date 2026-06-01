# Ntu_jinao 部署说明（/home1/banghai/IsaacLab）

本文档记录 **Ntu_jinao** 项目在本机 Isaac Lab 工作区中的部署状态与版本差异。

---

## 1. 部署位置

```text
/home1/banghai/IsaacLab/
├── isaaclab.sh                          # Isaac Lab 启动脚本
├── VERSION                              # Isaac Lab 2.3.2
├── ntu_jinao_repo/                      # ← 本项目（从 GitHub 移植）
│   ├── README.md
│   ├── docs/
│   │   ├── AI_HANDOFF.md                # 项目现状与后续任务（原仓库文档）
│   │   └── DEPLOYMENT.md                # 本文件
│   ├── source/standalone/franka_state_machine_cerebellum/
│   ├── skill_performance_predictor/
│   └── vlm_brain/
└── datasets/                            # 仿真采集输出（运行时生成，不提交 Git）
```

来源仓库：[https://github.com/ysdjy/Ntu_jinao](https://github.com/ysdjy/Ntu_jinao)

---

## 2. 版本差异（重要）

| 项目 | 原开发环境 | 本机环境 |
|------|-----------|---------|
| Isaac Sim | **4.5** | **5.1.0.0** |
| Isaac Lab | v2.0.x / v2.1.x | **2.3.2** |
| Conda 环境 | `isaaclab45` | **`env_isaaclab`** |
| Python | 3.10 | 3.11 |
| 原机器 GPU | RTX 3070 Ti 8GB | 待修复驱动后确认 |

**说明：**

- 原项目 README 写的是 Isaac Sim 4.5；本机 Isaac Lab 2.3.2 对应 **Isaac Sim 5.1**，属于跨大版本迁移。
- 仿真模块以 standalone 脚本调用 `Isaac-Lift-Cube-Franka-IK-Rel-v0`，该任务在本仓库中仍存在，API 结构（`arm_action` + `gripper_action`、`scene["robot"]` / `object` / `ee_frame`）与代码一致。
- 若 Stage II 运行报错，优先对照 Lift IK-Rel 环境配置是否变更，再查 `state_machine_cerebellum.py` 中的 scene 键名。
- **VLM（Qwen3-VL）必须在独立 Conda 环境运行**，不要升级 `env_isaaclab` 里的 `transformers`，以免影响 Isaac Sim。

---

## 3. 本机环境激活

```bash
cd /home1/banghai/IsaacLab
conda activate env_isaaclab
```

验证：

```bash
python -c "import isaaclab; import isaaclab_tasks; print('Isaac Lab OK')"
python -c "from importlib.metadata import version; print('isaacsim', version('isaacsim'))"
```

---

## 4. 快速验证（无需完整仿真）

### Stage IV dry-run（VLM + predictor 骨架）

```bash
cd /home1/banghai/IsaacLab/ntu_jinao_repo

python vlm_brain/vlm_predictor_loop.py \
  --config vlm_brain/configs/qwen3vl_8b_infer.json \
  --scene_state vlm_brain/examples/sample_scene_state.json \
  --task "pick the cube and place it on the target" \
  --output_dir vlm_brain/outputs/loop_dry_run \
  --dry_run true \
  --mock_predictor true \
  --max_refine_iters 1
```

### 蓝图校验

```bash
python vlm_brain/blueprint_validator.py \
  --blueprint source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint_parallel.json
```

---

## 5. Stage II 仿真采集（需 GPU + Isaac Sim 正常）

```bash
cd /home1/banghai/IsaacLab
conda activate env_isaaclab

TERM=xterm ./isaaclab.sh -p ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_episodes 1 \
  --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --blueprint_path ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint_parallel.json \
  --output_dir ./datasets/stage2_smoke_test \
  --target_mode custom_tabletop \
  --seed 0 \
  --save_trajectory true
```

输出目录应包含 `predictor_dataset.jsonl`、`episodes.jsonl`、`summary.json`。

---

## 6. 已知本机问题

- **NVIDIA 驱动**：部署时检测到 `Failed to initialize NVML: Driver/library version mismatch`，需重启或对齐驱动后再跑仿真。
- **项目不完整**：Stage I–IV 代码骨架已在原仓库验证；真实 Qwen3-VL 推理、500+ episode 数据采集、predictor_v1 正式训练尚未完成。详见 [`AI_HANDOFF.md`](AI_HANDOFF.md) 第 9 节优先任务（P0–P5）。

---

## 7. 后续工作优先级（摘自 AI_HANDOFF）

| 优先级 | 任务 |
|--------|------|
| **P0** | 大显存机器上真实 Qwen3-VL 推理 → 合法 `skill_blueprint` JSON |
| **P1** | 稳定 VLM JSON 输出（prompt / repair） |
| **P2** | Stage II 批量采集 500+ episodes → `predictor_dataset.jsonl` |
| **P3** | 训练 `predictor_v1` |
| **P4** | VLM → predictor → 修正蓝图 → Stage II 闭环对比 |
| **P5** | VLM SFT / LoRA 数据与训练 |

完整步骤与命令见 [`AI_HANDOFF.md`](AI_HANDOFF.md)。

---

## 8. 路径引导修复

原仓库 `_bootstrap_repo_source_paths()` 假定脚本位于 `IsaacLab/source/standalone/...`（向上 3 层即 Isaac Lab 根目录）。  
部署在 `ntu_jinao_repo/` 子目录时向上 3 层会停在 `ntu_jinao_repo` 而非 Isaac Lab 根目录。

已新增 `bootstrap_paths.py`：向上查找含 `isaaclab.sh` 与 `source/isaaclab` 的目录，两种布局均可运行。
