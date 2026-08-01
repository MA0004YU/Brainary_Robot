# _deliveries —— 各模块交付备份档

本目录**只存原始交付**(协作者每次发来的 zip / 解包件),按模块分文件夹、按日期命名。
**不是运行代码**;运行的永远是 `brainary/<模块>/` 下的 canonical 版本。

## 目录结构

```
_deliveries/
  perception/   memory/   planning/   monitor/   simulation/
    <YYYYMMDD>_<简述>.zip              原始交付(不可变,别改)
    <YYYYMMDD>_<简述>_unpacked/        需要时解包留档
```

当前归档:
| 模块 | 交付件 | 说明 |
|---|---|---|
| planning | `20260706_base_pipeline.zip` | 早期 感知+记忆+规划 离线管线 |
| planning | `20260711_ltm_pipeline.zip` + `_unpacked/` | 加了 EmbodiedLTM 的规划(当前 canonical 源) |
| planning | `_working_offline_pipeline/` | 我方合并/修复过的离线工作副本(留档) |
| monitor  | `20260719_monitor.zip` | SSP + SafetyCritic 初版 |
| monitor  | `Perception_memory_planner_monitor_pipeline-V1.1.zip` + `20260724_v1.1_unpacked/` | **V1.1(当前 canonical 源)**:SSP 修复风险爆炸(三态 latent/uncertain,消 N² 笛卡尔积);已接进 planner 之前回灌约束 |
| simulation | `20260706_simulation.zip` | SAPIEN 物理校验沙盒(初版) |
| simulation | `simulation(1).zip` + `requirements.txt` | **最新**:代码与初版一致,更新=重型 CUDA 依赖清单 + weights/cg_weights 权重下载 + PhysicalValidator 接入契约 |
| project_management | `PM_delivery_to_banghai_20260722_181318.zip` + `20260722_pm_unpacked/` | **PM 项目管理**(当前 canonical 源):plan 别名解析(中文名→仿真ID)+ 依赖分波调度 + 执行(dry_run/真机) |

## canonical(实际运行的)在哪

```
brainary/
  perception/ memory/ planning/ project_management/ monitor/ simulation/   ← 每模块最新代码
  run_brainary.py                                           ← 唯一编排,跑上面 5 个
  output/<时间戳>/{sim,perception,memory,planning,monitor}/ ← 每次运行的输出
```

一键运行(用最新仿真场景):
```
conda activate env_isaaclab
./isaaclab.sh -p brainary/run_brainary.py
```

## 更新迁移工作流(每次有人发新版就照做)

1. **归档**:把交付 zip 放进 `_deliveries/<模块>/<日期>_<简述>.zip`(原件不动)。
2. **对比**:解包后和 canonical `brainary/<模块>/` diff,看改了哪些文件。
3. **移植**:只把**该模块自己的代码**覆盖进 canonical;
   - ⚠️ **别带进别的模块的捆绑副本**(交付里常夹带旧的 planning/ 或 memory_pkg/,是旧版,会污染 import)。
   - ⚠️ **保留我方已打的中转修复**:`planning/llm_client.py`(走中转 gpt-5.5)、`monitor/Monitor/safety_critic/pipeline_llm.py`(VLM_BASE_URL 归一化)。若新版覆盖了它们,重新打上修复。
4. **验证**:跑一次 `run_brainary.py`(或离线只跑改动的阶段),确认 `output/<新时间戳>/` 产物正常。

## 已知需要对方修的(见对话记录)

- ~~monitor/SSP 风险爆炸~~ **已在 V1.1 修复**(三态 latent/uncertain + 消 N² 笛卡尔积;本场景 7 物体 activated 0/latent 7 → 14 约束),已接进 planner 之前回灌。
- **simulation**:缺 sapien/GSA、无环境说明、需要深度+相机内外参、目标名中英对不上 → 暂未接入 run_brainary。
