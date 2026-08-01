# Simulation 物理沙盒接入(PhysicalValidator)

## 1. 契约(simulation 作者给的)

在大模型每次生成 plan 后,把 `plan_dag` + 多视角 RGB/深度/相机内外参传给
`PhysicalValidator.verify_local_plan(...)`,引擎即时重构局部物理沙盒预演:

- `success == True`  → 直接把 plan 下发下游(PM / 执行);
- `success == False` → 把返回的 `llm_reflection_prompt` 原样喂回 planning agent 做 replan。

流水线位置:**规划之后、PM/执行之前**。

```
感知 → 记忆 → SSP → 规划 → [物理校验 verify]──success──▶ PM → 监控 → 执行
                          └──False: llm_reflection_prompt──▶ 回规划 replan(最多 N 次)
```

## 2. 现成接入代码:`simulation/verify_adapter.py`

```python
from engine_interface import PhysicalValidator      # 需在 simulation/ 目录、sim CUDA 环境里
from verify_adapter import verify_then_replan

validator = PhysicalValidator(config_path="config/global_config.yaml")
plan, res = verify_then_replan(validator, sim, planner, planning_input, task, max_replans=2)
# res["success"] 为 True 时 plan 可下发;planner 需能读 planning_input["physics_reflection"] 做反思
```

- `gather_camera_inputs(sim)`:从 `BrainaryAPI.get_all_cameras()` 一次取
  `{cam:{rgb,depth,intrinsics,pose}}` 并转成 verify 要的 `rgb_views/depth_views/内参3x3/外参4x4`。
- `verify_plan(validator, sim, plan_dag)`:采集相机数据 → 调 `verify_local_plan`。
- `verify_then_replan(...)`:完整的 规划→校验→(失败注入反馈 replan)循环。

## 3. ⚠️ 三个运行前提(缺一不可,当前 env_isaaclab 都不满足)

1. **独立 CUDA 环境**:`sapien==3.0.1 / torch2.1.2+cu121 / pytorch3d / nvdiffrast / groundingdino / segment_anything / conceptgraphs`(见 `requirements.txt`)。
   **绝不能装进 env_isaaclab**(依赖互斥,会拖坏 Isaac)。建议单开 conda 环境,像 scene_describer 那样。
2. **权重**:先下 `weights/cg_weights/` 的两个权重(见 `WEIGHTS_DOWNLOAD.md`)。
3. **环境变量** `GSA_PATH` 指向 Grounded-Segment-Anything(否则 `PhysicalValidator.__init__` 直接抛错)。

## 4. 因环境互斥,推荐的部署形态

`PhysicalValidator` 跑在自己的 CUDA 环境、且每次调用会重建 SAPIEN 沙盒(重)。它和 Isaac(env_isaaclab)
**不能同进程**。两种落地方式:

- **A) 服务化(推荐,类比 scene_describer:5599)**:在 sim 环境起一个 HTTP 服务包住 `verify_plan`,
  `run_brainary` 规划后 POST(plan + 相机数据)过去,拿回 `{success, llm_reflection_prompt}`。
- **B) 同环境**:若你把 Isaac 与 sim 依赖合进一个环境(不推荐,易冲突),可直接在 `run_brainary`
  规划后调 `verify_then_replan`。

`run_brainary.py` 已预留 `--verify` 钩子(默认关):开时在规划后尝试 import 物理校验,缺依赖则打印提示并跳过,
不影响其余阶段。真正启用需按上面 A/B 之一搭好 sim 环境。

## 5. 数据契约缺口(offline 场景需补)

`verify_local_plan` 要**相机内外参**。Isaac 方式(`run_brainary`)可由 `BrainaryAPI.get_all_cameras()`
直接给;但离线的 `sample_data/sim/scene_state.json` 目前**没存相机内外参**,所以离线方式暂不能跑物理校验
(需仿真侧在导出 sim 数据时把 intrinsics/pose 一并存进 scene_state)。

## 6. HTTP 服务(已实现,推荐 —— 与 Isaac 零冲突、不动驱动)

物理沙盒需 `brainary_sim` 环境(sapien 等),与 `env_isaaclab` 互斥、不能同进程。解决办法:
把它跑成**独立进程的 HTTP 服务**(类比感知 scene_describer:5599),主流程只发网络请求。

**① 起服务(单独一个终端,brainary_sim 环境):**
```bash
conda activate brainary_sim
export CUDA_HOME=/home1/banghai/miniconda3/envs/brainary_sim
export GSA_PATH=<...>/brainary/simulation/src/perception/third_party/Grounded-Segment-Anything
export CUDA_VISIBLE_DEVICES=1          # 用第 2 张卡,避开 Isaac 用的 cuda:0(可选)
cd <...>/brainary/simulation
python serve.py                        # 加载 GDINO+SAM,然后 READY on 0.0.0.0:5600
```

**② 主流程调用(另一个终端,env_isaaclab,照常):**
```bash
./isaaclab.sh -p brainary/run_brainary.py --verify   # 规划后 POST 给 :5600,不过就反思 replan
```
`run_brainary` 的 `stage_verify` 只 `import verify_client`(纯 numpy/PIL/requests,**不 import sapien**),
从 `BrainaryAPI.get_all_cameras()` 取 rgb/深度/内外参 → POST `/verify` → 拿 `{success, llm_reflection_prompt}`。
**服务没起就自动跳过**(`--verify` 变空操作),对其它模块零影响。

- 服务端:`simulation/serve.py`（`GET /health`、`POST /verify`）。
- 客户端:`simulation/verify_client.py`（Isaac 侧,`service_up()` / `verify_via_service()`）。
- 地址可用 `SIM_VERIFY_ADDR` 覆盖(默认 `http://127.0.0.1:5600`)。

**隔离性**:两进程各用各的 conda 环境,只用 HTTP 说话;共用同一个内核 NVIDIA 驱动(无需改驱动);
不启动服务 = 和现在完全一样。
