# Brainary —— 具身大脑闭环项目

把 **7 个模块**串成一个机器人"大脑":从 Isaac Sim 真实仿真取画面,经感知/记忆/规划,再由安全与物理
两道校验把关,最后调度执行到机器人。目标是**从真实仿真器出发的一条可闭环、可替换、可分工的具身智能流水线**。

一条指令跑通全流程,每次运行把每个模块的输入/输出分门别类存进 `output/<时间戳>/`。

---

## 1. 架构总览(注意:不是纯线性,含多处反馈环)

```
              ┌─────────────────────────── Isaac Sim 真实仿真(sim 模块)───────────────────────────┐
              │  感知输入:5相机 RGB+深度+内外参 / 场景状态      执行输出:grasp/place/开抽屉/开门…      │
              └───────┬─────────────────────────────────────────────────────▲──────────────────────┘
                      │ RGB/状态                                              │ 动作(真机执行)
                      ▼                                                       │
   ①感知 perception ──► ②记忆 memory ──► planning_input ──────────┐          │
   (物体+关系)          (三层记忆+快照)                          │          │
                             │                                     ▼          │
                             │  memory_snapshot          ④规划 planning ◄──┐  │
                             ▼                            (Intent→SDG→落地) │  │
              监控·SSP 场景安全解析 ──候选安全约束──────► (回灌 planning_input) │  ← 反馈环A:约束回流
                    (监控模块的一部分)                            │          │  │
                                                                  ▼          │  │
                                                    ⑤物理校验 simulation ────┘  │  ← 反馈环B:不过则
                                                    (PhysicalValidator)          │     反思 replan
                                                       success│                  │
                                                             ▼                   │
                                          项目管理 PM ──别名解析+分波调度──────┘
                                          (含"执行":调 API 抓放)  ├─► 真机执行(--execute)
                                                       │
                                                       ▼
                                          监控·SafetyCritic ── 逐动作判 malicious(可 halt)  ← 侧路裁判/门控
```

> 说明:**SSP 和 SafetyCritic 都属于「监控」模块**(SSP 在规划前喂约束、SafetyCritic 在执行时逐动作裁判);
> **"执行/调 API 抓放"属于「项目管理」模块**。这两条只是监控/项目管理各自内部的职能,不是独立模块。

**三处"非线性":**
- **反馈环 A(监控·SSP→规划)**:SSP 在规划**之前**跑,但它的产出(候选安全约束)**回灌进 planning_input**,
  再喂给规划器——是"约束回流",不是简单前后串联。
- **反馈环 B(物理校验→重规划)**:规划出 plan 后先送物理沙盒预演;**不通过则把 `llm_reflection_prompt`
  喂回规划器 replan**,构成"规划⇄校验"小循环(`--verify`)。
- **侧路门控(监控·SafetyCritic)**:不在主数据流里改 plan,而是**逐动作旁路裁判**,遇 malicious 记录中断点。
- **项目管理并行调度**:PM 把动作按依赖关系分成**并行波(wave)**,不是逐条线性执行。

> 完整闭环(执行→再感知→再规划)是设计目标;当前主流程是单轮 + 上述局部反馈环,尚未整圈自动闭合。

---

## 2. 两种运行方式

### A. 有 Isaac Sim(接真实仿真,完整闭环)
```bash
cd <IsaacLab 根目录>
conda activate env_isaaclab
./isaaclab.sh -p brainary/run_brainary.py                 # 默认:感知auto,PM dry_run,不做物理校验

# 真实下发机器人执行 grasp/place(建议先降速,避免搬运甩物体):
export SKILL_TEST_RUNNER_SPEED=1.5    # 抓/放整体速度(默认3.0会把每步关节限幅顶到0.5rad,偏快)
export SKILL_TEST_CARRY_STEP=0.10     # 搬运物体时的每步关节限幅(越小越稳,不甩物体)
./isaaclab.sh -p brainary/run_brainary.py --execute

# 规划后加物理沙盒校验:先在另一终端起 sim 服务(见 §5 / simulation/INTEGRATION.md §6):
#   conda activate brainary_sim && python brainary/simulation/serve.py   # 等 READY on :5600
./isaaclab.sh -p brainary/run_brainary.py --verify        # 规划后 POST 给 :5600;服务没起则自动跳过

# ★可视化一键测试 GUI(加载后点一个按钮跑整条流水线,7 阶段实时显示状态/结果 + 展示方案+安全裁决):
./isaaclab.sh -p brainary/sim/brainary_brain_ui.py --device cuda:0
```

### B. 没有 Isaac Sim(别人的电脑,测自己模块)
感知输入直接读静态仿真数据(仓库自带 `sample_data/sim/`),跳过仿真与需 Isaac 的阶段:
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r brainary/requirements-offline.txt
python brainary/run_offline.py                            # 感知→记忆→SSP→规划→PM→监控
```
详见 **[DEPLOY_OFFLINE.md](DEPLOY_OFFLINE.md)**。

---

## 3. 模块详解(功能 / 接真实 Isaac 的角色 / 现存问题)

> **整个项目就 7 个模块,每个对应一位负责人**(别拆多余的):
> IsaacSim 底座(含技能)/ 感知 / 记忆 / 规划 / **监控(含 SSP + SafetyCritic)**/ Simulation / **项目管理(含执行)**。
> 注:SSP 属于**监控**(不是独立模块);"执行/调 API 抓放"属于**项目管理**(不是独立模块)。

### ① IsaacSim 底座(含技能)—— 仿真接口(`sim/`)
- **功能**:`BrainaryAPI`(包 `SimInterface`)是真实仿真的**唯一出入口**。
  - 感知侧(只读):`get_all_cameras()` 一次给 5 相机 `{rgb, depth, intrinsics, pose}`;`get_robot_state/get_object_pose` 等。
  - 执行侧:`grasp/place/open_drawer/close_drawer/open_door/close_door/operate_coffee/go_home/open_gripper/close_gripper`,
    每个都带**物理失败检测**(如 grasp 返回 `ok/holding/gripper_width`)。
- **接真实 Isaac**:它就是"真实"那一层——底层封装 `projects/franka_v1_skill_lab` 的技能与场景。
- **现存问题**:必须在 IsaacLab 根目录用 `./isaaclab.sh` 跑;`operate_coffee` 等技能是否可用要以
  `list_skills()` 为准;离线导出的 `scene_state.json` **未存相机内外参**(物理校验因此离线跑不了)。

### ② perception —— 感知(`perception/`)
- **功能**:多视角 RGB → `{objects[{name,category,appearance,location}], relations}`。三个后端:
  ChatGPT(`scene_describer`:5599,默认)/ 本地 Qwen2.5-VL(:5601)/ 仿真 GT-mock(不联网兜底)。
- **接真实 Isaac**:吃 `sim` 抓的 RGB;产物 `perception.json` 喂记忆。
- **现存问题**:①**不输出物理属性**(energy/stability/orientation/containment 等,SSP 只能靠映射表推断);
  ②产出的 `relations[]` 目前在记忆阶段被丢弃;③GPT 给的是友好名(如 `yellow_mug`),**≠ 仿真 ID**(`Prop_SM_Mug_C1`),
  靠 PM 的别名表桥接;④漏小物体的老问题。

### ③ memory —— 三层记忆(`memory/`)
- **功能**:感知结果入 working/episodic/semantic 三层记忆,导出规划要的
  `planning_input.json`(`task_instruction / manipulable_objects / available_skills / constraints`)+ `memory_snapshot.json`。
- **接真实 Isaac**:纯本地逻辑,不调模型;是感知↔规划/监控之间的结构化中枢。
- **现存问题**:**不存实体间关系,也不存物体物理状态/稳定 id**——SSP 需要的这些当前由 SSP 适配层"补/写死"。

### ④ planning —— 规划(`planning/`)
- **功能**:三段式 LLM 规划 `Intent(深层意图)→SDG(状态依赖图)→落地动作序列`,产
  `plan.json` + `planned_actions.json`(`[{id,action,target,depends_on}]`)。走中转 gpt-5.5;无 key 退回规则分拣兜底。
- **接真实 Isaac**:读记忆(+SSP约束)出 plan;action 词表 `grasp/place` 与 sim 技能对齐。
- **现存问题**:①依赖 LLM;②`--verify` 的反思 replan 已把 `physics_reflection` 注入 planning_input,
  但**规划器 prompt 侧尚需读取该字段**才真正闭环;③可选长期记忆 EmbodiedLTM(`--use-ltm`,需另起 :8000 服务)。

### ⑤ monitor —— 监控(`monitor/`,**含 SSP + SafetyCritic 两条职能,是一个模块**)
- **SSP(场景安全解析,规划之前;反馈环A)**:记忆快照 → 强类型场景图 `PerceptualGraph` → 跑 14 类风险模板 →
  候选安全约束(CT-*),**回灌进 planning_input 的 `constraints.safety_constraints`**,让规划带着安全约束生成动作。
  V1.1 已修风险爆炸(三态 activated/latent/uncertain + 消 N² 笛卡尔积,本场景 activated 0/latent 7/14 约束);
  代码在 `monitor/Monitor/ssp_*`。
- **SafetyCritic(逐动作裁判,执行前/中;侧路门控)**:读动作序列 + 记忆快照,用 LLM **逐动作**判
  `malicious / not malicious`,记首个中断点,产 `safety_critic_review.json`;代码在 `monitor/Monitor/safety_critic`。
- **接真实 Isaac**:SSP 在规划前喂约束,SafetyCritic 在执行前后做安全门控。
- **现存问题**:SSP 与 SafetyCritic 尚未联动(候选约束未喂给 critic);SSP 的物体属性/实体关系仍由适配层
  `object_attribute_map.py`/`demo_relations.py` **DEMO 写死**,真机应由感知/记忆产出;critic 裁决偏保守。

### ⑥ simulation —— 物理沙盒校验(`simulation/`,反馈环B,◐ 可选)
- **功能**:`PhysicalValidator.verify_local_plan(plan_dag, rgb, depth, 内参, 外参)` 即时重构**局部 SAPIEN 物理沙盒**
  预演 plan,检碰撞/倾覆/滑脱/夹碎/关节死锁;`success=True` 放行,`False` 返回 `llm_reflection_prompt` 供 replan。
- **接真实 Isaac**:规划后、执行前的**物理关**;相机数据由 `BrainaryAPI.get_all_cameras()` 提供
  (`simulation/verify_adapter.py` 负责转格式)。
- **已跑通(HTTP 服务)**:独立 CUDA 环境 `brainary_sim` 装好并验证——torch cu121 + cuda12.1 +
  GroundingDINO(_C 编译)+ SAM + pytorch3d/nvdiffrast + sapien,权重(`weights/cg_weights/`)已下好。
  `simulation/serve.py`(:5600)在该环境常驻,主流程(env_isaaclab)`--verify` 时经 `verify_client.py`
  POST 相机数据+plan 过去,拿回 `{success, llm_reflection_prompt}`——**两进程各用各的 conda 环境、只用 HTTP
  说话、共用同一内核 NVIDIA 驱动(不改驱动)**。`serve.py` 启动即 `chdir` 到模块根,从任意目录起都能找到
  `panda.urdf` 等相对资产。`physics_dictionary.py` 已补 brainary_test 场景物体(蓝杯/黄杯/香蕉/橙子/剪刀/
  饼干盒/肉罐)供 GroundingDINO 检测。配置见 `simulation/SETUP_ENV.md` / `INTEGRATION.md §6`。
- **现存问题(名字桥接)**:沙盒有**自己独立的一套感知**(GroundingDINO + `physics_dictionary`),其实体名
  (`blue mug`/`orange`…)与规划器用的名字(来自 scene_describer 的 `blue_small_cup`/`orange_ball`…)**对不上**
  → verify 现在会在第一个动作就判 `TARGET_NOT_FOUND`、反馈无法被规划器有效利用(空转告警)。**下一步**:在
  `verify_client.py` 发送前做名字归一/共用同一套感知,让物理校验真正把关。此外 action 词表只实现 `grasp/place`;
  离线 `scene_state` 未存相机内外参(离线不可用)。

### ⑦ project_management(PM)—— 调度与**执行**(`project_management/`,**"执行/调 API 抓放"属于本模块**)
- **功能**:plan → **别名解析**(感知名/中文名→仿真 ID,`object_aliases.json` + 模糊匹配)→ **依赖分波并行调度** →
  **执行(调 API 抓放)**:`sim.grasp/place/...`,**抓成功才放、没抓到就跳过对应放置**;产
  `pm_execution_result.json` + `pm_planned_actions.json`。
- **接真实 Isaac**:这就是"**规划→机器人执行**"那一环("执行"是本模块的职能,不是独立模块);
  `sim=None`→dry_run 只调度,传真实 `sim`(`--execute`)→真机动。
- **分拣归一化**:感知阶段(`run_brainary._normalize_category`)会把 GPT 给的宽松类(`ball`/`package`/`cup`…)
  按 name/appearance 关键词归一到分拣分类法(水果/工具/杯具/食品),否则规划器会把 `orange_ball` 当"球"直接漏排。
  修后橙子归水果→KLT_3,**三个篮子都用上**。`object_aliases.json` 已补 `blue_small_cup`/`red_package` 等,
  确保 `--execute` 时每个 plan 目标都能解析到真 prop。
- **现存问题**:别名表按场景手写(换场景要更新/自动化);真机执行受**可达性**影响(物体摆太远 approach 够不到,需摆进可达区)。

---

## 4. 完整数据契约(替换模块时对齐)

| 边 | 文件 | 关键字段 |
|---|---|---|
| sim→感知 | `sim/scene_state.json` + `sim/rgb/*.png`(+depth) | task/graspable/baskets/objects{name:pose}/robot(+相机内外参*) |
| 感知→记忆 | `perception.json` | `objects[{name,category,appearance,location}], relations, perception_backend` |
| 记忆→SSP/规划/监控 | `planning_input.json` + `memory_snapshot.json` | `task_instruction/manipulable_objects/available_skills/constraints`;`working.observation.*` |
| SSP→规划 | `planning_input.json`(回写) | `constraints.safety_constraints`(候选 CT-*) |
| 规划→校验/PM/监控 | `plan.json` + `planned_actions.json` | `[{id,action,target,depends_on}]` |
| 校验→规划 | (内存) | `success` / `llm_reflection_prompt`(replan) |
| PM→监控/执行 | `pm_planned_actions.json` + `pm_execution_result.json` | 别名解析后的动作(仿真 ID)+ 执行报告 |
| 监控→输出 | `safety_critic_review.json` | 逐动作裁决 + `first_halt_index` |

\* 相机内外参:Isaac 方式由 `get_all_cameras()` 实时提供;离线 `scene_state.json` 暂未存(物理校验离线不可用)。

---

## 5. 环境与部署

| 用途 | 环境 | 说明 |
|---|---|---|
| 方式A 完整闭环 | `env_isaaclab` | Isaac Sim + torch;`./isaaclab.sh` |
| 方式B 离线(感知→监控) | 任意带 `requirements-offline.txt` 的 venv | 无需 Isaac;干净 venv 实测 |
| 规划/监控 LLM | 环境变量 `API_zhongzhuan` | 中转 gpt-5.5;无 key 则规划退兜底、监控跳过 |
| 感知 ChatGPT(可选) | `perception/scene_describer/.venv_vlm` | openai+httpx,不占 GPU |
| 物理校验 simulation(可选) | **独立 CUDA 环境** `brainary_sim` | sapien/pytorch3d/nvdiffrast/GSA/SAM;配置见 **simulation/SETUP_ENV.md**;权重(cg_weights)已下好;+ GSA_PATH |

各模块 canonical 代码在 `brainary/{perception,memory,planning,project_management,monitor,simulation}/`;
协作者原始交付归档在 `_deliveries/<模块>/`(更新流程见其 README)。

---

## 6. 现存问题总清单(接真实 Isaac 视角)

1. **感知语义↔仿真 id 的桥接**:感知给友好名/中文名,执行要仿真 id;现靠 PM 手写别名表(已覆盖本场景物体)+
   感知阶段类别归一化,应进一步做成自动匹配(位置+类别)。
2. **感知未产物理属性 / 记忆未存关系**:SSP 现在靠适配层写死,真机应由感知输出 stability/containment… + 记忆存 near/supports。
3. **物理校验的名字桥接**:sim HTTP 服务(:5600)已跑通,但沙盒自带感知的实体名 ≠ 规划器用的名字 →
   verify 现在总在首个动作判 `TARGET_NOT_FOUND`。需在 `verify_client.py` 发送前归一名字/共用感知,才能真正把关。
4. **反思 replan 未完全闭合**:`--verify` 已把物理反馈注入 planning_input,但规划器 prompt 需读取 `physics_reflection`
   (且依赖上面第 3 条修好后 verify 反馈才有意义)。
5. **整圈闭环未自动化**:执行→再感知→再规划 的大闭环尚未串;当前是单轮 + 局部反馈环。
6. **监控与 SSP 未联动**:SafetyCritic 未消费 SSP 的候选约束。
7. **真机执行链路**:PM `--execute` + 别名/降速就位,可端到端跑;仍受场景**可达性**制约(物体需摆进可达区)。
