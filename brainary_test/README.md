# Brainary —— 具身大脑闭环项目

把 **5 个模块串成闭环**的机器人"大脑":在桌面分拣场景里
**仿真**出画面 → **感知**识别物体 → **记忆**积累汇总 → **规划**产出动作序列 → **监控**逐动作判安全。

一条指令跑通全流程,每次运行把每个模块的输入/输出分门别类存进 `output/<时间戳>/`。

---

## 两种运行方式

### A. 有 Isaac Sim(接真实仿真环境,完整闭环)

```bash
cd <IsaacLab 根目录>
conda activate env_isaaclab
./isaaclab.sh -p brainary/run_brainary.py
```
跑完 **仿真→感知→记忆→规划→监控** 五阶段,输出在 `brainary/output/<时间戳>/`(`output/latest` 软链指向最近一次)。

常用变体:
```bash
./isaaclab.sh -p brainary/run_brainary.py --perception mock   # 感知用仿真GT(不联网,验证管线最快)
./isaaclab.sh -p brainary/run_brainary.py --perception gpt    # 感知强制走 ChatGPT
```

### B. 没有 Isaac Sim(别人的电脑,一键跑"仿真之后的所有模块")

感知的输入**直接读静态的仿真数据图片**(仓库自带 `sample_data/sim/`),给各模块负责人测自己的模块:

```bash
# 1) 装依赖(约 2 步,见 DEPLOY_OFFLINE.md)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r brainary/requirements-offline.txt

# 2) 一键运行(默认 mock 感知,零配置)
python brainary/run_offline.py
```
跑完 **感知→记忆→规划→监控** 四阶段(跳过需要 Isaac 的仿真阶段),输出结构与方式 A 完全一致。
换自己的静态数据:`python brainary/run_offline.py --sim-data <某个 sim 目录>`。

> 详细部署见 **[DEPLOY_OFFLINE.md](DEPLOY_OFFLINE.md)**。

---

## 模块与接入状态

| 阶段 | 模块 | 目录 | 状态 | 说明 |
|---|---|---|---|---|
| 1 | 仿真 sim | `sim/` | ✅(仅方式A) | Isaac 场景,抓 5 视角 RGB+深度+scene_state |
| 2 | 感知 perception | `perception/` | ✅ | ChatGPT(scene_describer:5599)/ 本地Qwen(:5601)/ 仿真GT-mock |
| 3 | 记忆 memory | `memory/` | ✅ | 三层记忆,产 planning_input + memory_snapshot |
| 4 | 规划 planning | `planning/` | ✅ | LTM 真规划(gpt-5.5 中转,Intent→SDG→落地),规则分拣兜底 |
| 5 | 监控 monitor | `monitor/` | ✅(部分) | **SafetyCritic** 逐动作判安全;**SSP** 暂未接(风险因子爆炸,待作者修) |
| — | 物理沙盒 simulation | `simulation/` | ⛔ 未接 | SAPIEN 物理校验;缺 sapien/GSA、需深度+相机内外参、目标名待桥接 |

---

## 目录结构

```
brainary/
├── run_brainary.py           ★方式A:Isaac 闭环编排(5 阶段)
├── run_offline.py            ★方式B:无 Isaac 一键跑感知之后所有模块(读静态数据)
├── requirements-offline.txt  离线运行依赖(已干净 venv 实测)
├── README.md / DEPLOY_OFFLINE.md
│
├── perception/  memory/  planning/  monitor/  simulation/   ← 各模块【最新 canonical 代码】
├── sim/                      仿真 API + 场景 GUI(方式A用;底层 import projects/ 里的 Isaac 框架)
├── scene/brainary_test/      仿真场景数据(USD + manifest + grasp/waypoints)
├── sample_data/sim/          样例静态仿真数据(方式B输入:5 RGB + 深度 + scene_state)
│
├── _deliveries/              各模块【原始交付备份档】(按模块/日期,见其 README)
└── output/<时间戳>/          每次运行输出:{sim,perception,memory,planning,monitor}/ + run_summary.json
```

**canonical vs 备份档**:`brainary/<模块>/` 是实际运行的最新代码;协作者每次发来的原始交付归档在
`_deliveries/<模块>/`,更新流程见 [`_deliveries/README.md`](_deliveries/README.md)。

> 方式 A 需在 IsaacLab 根目录用 `./isaaclab.sh` 跑(仿真框架本体留在 `projects/`,靠 import 使用)。
> 方式 B 与 Isaac 完全解耦,任意目录 `python brainary/run_offline.py` 即可。

---

## 运行环境

| 用途 | 环境 | 说明 |
|---|---|---|
| 方式A 完整闭环 | `env_isaaclab` | Isaac Sim + torch;`./isaaclab.sh` 已用它 |
| 方式B 离线(感知/记忆/规划/监控) | 任意带 `requirements-offline.txt` 的 venv | 无需 Isaac;见 DEPLOY_OFFLINE.md |
| 感知 ChatGPT 后端(:5599) | `perception/scene_describer/.venv_vlm` | openai+httpx,不占 GPU;需 API key(可选) |
| 感知 Qwen 后端(:5601,备选) | conda `qwen3vl` | 本地 Qwen2.5-VL-3B,占一张卡(可选) |

---

## 模块间数据契约(替换模块时对齐这些字段)

- **仿真 → 感知**:`sim/scene_state.json`(task/graspable/baskets/objects{name:pose}/robot)+ `sim/rgb/*.png` 5 视角。
- **感知 → 记忆**:`perception/perception.json`
  `{scene_summary, objects:[{name,category,appearance,location,position?}], relations, perception_backend}`。
- **记忆 → 规划/监控**:`memory/planning_input.json`
  `{task_instruction, manipulable_objects:{name:[skills]}, available_skills, constraints:{category_rules,...}}`
  + `memory/memory_snapshot.json`(working.observation.* —— 监控读它判安全)。
- **规划 → 监控/执行**:`planning/plan.json`(富信息)+ `planning/planned_actions.json`
  `[{id, action, target, depends_on}]` —— 监控读后者;执行时可喂回 `BrainaryAPI.grasp/place`。
- **监控 → 输出**:`monitor/safety_critic_review.json`(逐动作 malicious/not malicious + 总体裁决)。

---

## LLM 说明(规划 / 监控)

- 规划、监控直连**中转站** gpt-5.5,读环境变量 `API_zhongzhuan`(或 `OPENAI_API_KEY`)。
- 感知走独立 scene_describer:5599 服务器(自带 venv);无 key/服务器时自动退回仿真 GT-mock。
- **无 `API_zhongzhuan` 也能跑通**:规划退回规则分拣、监控自动跳过,前面阶段照常出产物。

---

## 各模块负责人怎么调试/优化

- **感知**(perception/):看 `output/latest/perception/perception.json`;GPT 漏小物体 → 改
  `scene_describer/schema.py` 的 prompt,或切 Qwen(:5601)。
- **记忆**(memory/):三层记忆在 `embodiedbench/memory_manip/`,对外接口 `memory_module/`;看
  `output/latest/memory/memory_report.md`。出新版:替换 `memory/` 内容(接口向后兼容)。
- **规划**(planning/):`planning/task_planner.py` 三段式(Intent→SDG→落地);中转 LLM 在 `llm_client.py`。
  看 `output/latest/planning/plan.json`。开长期记忆:`--use-ltm`(需另起 EmbodiedLTM :8000 服务)。
- **监控**(monitor/):`Monitor/safety_critic` 是逐动作裁判;看 `output/latest/monitor/safety_critic_review.json`。
  SSP(`Monitor/ssp_adapter`)可独立跑 `python -m Monitor.ssp_adapter --dry-run`。

---

## 常见问题

- **方式A import 报错**:必须在 IsaacLab 根目录用 `./isaaclab.sh` 跑。
- **方式B 缺依赖**:`pip install -r brainary/requirements-offline.txt`(torch 单独装 CPU 版)。
- **感知一直是 mock**:没连上 ChatGPT 服务器(:5599 没起/无 key)。方式B 默认就是 mock,正常。
- **规划只有 1 步 / 报 401**:LLM 没连上中转(检查 `API_zhongzhuan`);会自动退回规则分拣。
- **每次运行的产物**:都在 `output/<时间戳>/`;`output/latest` 软链指最近一次。
