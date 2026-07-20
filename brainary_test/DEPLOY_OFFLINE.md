# Brainary 离线部署指南(无需 Isaac Sim)

给**没有仿真器**的协作者:在自己电脑上快速跑通"仿真之后的所有模块"(感知→记忆→规划→监控),
用来测试/调试自己负责的模块。感知的输入直接读仓库自带的**静态仿真数据图片**,不需要跑仿真。

> 已在**干净 venv 实测**:下面的步骤装完即可一键跑通(torch CPU 2.13 + 8 个小依赖)。

---

## 1. 前置

- Python 3.9 ~ 3.13(实测 3.13 OK)
- 能联网装 pip 包
- (可选)中转站 API key `API_zhongzhuan` —— 规划/监控要调 gpt-5.5 才需要;不配也能跑(见 §4)

---

## 2. 安装(2 步)

```bash
cd <Brainary_Robot 仓库>/brainary_test        # 即本项目所在目录

python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate

# torch 装 CPU 版(小、无需显卡;记忆模块 import torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r brainary/requirements-offline.txt
```

依赖清单(`brainary/requirements-offline.txt`,核心 8 个):
`numpy  Pillow  requests  urllib3  pydantic>=2  networkx  structlog  PyYAML` + `torch`(CPU)。

---

## 3. 运行(1 条指令)

```bash
# 配 key(要用 gpt-5.5 规划/监控;不配见 §4)
export API_zhongzhuan=<你的中转key>

python brainary/run_offline.py
```

- 默认读 `brainary/sample_data/sim/`(最新桌面分拣场景:7 物品 + 3 篮子的 5 视角 RGB+深度+状态)。
- 感知默认 `mock`(用仿真 GT,不联网、零配置)。
- 输出在 `brainary/output/<时间戳>/{perception,memory,planning,monitor}/`,`output/latest` 指最近一次。

跑完看结果:
```bash
cat brainary/output/latest/planning/plan.json              # 规划出的动作序列
cat brainary/output/latest/monitor/safety_critic_review.json  # 安全裁判逐动作结论
```

换成你自己的静态数据(目录里要有 `rgb/*.png` + `scene_state.json`):
```bash
python brainary/run_offline.py --sim-data <你的sim目录>
```

---

## 4. 无 API key 时

规划会自动**退回规则分拣**、监控会**自动跳过**,前面感知/记忆照常出产物——**管线仍跑通**,
只是规划不走 LLM。适合只测感知/记忆的同学。

---

## 5. 各模块负责人怎么测自己的模块

| 你负责 | 改哪 | 看哪个产物 |
|---|---|---|
| 感知 | `perception/`(或直接改 `run_offline.py` 的感知阶段) | `output/latest/perception/perception.json` |
| 记忆 | `memory/memory_module/`、`memory/embodiedbench/` | `output/latest/memory/planning_input.json` + `memory_report.md` |
| 规划 | `planning/task_planner.py`、`prompt_templates.py` | `output/latest/planning/plan.json` + `planned_actions.json` |
| 监控 | `monitor/Monitor/safety_critic/` | `output/latest/monitor/safety_critic_review.json` |

模块间字段契约见主 [README.md](README.md) 的「数据契约」一节。改自己模块时**保持输出字段不变**即可与上下游对齐。

---

## 6. 可选增强(默认都不需要)

- **感知用真 ChatGPT**(而非 mock):`python brainary/run_offline.py --perception gpt`
  需起 `perception/scene_describer` 服务器(自带 `.venv_vlm`,装 openai)+ `API_zhongzhuan`。
- **规划开长期记忆 EmbodiedLTM**:需另起 `planning/EmbodiedLTM` 的 :8000 服务(见其 README,建议单独 conda 环境)。
- **感知用本地 Qwen**:`perception/perception_qwen`(需 GPU + `qwen3vl` 环境)。

---

## 7. 有 Isaac Sim 的话

在 IsaacLab 根目录直接跑完整闭环(含真实仿真):
```bash
./isaaclab.sh -p brainary/run_brainary.py
```
见主 [README.md](README.md) 方式 A。
