# perception_memory_pipeline —— 感知 + 记忆 一键流水线

读 `input/` 里的 **5 张相机视角图** → **GPT‑5.5 感知**(枚举所有物体 + 关系）→ 喂 **记忆模块**（三层记忆）→
把感知输出、记忆给规划模块的输出 **全部写成 `output/` 下的 JSON**。一条命令 / 直接运行 `main.py` 一键完成。

## 目录结构
```
perception_memory_pipeline/
├── main.py                 ← 主入口(一键跑)
├── input/                  ← 放 5 张视角图(front/wrist/left/right/top .png)
├── output/                 ← 运行后生成的 JSON(见下)
├── memory_pkg/             ← 内置(vendored)的记忆模块(自包含,不依赖外部路径)
├── planning/               ← 规划模块相关代码
│   └── run_planner.py      ← 规划模块独立测试运行脚本
└── README.md
```

## 依赖(已逐文件 import 核对 + 实测,精确到 4 个)
只需要 **4 个第三方包**:`requests`、`torch`、`Pillow`、`numpy`。
```bash
pip install -r requirements.txt          # 或: pip install requests torch Pillow numpy
```
为什么是这 4 个:
- `requests` —— 感知步 HTTP 调中转 Responses API(`urllib3` 随它一起装,无需单列)。
- `torch` + `Pillow` —— vendored 的 `memory_pkg/perception_vlm.py` **顶层无条件** `import torch` / `from PIL import ...`,
  而 `main.py` 直接 import 了它的 `RecognitionResult`,所以这两个是硬依赖(**Pillow 容易漏,别忘**)。
- `numpy` —— 记忆模块(perception_adapter / agent_memory / working_memory)顶层 `import numpy`。

**不需要**(别装):`transformers` / `accelerate` / `qwen-vl-utils` —— 它们在 `perception_vlm.py` 里被
`try/except ImportError` 包住,只在改用【本地 Qwen 感知】时才要;本流水线用 GPT‑5.5,用不到。
`scipy`/`scikit-learn`/`pandas` 只是 transformers 的传递依赖,同样不需要。

> 本机现成可用:`env_isaaclab` conda 环境(这 4 个都在)。

## 一键运行
```bash
cd projects/perception_memory_pipeline

# 1) API key:你的真 key 已在 ~/.bashrc 的 API_zhongzhuan,交互终端会自动加载 —— 通常【什么都不用设】。
#    ★ 别照抄下面的 sk-...(那是占位符!照抄会把真 key 覆盖成假的 -> 401 Unauthorized)。
#    只有在 key 没自动加载时,才用【你自己的完整 key】设:
#      export API_zhongzhuan=<你的完整key>        # 或 export OPENAI_API_KEY=<你的完整key>
#    验证 key 在不在(应显示 长度=67 开头=sk-a98):
#      echo "len=${#API_zhongzhuan} head=${API_zhongzhuan:0:6}"

# 2) 跑(用带 torch 的 python)
/home1/banghai/miniconda3/envs/env_isaaclab/bin/python main.py
#   或:conda activate env_isaaclab && python main.py
```
> 想换图:把新的 5 张图覆盖进 `input/`(命名 front/wrist/left/right/top.png)再跑即可。

## 输出(`output/`)
| 文件 | 是什么 |
|---|---|
| `perception.json` | **感知模块输出**(GPT‑5.5):`scene_summary` + `objects[]`(名称/类别/外观/位置/可见视角)+ `relations[]` |
| **`memory_planning_input.json`** | **★ 记忆给规划模块的文件**:`task_instruction` + `manipulable_objects`(物体+可供性)+ `available_skills` + `constraints`(分类规则/不混放/避碰) |
| `memory_planning_context.json` | 记忆产出的完整 `PlanningContext`(所有字段) |
| `memory_snapshot.json` | 记忆三层快照:working / episodic / semantic |
| `goal_intent.json` | 规划内部产出: 抽象深层意图 (Goal Reasoner) |
| `sdg_plan.json` | 规划内部产出: 抽象状态依赖图 (SDG Planner) |
| `planned_actions.json` | **★ 规划最终输出**: 包含思维链(CoT)推理过程与具体抓取/放置序列 |

规划模块只需读 **`memory_planning_input.json`** 即可在内部完成 `意图提取 -> SDG 状态规划 -> CoT 动作接地` 的完整闭环。

## 可配置(环境变量)
| 变量 | 默认 | 说明 |
|---|---|---|
| `API_zhongzhuan` / `OPENAI_API_KEY` | (必填) | 中转/OpenAI key |
| `VLM_BASE_URL` | `https://165.154.193.90` | 中转 Responses API 端点根 |
| `VLM_MODEL` | `gpt-5.5` | 感知模型 |
| `VLM_REASONING` | `high` | 推理强度(high 更全但更慢;medium 更快) |
| `TASK` | `把桌面物品按类别分拣进三个篮子` | 任务指令(写进记忆) |

## 流程图
```
input/*.png (5 视角)
    │
    ▼  run_perception()  —— GPT-5.5 中转 Responses API(结构化 JSON)
perception.json  {scene_summary, objects[], relations[]}
    │
    ▼  run_memory()  —— 喂三层记忆 + 注入任务约束
output/  memory_planning_input.json   (给规划的三样)
         memory_planning_context.json (完整 PlanningContext)
         memory_snapshot.json         (三层记忆快照)
    │
    ▼  planner.generate_plan() —— (封装执行 Intent抽取 -> SDG状态蓝图 -> CoT思维链泛化落地)
output/  goal_intent.json             (任务深层意图)
         sdg_plan.json                (基于类别约束的抽象状态图)
         planned_actions.json         (带有自我推理的具象可执行动作)
```

## 说明
- **自包含**:记忆模块已 vendored 进 `memory_pkg/`,不依赖仓库外路径。感知走 HTTP 调中转,不需要本地大模型。
- 不需要 Isaac / 仿真:输入是**预先抓好的 5 张图**;要换场景就换 `input/` 里的图。
- 一个已知口径:记忆默认给每个物体 `[grasp, place]` 可供性(含桌子/柜子等);如需按类别区分(家具不可抓),
  在 `run_memory` 里按 `objects[].category` 定制,或在感知侧只把"可抓物品"列进 objects。
