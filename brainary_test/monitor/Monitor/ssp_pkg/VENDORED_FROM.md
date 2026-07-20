# Vendored SSP (Scene Safety Parser)

本目录是从「场景安全解析器 SSP」仓库 **vendor 复制**进来的核心最小集，
对齐本仓库既有的 `memory_pkg/` vendor 模式（通过 `sys.path` 注入，绝对 import `from ssp.xxx`）。

## 来源

| 项 | 值 |
|---|---|
| 源仓库路径 | `/mnt/d/Postdoctoral_Research/Embodied_monitor/M2_Scene_safety_parser` |
| 源 commit | `7501079937c71f8beec8ee317b25b7c6ace5b49e` |
| SSP `__version__` | `0.1.0` |
| 复制日期 | 2026-07-18 |

## 复制范围（import 闭包最小集）

只复制 `SceneSafetyParser.query_risk` 运行所需的最小集合：

- `ssp/__init__.py`, `ssp/parser.py`（入口 `SceneSafetyParser`）
- `ssp/ontology/`：`__init__.py, entities.py, relations.py, risk_events.py, schema.py, template_registry.py`
- `ssp/graph/`：`__init__.py, g_percept.py, g_reason.py, lift.py, g_activated.py, g_constraint.py`
  - 注：`g_activated.py` / `g_constraint.py` 本身不在核心调用路径上，但 `graph/__init__.py`
    会 eager import 它们，为保证零改动 import 一并复制。
- `ssp/propagation/`：全部（`__init__, operator, fixed_point, admissibility, aggregation, params`）
- `ssp/query/`：`__init__.py, activate.py, predicates.py, result.py`
- `configs/re_templates/*.yaml`：全部 14 个风险事件（RE）模板

## 有意省略（非核心）

- `ssp/utils/`（空壳 stub，核心模块直接用 `structlog.get_logger()`，无人 import）
- `ssp/bridge/`（`constitution.py` / `ltl_skeleton.py` 均为空壳 stub —— 未来 CT-id→LTL 的桥接层）
- `ssp/ablation/`（消融实验用的 parser 子类）
- `ssp/ontology/bench_adapter.py`, `bench_schema.py`, `rule_derived_gt.py`（benchmark 专用，
  本仓库改用 `adapters/` 下自研的 memory→G_P 适配器）

## import 风格

源仓库全程绝对 import（`from ssp.xxx import ...`），且本 vendor 保持顶层包名 `ssp` 不变，
因此**无需改任何 import**。调用方（`main.py` / `adapters/`）通过
`sys.path.insert(0, "ssp_pkg")` 把本目录当作 site-packages 根注入。

## 运行时依赖

`pydantic>=2`, `networkx`, `structlog`, `PyYAML`（已追加进本仓库 `requirements.txt`）。
