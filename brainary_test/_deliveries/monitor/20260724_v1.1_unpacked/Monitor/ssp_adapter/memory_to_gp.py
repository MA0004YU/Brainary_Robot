"""核心适配器：记忆快照 -> SSP PerceptualGraph（适配层主体）。

读 output/memory_snapshot.json 的 working.observation.memory_objects，
经「属性映射」（object_attribute_map）+「关系注入」（demo_relations），
产出合法的、通过 g_p.validate() 的 PerceptualGraph。

设计要点（对齐用户要求）：
  - 可独立运行：只吃现成 memory_snapshot.json，不重跑感知/记忆、不碰 torch。
  - 中文物体名 -> 稳定 ASCII id（拼音表 + 序号后缀），保留 id<->中文名 双向映射用于回写。
  - 严格遵守 SSP 闭集契约：Node.type/subtype 用枚举，物理状态进 StateSchema。

⚠️ 边界：本适配器只负责"造合法输入"。是否出风险由 SSP 的模板 + 场景本身决定；
   当前 DEMO 无 human/animal，多数终端风险无 victim，必然 0 activated —— 这是预期正确行为。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ssp.graph.g_percept import PerceptualGraph
from ssp.ontology.entities import EntityType, PhysicalObjectSubtype
from ssp.ontology.relations import L0Relation
from ssp.ontology.schema import (
    ContainmentState,
    Edge,
    Node,
    StateSchema,
    UncertaintyTag,
)

from Monitor.ssp_adapter.demo_relations import (
    TABLE_SURFACE_ID,
    TABLE_SURFACE_NAME,
    build_demo_relations,
)
from Monitor.ssp_adapter.object_attribute_map import build_state_schema, resolve_spec

# 中文 -> 拼音音译表，覆盖 DEMO 已知物体的字。未覆盖的字回退到序号。
_PINYIN: dict[str, str] = {
    "绿": "lv", "色": "se", "方": "fang", "块": "kuai",
    "黄": "huang", "杯": "bei", "子": "zi",
    "蓝": "lan", "红": "hong", "碗": "wan",
    "剪": "jian", "刀": "dao", "折": "zhe", "叠": "die",
    "橙": "cheng", "茶": "cha", "盒": "he",
    "香": "xiang", "蕉": "jiao", "咖": "ka", "啡": "fei", "机": "ji",
    "抽": "chou", "屉": "ti", "柜": "gui", "桌": "zhuo", "面": "mian",
}


def _to_ascii_id(name: str, seq: int) -> str:
    """中文名 -> 稳定 ASCII id：拼音拼接 + 序号后缀（序号保证唯一 & 稳定）。

    序号后缀让 id 与快照中物体顺序绑定，即使音译表未覆盖也不会碰撞。
    """
    parts = [_PINYIN.get(ch, "") for ch in name]
    stem = "".join(parts) or "obj"
    return f"{stem}_{seq}"


@dataclass
class GPBuildResult:
    """适配器产物：PerceptualGraph + 回写用的辅助映射。"""

    graph: PerceptualGraph
    id_to_name: dict[str, str] = field(default_factory=dict)  # ASCII id -> 中文名
    name_to_id: dict[str, str] = field(default_factory=dict)  # 中文名 -> ASCII id
    notes: list[str] = field(default_factory=list)            # DEMO 写死项/降级说明

    def to_debug_dict(self) -> dict:
        """序列化成可写 output/ssp_perceptual_graph.json 的调试结构。"""
        return {
            "nodes": [
                n.model_dump(mode="json", exclude_none=True)
                for n in self.graph.nodes.values()
            ],
            "edges": [e.model_dump(mode="json", exclude_none=True) for e in self.graph.edges],
            "id_to_name": self.id_to_name,
            "demo_notes": self.notes,
        }


def load_memory_objects(snapshot_path: str | Path) -> dict[str, dict]:
    """从 memory_snapshot.json 读出 working.observation.memory_objects（中文名 -> 属性 dict）。"""
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    try:
        return snap["working"]["observation"]["memory_objects"]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"memory_snapshot.json 结构不符合预期（缺 working.observation.memory_objects）: {e}"
        ) from e


def build_graph_from_memory(snapshot_path: str | Path) -> GPBuildResult:
    """主入口：读记忆快照 -> 属性映射 + 关系注入 -> 合法 PerceptualGraph。"""
    memory_objects = load_memory_objects(snapshot_path)
    return build_graph_from_objects(memory_objects)


def build_graph_from_objects(memory_objects: dict[str, dict]) -> GPBuildResult:
    """从 memory_objects（中文名->属性）构建 G_P。拆出来便于单测/复用。"""
    result = GPBuildResult(graph=PerceptualGraph(nodes=[], edges=[]))
    nodes: list[Node] = []
    edges: list[Edge] = []

    object_names = list(memory_objects.keys())

    # 1) table 支撑平面节点（DEMO 写死；关系依赖它做 supports 源）
    result.id_to_name[TABLE_SURFACE_ID] = TABLE_SURFACE_NAME
    result.name_to_id[TABLE_SURFACE_NAME] = TABLE_SURFACE_ID
    nodes.append(
        Node(
            id=TABLE_SURFACE_ID,
            type=EntityType.SURFACE,
            subtype=None,
            attributes=StateSchema(),
            uncertainty=UncertaintyTag.ASSUMED,  # 写死推断，非观测
            source="adapter_demo",
        )
    )
    result.notes.append(
        "DEMO 写死: 追加了 table_surface 平面节点 (SURFACE)，真机应由感知产出支撑面。"
    )

    # 2) 每个记忆物体 -> Node（属性映射）
    for seq, name in enumerate(object_names, start=1):
        meta = memory_objects[name] or {}
        category = meta.get("category")
        spec = resolve_spec(name, category)

        node_id = _to_ascii_id(name, seq)
        result.id_to_name[node_id] = name
        result.name_to_id[name] = node_id

        nodes.append(
            Node(
                id=node_id,
                type=spec.entity_type,
                subtypes=[s.value for s in spec.subtypes],
                attributes=build_state_schema(spec),
                # 属性来自映射表推断而非直接观测 -> 标 INFERRED
                uncertainty=UncertaintyTag.INFERRED,
                source="adapter_attr_map",
            )
        )

        # 2b) 含液覆盖：杯/碗 DEMO 视为含液 -> 补一个 substance 水节点 + contains 边
        if spec.has_liquid:
            liquid_id = f"{node_id}_liquid"
            liquid_name = f"{name}_液体"
            result.id_to_name[liquid_id] = liquid_name
            result.name_to_id[liquid_name] = liquid_id
            nodes.append(
                Node(
                    id=liquid_id,
                    type=EntityType.SUBSTANCE,
                    subtype=None,
                    attributes=StateSchema(containment=ContainmentState.OPEN),
                    uncertainty=UncertaintyTag.ASSUMED,  # DEMO 写死"含液"
                    source="adapter_demo",
                )
            )
            edges.append(
                Edge(src=node_id, dst=liquid_id, relation=L0Relation.CONTAINS, sign="+")
            )
            result.notes.append(
                f"DEMO 写死: 容器 '{name}' 视为含液，补 substance 节点 '{liquid_name}' + contains 边；"
                "真机应由感知判断容器内容物。"
            )

    # 3) 关系注入（demo_relations 给的是中文名三元组，映射成 id）
    present_names = set(result.name_to_id.keys())
    for src_name, dst_name, rel in build_demo_relations(object_names):
        if src_name not in present_names or dst_name not in present_names:
            continue
        edges.append(
            Edge(
                src=result.name_to_id[src_name],
                dst=result.name_to_id[dst_name],
                relation=L0Relation(rel),
                sign="+",
                uncertainty=UncertaintyTag.ASSUMED,  # DEMO 写死关系
            )
        )
    result.notes.append(
        "DEMO 写死: 实体间 near/supports 关系由 demo_relations 硬编码，真机应由感知/记忆产出。"
    )

    graph = PerceptualGraph(nodes=nodes, edges=edges)
    graph.validate()  # 契约校验：边端点存在、relation 合法、sign=+
    result.graph = graph
    return result
