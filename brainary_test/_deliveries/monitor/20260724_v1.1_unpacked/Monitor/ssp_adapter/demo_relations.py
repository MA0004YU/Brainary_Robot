"""DEMO 物体间 L0 关系（适配层 · 用户需求 2）。

SSP 的风险模板要靠实体间关系（near / supports / reachable / contains ...）来实例化风险，
但当前记忆里**一条实体间关系都没存**（perception.json 的 relations[] 只是"物体→桌面方位"，
且在 run_memory 阶段被丢弃）。所以 DEMO 阶段这里把关系硬编码。

关系依据当前 5 图 DEMO 场景的 location 字段推断（同一方位的物体互为 near，
所有桌面物体由 table 平面 supports）。

⚠️⚠️⚠️  DEMO 写死项  ⚠️⚠️⚠️
本文件所有关系都是硬编码。真机 / 未来版本应由**感知或记忆**产出实体间空间关系
（几何 near、支撑 supports、可达 reachable），不应写死。详见 docs/SSP_INTEGRATION.md。

关系语义（对齐 ssp.ontology.relations.L0Relation）：
    supports(a, b)  : a 平面支撑 b（table -> 每个桌面物体）
    near(a, b)      : a、b 空间邻近（同一 location 分组内两两互连）
    reachable(a, b) : b 对 a 可达 —— 本 DEMO **不注入**，因为没有 human/robot victim，
                      注入 reachable 也无终端风险 target（见 docs 说明）。
"""

from __future__ import annotations

# DEMO 场景的一张桌面支撑平面。id 用固定 ASCII，中文名给回写用。
TABLE_SURFACE_ID = "table_surface"
TABLE_SURFACE_NAME = "桌面"

# 按 memory_snapshot.json 的 location 字段分组（同组视为空间邻近）
# TODO(真机): 用感知输出的物体位姿/包围盒算几何邻近，替换此写死分组
_LOCATION_GROUPS: list[list[str]] = [
    ["绿色方块", "折叠刀"],            # 桌面中间
    ["剪刀", "橙子"],                 # 桌面中间偏左
    ["黄色杯子", "蓝色杯子", "茶盒", "香蕉"],  # 桌面左侧
    ["红色碗"],                       # 桌面前侧
    ["咖啡机", "抽屉柜"],             # 桌面右上角
]


def build_demo_relations(object_names: list[str]) -> list[tuple[str, str, str]]:
    """产出 DEMO 硬编码关系，以中文名表示（memory_to_gp 负责映射成 ASCII id）。

    返回 (src_name, dst_name, relation) 三元组列表，relation 为 L0Relation 的字符串值。
    只对实际出现在 object_names 里的物体建边（对齐当前记忆快照）。

    TODO(真机): 整个函数应被"感知/记忆产出的关系"取代，此处仅为打通管线。
    """
    present = set(object_names)
    edges: list[tuple[str, str, str]] = []

    # 1) table 平面 supports 每个桌面物体
    #    TODO(真机): supports 应由感知的支撑面检测产出
    for name in object_names:
        edges.append((TABLE_SURFACE_NAME, name, "supports"))

    # 2) 同一 location 分组内的物体两两 near（无向 -> 建单向即可，SSP 按需匹配）
    #    TODO(真机): near 应由感知的物体间距离阈值产出
    for group in _LOCATION_GROUPS:
        members = [n for n in group if n in present]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j], "near"))

    return edges
