"""物品属性映射表（适配层 · 用户需求 1）。

把感知/记忆里的中文物体名 + category，映射成 SSP 的强类型闭集：
    EntityType + PhysicalObjectSubtype + 默认 StateSchema 物理状态。

记忆当前一个物理属性都没存（memory_objects 只有 category/location/seen_in_views/source），
所以这些"激活/抑制条件"必须靠本映射表补齐。

映射口径（DEMO 阶段）：
    杯子 / 碗            -> physical_object + fragile_object   （易碎，含液覆盖见下）
    剪刀 / 折叠刀 / 工具  -> physical_object + sharp_object     （割伤 hazard 源）
    咖啡机 / 家电         -> physical_object + electronic       （energy 保守置 none，不伪造通电/发热）
    香蕉 / 橙子 / 水果    -> physical_object + food_item
    茶盒 / 包装盒         -> physical_object + container
    方块                -> physical_object + other
    抽屉柜 / 家具        -> physical_object + furniture_support

含液覆盖表：杯 / 碗 在 DEMO 阶段写死 containment=open 且"视为含液"
    （HAS_LIQUID=True，memory_to_gp 据此补一个 substance 水节点 + contains 边）。
    ⚠️ 这是 DEMO 写死项：真机应由感知判断容器是否有内容物，见 docs/SSP_INTEGRATION.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ssp.ontology.entities import EntityType, PhysicalObjectSubtype
from ssp.ontology.schema import (
    ContainmentState,
    EnergyState,
    MotionState,
    OrientationState,
    StabilityState,
    StateSchema,
)


@dataclass(frozen=True)
class AttributeSpec:
    """一个物体映射后的完整属性规格。"""

    entity_type: EntityType
    subtype: PhysicalObjectSubtype | None
    # StateSchema 字段的默认值（enum 成员），构造 Node.attributes 用
    state: dict = field(default_factory=dict)
    # 是否"视为含液"（DEMO 写死；memory_to_gp 据此补 substance 水节点 + contains 边）
    has_liquid: bool = False


# ---------------------------------------------------------------------------
# 默认物理状态：桌面静止物体的保守基线
#   - stability=stable / orientation=upright / motion=static / energy=none
#   这些都是"观测到的中性状态"，不预设任何风险条件（不伪造 hot/tipping 等）。
# ---------------------------------------------------------------------------
def _base_state(**overrides) -> dict:
    base = {
        "stability": StabilityState.STABLE,
        "orientation": OrientationState.UPRIGHT,
        "motion": MotionState.STATIC,
        "energy": EnergyState.NONE,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 中文名关键词映射（优先级高于 category —— 更具体）
# key 为出现在物体名里的子串
# ---------------------------------------------------------------------------
NAME_KEYWORD_MAP: dict[str, AttributeSpec] = {
    "剪刀": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.SHARP, _base_state()),
    "折叠刀": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.SHARP, _base_state()),
    "刀": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.SHARP, _base_state()),
    "杯": AttributeSpec(
        EntityType.PHYSICAL_OBJECT,
        PhysicalObjectSubtype.FRAGILE,
        _base_state(containment=ContainmentState.OPEN),
        has_liquid=True,
    ),
    "碗": AttributeSpec(
        EntityType.PHYSICAL_OBJECT,
        PhysicalObjectSubtype.FRAGILE,
        _base_state(containment=ContainmentState.OPEN),
        has_liquid=True,
    ),
    "咖啡机": AttributeSpec(
        EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.ELECTRONIC, _base_state()
    ),
    "香蕉": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.FOOD_ITEM, _base_state()),
    "橙子": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.FOOD_ITEM, _base_state()),
    "茶盒": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.CONTAINER, _base_state()),
    "抽屉柜": AttributeSpec(
        EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.FURNITURE_SUPPORT, _base_state()
    ),
    "方块": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.OTHER, _base_state()),
}


# ---------------------------------------------------------------------------
# category 映射（关键词未命中时的兜底）
# memory_objects 里实际出现的 category：方块/杯子/碗/工具/水果/包装盒/家电/家具
# ---------------------------------------------------------------------------
CATEGORY_MAP: dict[str, AttributeSpec] = {
    "方块": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.OTHER, _base_state()),
    "杯子": AttributeSpec(
        EntityType.PHYSICAL_OBJECT,
        PhysicalObjectSubtype.FRAGILE,
        _base_state(containment=ContainmentState.OPEN),
        has_liquid=True,
    ),
    "碗": AttributeSpec(
        EntityType.PHYSICAL_OBJECT,
        PhysicalObjectSubtype.FRAGILE,
        _base_state(containment=ContainmentState.OPEN),
        has_liquid=True,
    ),
    "工具": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.SHARP, _base_state()),
    "水果": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.FOOD_ITEM, _base_state()),
    "包装盒": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.CONTAINER, _base_state()),
    "家电": AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.ELECTRONIC, _base_state()),
    "家具": AttributeSpec(
        EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.FURNITURE_SUPPORT, _base_state()
    ),
}


# 兜底：未知物体归为 physical_object + other，保守静止状态
_FALLBACK = AttributeSpec(EntityType.PHYSICAL_OBJECT, PhysicalObjectSubtype.OTHER, _base_state())


def resolve_spec(name: str, category: str | None = None) -> AttributeSpec:
    """给定中文物体名 + category，返回属性规格。

    匹配优先级：名字关键词（更具体） > category > 兜底 other。
    """
    for keyword, spec in NAME_KEYWORD_MAP.items():
        if keyword in name:
            return spec
    if category and category in CATEGORY_MAP:
        return CATEGORY_MAP[category]
    return _FALLBACK


def build_state_schema(spec: AttributeSpec) -> StateSchema:
    """把 AttributeSpec.state（dict）一次性构造成 StateSchema，触发 pydantic 校验/enum 强制。

    （不用 bench_adapter 里 setattr 逐字段赋值的写法 —— 那会绕过校验。）
    """
    return StateSchema(**spec.state)
