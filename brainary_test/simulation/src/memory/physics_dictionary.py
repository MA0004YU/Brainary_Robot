import os

# ---------------------------------------------------------------------
# 原始物理先验配置 (结合了质量、摩擦力、受力极限与几何体积先验)
# ★ 只登记【当前 brainary_test 场景真实存在的交互物体】(7 个可抓物 + 3 个 place 篮子)。
#   字典的 key 会被拼成 GroundingDINO 的检测提示词(get_dino_prior_prompt),所以这里 = 场景物体全集,
#   一一对应、无幽灵项(原 red/blue/green cube + yellow box 是沙盒作者的旧 cube 测试场景物,本场景没有,已删)。
# ---------------------------------------------------------------------
_RAW_DICT = {
    # -----------------------------------------------------------------
    # brainary_test 场景实际物体（key = GroundingDINO 检测提示词，须是自然语言短语）。
    # 物理值为【工程估计】(mass/friction/virtual_pain_limit 单位: kg / 无量纲 / N)，
    # 精确标定应由 simulation 作者按真实资产细调。size_bounds 为体积消歧的粗过滤上下限(m³)。
    # 对应关系: blue mug=Prop_SM_Mug_D1, yellow mug=Prop_SM_Mug_C1, banana=Prop_011_banana,
    #           orange=Prop_orange_01, scissors=Prop_037_scissors,
    #           cracker box=Prop_003_cracker_box, meat can=Prop_010_potted_meat_can。
    # -----------------------------------------------------------------
    "blue mug": {          # 蓝色陶瓷马克杯：中等重量、握持稳、能承受夹持
        "mass": 0.15, "friction": 0.7, "virtual_pain_limit": 60.0,
        "size_bounds": {"min_vol_m3": 0.0001, "max_vol_m3": 0.005}
    },
    "yellow mug": {        # 黄色陶瓷马克杯：同上
        "mass": 0.15, "friction": 0.7, "virtual_pain_limit": 60.0,
        "size_bounds": {"min_vol_m3": 0.0001, "max_vol_m3": 0.005}
    },
    "banana": {            # 香蕉：软、易压伤 -> 痛觉阈值很低，夹持力必须小
        "mass": 0.12, "friction": 0.6, "virtual_pain_limit": 15.0,
        "size_bounds": {"min_vol_m3": 0.00005, "max_vol_m3": 0.002}
    },
    "orange": {            # 橙子：球形水果，稍软
        "mass": 0.15, "friction": 0.6, "virtual_pain_limit": 25.0,
        "size_bounds": {"min_vol_m3": 0.0001, "max_vol_m3": 0.003}
    },
    "scissors": {          # 剪刀：金属、薄、体积小、较硬
        "mass": 0.08, "friction": 0.5, "virtual_pain_limit": 50.0,
        "size_bounds": {"min_vol_m3": 0.00001, "max_vol_m3": 0.001}
    },
    "cracker box": {       # 饼干盒：纸盒、轻、可压瘪
        "mass": 0.4, "friction": 0.6, "virtual_pain_limit": 30.0,
        "size_bounds": {"min_vol_m3": 0.0002, "max_vol_m3": 0.006}
    },
    "meat can": {          # 肉罐头：金属罐、较重、耐夹
        "mass": 0.35, "friction": 0.6, "virtual_pain_limit": 80.0,
        "size_bounds": {"min_vol_m3": 0.00005, "max_vol_m3": 0.002}
    },
    # -----------------------------------------------------------------
    # 分拣篮子(place 目标):KLT 塑料筐,是 place 动作的【容器】,必须建模否则"放入篮子"没有容器可验。
    # 颜色对应仿真: green basket=Prop_KLT_1(绿) / blue basket=Prop_KLT_2(蓝) /
    #              purple basket=Prop_KLT_3(原色,GPT/DINO 视作紫红)。
    # 物理:塑料筐较重、稳固、耐夹,体积上下限放宽(点云易粘连,AABB 会偏大)。
    # 若 DINO 对 'basket' 召回差,可改成 'bin'/'crate'/'container' 试(它是长方塑料筐)。
    # -----------------------------------------------------------------
    "green basket": {      # 绿色 KLT 筐 = Prop_KLT_1
        "mass": 0.5, "friction": 0.7, "virtual_pain_limit": 200.0,
        "size_bounds": {"min_vol_m3": 0.001, "max_vol_m3": 0.05}
    },
    "blue basket": {       # 蓝色 KLT 筐 = Prop_KLT_2
        "mass": 0.5, "friction": 0.7, "virtual_pain_limit": 200.0,
        "size_bounds": {"min_vol_m3": 0.001, "max_vol_m3": 0.05}
    },
    "purple basket": {     # 原色 KLT 筐 = Prop_KLT_3(GPT 感知视作紫)
        "mass": 0.5, "friction": 0.7, "virtual_pain_limit": 200.0,
        "size_bounds": {"min_vol_m3": 0.001, "max_vol_m3": 0.05}
    }
}

# =====================================================================
# 🚀 自动兼容双格式的运行时字典
# =====================================================================
PHYSICS_DICTIONARY = {}
for k, v in _RAW_DICT.items():
    PHYSICS_DICTIONARY[k] = v
    PHYSICS_DICTIONARY[k.replace(" ", "_")] = v

def get_physics_properties(object_label: str) -> dict:
    clean_label = object_label.lower().strip().replace(".", "")
    if clean_label in PHYSICS_DICTIONARY:
        limit = PHYSICS_DICTIONARY[clean_label]["virtual_pain_limit"]
        print(f"[Physics_Dict] 🎯 精确命中先验: '{clean_label}' (虚拟痛觉阈值: {limit}N)")
        return PHYSICS_DICTIONARY[clean_label]

    print(f"[Physics_Dict] ⚠️ 未知物体侵入: '{object_label}'，启动兜底防御属性")
    return {
        "mass": 0.1,
        "friction": 0.8,
        "virtual_pain_limit": 100.0,
        "size_bounds": {"min_vol_m3": 1e-6, "max_vol_m3": 1.0}
    }

def get_dino_prior_prompt() -> str:
    # 提示词 = 当前场景全部交互物体(7 可抓物 + 3 篮子),给 GroundingDINO 逐一检测。
    return " . ".join(_RAW_DICT.keys()) + " ."
