import os

# ---------------------------------------------------------------------
# 原始物理先验配置 (结合了质量、摩擦力、受力极限与几何体积先验)
# ---------------------------------------------------------------------
_RAW_DICT = {
    "red cube": {
        "mass": 0.1,
        "friction": 0.8,
        "virtual_pain_limit": 40.0,
        "size_bounds": {
            "min_vol_m3": 0.00001,  # 远小于积木的碎片直接丢弃
            "max_vol_m3": 0.0005    # 把大号物体误认为积木直接拦截
        }
    },
    "blue cube": {
        "mass": 0.1,
        "friction": 0.8,
        "virtual_pain_limit": 40.0,
        "size_bounds": {
            "min_vol_m3": 0.00001,
            "max_vol_m3": 0.0005
        }
    },
    "green cube": {
        "mass": 0.1,
        "friction": 0.8,
        "virtual_pain_limit": 40.0,
        "size_bounds": {
            "min_vol_m3": 0.00001,
            "max_vol_m3": 0.0005
        }
    },
    "yellow box": {
        "mass": 0.5,
        "friction": 0.8,
        "virtual_pain_limit": 200.0,
        "size_bounds": {
            "min_vol_m3": 0.0001,
            # 🚀 盘子体积上限放宽到 0.2，防止因为点云粘连导致的体积计算异常
            # 这是一个通用框架应该具备的鲁棒性：允许感知有误差，但严禁感知彻底离谱
            "max_vol_m3": 0.2
        }
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
    # 咒语将变成: "red cube . blue cube . green cube . yellow box ."
    return " . ".join(_RAW_DICT.keys()) + " ."
