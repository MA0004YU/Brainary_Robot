# simulation/src/memory/physics_dictionary.py
# src/memory/physics_dictionary.py

"""
本文件属于只读常识库。提供材料固有的理学常量（密度、均匀摩擦系数、屈服应力上限、拓扑结构形态等），
供 SceneBuilder 赋予刚体物理属性，以及供 PhysicsBoundaryDetectors 进行微观碰撞探针高频比对。
"""

PHYSICS_DICTIONARY = {

    # 异形薄板中空塑料收纳盒
    "plastic_box": {
        "semantic_category": "movable_object",  # 动力学属性：属于活动件，受重力影响，可被机械臂抓取或撞飞
        "structure_type": "hollow_open_top",  # 拓扑形态：无盖中空结构，触发 SceneBuilder 执行5面薄板拼接算法
        "wall_thickness_m": 0.005,  # 物理壁厚：5mm
        "density_kg_m3": 1200.0,  # 材料密度：标准 ABS/PP 塑料密度 (kg/m³)
        "friction_uniform": 0.45,  # 表面均匀摩擦系数 (无量纲)
        "yield_stress_N": 80.0  # 刚度损毁红线：外力挤压超过 80N 触发刚度损毁熔断探针
    },

    # 规整的木质收纳层板/立柱/挡板
    "wooden_rack": {
        "semantic_category": "environment_structure",  # 动力学属性：固定的静态环境，SAPIEN 将其编译为 Kinematic 钢印，雷打不动
        "structure_type": "solid",  # 拓扑形态：实心立方体结构，触发标准单一 Box 碰撞体构建
        "density_kg_m3": 700.0,  # 材料密度：规则硬木密度 (kg/m³)
        "friction_uniform": 0.55,  # 表面均匀摩擦系数（木质表面粗糙度较高）
        "yield_stress_N": 5000.0  # 破坏上限：环境件极难被肉体破坏
    },

    # 高密度实心规则铁块
    "iron_block": {
        "semantic_category": "movable_object",  # 动力学属性：活动件，通常作为压在塑料盒上方的重物或密集堆叠障碍物
        "structure_type": "solid",  # 拓扑形态：实心立方体结构
        "density_kg_m3": 7800.0,  # 材料密度：标准碳钢/纯铁密度 (kg/m³)，体积虽小但质量极大
        "friction_uniform": 0.20,  # 表面均匀摩擦系数（钢结构表面较为平滑）
        "yield_stress_N": 10000.0  # 破坏上限：极高，基本不可能发生材质损毁，通常只会触发死锁或倾倒
    }
}
