# simulation/src/simulator/scene_builder.py

import sapien.core as sapien
import numpy as np
from typing import Dict, Any, List


class SceneBuilder:
    def __init__(self, config: Dict[str, Any], physics_dictionary: Dict[str, Any]):
        self.config = config["simulator_config"]
        self.physics_dict = physics_dictionary

        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        self.scene = self.engine.create_scene()
        self.scene.set_timestep(self.config["time_step"])

        # 物理地面，摩擦力设为 1.0 防止物体滑动
        ground_material = self.scene.create_physical_material(1.0, 1.0, 0.0)
        self.scene.add_ground(altitude=0.0, material=ground_material)

    def build_twin_world(self, perception_json: Dict[str, Any]) -> List[sapien.Actor]:
        instantiated_actors = []

        for entity in perception_json["scene_entities"]:
            entity_id = entity["id"]
            label = entity_id.split("_")[0]

            mat_props = self.physics_dict.get(label, self.physics_dict["plastic_box"])
            density = mat_props["density_kg_m3"]
            friction = mat_props["friction_uniform"]
            structure_type = mat_props.get("structure_type", "solid")

            # 刚体约束：动静摩擦一致，恢复系数绝对为 0
            rigid_material = self.scene.create_physical_material(friction, friction, 0.0)
            builder = self.scene.create_actor_builder()

            if entity["type"] == "primitive_box":
                w, h, d = entity["size"]
                hw, hh, hd = w / 2.0, h / 2.0, d / 2.0

                if structure_type == "hollow_open_top":
                    t = mat_props["wall_thickness_m"]
                    ht = t / 2.0

                    # SAPIEN 5 面薄板拼接 (底面 + 四周墙壁)，确保碰撞体内部真实中空
                    builder.add_box_collision(half_size=[hw, hd, ht], pose=sapien.Pose([0, 0, -hh + ht]),
                                              material=rigid_material)
                    builder.add_box_visual(half_size=[hw, hd, ht], pose=sapien.Pose([0, 0, -hh + ht]))

                    builder.add_box_collision(half_size=[ht, hd, hh - ht], pose=sapien.Pose([-hw + ht, 0, ht]),
                                              material=rigid_material)
                    builder.add_box_visual(half_size=[ht, hd, hh - ht], pose=sapien.Pose([-hw + ht, 0, ht]))

                    builder.add_box_collision(half_size=[ht, hd, hh - ht], pose=sapien.Pose([hw - ht, 0, ht]),
                                              material=rigid_material)
                    builder.add_box_visual(half_size=[ht, hd, hh - ht], pose=sapien.Pose([hw - ht, 0, ht]))

                    builder.add_box_collision(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, -hd + ht, ht]),
                                              material=rigid_material)
                    builder.add_box_visual(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, -hd + ht, ht]))

                    builder.add_box_collision(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, hd - ht, ht]),
                                              material=rigid_material)
                    builder.add_box_visual(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, hd - ht, ht]))

                    # 根据复合几何体和密度推算动力学惯性张量
                    builder.set_mass_and_inertia_from_density(density)
                else:
                    builder.add_box_collision(half_size=[hw, hh, hd], material=rigid_material)
                    builder.add_box_visual(half_size=[hw, hh, hd])
                    builder.set_mass_and_inertia_from_density(density)

            elif entity["type"] == "reconstructed_mesh":
                mesh_path = entity["mesh_path"]
                # 载入凹共形网格文件
                builder.add_collision_from_file(filename=mesh_path, material=rigid_material)
                builder.add_visual_from_file(filename=mesh_path)

            pose_data = entity["pose"]
            actor = builder.build(name=entity_id)
            actor.set_pose(sapien.Pose(pose_data["pos"], pose_data["quat"]))

            if entity["type"] == "reconstructed_mesh":
                actor.set_kinematic(True)  # 收纳柜等环境件固定死

            instantiated_actors.append(actor)

        # 重力沉降消穿模
        self._execute_gravity_settling(instantiated_actors)
        return instantiated_actors

    def _execute_gravity_settling(self, actors: List[sapien.Actor]):
        """真实阻尼挂载与物理步进"""
        for actor in actors:
            if not actor.is_kinematic:
                actor.set_damping(self.config["settling_linear_damping"], self.config["settling_angular_damping"])

        for _ in range(self.config["settling_steps"]):
            self.scene.step()

        for actor in actors:
            if not actor.is_kinematic:
                actor.set_damping(0.0, 0.0)  # 沉降完毕，恢复自由动力学状态
