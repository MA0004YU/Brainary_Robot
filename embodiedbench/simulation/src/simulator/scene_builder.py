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

        ground_material = self.scene.create_physical_material(1.0, 1.0, 0.0)
        self.scene.add_ground(altitude=0.0, material=ground_material)

    def build_twin_world(self, perception_json: Dict[str, Any]) -> List[sapien.Actor]:
        instantiated_actors = []

        if "objects" not in perception_json:
            raise KeyError("感知数据结构缺失 'objects' 数组，无法构建三维环境。")

        for entity in perception_json["objects"]:
            entity_id = entity["name"]
            label = entity_id.rsplit("_", 1)[0]
            
            mat_props = self.physics_dict.get(label, self.physics_dict["plastic_box"])
            density = mat_props["density_kg_m3"]
            friction = mat_props["friction_uniform"]
            structure_type = mat_props.get("structure_type", "solid")
            
            rigid_material = self.scene.create_physical_material(friction, friction, 0.0)
            builder = self.scene.create_actor_builder()

            w, h, d = entity["size_whd"]
            hw, hh, hd = w / 2.0, h / 2.0, d / 2.0

            if structure_type == "hollow_open_top":
                t = mat_props["wall_thickness_m"]
                ht = t / 2.0
                builder.add_box_collision(half_size=[hw, hd, ht], pose=sapien.Pose([0, 0, -hh + ht]), material=rigid_material)
                builder.add_box_visual(half_size=[hw, hd, ht], pose=sapien.Pose([0, 0, -hh + ht]))
                builder.add_box_collision(half_size=[ht, hd, hh - ht], pose=sapien.Pose([-hw + ht, 0, ht]), material=rigid_material)
                builder.add_box_visual(half_size=[ht, hd, hh - ht], pose=sapien.Pose([-hw + ht, 0, ht]))
                builder.add_box_collision(half_size=[ht, hd, hh - ht], pose=sapien.Pose([hw - ht, 0, ht]), material=rigid_material)
                builder.add_box_visual(half_size=[ht, hd, hh - ht], pose=sapien.Pose([hw - ht, 0, ht]))
                builder.add_box_collision(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, -hd + ht, ht]), material=rigid_material)
                builder.add_box_visual(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, -hd + ht, ht]))
                builder.add_box_collision(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, hd - ht, ht]), material=rigid_material)
                builder.add_box_visual(half_size=[hw - 2 * ht, ht, hh - ht], pose=sapien.Pose([0, hd - ht, ht]))
                builder.set_mass_and_inertia_from_density(density)
            else:
                builder.add_box_collision(half_size=[hw, hh, hd], material=rigid_material)
                builder.add_box_visual(half_size=[hw, hh, hd])
                builder.set_mass_and_inertia_from_density(density)

            # 【漏洞修复】：区分环境静态钢印件与动态受力件
            if mat_props.get("semantic_category") == "environment_structure":
                actor = builder.build_kinematic(name=entity_id)
            else:
                actor = builder.build(name=entity_id)

            pose_data = entity["pose"]
            actor.set_pose(sapien.Pose(pose_data[:3], pose_data[3:]))
            instantiated_actors.append(actor)

        self._execute_gravity_settling(instantiated_actors)
        return instantiated_actors

    def _execute_gravity_settling(self, actors: List[sapien.Actor]):
        """强制高阻尼沉降，抹平视觉模块的 3D Z 轴悬空误差"""
        for actor in actors:
            if not actor.is_kinematic:
                actor.set_damping(self.config["settling_linear_damping"], self.config["settling_angular_damping"])

        for _ in range(self.config["settling_steps"]):
            self.scene.step()

        for actor in actors:
            if not actor.is_kinematic:
                actor.set_damping(0.0, 0.0)
