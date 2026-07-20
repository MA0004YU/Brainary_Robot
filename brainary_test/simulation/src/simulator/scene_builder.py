import os
import cv2
import random
import numpy as np
import sapien.core as sapien
import sapien.render
from typing import Dict, Any, List


class SceneBuilder:
    def __init__(self, config: Dict[str, Any], physics_dictionary: Dict[str, Any]):
        self.config = config["simulator_config"]
        self.physics_dict = physics_dictionary

        self.scene = sapien.Scene()
        self.scene.set_timestep(self.config["time_step"])

        # =================================================================
        # 🌞 物理级真实光影重构 (物理场景构建的核心，破除 Sim2Real 域间隙)
        # =================================================================
        # 1. 压暗全局底光，给阴影留出呈现的对比空间 (告别发灰的“均光室”)
        self.scene.set_ambient_light([0.1, 0.1, 0.1])

        # 2. 注入带物理阴影的主平行光 (模拟明亮的实验室顶灯)
        # SAPIEN 3.x 只需要 shadow=True 即可，底层会自动处理分辨率和视锥边界
        self.scene.add_directional_light(
            direction=[0.5, 0.5, -1.0],
            color=[0.9, 0.9, 0.9],
            shadow=True
        )

        # 3. 增加辅助补光灯 (模拟斜侧方的操作台台灯)
        self.scene.add_point_light(
            position=[0.0, 0.0, 1.0],
            color=[0.3, 0.3, 0.3],
            shadow=False
        )
        # =================================================================

        ground_material = self.scene.create_physical_material(1.0, 1.0, 0.0)
        self.scene.add_ground(altitude=0.0, material=ground_material)

    def build_twin_world(self, perception_json: Dict[str, Any]) -> List["Any"]:
        instantiated_actors = []
        # 🚀 维护一个已通过校验的实体状态字典，用于 3D NMS (非极大值抑制)
        accepted_entities_info = []

        if "objects" not in perception_json:
            raise KeyError("感知数据结构缺失 'objects' 数组，无法构建三维环境。")

        for entity in perception_json["objects"]:
            entity_name = entity.get("name", f"unknown_obj_{len(instantiated_actors)}")
            entity_label = entity.get("label", entity_name.rsplit("_", 1)[0])

            # -----------------------------------------------------------------
            # 🚀 Step 1: 提前解析位姿，用于物理排斥校验
            # -----------------------------------------------------------------
            if "coarse_pose" in entity:
                import scipy.spatial.transform as sst
                matrix_4x4 = np.array(entity["coarse_pose"])
                parsed_pos = matrix_4x4[:3, 3]
                quat_xyzw = sst.Rotation.from_matrix(matrix_4x4[:3, :3]).as_quat()
                parsed_quat = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            else:
                pose_data = entity.get("pose", {})
                if isinstance(pose_data, dict):
                    parsed_pos = pose_data.get("pos", pose_data.get("p", pose_data.get("position", [0.0, 0.0, 0.0])))
                    parsed_quat = pose_data.get("quat",
                                                pose_data.get("q", pose_data.get("quaternion", [1.0, 0.0, 0.0, 0.0])))
                else:
                    parsed_pos = pose_data[:3]
                    parsed_quat = pose_data[3:]

            pos_np = np.array(parsed_pos)

            # -----------------------------------------------------------------
            # 🚀 Step 2: 物理孪生沙盒防线 (跨语义 3D 绝对排他去重)
            # -----------------------------------------------------------------
            is_duplicate = False
            for prev_info in accepted_entities_info:
                dist = np.linalg.norm(pos_np - prev_info["pos"])
                # 只要物理中心距离小于 4cm，绝对是同一个物体，直接过滤，无视标签！
                if dist < 0.04:
                    is_duplicate = True
                    print(
                        f"🧹 [物理去重] 拒绝实例化 {entity_name} ({entity_label}): 它与 {prev_info['name']} ({prev_info['label']}) 发生了严重的物理空间重叠 (距离 {dist:.4f}m)！")
                    break

            if is_duplicate:
                continue

            # -----------------------------------------------------------------
            # 🚀 Step 3: 通过校验，正式构建刚体
            # -----------------------------------------------------------------
            print(f"👀 [Debug 视觉透视] 实例化刚体: {entity_name} | 映射物理字典标签: {entity_label}")

            # 结合已提取的物理物体属性（密度、摩擦力等）进行精确建模
            mat_props = self.physics_dict.get(entity_label, self.physics_dict.get("plastic_box", {}))
            density = mat_props.get("density_kg_m3", 1000.0)
            friction = mat_props.get("friction_uniform", 0.5)
            structure_type = mat_props.get("structure_type", "solid")

            rigid_material = self.scene.create_physical_material(friction, friction, 0.0)
            builder = self.scene.create_actor_builder()

            size_list = entity.get("size_whd", entity.get("size", [0.01, 0.01, 0.01]))
            w, h, d = size_list[0], size_list[1], size_list[2]
            hw, hh, hd = w / 2.0, h / 2.0, d / 2.0

            render_mat = sapien.render.RenderMaterial()
            render_mat.set_base_color([random.random(), random.random(), random.random(), 1.0])
            render_mat.set_roughness(0.5)

            if structure_type == "solid":
                builder.add_box_collision(half_size=[hw, hh, hd], material=rigid_material, density=density)
                builder.add_box_visual(half_size=[hw, hh, hd], material=render_mat)

            actor = builder.build(name=entity_name)
            actor.set_pose(sapien.Pose(parsed_pos, parsed_quat))

            instantiated_actors.append(actor)
            accepted_entities_info.append({
                "name": entity_name,
                "label": entity_label,
                "pos": pos_np
            })

        self._execute_gravity_settling(instantiated_actors)
        return instantiated_actors

    def _execute_gravity_settling(self, actors: List["Any"]):
        def get_dyn(ent):
            return next((c for c in ent.components if "RigidDynamicComponent" in type(c).__name__), None)

        for actor in actors:
            dyn = get_dyn(actor)
            if dyn:
                dyn.linear_damping = 0.5
                dyn.angular_damping = 0.5

        for _ in range(50):
            self.scene.step()

    def export_debug_snapshot(self, output_path="inputs/twin_world_debug.png"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cam = self.scene.add_camera(name="debug_cam", width=1280, height=720, fovy=1.0, near=0.01, far=10.0)
        cam.set_pose(sapien.Pose([0.55, 0.0, 0.70], [0.7071, 0.0, 0.7071, 0.0]))
        self.scene.update_render()
        cam.take_picture()
        rgba = cam.get_picture("Color")
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"📸 [Debug 探针] 脑内沙盒 3D 重建快照已保存至: {output_path}")
