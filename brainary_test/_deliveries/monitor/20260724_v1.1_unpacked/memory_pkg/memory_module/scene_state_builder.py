"""
SceneStateBuilder: constructs spine brain scene_state dicts from perception outputs.

The spine brain (franka_state_machine_cerebellum) and VLM Brain both expect a
scene_state dict in this format:

    {
        "cube_pose":        [x, y, z, qw, qx, qy, qz],   # 7D pose
        "target_pose":      [x, y, z, qw, qx, qy, qz],   # 7D pose
        "ee_pose":          [x, y, z, qw, qx, qy, qz],   # end-effector pose
        "gripper_width":    float,                         # 0.0 – 1.0
        "robot_joint_pos":  [q0, q1, q2, q3, q4, q5, q6], # joint angles
        "step_index":       int,
    }

When only RGB perception is available (no SLAM or proprioception),
build_stub_scene_state() fills unknown poses with NaN so downstream code can
detect and handle missing fields gracefully.

When SLAM or Isaac Sim provides poses later, merge_with_sensor_data() overlays
real data onto the perception stub.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


_NAN_POSE_7D = [float("nan")] * 7
_NAN_JOINTS_7 = [float("nan")] * 7


def build_stub_scene_state(
    visible_objects: Sequence[str],
    step_index: int = 0,
    ee_pose: Optional[List[float]] = None,
    gripper_width: Optional[float] = None,
    joint_angles: Optional[List[float]] = None,
    extra_object_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Build a best-effort scene_state from perception-only information.

    Pose fields not available from perception are set to NaN lists.
    Downstream code should check for NaN before using a pose.

    Parameters
    ----------
    visible_objects      : object names detected by VisionPerceptionAdapter
    step_index           : current episode step index
    ee_pose              : end-effector pose from proprioception, or None
    gripper_width        : gripper aperture [0.0, 1.0] from proprioception, or None
    joint_angles         : joint angles from proprioception, or None
    extra_object_names   : additional object names to add NaN pose entries for
                           (e.g., ["cube", "target"] to guarantee standard keys exist)

    Returns
    -------
    dict conforming to spine brain scene_state format,
    with NaN for unknown pose fields.
    """
    state: Dict[str, Any] = {
        "step_index": step_index,
        "ee_pose": list(ee_pose) if ee_pose is not None else list(_NAN_POSE_7D),
        "gripper_width": float(gripper_width) if gripper_width is not None else float("nan"),
        "robot_joint_pos": list(joint_angles) if joint_angles is not None else list(_NAN_JOINTS_7),
        "visible_objects_from_perception": list(visible_objects),
    }

    # Add NaN pose entries for detected objects
    all_names = list(visible_objects) + list(extra_object_names or [])
    seen = set()
    for name in all_names:
        key = f"{name}_pose"
        if key not in seen:
            state[key] = list(_NAN_POSE_7D)
            seen.add(key)

    # Guarantee the two standard spine brain keys always exist
    for standard in ("cube", "target"):
        key = f"{standard}_pose"
        if key not in state:
            state[key] = list(_NAN_POSE_7D)

    return state


def merge_with_sensor_data(
    perception_state: Dict[str, Any],
    sensor_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Overlay real sensor data onto a perception-derived stub scene_state.

    Real data (from SLAM, Isaac Sim, or the robot controller) overwrites
    NaN stub values. Keys present only in sensor_data are added as-is.

    Parameters
    ----------
    perception_state : dict from build_stub_scene_state()
    sensor_data      : dict from SLAM / Isaac Sim / proprioception.
                       Any key whose value is not None overrides the stub.

    Returns
    -------
    Merged scene_state dict ready for memory.ingest_scene_state()
    """
    merged = dict(perception_state)
    for key, value in sensor_data.items():
        if value is not None:
            merged[key] = value
    return merged


def from_perception_results(
    perception_results: Sequence[Any],
    step_index: int = 0,
    ee_pose: Optional[List[float]] = None,
    gripper_width: Optional[float] = None,
    joint_angles: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Build a scene_state dict directly from RecognitionResult objects.

    Convenience wrapper that extracts visible object names from
    RecognitionResult.objects and calls build_stub_scene_state().

    Parameters
    ----------
    perception_results : List[RecognitionResult] from QwenVLMPerception
    step_index         : current episode step index
    ee_pose            : end-effector pose, or None
    gripper_width      : gripper aperture [0.0, 1.0], or None
    joint_angles       : joint angles, or None

    Returns
    -------
    scene_state dict
    """
    visible: List[str] = []
    seen: set = set()
    for r in perception_results:
        label = r.primary_label.strip()
        if label and label not in ("unknown", "") and label not in seen:
            visible.append(label)
            seen.add(label)
        for obj in r.objects:
            name = obj.get("name", "").strip()
            conf = float(obj.get("confidence", 0.0))
            if name and conf >= 0.40 and name not in seen:
                visible.append(name)
                seen.add(name)

    return build_stub_scene_state(
        visible_objects=visible,
        step_index=step_index,
        ee_pose=ee_pose,
        gripper_width=gripper_width,
        joint_angles=joint_angles,
        extra_object_names=["cube", "target"],
    )
