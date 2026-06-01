"""Environment config for Stage 1.1 constrained Franka position reach."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_CFG

from .commands import PositionReachCommandCfg
from .markers import TARGET_MARKER_CFG
from . import position_reach_mdp as mdp


@configclass
class PositionReachSceneCfg(InteractiveSceneCfg):
    """Scene with table + Franka for constrained position reach."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    """Command terms."""

    ee_pose = PositionReachCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(12.0, 12.0),  # episode-level command for stable tolerance-conditioned learning
        debug_vis=False,  # reduce training overhead; can be enabled for debug runs
        ranges=PositionReachCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.30, 0.30),
            pos_z=(0.20, 0.55),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-0.5, 0.5),
        ),
        goal_pose_visualizer_cfg=TARGET_MARKER_CFG,
        current_pose_visualizer_cfg=TARGET_MARKER_CFG.replace(prim_path="/Visuals/CerebellumRL/current"),
    )


@configclass
class ActionsCfg:
    """Action specs."""

    arm_action: ActionTerm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.02,
        use_zero_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specs."""

    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(
            func=mdp.constrained_reach_obs,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms."""

    stage1_reward = RewTerm(
        func=mdp.stage1_position_reach_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]), "command_name": "ee_pose"},
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.position_success,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "command_name": "ee_pose",
        },
    )
    safety_abort = DoneTerm(
        func=mdp.nan_or_inf_abort,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )


@configclass
class FrankaPositionReachEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1.1 Position Reach config."""

    scene: PositionReachSceneCfg = PositionReachSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.0, 2.5, 2.4)
        self.viewer.lookat = (0.55, 0.0, 0.35)

