"""Local gym task registration for cerebellum RL tasks."""

from __future__ import annotations

import gymnasium as gym

TASK_ID = "Isaac-ConstrainedReach-Position-Franka-v0"


def register() -> None:
    """Register local constrained-reach task if not already registered."""
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "cerebellum_rl.configs.franka_position_reach_cfg:FrankaPositionReachEnvCfg",
            "rsl_rl_cfg_entry_point": "cerebellum_rl.configs.rsl_rl_ppo_cfg:FrankaPositionReachPPORunnerCfg",
        },
        disable_env_checker=True,
    )


register()

