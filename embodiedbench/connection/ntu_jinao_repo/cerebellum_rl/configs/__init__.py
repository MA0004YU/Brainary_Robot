"""Configs for cerebellum RL local tasks."""

from .franka_position_reach_cfg import FrankaPositionReachEnvCfg
from .rsl_rl_ppo_cfg import FrankaPositionReachPPORunnerCfg

__all__ = ["FrankaPositionReachEnvCfg", "FrankaPositionReachPPORunnerCfg"]

