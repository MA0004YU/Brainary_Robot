"""Custom command terms for constrained reach."""

from __future__ import annotations

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.utils import configclass


@configclass
class PositionReachCommandCfg(UniformPoseCommandCfg):
    """Position-only reach command with fixed orientation fields."""

