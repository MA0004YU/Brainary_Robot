"""Marker helpers for constrained reach task."""

from __future__ import annotations

from isaaclab.markers.config import FRAME_MARKER_CFG


TARGET_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/CerebellumRL/target")
TARGET_MARKER_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

