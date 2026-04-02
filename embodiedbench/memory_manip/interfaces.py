"""
External I/O ports for the embodied manipulation stack.

Colleagues implement perception (vision tokens), simulation (forward models), and
monitoring (safety / progress checks). Stubs keep imports runnable before integration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PerceptionInterface(ABC):
    """
    Vision-backed perception. Expected feature layout: (T, P, D)
    T = temporal frames, P = patches (or tokens) per frame, D = channel dimension.

    Implementations should call `memory.on_perception_features` on the parent memory system
    or return tensors for the agent to route manually.
    """

    @abstractmethod
    def extract_features(
        self, observation: Any, *, meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Return array shaped (T, P, D) as float32 (or a subclass contract)."""


class SimulationInterface(ABC):
    """
    Forward simulation / mental model rollouts (contacts, object dynamics).

    Placeholder: wire your physics engine or learned world model here.
    """

    @abstractmethod
    def predict(
        self,
        state_summary: Dict[str, Any],
        action: Any,
        *,
        horizon: int = 1,
    ) -> Dict[str, Any]:
        """Return a predicted state or distribution summary."""


class MonitorInterface(ABC):
    """
    Runtime monitoring: success predicates, constraint violations, drift detection.

    Placeholder: connect proprioception, force torque, or task KPIs.
    """

    @abstractmethod
    def check(
        self,
        observation: Any,
        memory_snapshot: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Return (ok, optional_alert_message)."""


class NullPerception(PerceptionInterface):
    """Returns zeros (1,1,1) until a real encoder is available."""

    def extract_features(
        self, observation: Any, *, meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        return np.zeros((1, 1, 1), dtype=np.float32)


class NullSimulation(SimulationInterface):
    """Echoes the input action as a trivial prediction."""

    def predict(
        self,
        state_summary: Dict[str, Any],
        action: Any,
        *,
        horizon: int = 1,
    ) -> Dict[str, Any]:
        return {"echo_action": action, "horizon": horizon, "note": "null_simulation"}


class NullMonitor(MonitorInterface):
    """Always reports OK."""

    def check(
        self,
        observation: Any,
        memory_snapshot: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        return True, None
