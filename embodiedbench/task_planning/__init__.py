"""Task planning control loop aligned with repair/brain_api_example; pluggable robot body (real or Sapien)."""

from .planning_loop import TaskPlanningLoop
from .heuristic_policy import BrightnessAvoidancePolicy

# Import RepairWalkBody / SapienVelocityBody from submodules to avoid heavy deps at package import.

__all__ = ["TaskPlanningLoop", "BrightnessAvoidancePolicy"]
