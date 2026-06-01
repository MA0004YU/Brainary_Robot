# Connection Module

This folder stores the interface contract between the Brain module (VLM or any replacement planner) and the robot skill executor.

## Start Here

- `VLM_BRAIN_INTERFACE.md`: detailed I/O protocol, blueprint schema contract, skill-by-skill parameter semantics, and output template.

## Goal

Enable other Brain implementations to replace the current VLM generation stage while keeping the downstream skill execution stack unchanged.
