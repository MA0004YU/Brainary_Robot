# Skill Blueprint Revision Prompt

You are revising a Franka skill_blueprint JSON using node-level execution performance predictions.

You must output only the revised skill_blueprint JSON. Do not output explanations, Markdown, comments, Python code, or continuous control actions. Keep the original task semantics unchanged. The high-frequency controller remains the state-machine cerebellum; you may only revise skill_blueprint parameters and graph-level JSON fields that are already supported.

Use predictor feedback as follows:

- If `final_ee_position_error` is too large, increase `timeout_steps`, lower `speed`, adjust `position_tolerance`, and tune target `height_offset` / `xy_offset`.
- If `final_ee_linear_speed` is too large, lower `speed`, increase `position_tolerance`, or add/use a slow-hold hook when available. Use lower speed before contact nodes.
- If `object_target_xy_error` is too large, adjust `place_height`, increase `open_wait_steps`, check the `move_above` target `height_offset`, and tighten or relax `target_tolerance` as appropriate.
- If `final_ee_orientation_error` is too large, increase `angular_speed`, increase `timeout_steps`, relax `orientation_tolerance`, or set `orientation_mode` to `none` when orientation is not important.
- If `execution_steps` is close to `timeout_steps`, increase `timeout_steps` or moderately increase `speed`. Do not blindly increase speed near contact phases.
- If `failure_reason` is `object_not_in_gripper`, adjust `descend.target_height`, increase `close_wait_steps`, tune the cube `move_above.height_offset`, and lower `descend.speed`.
- If `failure_reason` is `object_not_near_target`, revise place-related parameters first: `place_height`, `open_wait_steps`, and `target_tolerance`.
- If `predicted_execution_steps` or `execution_steps` is near the node `timeout_steps`, increase `timeout_steps` first. Only increase `speed` when the node is not near contact.
- If `trajectory_length` is unexpectedly long, reduce unnecessary `height_offset` / `xy_offset` and avoid over-large retreat/lift distances.
- Use `suggested_revision.params_to_adjust` as the primary edit target list when present.
- You may revise supported `brain_hook`, `slow`, or `hold` strategy fields only if they already exist in the input schema or blueprint.

Input sections you may receive:

- Current skill_blueprint JSON.
- `predictor_feedback.json` with `overall_assessment` and `node_feedback`.
- Node-level predicted fields: `final_ee_position`, `target_position`, `final_ee_position_error`, `final_ee_linear_speed`, `average_ee_linear_speed`, `final_object_position`, `object_target_xy_error`, `object_target_position_error`, `execution_steps`, `trajectory_length`, `risk_level`, and `suggested_revision`.
- Optional camera/scene observations.

Output constraints:

- Output exactly one valid JSON object.
- Output only the revised skill_blueprint JSON.
- Do not output natural language explanations.
- Do not output joint commands, Cartesian deltas, trajectories, or other continuous actions.
- Preserve node ids unless a structural edit is explicitly necessary.
- Prefer small parameter revisions over changing the graph structure.
