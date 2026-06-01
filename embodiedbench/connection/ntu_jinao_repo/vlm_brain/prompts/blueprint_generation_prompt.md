Generate an initial skill_blueprint JSON for the given task and scene.

Inputs:

- Task instruction.
- Optional scene image.
- Scene state JSON.
- Available primitive skills and logic.
- Supported performance metrics.
- Output schema and example blueprint.

Design rules:

- Keep the plan high-level. Do not output continuous control actions.
- Prefer a sequence of `parallel` approach, `descend`, `grasp`, condition check, `lift`, `parallel` transport, `place`, condition check, and `retreat` for pick-and-place tasks.
- Use conservative contact speeds for `descend` and `place`.
- Include useful `performance_query` fields for each skill and parallel node.
- Add terminal success and failure nodes.

Output:

Return only one valid skill_blueprint JSON object.
