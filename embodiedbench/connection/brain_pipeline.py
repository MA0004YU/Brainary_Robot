"""
Brain-Spine pipeline orchestrator.

Connects three-layer memory → VLM Brain → spine brain (franka_state_machine_cerebellum).

Two-phase design (Isaac Lab must run as a separate process):

  Phase 1  prepare  — memory context export + VLM generation → Blueprint JSON
  Phase 2  record   — read Isaac Lab result → update memory

─────────────────────────────────────────────────────────────────────────────
CLI usage
─────────────────────────────────────────────────────────────────────────────

# Phase 1: generate Blueprint before handing off to Isaac Lab
python brain_pipeline.py prepare \\
    --task "pick up the cube and place it on the target" \\
    --scene_state path/to/scene_state.json \\
    --output_dir path/to/run_001 \\
    --vlm_config path/to/vlm_config.json \\
    [--image path/to/rgb.png] \\
    [--dry_run true] \\
    [--use_predictor true] \\
    [--predictor_checkpoint path/to/ckpt.pth]

# Hand Blueprint to Isaac Lab (run separately in Isaac Lab environment):
#   ./isaaclab.sh -p run_skill_blueprint_executor.py \\
#       --blueprint_path path/to/run_001/generated_blueprint.json \\
#       --output_dir path/to/run_001/isaac_output \\
#       --num_episodes 1

# Phase 2: record Isaac Lab result back into memory
python brain_pipeline.py record \\
    --task "pick up the cube and place it on the target" \\
    --blueprint path/to/run_001/generated_blueprint.json \\
    --success true \\
    --vlm_config path/to/vlm_config.json \\
    [--scene_state path/to/run_001/final_scene_state.json] \\
    [--isaac_output path/to/run_001/isaac_output]

─────────────────────────────────────────────────────────────────────────────
Python API usage
─────────────────────────────────────────────────────────────────────────────

    from embodiedbench.connection.brain_pipeline import BrainSpinePipeline

    pipeline = BrainSpinePipeline(vlm_config_path="vlm_config.json")

    # Phase 1
    blueprint_path = pipeline.prepare(
        task="pick up the cube and place it on the target",
        scene_state={"cube_pose": [...], "target_pose": [...], ...},
        output_dir="run_001",
    )

    # → run Isaac Lab externally with blueprint_path →

    # Phase 2
    pipeline.record(
        task="pick up the cube and place it on the target",
        blueprint_path=blueprint_path,
        success=True,
        scene_state=final_scene_state,
    )
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Path bootstrap: make memory_manip and vlm_brain importable from anywhere
# ─────────────────────────────────────────────────────────────────────────────
_CONNECTION_DIR = Path(__file__).resolve().parent          # embodiedbench/connection/
_EMBODIED_ROOT = _CONNECTION_DIR.parent.parent             # Brainary_Robot/
_JINAO_REPO = _CONNECTION_DIR / "ntu_jinao_repo"           # .../ntu_jinao_repo/
_VLM_DIR = _JINAO_REPO / "vlm_brain"                      # .../ntu_jinao_repo/vlm_brain/

for _p in [str(_EMBODIED_ROOT), str(_JINAO_REPO), str(_VLM_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
# Imports (after path setup)
# ─────────────────────────────────────────────────────────────────────────────
from embodiedbench.memory_manip import EmbodiedManipulationMemorySystem, MemorySystemConfig
from run_vlm_inference import build_generation_prompt, _read_json, _write_json, _str_to_bool
from blueprint_parser import extract_json_from_text
from blueprint_validator import validate_blueprint


# ─────────────────────────────────────────────────────────────────────────────
# BrainSpinePipeline
# ─────────────────────────────────────────────────────────────────────────────

class BrainSpinePipeline:
    """
    Orchestrates memory ↔ VLM Brain ↔ spine brain for one task.

    Initialize once per session; reuse across multiple tasks/episodes.
    Memory state accumulates automatically across calls to prepare/record.
    """

    def __init__(
        self,
        vlm_config_path: str | Path,
        memory_config: Optional[MemorySystemConfig] = None,
    ) -> None:
        self.vlm_config = _read_json(Path(vlm_config_path))
        self.memory = EmbodiedManipulationMemorySystem(memory_config or MemorySystemConfig())

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: generate Blueprint
    # ─────────────────────────────────────────────────────────────────────────

    def prepare(
        self,
        task: str,
        scene_state: Dict[str, Any],
        output_dir: str | Path,
        image_path: Optional[str] = None,
        dry_run: bool = False,
        use_predictor: bool = False,
        predictor_checkpoint: Optional[str] = None,
    ) -> Path:
        """
        Export memory context + run VLM Brain → write Blueprint JSON.

        Parameters
        ----------
        task                 : natural language task instruction
        scene_state          : spine brain scene_state dict
                               {cube_pose, target_pose, ee_pose, gripper_width, robot_joint_pos}
        output_dir           : directory for all outputs of this run
        image_path           : optional RGB image path for the VLM
        dry_run              : skip VLM call, copy example blueprint instead
        use_predictor        : run predictor feedback + revision loop after initial generation
        predictor_checkpoint : path to trained predictor .pth file (uses mock if absent)

        Returns
        -------
        Path to the final Blueprint JSON ready for Isaac Lab.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Persist scene_state for traceability
        _write_json(output_dir / "scene_state.json", scene_state)

        # Start episode lifecycle so working memory is properly initialized
        self.memory.reset_episode(scene_id=str(output_dir.name))
        self.memory.begin_task(task)

        # Ingest scene into memory working layer (robot_state + observation)
        self.memory.ingest_scene_state(scene_state)

        # Export memory context → JSON file (atomic write)
        # Reads from semantic + episodic memory accumulated across past episodes
        memory_context = self.memory.export_vlm_context(
            task, output_dir / "memory_context.json"
        )
        print(f"[prepare] memory context → {output_dir / 'memory_context.json'}")

        # Build prompt with memory context
        prompt = build_generation_prompt(task, scene_state, memory_context)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        blueprint_path = output_dir / "generated_blueprint.json"

        if dry_run:
            blueprint_path = self._dry_run_blueprint(output_dir, blueprint_path)
        else:
            blueprint_path = self._generate_blueprint(
                task, scene_state, image_path, prompt, output_dir, blueprint_path
            )

        # Optional predictor-based revision loop
        if use_predictor and not dry_run:
            blueprint_path = self._run_predictor_loop(
                blueprint_path, scene_state, output_dir, predictor_checkpoint
            )

        print(f"[prepare] Blueprint ready → {blueprint_path}")
        return blueprint_path

    def _dry_run_blueprint(self, output_dir: Path, blueprint_path: Path) -> Path:
        import shutil
        example = _VLM_DIR / "examples" / "sample_blueprint.json"
        if example.exists():
            shutil.copyfile(example, blueprint_path)
            print(f"[dry_run] Example blueprint copied to {blueprint_path}")
        else:
            print(f"[dry_run] Warning: example blueprint not found at {example}")
        return blueprint_path

    def _generate_blueprint(
        self,
        task: str,
        scene_state: Dict[str, Any],
        image_path: Optional[str],
        prompt: str,
        output_dir: Path,
        blueprint_path: Path,
    ) -> Path:
        from qwen3vl_loader import generate_response, load_qwen3vl_model

        model, processor = load_qwen3vl_model(self.vlm_config)
        raw = generate_response(model, processor, image_path, prompt, self.vlm_config)
        (output_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

        parse_result = extract_json_from_text(raw)
        if parse_result.extracted_text:
            (output_dir / "parsed_blueprint.json").write_text(
                parse_result.extracted_text + "\n", encoding="utf-8"
            )

        if not parse_result.ok or parse_result.blueprint is None:
            raise RuntimeError(f"VLM output parse failed: {parse_result.error}\nRaw:\n{raw[:500]}")

        validation = validate_blueprint(parse_result.blueprint)
        _write_json(output_dir / "validation_report.json", validation.to_dict())
        if not validation.valid:
            raise RuntimeError(f"Blueprint validation failed: {validation.errors}")

        _write_json(blueprint_path, parse_result.blueprint)
        return blueprint_path

    def _run_predictor_loop(
        self,
        blueprint_path: Path,
        scene_state: Dict[str, Any],
        output_dir: Path,
        predictor_checkpoint: Optional[str],
    ) -> Path:
        from predictor_bridge import build_predictor_feedback

        ckpt = Path(predictor_checkpoint).expanduser() if predictor_checkpoint else None
        use_mock = ckpt is None or not ckpt.exists()
        if ckpt and not ckpt.exists():
            print(f"[predictor] Checkpoint not found, using mock: {ckpt}")

        feedback = build_predictor_feedback(
            blueprint_path=blueprint_path,
            checkpoint_path=ckpt,
            scene_state=scene_state,
            use_mock=use_mock,
        )
        _write_json(output_dir / "predictor_feedback.json", feedback)

        high_risk = feedback.get("overall_assessment", {}).get("high_risk_nodes", [])
        if not high_risk:
            print("[predictor] No high-risk nodes — Blueprint accepted as-is")
            return blueprint_path

        print(f"[predictor] High-risk nodes: {high_risk} — requesting revision from VLM")
        revised_path = self._revise_blueprint(
            blueprint_path, feedback, scene_state, output_dir
        )
        return revised_path

    def _revise_blueprint(
        self,
        blueprint_path: Path,
        feedback: Dict[str, Any],
        scene_state: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        from qwen3vl_loader import generate_response, load_qwen3vl_model
        from run_vlm_inference import _read_text

        blueprint = _read_json(blueprint_path)
        revision_prompt_template = _read_text(_VLM_DIR / "prompts" / "blueprint_revision_prompt.md")
        prompt = (
            f"{revision_prompt_template}\n\n"
            f"Original skill_blueprint:\n{json.dumps(blueprint, indent=2, ensure_ascii=False)}\n\n"
            f"Predictor feedback:\n{json.dumps(feedback, indent=2, ensure_ascii=False)}\n\n"
            f"Scene state:\n{json.dumps(scene_state, indent=2, ensure_ascii=False)}\n"
        )
        (output_dir / "revision_prompt.txt").write_text(prompt, encoding="utf-8")

        model, processor = load_qwen3vl_model(self.vlm_config)
        raw = generate_response(model, processor, None, prompt, self.vlm_config)
        (output_dir / "revision_raw_response.txt").write_text(raw, encoding="utf-8")

        parse_result = extract_json_from_text(raw)
        if not parse_result.ok or parse_result.blueprint is None:
            print(f"[predictor] Revision parse failed — keeping original blueprint")
            return blueprint_path

        validation = validate_blueprint(parse_result.blueprint)
        if not validation.valid:
            print(f"[predictor] Revised blueprint invalid {validation.errors} — keeping original")
            return blueprint_path

        revised_path = output_dir / "revised_blueprint.json"
        _write_json(revised_path, parse_result.blueprint)
        print(f"[predictor] Revised Blueprint → {revised_path}")
        return revised_path

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: record execution result back to memory
    # ─────────────────────────────────────────────────────────────────────────

    def record(
        self,
        task: str,
        blueprint_path: str | Path,
        success: bool,
        scene_state: Optional[Dict[str, Any]] = None,
        isaac_output_dir: Optional[str | Path] = None,
    ) -> None:
        """
        Feed spine brain execution result back into memory.

        Extracts the executed skill sequence from the Blueprint, then calls
        memory.record_blueprint_execution() to update TaskSchema so that the
        next call to prepare() can reference a known-good skill ordering.

        Parameters
        ----------
        task              : same instruction string used in prepare()
        blueprint_path    : path to the Blueprint JSON that was executed
        success           : whether the task was completed successfully
        scene_state       : optional final scene_state dict from Isaac Lab
        isaac_output_dir  : if provided, attempt to read success/metrics from
                            Isaac Lab's episode_summary.json automatically
        """
        blueprint = _read_json(Path(blueprint_path))
        skills = _extract_skill_sequence(blueprint)

        # Try to read actual success from Isaac Lab output if directory provided
        if isaac_output_dir is not None:
            isaac_success = _read_isaac_success(Path(isaac_output_dir))
            if isaac_success is not None:
                success = isaac_success
                print(f"[record] success overridden from Isaac Lab output: {success}")

        # Close the episode: creates a full EpisodeRecord in episodic memory,
        # attaches blueprint_skills, computes importance score, and triggers
        # batch generalization to semantic memory every N episodes.
        # This is what makes query_vlm_context() → similar_episodes work over time.
        self.memory.end_episode(success=success, blueprint_skills=skills)

        # Also call record_blueprint_execution for immediate semantic TaskSchema update
        # (end_episode batches generalization every N episodes; this ensures the
        # recommended skill sequence is available for the very next prepare() call)
        self.memory.record_blueprint_execution(
            task_instruction=task,
            blueprint_skills=skills,
            success=success,
            scene_state=scene_state,
        )
        print(f"[record] task='{task[:60]}' skills={skills} success={success} → memory updated")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_skill_sequence(blueprint: Dict[str, Any]) -> List[str]:
    """
    Walk the execution_graph from 'start', following the happy path
    (next / if_true links), and collect skill names in execution order.
    """
    graph = blueprint.get("execution_graph", {})
    nodes = graph.get("nodes", {})
    start = graph.get("start", "")

    skills: List[str] = []
    visited = set()
    node_id = start

    while node_id and node_id not in visited:
        visited.add(node_id)
        node = nodes.get(node_id)
        if node is None:
            break
        ntype = node.get("type")
        if ntype == "skill":
            skill_name = node.get("skill", "")
            if skill_name:
                skills.append(skill_name)
            node_id = node.get("next", "")
        elif ntype == "condition":
            # Follow happy path (if_true)
            node_id = node.get("if_true", "")
        elif ntype == "parallel":
            # Record sub-skills inside parallel node
            for goal in (node.get("goals") or {}).values():
                if isinstance(goal, dict):
                    sub_skill = goal.get("skill", "")
                    if sub_skill:
                        skills.append(sub_skill)
            node_id = node.get("next", "")
        else:
            # terminal or unknown — stop
            break

    return skills


def _read_isaac_success(isaac_output_dir: Path) -> Optional[bool]:
    """Try to parse success from Isaac Lab's episode output files."""
    for candidate in ["episode_summary.json", "summary.json", "metrics.json"]:
        p = isaac_output_dir / candidate
        if p.exists():
            try:
                data = _read_json(p)
                if "success" in data:
                    return bool(data["success"])
                if "task_success" in data:
                    return bool(data["task_success"])
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli_prepare(args: argparse.Namespace) -> None:
    scene_state = _read_json(Path(args.scene_state))
    pipeline = BrainSpinePipeline(args.vlm_config)
    pipeline.prepare(
        task=args.task,
        scene_state=scene_state,
        output_dir=args.output_dir,
        image_path=args.image,
        dry_run=_str_to_bool(args.dry_run),
        use_predictor=_str_to_bool(args.use_predictor),
        predictor_checkpoint=args.predictor_checkpoint,
    )


def _cli_record(args: argparse.Namespace) -> None:
    scene_state = _read_json(Path(args.scene_state)) if args.scene_state else None
    pipeline = BrainSpinePipeline(args.vlm_config)
    pipeline.record(
        task=args.task,
        blueprint_path=args.blueprint,
        success=_str_to_bool(args.success),
        scene_state=scene_state,
        isaac_output_dir=args.isaac_output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brain-Spine pipeline: memory ↔ VLM Brain → Blueprint → spine brain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vlm_config", required=True,
                        help="Path to VLM model config JSON (e.g. vlm_brain/configs/qwen3vl_config.json).")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── prepare ──────────────────────────────────────────────────────────────
    p_prepare = sub.add_parser(
        "prepare",
        help="Phase 1: export memory context + generate Blueprint JSON.",
    )
    p_prepare.add_argument("--task", required=True,
                           help="Natural language task instruction.")
    p_prepare.add_argument("--scene_state", required=True,
                           help="Path to scene_state.json.")
    p_prepare.add_argument("--output_dir", required=True,
                           help="Output directory for all generated files.")
    p_prepare.add_argument("--image", default=None,
                           help="Optional path to RGB image for the VLM.")
    p_prepare.add_argument("--dry_run", default="false",
                           help="Skip VLM call and use example blueprint (true/false).")
    p_prepare.add_argument("--use_predictor", default="false",
                           help="Run predictor feedback + revision loop (true/false).")
    p_prepare.add_argument("--predictor_checkpoint", default=None,
                           help="Path to trained predictor .pth file. Mock used if absent.")

    # ── record ───────────────────────────────────────────────────────────────
    p_record = sub.add_parser(
        "record",
        help="Phase 2: record spine brain execution result back to memory.",
    )
    p_record.add_argument("--task", required=True,
                          help="Same task instruction used in prepare.")
    p_record.add_argument("--blueprint", required=True,
                          help="Path to the Blueprint JSON that was executed.")
    p_record.add_argument("--success", required=True,
                          help="Whether execution succeeded (true/false).")
    p_record.add_argument("--scene_state", default=None,
                          help="Optional path to final scene_state.json from Isaac Lab.")
    p_record.add_argument("--isaac_output", default=None,
                          help="Optional path to Isaac Lab output directory (reads success automatically).")

    args = parser.parse_args()
    if args.command == "prepare":
        _cli_prepare(args)
    elif args.command == "record":
        _cli_record(args)


if __name__ == "__main__":
    main()
