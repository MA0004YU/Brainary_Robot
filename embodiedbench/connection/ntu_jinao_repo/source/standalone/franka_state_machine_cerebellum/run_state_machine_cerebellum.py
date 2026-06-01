"""Run stage-1 Franka state-machine cerebellum data collection.

Example:
    ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_state_machine_cerebellum.py \
        --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
        --num_episodes 20 \
        --headless \
        --output_dir ./datasets/state_machine_stage1 \
        --save_trajectory true
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


from bootstrap_paths import bootstrap_isaaclab_paths

bootstrap_isaaclab_paths(__file__)

from isaaclab.app import AppLauncher


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


parser = argparse.ArgumentParser(description="Franka state-machine cerebellum dataset generator.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0", help="Isaac Lab task id.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to generate.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./datasets/state_machine_stage1",
    help="Directory for episodes.jsonl, summary.json, and trajectories.",
)
parser.add_argument(
    "--skill_plan",
    type=str,
    default=None,
    help="Path to JSON skill plan. Defaults to this module's configs/example_skill_plan.json.",
)
parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible episode generation.")
parser.add_argument("--save_trajectory", type=_str_to_bool, default=True, help="Whether to save trajectory JSONL files.")
parser.add_argument("--max_episode_steps", type=int, default=3000, help="Maximum state-machine steps per episode.")
parser.add_argument(
    "--max_delta_pos",
    type=float,
    default=0.08,
    help="Maximum per-step IK-Rel position delta before env action scaling (raw action units).",
)
parser.add_argument(
    "--max_delta_rot",
    type=float,
    default=0.20,
    help="Maximum per-step IK-Rel orientation delta in axis-angle radians.",
)
parser.add_argument(
    "--target_mode",
    type=str,
    default="custom_tabletop",
    choices=["custom_tabletop", "official_command"],
    help="Target source: sampled tabletop target or official Lift command target.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

module_dir = Path(__file__).resolve().parent
skill_plan_path = Path(args_cli.skill_plan).expanduser() if args_cli.skill_plan else module_dir / "configs/example_skill_plan.json"

from skill_interface import SkillPlan  # noqa: E402

try:
    parsed_skill_plan = SkillPlan.from_json(skill_plan_path)
except Exception as exc:
    parser.error(f"Failed to load --skill_plan '{skill_plan_path}': {exc}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from logger import StateMachineDatasetLogger, SummaryAccumulator  # noqa: E402
from state_machine_cerebellum import StateMachineCerebellum  # noqa: E402


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser()

    skill_plan = parsed_skill_plan
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    # Disable env timeout so the state machine controls episode length.
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        if args_cli.target_mode == "custom_tabletop":
            # The Lift command marker is an airborne lift target and is not the tabletop placement goal.
            env_cfg.commands.object_pose.debug_vis = False
    # Extend episode horizon for pick + place.
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        args_cli.max_episode_steps * float(env_cfg.sim.dt) * int(env_cfg.decimation),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    target_marker = None
    if args_cli.target_mode == "custom_tabletop":
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/StateMachineTabletopTarget"
        marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        target_marker = VisualizationMarkers(marker_cfg)
        target_marker.set_visibility(True)
    logger = StateMachineDatasetLogger(output_dir=output_dir, save_trajectory=args_cli.save_trajectory)
    summary = SummaryAccumulator()

    try:
        for episode_idx in range(args_cli.num_episodes):
            episode_id = f"episode_000{episode_idx:03d}" if episode_idx < 1000 else f"episode_{episode_idx:06d}"
            episode_seed = args_cli.seed + episode_idx
            plan_for_episode = skill_plan.with_episode_id(episode_id)
            cerebellum = StateMachineCerebellum(
                env=env,
                skill_plan=plan_for_episode,
                device=env.unwrapped.device,
                logger=logger,
                env_id=args_cli.task,
                target_mode=args_cli.target_mode,
                seed=args_cli.seed,
                max_episode_steps=args_cli.max_episode_steps,
                max_delta_pos=args_cli.max_delta_pos,
                max_delta_rot=args_cli.max_delta_rot,
                target_marker=target_marker,
            )
            episode = cerebellum.execute_episode(episode_id=episode_id, episode_seed=episode_seed)
            summary.add_episode(episode)
            print(
                f"[INFO] {episode_id}: success={episode['final_result']['success']} "
                f"reason={episode['final_result']['failure_reason']} "
                f"steps={sum(item['num_steps'] for item in episode['execution_trace'])}"
            )

        summary_dict = summary.to_dict()
        logger.write_summary(summary_dict)
        print("[INFO] Dataset generation summary:")
        print(f"  total episodes: {summary_dict['total_episodes']}")
        print(f"  total success rate: {summary_dict['success_rate']:.3f}")
        print(f"  pick success rate: {summary_dict['pick_success_rate']:.3f}")
        print(f"  place success rate: {summary_dict['place_success_rate']:.3f}")
        print(f"  average steps: {summary_dict['average_steps']:.1f}")
        print(f"  failure reason counts: {summary_dict['failure_reason_counts']}")
        print(f"[INFO] Wrote dataset to: {output_dir.resolve()}")
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] State-machine cerebellum failed: {exc}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
