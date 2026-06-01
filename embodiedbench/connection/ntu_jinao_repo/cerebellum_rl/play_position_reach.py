"""Play/evaluate Stage 1.1 constrained Franka position reach checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# --- local path bootstrap ---
CURRENT_FILE = Path(__file__).resolve()
CEREBELLUM_ROOT = CURRENT_FILE.parent
NTU_REPO_ROOT = CEREBELLUM_ROOT.parent
if str(NTU_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(NTU_REPO_ROOT))

import cerebellum_rl.register_tasks  # noqa: F401


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL args compatible with Isaac Lab play script."""
    group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    group.add_argument("--experiment_name", type=str, default=None, help="Experiment folder name.")
    group.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
    group.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint.")
    group.add_argument("--load_run", type=str, default=None, help="Run folder for resume.")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file for resume.")
    group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module.",
    )
    group.add_argument("--log_project_name", type=str, default=None, help="Logger project name.")


def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Update RSL-RL cfg from CLI args."""
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg

parser = argparse.ArgumentParser(description="Play constrained position reach with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=300, help="Video length.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of envs.")
parser.add_argument("--task", type=str, default="Isaac-ConstrainedReach-Position-Franka-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Real-time play.")
parser.add_argument(
    "--no_target_vis",
    action="store_true",
    default=False,
    help="Disable target/current pose markers in GUI.",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=60,
    help="Print success/error metrics every N environment steps.",
)
add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from cerebellum_rl.constrained_reach.utils import (  # noqa: E402
    get_ee_and_target_position_env,
    get_stage1_orientation_terms,
    get_stage1_orientation_tolerance,
    get_stage1_position_tolerance,
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    enable_target_vis = not args_cli.no_target_vis
    if enable_target_vis:
        env_cfg.commands.ee_pose.debug_vis = True

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if enable_target_vis:
        env.unwrapped.command_manager.set_debug_vis(True)
        print("[INFO] Target/current pose markers enabled for play visualization.")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    step_dt = env.unwrapped.step_dt
    steps = 0
    episode_success_count = 0
    episode_count = 0
    robot_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
    command_name = "ee_pose"

    def _tolerance_success_mask(base_env) -> tuple[torch.Tensor, ...]:
        ee_pos, target_pos = get_ee_and_target_position_env(base_env, robot_cfg, command_name)
        pos_tol = get_stage1_position_tolerance(base_env)
        pos_error_vec = target_pos - ee_pos
        pos_error = torch.norm(pos_error_vec, dim=1)
        normalized_pos_error = torch.norm(pos_error_vec / torch.clamp(pos_tol, min=1e-6), dim=1)
        _, orientation_error, _, _, quat_dot_abs = get_stage1_orientation_terms(base_env, robot_cfg, command_name)
        ori_tol = get_stage1_orientation_tolerance(base_env)
        normalized_ori_error = orientation_error / torch.clamp(ori_tol[:, 0], min=1e-6)
        position_success = normalized_pos_error < 1.0
        orientation_success = normalized_ori_error < 1.0
        pose_success = position_success & orientation_success
        return (
            pos_error,
            normalized_pos_error,
            position_success,
            orientation_error,
            normalized_ori_error,
            orientation_success,
            pose_success,
            quat_dot_abs,
        )

    def _print_play_metrics(tag: str) -> None:
        base_env = env.unwrapped
        (
            pos_error,
            normalized_pos_error,
            position_success,
            orientation_error,
            normalized_ori_error,
            orientation_success,
            pose_success,
            quat_dot_abs,
        ) = _tolerance_success_mask(base_env)
        pos_tol = get_stage1_position_tolerance(base_env)
        mean_pos_error = pos_error.mean().item()
        mean_tol = pos_tol[:, 0].mean().item()
        print(
            f"[PLAY][{tag}] mean_position_error={mean_pos_error:.4f} m | "
            f"mean_position_tolerance={mean_tol:.4f} m | "
            f"mean_normalized_position_error={normalized_pos_error.mean().item():.4f} | "
            f"position_success_rate={position_success.float().mean().item():.4f} | "
            f"mean_orientation_error={orientation_error.mean().item():.4f} rad | "
            f"mean_normalized_orientation_error={normalized_ori_error.mean().item():.4f} | "
            f"orientation_success_rate={orientation_success.float().mean().item():.4f} | "
            f"pose_success_rate={pose_success.float().mean().item():.4f} | "
            f"mean_quat_dot_abs={quat_dot_abs.mean().item():.4f} | "
            f"min_quat_dot_abs={quat_dot_abs.min().item():.4f} | "
            f"episodes={episode_count} | episode_success_rate={episode_success_count / max(episode_count, 1):.4f}"
        )

    print("[INFO] Play success criterion: normalized_position_error < 1.0 and normalized_orientation_error < 1.0")
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
        steps += 1

        if steps == 1 or (args_cli.log_interval > 0 and steps % args_cli.log_interval == 0):
            if "log" in extras and "stage1/mean_position_error" in extras["log"]:
                def _scalar(x):
                    return x.item() if isinstance(x, torch.Tensor) else x

                mean_err = _scalar(extras["log"]["stage1/mean_position_error"])
                mean_tol = _scalar(extras["log"].get("stage1/mean_position_tolerance", 0.0))
                norm_err = _scalar(extras["log"].get("stage1/mean_normalized_position_error", 0.0))
                pos_succ = _scalar(extras["log"].get("stage1/position_success_rate", 0.0))
                ori_err = _scalar(extras["log"].get("stage1/mean_orientation_error", 0.0))
                norm_ori_err = _scalar(extras["log"].get("stage1/mean_normalized_orientation_error", 0.0))
                ori_succ = _scalar(extras["log"].get("stage1/orientation_success_rate", 0.0))
                pose_succ = _scalar(extras["log"].get("stage1/pose_success_rate", 0.0))
                quat_dot_mean = _scalar(extras["log"].get("stage1/mean_quat_dot_abs", 0.0))
                quat_dot_min = _scalar(extras["log"].get("stage1/min_quat_dot_abs", 0.0))
                print(
                    f"[PLAY][step {steps}] mean_position_error={mean_err:.4f} m | "
                    f"mean_position_tolerance={mean_tol:.4f} m | "
                    f"mean_normalized_position_error={norm_err:.4f} | "
                    f"position_success_rate={pos_succ:.4f} | "
                    f"mean_orientation_error={ori_err:.4f} rad | "
                    f"mean_normalized_orientation_error={norm_ori_err:.4f} | "
                    f"orientation_success_rate={ori_succ:.4f} | "
                    f"pose_success_rate={pose_succ:.4f} | "
                    f"mean_quat_dot_abs={quat_dot_mean:.4f} | "
                    f"min_quat_dot_abs={quat_dot_min:.4f}"
                )
            else:
                _print_play_metrics(f"step {steps}")

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            base_env = env.unwrapped
            _, _, _, _, _, _, pose_success, _ = _tolerance_success_mask(base_env)
            for env_id in done_ids.tolist():
                episode_count += 1
                if pose_success[env_id]:
                    episode_success_count += 1
            print(
                f"[PLAY][episode done] env_ids={done_ids.tolist()} | "
                f"cumulative_episode_success_rate={episode_success_count / max(episode_count, 1):.4f}"
            )

        if args_cli.video and steps >= args_cli.video_length:
            break
        sleep_time = step_dt - (time.time() - start)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

