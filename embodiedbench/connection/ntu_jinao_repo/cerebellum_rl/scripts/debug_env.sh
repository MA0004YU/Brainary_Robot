#!/usr/bin/env bash
set -e
cd /home1/banghai/IsaacLab
./isaaclab.sh -p ntu_jinao_repo/cerebellum_rl/train_position_reach.py \
  --task Isaac-ConstrainedReach-Position-Franka-v0 \
  --num_envs 16

