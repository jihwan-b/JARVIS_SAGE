"""
RSL-RL PPO training entry point for JarvisMultiTool-v0.

Usage (Windows-native Isaac Lab):
    isaaclab.bat -p rl/train.py --task JarvisMultiTool-v0 --num_envs 4096 --headless
    isaaclab.bat -p rl/train.py --task JarvisMultiTool-v0 --resume --tool new_tool --max_iter 100

`--resume` warm-starts from a converged checkpoint and runs a short adaptation
on a new tool's latent — the ~100-iteration adaptation behind Stage 2.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="JarvisMultiTool-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iter", type=int, default=2000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--tool", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs after the simulator is up -----------------------
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import jarvis_env  # noqa: F401  (registers JarvisMultiTool-v0)

PPO_CFG = {
    "num_steps_per_env": 24,
    "max_iterations": args.max_iter,
    "policy": {"class_name": "ActorCritic",
               "actor_hidden_dims": [256, 128, 64],
               "critic_hidden_dims": [256, 128, 64],
               "activation": "elu"},
    "algorithm": {"class_name": "PPO", "gamma": 0.99, "lam": 0.95,
                  "learning_rate": 1e-3, "entropy_coef": 0.005,
                  "num_learning_epochs": 5, "num_mini_batches": 4},
}


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    runner = OnPolicyRunner(env, PPO_CFG, log_dir="logs/rsl_rl/jarvis_tool",
                            device=args.device)
    if args.resume and args.checkpoint:
        runner.load(args.checkpoint)   # warm-start for new-tool adaptation
        print(f"warm-start from {args.checkpoint} → {args.max_iter} iters on '{args.tool}'")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
