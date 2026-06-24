"""
Convert RL policy rollouts into a LeRobot v3 dataset for SmolVLA fine-tuning.

The tool-conditioned policy is rolled out in jarvis_env to collect ~500 episodes
(~20 min), each with two camera streams and the joint state/action. The dataset
is written in LeRobot v3 format and pushed to the HuggingFace Hub.

Camera keys (must match the SmolVLA fine-tune rename_map):
    observation.images.wrist      → camera1
    observation.images.external   → camera2
"""

from __future__ import annotations

import argparse

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,)},
    "action": {"dtype": "float32", "shape": (6,)},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3)},
    "observation.images.external": {"dtype": "video", "shape": (480, 640, 3)},
}


def build_dataset(rollouts: list, repo_id: str, fps: int = 30, task: str = "pick up the tool"):
    """Write episodes to a LeRobot v3 dataset.

    Args:
        rollouts: list of episodes, each a dict with keys
            state (T,6), action (T,6), wrist (T,H,W,3), external (T,H,W,3).
    """
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=FEATURES,
                               use_videos=True)
    for ep in rollouts:
        T = len(ep["action"])
        for t in range(T):
            ds.add_frame({
                "observation.state": np.asarray(ep["state"][t], np.float32),
                "action": np.asarray(ep["action"][t], np.float32),
                "observation.images.wrist": ep["wrist"][t],
                "observation.images.external": ep["external"][t],
            }, task=task)
        ds.save_episode()
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="dir of rollout .npz episodes")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. davekim0323/jarvis-screwdriver-v1")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    import glob, os
    eps = []
    for f in sorted(glob.glob(os.path.join(args.rollouts, "*.npz"))):
        d = np.load(f)
        eps.append({k: d[k] for k in ("state", "action", "wrist", "external")})

    ds = build_dataset(eps, args.repo, args.fps)
    print(f"built {len(eps)} episodes → {args.repo}")
    if args.push:
        ds.push_to_hub()
        print("pushed to HuggingFace Hub")


if __name__ == "__main__":
    main()
