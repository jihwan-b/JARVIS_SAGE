"""
Trajectory diversity analysis.

Quantifies how diverse a set of manipulation trajectories is, to test whether
RL-generated rollouts cover more of the behavior space than teleoperated demos.

Method:
  * Truncated path signatures (`iisignature`) as a reparameterization-invariant
    trajectory descriptor — invariant to speed/time-warping, which matters when
    comparing smooth RL motion against stepwise teleop.
  * **Per-trajectory L2 normalization** of the signature feature, giving a
    normalized linear signature kernel  K(i,j) = <s_i, s_j>.
  * Vendi score over K = effective number of distinct trajectories.
  * Geometric workspace-coverage as an interpretable fallback metric.

Important: normalization is *per trajectory*. Across-set standardization
inflates the score for near-identical sets and must not be used here.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

try:
    import iisignature
    _HAS_IISIG = True
except Exception:  # pragma: no cover
    _HAS_IISIG = False


# --------------------------------------------------------------------------- #
# Signature features
# --------------------------------------------------------------------------- #
def path_signature(traj: np.ndarray, depth: int = 3) -> np.ndarray:
    """Truncated path signature of a (T, D) trajectory up to `depth`."""
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 2:
        raise ValueError(f"trajectory must be (T>=2, D), got {traj.shape}")
    if not _HAS_IISIG:
        raise ImportError("iisignature is required: pip install iisignature")
    return iisignature.sig(traj, depth)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-vector L2 normalization."""
    n = np.linalg.norm(x)
    return x / (n + eps)


def signature_features(trajectories, depth: int = 3) -> np.ndarray:
    """Stack per-trajectory L2-normalized signatures into (N, sig_dim)."""
    feats = [l2_normalize(path_signature(t, depth)) for t in trajectories]
    return np.stack(feats, axis=0)


# --------------------------------------------------------------------------- #
# Vendi score
# --------------------------------------------------------------------------- #
def vendi_score(kernel: np.ndarray) -> float:
    """Effective number of distinct items via the Vendi score (exp-entropy of
    the normalized kernel eigenvalues). Identical sets → ~1.0."""
    n = kernel.shape[0]
    if n == 0:
        return 0.0
    K = kernel / n  # trace-normalize so eigenvalues sum to 1
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    w = w / w.sum()
    entropy = -np.sum(w * np.log(w))
    return float(np.exp(entropy))


def diversity_from_trajectories(trajectories, depth: int = 3) -> float:
    feats = signature_features(trajectories, depth)        # (N, d), L2-normalized
    kernel = feats @ feats.T                               # linear signature kernel
    return vendi_score(kernel)


# --------------------------------------------------------------------------- #
# Geometric coverage fallback
# --------------------------------------------------------------------------- #
def workspace_coverage(trajectories, bins: int = 12) -> float:
    """Fraction of an occupancy grid (over the first 3 columns) that the
    trajectory set visits. Interpretable companion to the Vendi score."""
    pts = np.concatenate([np.asarray(t)[:, :3] for t in trajectories], axis=0)
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    idx = ((pts - lo) / span * (bins - 1)).astype(int)
    occupied = {tuple(p) for p in idx}
    return len(occupied) / float(bins ** 3)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_csv_dir(path: str, columns) -> list:
    """Load every CSV in `path` as a trajectory, keeping `columns`."""
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    trajs = []
    for f in files:
        arr = np.genfromtxt(f, delimiter=",", names=True)
        cols = [arr[c] for c in columns]
        trajs.append(np.stack(cols, axis=1))
    return trajs


# --------------------------------------------------------------------------- #
# Self-test (monotone ground truth)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    rng = np.random.default_rng(0)

    def make_set(spread, n=24, T=40):
        base = np.linspace(0, 1, T)[:, None] * np.array([1.0, 1.0, 1.0])
        return [base + rng.normal(0, spread, base.shape) for _ in range(n)]

    identical = [np.linspace(0, 1, 40)[:, None] * np.ones(3)] * 24
    levels = {
        "identical": diversity_from_trajectories(identical),
        "narrow": diversity_from_trajectories(make_set(0.02)),
        "medium": diversity_from_trajectories(make_set(0.10)),
        "wide": diversity_from_trajectories(make_set(0.30)),
    }
    for k, v in levels.items():
        print(f"  {k:>10s}: Vendi = {v:.3f}")
    assert levels["identical"] < 1.05
    assert levels["identical"] < levels["narrow"] < levels["medium"] < levels["wide"]
    print("selftest OK — monotone diversity confirmed")


def main():
    ap = argparse.ArgumentParser(description="Trajectory diversity (signature + Vendi).")
    ap.add_argument("--rl-dir", type=str, default=None)
    ap.add_argument("--teleop-dir", type=str, default=None)
    ap.add_argument("--columns", nargs="+", default=["ee_x", "ee_y", "ee_z"])
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name, d in (("RL", args.rl_dir), ("teleop", args.teleop_dir)):
        if not d:
            continue
        trajs = load_csv_dir(d, args.columns)
        v = diversity_from_trajectories(trajs, args.depth)
        cov = workspace_coverage(trajs)
        print(f"[{name}] n={len(trajs)}  Vendi={v:.3f}  coverage={cov:.3f}")


if __name__ == "__main__":
    main()
