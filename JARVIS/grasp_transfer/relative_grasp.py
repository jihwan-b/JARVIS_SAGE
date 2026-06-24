"""
Cluster-supervised grasp transfer.

A grasp is annotated once per cluster representative. To transfer it to any other
tool in (or near) that cluster, the grasp point is stored in a **bbox-normalized,
principal-axis-aligned** frame, then mapped onto the target tool's geometry.

  1. PCA on the tool point cloud → principal axes (handle direction, etc.).
  2. Express the representative's grasp point in normalized [0,1]^3 bbox coords
     within that aligned frame.
  3. For a new tool, recover its aligned bbox and read the grasp back out at the
     same normalized coordinate.

Caveat: tools whose proportions differ a lot from the representative (e.g. a much
longer handle) can misalign — verify with verify_grasp before trusting it.
"""

from __future__ import annotations

import numpy as np


def principal_frame(points: np.ndarray):
    """Return (centroid, R) where R's columns are the principal axes."""
    c = points.mean(0)
    cov = np.cov((points - c).T)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]            # major → minor
    R = V[:, order]
    if np.linalg.det(R) < 0:              # keep right-handed
        R[:, -1] *= -1
    return c, R


def to_normalized(points: np.ndarray, grasp_world: np.ndarray):
    """Encode a world grasp point as normalized bbox coords in the aligned frame."""
    c, R = principal_frame(points)
    local = (points - c) @ R
    lo, hi = local.min(0), local.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    g_local = (grasp_world - c) @ R
    return (g_local - lo) / span          # in ~[0,1]^3


def from_normalized(points: np.ndarray, norm_coord: np.ndarray) -> np.ndarray:
    """Decode normalized bbox coords back to a world grasp point on a new tool."""
    c, R = principal_frame(points)
    local = (points - c) @ R
    lo, hi = local.min(0), local.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    g_local = lo + norm_coord * span
    return c + g_local @ R.T              # back to world


def transfer_grasp(rep_points, rep_grasp_world, new_points) -> np.ndarray:
    """Map a representative's grasp onto a new tool of the same cluster."""
    norm = to_normalized(rep_points, rep_grasp_world)
    return from_normalized(new_points, norm)
