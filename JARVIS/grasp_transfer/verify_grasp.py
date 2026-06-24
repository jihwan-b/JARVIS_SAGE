"""
Verify a transferred grasp point before using it in RL.

Checks that the proposed grasp (a) lies close to the tool surface and (b) sits
near the tool's principal (handle) axis — the two failure modes when transferring
a grasp across tools with different proportions.
"""

from __future__ import annotations

import numpy as np

from relative_grasp import principal_frame


def surface_distance(points: np.ndarray, grasp: np.ndarray) -> float:
    """Distance from the grasp point to the nearest surface point."""
    return float(np.min(np.linalg.norm(points - grasp, axis=1)))


def axis_offset(points: np.ndarray, grasp: np.ndarray) -> float:
    """Perpendicular distance from the grasp to the major principal axis."""
    c, R = principal_frame(points)
    axis = R[:, 0]
    v = grasp - c
    return float(np.linalg.norm(v - np.dot(v, axis) * axis))


def verify(points: np.ndarray, grasp: np.ndarray,
           surf_thr: float = 0.01, axis_thr: float = 0.03):
    sd, ao = surface_distance(points, grasp), axis_offset(points, grasp)
    ok = sd <= surf_thr and ao <= axis_thr
    return ok, {"surface_dist": sd, "axis_offset": ao}


if __name__ == "__main__":
    import sys
    pts = np.load(sys.argv[1])        # (N,3) tool point cloud
    g = np.load(sys.argv[2])          # (3,) candidate grasp
    ok, info = verify(pts, g)
    print(("OK  " if ok else "FAIL"), info)
