"""
Cluster the tool latent space and pick K representatives.

K-means over the PCA-64 tool latents. Each cluster is scored on:
  * tightness  — mean distance of members to the centroid (lower = tighter)
  * purity     — dominant tool-family fraction within the cluster (higher = purer)

Loose / impure clusters are dropped; the remaining cluster centroids' nearest
real tools become the **representatives** that get a hand-specified grasp. This
gives O(K) grasp annotation instead of O(N).

The default distance threshold tends to yield ~11 raw clusters; keeping the 8
tight+pure ones gives the K=8 representative set used by the policy.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans


def cluster(latents: np.ndarray, k: int, seed: int = 0):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(latents)
    return km.labels_, km.cluster_centers_


def cluster_quality(latents, labels, centers, families=None):
    """Per-cluster (mean_dist_to_centroid, purity)."""
    out = {}
    for c in range(centers.shape[0]):
        members = latents[labels == c]
        if len(members) == 0:
            out[c] = (np.inf, 0.0)
            continue
        mean_dist = float(np.linalg.norm(members - centers[c], axis=1).mean())
        if families is not None:
            fam = np.asarray(families)[labels == c]
            _, counts = np.unique(fam, return_counts=True)
            purity = float(counts.max() / counts.sum())
        else:
            purity = 1.0
        out[c] = (mean_dist, purity)
    return out


def select_representatives(latents, labels, centers, quality,
                           dist_thr: float, purity_thr: float = 0.6):
    """Keep tight+pure clusters; representative = member nearest its centroid."""
    reps, kept = [], []
    for c, (md, pur) in quality.items():
        if md <= dist_thr and pur >= purity_thr:
            idx = np.where(labels == c)[0]
            nearest = idx[np.argmin(np.linalg.norm(latents[idx] - centers[c], axis=1))]
            reps.append(int(nearest))
            kept.append(c)
    return reps, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help="tool_latents.npy")
    ap.add_argument("--names", default=None, help="tool_names.json")
    ap.add_argument("--families", default=None, help="json: row→family label")
    ap.add_argument("--k", type=int, default=11, help="raw K-means clusters")
    ap.add_argument("--dist-thr", type=float, default=78.0)
    ap.add_argument("--purity-thr", type=float, default=0.6)
    ap.add_argument("--out", default="models/representatives.json")
    args = ap.parse_args()

    latents = np.load(args.latents)
    names = json.load(open(args.names)) if args.names else list(range(len(latents)))
    families = json.load(open(args.families)) if args.families else None

    labels, centers = cluster(latents, args.k)
    q = cluster_quality(latents, labels, centers, families)
    reps, kept = select_representatives(latents, labels, centers, q,
                                        args.dist_thr, args.purity_thr)

    print(f"raw clusters: {args.k}  →  kept: {len(kept)} (target K=8)")
    for c, (md, pur) in sorted(q.items()):
        flag = "keep" if c in kept else "drop"
        print(f"  C{c:<2d} dist={md:6.2f} purity={pur:4.2f}  [{flag}]")

    rep_names = [names[i] for i in reps]
    json.dump({"representatives": reps, "names": rep_names}, open(args.out, "w"), indent=2)
    print("representatives:", rep_names)


if __name__ == "__main__":
    main()
