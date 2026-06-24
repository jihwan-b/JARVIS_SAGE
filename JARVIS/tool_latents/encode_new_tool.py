"""
Encode a new (unseen) tool into the learned latent space.

Loads the frozen Uni3D encoder + the saved PCA model, and projects a new tool's
point cloud to a 64-d latent z_new. This z_new conditions the Stage-1 policy for
a short warm-start adaptation — no human demos required.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from build_tool_latents import encode_pointclouds, load_uni3d


def encode_new_tool(pc_path: str, pca_pkl: str, device: str = "cuda") -> np.ndarray:
    with open(pca_pkl, "rb") as f:
        m = pickle.load(f)
    pca, mean, std = m["pca"], m["mean"], m["std"]

    model = load_uni3d(device)
    emb = encode_pointclouds(model, [pc_path], device)   # (1, 1024)
    z = (emb - mean) / std
    return pca.transform(z)[0]                            # (64,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True)
    ap.add_argument("--pca", default="models/pca_model_dim64.pkl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = encode_new_tool(args.pointcloud, args.pca, args.device)
    print("z_new:", np.round(z, 3))
    if args.out:
        np.save(args.out, z)
        print("saved →", args.out)


if __name__ == "__main__":
    main()
