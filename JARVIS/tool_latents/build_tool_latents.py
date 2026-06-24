"""
Build the tool latent space.

Encodes a directory of tool point clouds with a **pretrained, frozen Uni3D**
encoder (1024-d), then fits a PCA projection down to 64-d. 64-d is the chosen
operating point: it retains ~79.5% of the variance while staying small enough
that the latent does not dominate the ~32-d RL observation when concatenated.

Outputs:
    tool_latents.npy        (N, 64)   PCA-reduced latents
    tool_names.json                   ordered names aligned to rows
    pca_model_dim64.pkl               {'pca', 'mean', 'std'} for new-tool encoding
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np

PCA_DIM = 64


def load_uni3d(device: str = "cuda"):
    """Load the pretrained Uni3D encoder in eval/frozen mode.

    Replace with the Uni3D checkpoint path used in the project. The encoder is
    never trained — only used to produce 1024-d geometry embeddings.
    """
    import torch
    from uni3d import create_uni3d  # project-local Uni3D wrapper

    model = create_uni3d(pretrained=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_pointclouds(model, pc_paths, device: str = "cuda") -> np.ndarray:
    """Encode each point cloud → (N, 1024) Uni3D embedding."""
    import torch
    import trimesh

    embs = []
    with torch.no_grad():
        for path in pc_paths:
            mesh = trimesh.load(path, force="mesh")
            pts = mesh.sample(8192) if hasattr(mesh, "sample") else np.asarray(mesh.vertices)
            pts = _normalize_unit_sphere(np.asarray(pts, dtype=np.float32))
            xyz = torch.from_numpy(pts)[None].to(device)
            feat = model.encode_pc(xyz)            # (1, 1024)
            embs.append(feat.squeeze(0).cpu().numpy())
    return np.stack(embs, axis=0)


def _normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-9
    return pts / scale


def fit_pca(emb: np.ndarray, dim: int = PCA_DIM):
    """Standardize then PCA to `dim`. Returns (latents, pca, mean, std)."""
    from sklearn.decomposition import PCA

    mean = emb.mean(0, keepdims=True)
    std = emb.std(0, keepdims=True) + 1e-9
    z = (emb - mean) / std
    pca = PCA(n_components=dim, random_state=0)
    latents = pca.fit_transform(z)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"PCA {emb.shape[1]} → {dim}d, variance retained = {var:.3f}")
    return latents, pca, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointclouds", required=True, help="dir of tool meshes/point clouds")
    ap.add_argument("--out", default="models/")
    ap.add_argument("--dim", type=int, default=PCA_DIM)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.pointclouds, "*")))
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    model = load_uni3d(args.device)
    emb = encode_pointclouds(model, paths, args.device)
    latents, pca, mean, std = fit_pca(emb, args.dim)

    np.save(os.path.join(args.out, "tool_latents.npy"), latents)
    with open(os.path.join(args.out, "tool_names.json"), "w") as f:
        json.dump(names, f, indent=2)
    with open(os.path.join(args.out, f"pca_model_dim{args.dim}.pkl"), "wb") as f:
        pickle.dump({"pca": pca, "mean": mean, "std": std}, f)
    print(f"saved {latents.shape} latents for {len(names)} tools → {args.out}")


if __name__ == "__main__":
    main()
