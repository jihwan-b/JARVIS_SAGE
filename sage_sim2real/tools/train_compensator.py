"""train_compensator.py — Phase 3 Task C: train the residual compensator and JIT-export it.

Small MLP (default 24->128->128->6) trained with MSE on the per-joint gap target produced by the
corrected `sage_to_training.py`. Inputs/outputs are normalized using the stats baked into train.npz.
Normalization is folded INTO the scripted module, so the exported `compensator.pt` takes a RAW
feature row in `.pos` units and returns a RAW gap prediction in `.pos` units — deploy needs only the
`.pt`, no stats files and no Isaac Sim.

Run (env C):
  python tools/train_compensator.py --data-dir runs/bong_v2 --epochs 200 --out runs/bong_v2/compensator.pt

Reports per-joint RMSE of the gap BEFORE correction (raw gap magnitude) vs AFTER (gap minus model
prediction), mirroring SAGE's per-joint metric format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import feature_spec as fs


class Compensator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...],
                 x_mean, x_std, y_mean, y_std):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        xn = (x_raw - self.x_mean) / self.x_std
        yn = self.net(xn)
        return yn * self.y_std + self.y_mean


def per_joint_rmse(residual: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(residual ** 2, axis=0))


def main() -> None:
    p = argparse.ArgumentParser(description="Train residual compensator (Phase 3 Task C, fixed)")
    p.add_argument("--data-dir", type=Path, default=Path("runs/bong_v2"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/bong_v2/compensator.pt"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = np.load(args.data_dir / "train.npz", allow_pickle=True)
    va = np.load(args.data_dir / "val.npz", allow_pickle=True)

    meta = json.loads(str(tr["meta"]))
    if meta["feature_dim"] != fs.FEATURE_DIM:
        raise RuntimeError(
            f"feature_spec mismatch: dataset built with feature_dim={meta['feature_dim']} but current "
            f"feature_spec has feature_dim={fs.FEATURE_DIM}. Rebuild the dataset or revert the spec."
        )

    Xt = torch.from_numpy(tr["X"].astype(np.float32))
    Yt = torch.from_numpy(tr["Y"].astype(np.float32))
    Xv = torch.from_numpy(va["X"].astype(np.float32))
    Yv = torch.from_numpy(va["Y"].astype(np.float32))

    dev = torch.device(args.device)
    model = Compensator(
        in_dim=fs.FEATURE_DIM, out_dim=fs.TARGET_DIM, hidden=tuple(args.hidden),
        x_mean=tr["x_mean"], x_std=tr["x_std"], y_mean=tr["y_mean"], y_std=tr["y_std"],
    ).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, Yt), batch_size=args.batch_size, shuffle=True)

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv.to(dev)), Yv.to(dev)).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  val_mse={vloss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- per-joint RMSE report (SAGE metric format): before vs after correction ----
    model.eval()
    with torch.no_grad():
        pred_v = model(Xv.to(dev)).cpu().numpy()
    Yv_np = Yv.numpy()
    rmse_before = per_joint_rmse(Yv_np)             # raw gap magnitude
    rmse_after = per_joint_rmse(Yv_np - pred_v)     # gap the model fails to explain

    print("\nper-joint gap RMSE (.pos units: arm deg, gripper 0..100)")
    print(f"{'joint':<16}{'before':>10}{'after':>10}{'improvement':>14}")
    for j, name in enumerate(fs.DEPLOY_JOINT_ORDER):
        b, a = rmse_before[j], rmse_after[j]
        imp = (1.0 - a / b) * 100.0 if b > 1e-9 else 0.0
        print(f"{name:<16}{b:>10.3f}{a:>10.3f}{imp:>13.1f}%")

    # ---- JIT export (deploy-ready; normalization folded in) ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.freeze(torch.jit.trace(model_cpu, torch.zeros(1, fs.FEATURE_DIM)))
    scripted.save(str(args.out))

    report = {
        "val_mse_best": best_val,
        "rmse_before": {n: float(rmse_before[j]) for j, n in enumerate(fs.DEPLOY_JOINT_ORDER)},
        "rmse_after": {n: float(rmse_after[j]) for j, n in enumerate(fs.DEPLOY_JOINT_ORDER)},
        "feature_dim": fs.FEATURE_DIM,
        "hidden": args.hidden,
        "dataset_meta": meta,
    }
    with open(args.out.parent / "compensator_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved TorchScript -> {args.out}")
    print(f"saved report     -> {args.out.parent / 'compensator_report.json'}")


if __name__ == "__main__":
    main()
