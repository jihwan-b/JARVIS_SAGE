"""verify_deploy.py — 배포 정합 검증. 학습 코드/stats 없이 .pt만 로드해 raw 피처로 동작 확인."""
import argparse, numpy as np, torch
import feature_spec as fs

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tools/compensator.pt")
args = ap.parse_args()

m = torch.jit.load(args.model); m.eval()

a    = np.array([10., -5., 12., -3., 2., 60.])
q    = a + np.array([0.8, -1.2, 1.5, -0.6, 0.4, 2.0])   # measured ≈ a + injected bias
qd   = np.zeros(6)
prev = a - np.array([1., 1., 1., 1., 1., 0.])
feat = fs.build_features(a, q, qdot_real=qd, prev_cmd=prev)
x = torch.from_numpy(feat).unsqueeze(0)

g_hat = m(x).detach().numpy()[0]
print("predicted gap :", np.round(g_hat, 3))
print("injected bias : [0.8, -1.2, 1.5, -0.6, 0.4, 2.0]")
print("corrected u   :", np.round(a - g_hat, 3))
print("input dim     :", x.shape[1], "== FEATURE_DIM", fs.FEATURE_DIM)
err = np.abs(g_hat - np.array([0.8,-1.2,1.5,-0.6,0.4,2.0])).max()
print("max abs error :", round(float(err),4), "-> PASS" if err < 0.1 else "-> FAIL")