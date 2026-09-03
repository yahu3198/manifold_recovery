"""Evaluate gates G1-G4 (spec Section 8) and write runs/spike_report.md."""
import _common  # noqa: F401
import argparse
import time

import numpy as np
from scipy.stats import spearmanr

from manifold_recovery.config import load
from manifold_recovery.scenario import ZONES, SPIKE_START_POSE, SPIKE_ZONE_ID
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.score.obstacles import DockField
from manifold_recovery.certify.surrogate import certify_batch, alpha_bar_policy
from manifold_recovery.certify.cluster import side_signature
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.features.condition import Standardizer
from manifold_recovery.analysis.plots import plot_trajectories

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--ckpt", default=str(_common.ROOT / "runs/ckpt.pt"))
p.add_argument("--skip-planner", action="store_true",
               help="skip G3 optimality vs B1 and G4 (fast smoke)")
a = p.parse_args()

import torch  # noqa: E402  (torch required for gate evaluation)
from manifold_recovery.model.train import load as load_ckpt  # noqa: E402

cfg = load(a.config)
model, meta = load_ckpt(a.ckpt)
std_om = Standardizer.from_dict(meta["std_omega"])
std_c = Standardizer.from_dict(meta["std_c"])
rtp = RTP(cfg.trajectory)
field = DockField()
zone = ZONES[SPIKE_ZONE_ID]
x0 = SPIKE_START_POSE
T_h, dt = rtp.horizon(x0, zone)
rng = np.random.default_rng(1)
sampler = SyntheticSampler()
report = ["# Spike report", f"config hash: {cfg.hash()}", ""]


def decode(h1, K):
    z = np.linspace(cfg.online.z_lo, cfg.online.z_hi, K)[:, None]
    c = std_c.transform(np.full((K, 1), h1, dtype=np.float32))
    om = model.decode(torch.tensor(z, dtype=torch.float32),
                      torch.tensor(c, dtype=torch.float32)).numpy()
    return std_om.inverse(om)


# ---- G1: training diagnostics ------------------------------------------
hist = meta["history"]
kl_f = float(np.mean(hist["kl"][-10:]))
rec_early = float(np.mean(hist["recon"][:max(3, len(hist["recon"]) // 5)]))
rec_late = float(np.mean(hist["recon"][-10:]))
g1 = (0.3 <= kl_f <= 5.0) and (rec_late < rec_early)
report += [f"## G1 training: {'PASS' if g1 else 'FAIL'}",
           f"final KL {kl_f:.2f} (band [0.3, 5]); recon {rec_early:.3f} -> {rec_late:.3f}", ""]

# ---- G2: conditioning works --------------------------------------------
h1_grid = np.array([0.9, 0.75, 0.5, 0.35, 0.15, 0.05])
cert_frac, align = [], []
for h1 in h1_grid:
    fr, al = [], []
    for _ in range(3):
        w, sig = sampler.sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
        w = w[:cfg.trajectory.N]
        om = decode(h1, cfg.online.K)
        wb = np.broadcast_to(w, (len(om),) + w.shape)
        cert = certify_batch(om, x0, zone, h1, 1.0, wb, sig, rtp, field, cfg)
        fr.append(cert.mask.mean())
        kin = rtp.kinematics(om, x0, zone)
        wdir = w[:, :2].mean(0); wdir /= max(np.linalg.norm(wdir), 1e-9)
        v = kin.vel / np.maximum(np.linalg.norm(kin.vel, axis=-1, keepdims=True), 1e-9)
        al.append(float((v @ wdir).mean()))
    cert_frac.append(np.mean(fr)); align.append(np.mean(al))
rho_c, _ = spearmanr(h1_grid, cert_frac)
rho_a, _ = spearmanr(1.0 - h1_grid, align)
g2 = (rho_c > 0.5) and (rho_a > 0.5)
report += [f"## G2 conditioning: {'PASS' if g2 else 'FAIL'}",
           f"certified fraction vs h1 Spearman = {rho_c:.2f} (> 0.5)",
           f"env-alignment vs severity Spearman = {rho_a:.2f} (> 0.5)",
           f"cert_frac(h1): " + ", ".join(f"{h:.2f}:{c:.2f}" for h, c in zip(h1_grid, cert_frac)), ""]

# ---- G3: quality at held-out severity h1 = 0.25 ------------------------
h1_ho = 0.25
w, sig = sampler.sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
w = w[:cfg.trajectory.N]
om = decode(h1_ho, 100)
wb = np.broadcast_to(w, (len(om),) + w.shape)
cert = certify_batch(om, x0, zone, h1_ho, 1.0, wb, sig, rtp, field, cfg)
g3a = cert.mask.mean() >= 0.30
report += [f"## G3 held-out h1=0.25: certified {cert.mask.mean():.2f} "
           f"(>= 0.30) -> {'PASS' if g3a else 'FAIL'} (part a)"]
g3b = None
g4 = None
if not a.skip_planner:
    from manifold_recovery.planner.energy_ocp import EnergyOCP
    from manifold_recovery.baselines.restarts import run_b1
    planner = EnergyOCP(zone, cfg)
    x0f = np.array([x0[0], x0[1], x0[2], 0, 0, 0])
    ab = float(alpha_bar_policy(sig, h1_ho, 1.0, cfg))
    b1 = run_b1(x0f, zone, np.array([h1_ho, 1.0]), ab, T_h, w, planner)
    ok = np.flatnonzero(cert.mask)
    if len(ok) and b1["best"]:
        best_i = ok[np.argmin(cert.max_d2[ok])]
        kin = rtp.kinematics(om[best_i][None], x0, zone)
        ft = planner.solve(x0f, np.array([h1_ho, 1.0]), ab, w, T_h,
                           warm_xi=kin.pos[0])
        g3b = ft.converged and ft.cost <= 1.10 * b1["best"]["cost"]
        report += [f"fine-tuned cost {ft.cost:.1f} vs B1 best {b1['best']['cost']:.1f} "
                   f"(<= 1.10x) -> {'PASS' if g3b else 'FAIL'} (part b)", ""]
    # ---- G4: class coverage --------------------------------------------
    g4_parts = []
    for h1c in (0.9, 0.5):
        w2, sg2 = sampler.sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
        w2 = w2[:cfg.trajectory.N]
        om2 = decode(h1c, cfg.online.K)
        wb2 = np.broadcast_to(w2, (len(om2),) + w2.shape)
        c2 = certify_batch(om2, x0, zone, h1c, 1.0, wb2, sg2, rtp, field, cfg)
        okk = np.flatnonzero(c2.mask)
        n_model = 0
        if len(okk):
            kin2 = rtp.kinematics(om2[okk], x0, zone)
            sgs = side_signature(kin2.pos, kin2.vel)
            n_model = len({tuple(s) for s in sgs})
        ab2 = float(alpha_bar_policy(sg2, h1c, 1.0, cfg))
        b1c = run_b1(x0f, zone, np.array([h1c, 1.0]), ab2, T_h, w2, planner)
        g4_parts.append(n_model >= len(b1c["classes"]))
        report += [f"G4 @ h1={h1c}: model classes {n_model} vs B1 {len(b1c['classes'])} "
                   f"-> {'PASS' if g4_parts[-1] else 'FAIL'}"]
    g4 = all(g4_parts)
    report += [""]

# ---- z-sweep figure -----------------------------------------------------
om_sweep = decode(0.5, 40)
kin = rtp.kinematics(om_sweep, x0, zone)
plot_trajectories(list(kin.pos), np.linspace(cfg.online.z_lo, cfg.online.z_hi, 40),
                  "Decoded manifold sweep, h1=0.5",
                  _common.ROOT / "runs/z_sweep.png", x0=x0, cbar_label="z")

verdict = "GO" if (g1 and g2 and g3a and (g3b is not False) and (g4 is not False)) else "EXTEND/STOP"
report += [f"## Verdict: {verdict}",
           f"(G1={g1}, G2={g2}, G3a={g3a}, G3b={g3b}, G4={g4})"]
out = _common.ROOT / "runs/spike_report.md"
out.write_text("\n".join(report))
print("\n".join(report))
print("\nreport ->", out)
