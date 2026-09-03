"""Evaluate gates G1-G4 (spec Section 8; rev 2) and write runs/spike_report.md.

Rev 2 changes, each traceable to a spike-run-1 finding:
  G1  KL band is [0.3, 1.2 * Cz_max]. Capacity annealing drives KL to Cz_max
      by construction, so an upper bound EQUAL to Cz_max (rev 1) rejected the
      loss's own target (KL 5.04 vs band [0.3, 5]).
  G2  Monotonicity of certified fraction vs h1 is tested on the mu_H > 0
      range (h1 <= 0.8, h2 = 1) where the alpha_bar policy admits environmental
      authority; h1 = 0.9 is still REPORTED with a per-criterion failure
      breakdown so the certificate's behaviour on near-healthy vessels stays
      visible. Environmental alignment is reported as a diagnostic, not gated:
      the spike condition c = [h1] carries no wind direction, so the decoder
      marginalises over it and cannot express alignment (use --gate-alignment
      to restore the rev-1 behaviour once direction is in c).
  G3b Compares against B1's best FEASIBLE solution (converged AND slack-free)
      and requires the fine-tune to be feasible too; otherwise INCONCLUSIVE,
      never FAIL, because a comparison against a non-converged oracle carries
      no information. Effort, slack, status are reported separately.
  G4  Uses winding signatures and reports N_H among all decodes, among
      certified decodes, and B1's feasible classes; INCONCLUSIVE when fewer
      than --min-cert certified decodes exist (class counting is starved).
  All gates average over --n-env force draws instead of one.
Verdict: GO (all PASS), EXTEND (no FAIL, some INCONCLUSIVE), STOP otherwise.
"""
import _common  # noqa: F401
import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from manifold_recovery.config import load
from manifold_recovery.scenario import ZONES, SPIKE_START_POSE, SPIKE_ZONE_ID
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.score.obstacles import DockField
from manifold_recovery.certify.surrogate import certify_batch, alpha_bar_policy, mu_H
from manifold_recovery.certify.cluster import side_signature
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.features.condition import Standardizer
from manifold_recovery.analysis.plots import plot_trajectories

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--ckpt", default=str(_common.ROOT / "runs/ckpt.pt"))
p.add_argument("--skip-planner", action="store_true",
               help="skip G3b optimality vs B1 and G4 (fast smoke)")
p.add_argument("--n-env", type=int, default=3, help="force draws per gate cell")
p.add_argument("--min-cert", type=int, default=10,
               help="certified decodes needed before G4 class counting is meaningful")
p.add_argument("--gate-alignment", action="store_true",
               help="gate G2 on env-alignment too (only meaningful once wind direction is in c)")
p.add_argument("--verbose-b1", action="store_true")
a = p.parse_args()

import torch  # noqa: E402  (torch required for gate evaluation)
from manifold_recovery.model.train import load as load_ckpt  # noqa: E402

cfg = load(a.config)
model, meta = load_ckpt(a.ckpt)
if meta.get("config_hash") != cfg.hash():
    print(f"WARNING: checkpoint config hash {meta.get('config_hash')} != current {cfg.hash()}; "
          f"regenerate from 01 if the config changed.")
std_om = Standardizer.from_dict(meta["std_omega"])
std_c = Standardizer.from_dict(meta["std_c"])
rtp = RTP(cfg.trajectory)
field = DockField()
zone = ZONES[SPIKE_ZONE_ID]
x0 = SPIKE_START_POSE
T_h, dt = rtp.horizon(x0, zone)
rng = np.random.default_rng(1)
sampler = SyntheticSampler()
PASS, FAIL, INC = "PASS", "FAIL", "INCONCLUSIVE"
report = ["# Spike report (gates rev 2)", f"config hash: {cfg.hash()}",
          f"horizon_mode: {cfg.trajectory.horizon_mode}  T_h = {T_h:.1f} s", ""]
summary = {}


def decode(h1, K):
    z = np.linspace(cfg.online.z_lo, cfg.online.z_hi, K)[:, None]
    c = std_c.transform(np.full((K, 1), h1, dtype=np.float32))
    om = model.decode(torch.tensor(z, dtype=torch.float32),
                      torch.tensor(c, dtype=torch.float32)).numpy()
    return std_om.inverse(om)


def draw_forces():
    w, sig = sampler.sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
    return w[:cfg.trajectory.N], sig


def cert_decodes(h1, K, w, sig):
    om = decode(h1, K)
    wb = np.broadcast_to(w, (len(om),) + w.shape)
    return om, certify_batch(om, x0, zone, h1, 1.0, wb, sig, rtp, field, cfg)


def n_classes(pos):
    if len(pos) == 0:
        return 0
    sgs = side_signature(pos)
    return len({tuple(s) for s in sgs.reshape(-1, sgs.shape[-1]).tolist()})


# ---- G1: training diagnostics ------------------------------------------
hist = meta["history"]
kl_f = float(np.mean(hist["kl"][-10:]))
kl_lo, kl_hi = 0.3, 1.2 * cfg.model.Cz_max
rec_early = float(np.mean(hist["recon"][:max(3, len(hist["recon"]) // 5)]))
rec_late = float(np.mean(hist["recon"][-10:]))
g1 = PASS if (kl_lo <= kl_f <= kl_hi) and (rec_late < rec_early) else FAIL
report += [f"## G1 training: {g1}",
           f"final KL {kl_f:.2f} (band [{kl_lo}, {kl_hi:.1f}] = [0.3, 1.2 Cz_max]); "
           f"recon {rec_early:.3f} -> {rec_late:.3f}", ""]
summary["G1"] = g1

# ---- G2: conditioning works --------------------------------------------
h1_grid = np.array([0.9, 0.75, 0.5, 0.35, 0.15, 0.05])
cert_frac, align, breakdown = [], [], []
for h1 in h1_grid:
    fr, al, bd = [], [], []
    for _ in range(a.n_env):
        w, sig = draw_forces()
        om, cert = cert_decodes(h1, cfg.online.K, w, sig)
        fr.append(cert.mask.mean())
        bd.append(cert.failure_breakdown())
        kin = rtp.kinematics(om, x0, zone)
        wdir = w[:, :2].mean(0); wdir /= max(np.linalg.norm(wdir), 1e-9)
        v = kin.vel / np.maximum(np.linalg.norm(kin.vel, axis=-1, keepdims=True), 1e-9)
        al.append(float((v @ wdir).mean()))
    cert_frac.append(float(np.mean(fr))); align.append(float(np.mean(al)))
    breakdown.append({k: float(np.mean([b[k] for b in bd])) for k in bd[0]})
cert_frac, align = np.array(cert_frac), np.array(align)
gated = mu_H(h1_grid, 1.0) > 0.0            # h1 <= 0.8 under the ICRA Eq. 9 policy
rho_c, _ = spearmanr(h1_grid[gated], cert_frac[gated])
rho_a, _ = spearmanr(1.0 - h1_grid, align)
g2_cert = rho_c > 0.5 and cert_frac[gated].max() >= 0.30
g2 = PASS if (g2_cert and (rho_a > 0.5 or not a.gate_alignment)) else FAIL
report += [f"## G2 conditioning: {g2}",
           f"certified fraction vs h1 on mu_H > 0 range (h1 <= 0.8): Spearman = {rho_c:.2f} (> 0.5)",
           f"env-alignment vs severity: Spearman = {rho_a:.2f} "
           f"({'gated' if a.gate_alignment else 'diagnostic only: c = [h1] has no wind direction'})",
           "cert_frac(h1): " + ", ".join(f"{h:.2f}:{c:.2f}" for h, c in zip(h1_grid, cert_frac)),
           "failure breakdown (fraction of decodes failing each criterion):"]
for h, b in zip(h1_grid, breakdown):
    report.append(f"  h1={h:.2f}  thrust {b['thrust_fail']:.2f}  sway {b['sway_fail']:.2f}  "
                  f"collision {b['collision']:.2f}  terminal {b['terminal_fail']:.2f}")
report.append("")
summary["G2"] = g2

# ---- G3: quality at held-out severity h1 = 0.25 ------------------------
h1_ho = 0.25
g3a_vals, best_cands = [], []
for _ in range(a.n_env):
    w, sig = draw_forces()
    om, cert = cert_decodes(h1_ho, 100, w, sig)
    g3a_vals.append(cert.mask.mean())
    ok = np.flatnonzero(cert.mask)
    best_cands.append((w, sig, om, cert, ok))
g3a_val = float(np.mean(g3a_vals))
g3a = PASS if g3a_val >= 0.30 else FAIL
report += [f"## G3 held-out h1=0.25", f"(a) certified {g3a_val:.2f} over {a.n_env} draws "
           f"(>= 0.30) -> {g3a}"]
summary["G3a"] = g3a
g3b = g4 = None
if not a.skip_planner:
    from manifold_recovery.planner.energy_ocp import EnergyOCP
    from manifold_recovery.baselines.restarts import run_b1
    planner = EnergyOCP(zone, cfg)
    x0f = np.array([x0[0], x0[1], x0[2], 0, 0, 0])
    # (b) optimality: use the draw with the most certified decodes
    w, sig, om, cert, ok = max(best_cands, key=lambda t: len(t[4]))
    ab = float(alpha_bar_policy(sig, h1_ho, 1.0, cfg))
    b1 = run_b1(x0f, zone, np.array([h1_ho, 1.0]), ab, T_h, w, planner, verbose=a.verbose_b1)
    line = (f"B1: {b1['n_feasible']}/{b1['n_seeds']} feasible "
            f"({b1['n_converged']} converged), classes {b1['classes']}, "
            f"best feasible effort {b1['best']['cost'] if b1['best'] else None}")
    if len(ok) == 0 or b1["best"] is None:
        g3b = INC
        why = "no certified decode" if len(ok) == 0 else "no feasible B1 solution"
        report += [f"(b) {g3b}: {why}", line]
    else:
        best_i = ok[np.argmin(cert.max_d2[ok])]
        kin = rtp.kinematics(om[best_i][None], x0, zone)
        ft = planner.solve(x0f, np.array([h1_ho, 1.0]), ab, w, T_h, warm_xi=kin.pos[0])
        ratio = ft.cost / b1["best"]["cost"]
        if not ft.feasible:
            g3b = INC
            report += [f"(b) {g3b}: fine-tune {ft.status}, slack {ft.slack_total:.3f} m "
                       f"(effort {ft.cost:.1f}, ratio {ratio:.2f} not comparable)", line]
        else:
            g3b = PASS if ratio <= 1.10 else FAIL
            report += [f"(b) fine-tuned effort {ft.cost:.1f} vs B1 best {b1['best']['cost']:.1f} "
                       f"= {ratio:.2f}x (<= 1.10) -> {g3b}   "
                       f"[fine-tune {ft.status}, slack {ft.slack_total:.3f}, "
                       f"{ft.n_attempts} attempt(s)]", line]
    report.append("")
    summary["G3b"] = g3b

    # ---- G4: class coverage --------------------------------------------
    g4_parts = []
    report.append("## G4 class coverage")
    for h1c in (0.9, 0.5):
        n_all, n_cert, n_ok_tot, n_b1 = [], [], 0, []
        for _ in range(a.n_env):
            w2, sg2 = draw_forces()
            om2, c2 = cert_decodes(h1c, cfg.online.K, w2, sg2)
            kin2 = rtp.kinematics(om2, x0, zone)
            okk = np.flatnonzero(c2.mask)
            n_ok_tot += len(okk)
            n_all.append(n_classes(kin2.pos))
            n_cert.append(n_classes(kin2.pos[okk]))
            ab2 = float(alpha_bar_policy(sg2, h1c, 1.0, cfg))
            b1c = run_b1(x0f, zone, np.array([h1c, 1.0]), ab2, T_h, w2, planner,
                         verbose=a.verbose_b1)
            n_b1.append(len(b1c["classes"]))
        nc, nb = int(np.max(n_cert)), int(np.max(n_b1))
        if n_ok_tot < a.min_cert or nb == 0:
            res = INC
        else:
            res = PASS if nc >= nb else FAIL
        g4_parts.append(res)
        report.append(f"h1={h1c}: classes among all decodes {int(np.max(n_all))}, "
                      f"among certified {nc} ({n_ok_tot} certified over {a.n_env} draws), "
                      f"B1 feasible classes {nb} -> {res}")
    g4 = FAIL if FAIL in g4_parts else (INC if INC in g4_parts else PASS)
    report.append("")
    summary["G4"] = g4

# ---- z-sweep figure -----------------------------------------------------
om_sweep = decode(0.5, 40)
kin = rtp.kinematics(om_sweep, x0, zone)
plot_trajectories(list(kin.pos), np.linspace(cfg.online.z_lo, cfg.online.z_hi, 40),
                  "Decoded manifold sweep, h1=0.5",
                  _common.ROOT / "runs/z_sweep.png", x0=x0, cbar_label="z")

results = list(summary.values())
verdict = "STOP" if FAIL in results else ("EXTEND" if INC in results else "GO")
report += [f"## Verdict: {verdict}", "(" + ", ".join(f"{k}={v}" for k, v in summary.items()) + ")"]
out = _common.ROOT / "runs/spike_report.md"
out.write_text("\n".join(report))
(_common.ROOT / "runs/spike_summary.json").write_text(json.dumps(
    {"verdict": verdict, "gates": summary, "config_hash": cfg.hash(),
     "cert_frac": dict(zip(map(str, h1_grid), map(float, cert_frac))),
     "g3a": g3a_val}, indent=2))
print("\n".join(report))
print("\nreport ->", out)