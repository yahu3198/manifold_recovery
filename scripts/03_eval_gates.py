"""Evaluate gates G1-G4 (spec Section 8; rev 4) and write runs/spike_report.md.

Rev 4: the decoder is zone-conditioned; every decode call splits K across the
three zones (Decoder.decode_zones) so V is comparable with the zone-balanced
proposal. G2 additionally prints a per-zone V / V_proposal table at h1 = 0.5.

Gate definitions (unchanged from rev 3):
  G1  KL in [0.3, 1.2 Cz_max] and reconstruction decreased.
  G2  Conditioning works = the manifold BEATS the naive proposal and respects
      severity: mean lift V / V_proposal over the h1 grid >= --min-lift, and
      cert_frac is non-increasing with degradation (tolerance --mono-tol).
      Reported per h1 with the failure breakdown, at the canonical start and
      at --n-starts random starts from the arc (generalisation over x0).
  G3  Held-out severity h1 = 0.25: (a) certified fraction >= 0.30;
      (b) fine-tuned effort of the best certified decode within 1.10x of
      B1's best FEASIBLE effort across all zones (same terminal semantics:
      inside a zone). INCONCLUSIVE if either side is infeasible.
  G4  Class coverage: (zone, winding) classes among certified decodes >=
      B1's feasible classes, at h1 = 0.9 and 0.5. INCONCLUSIVE below
      --min-cert certified decodes.
Verdict: GO (all PASS), EXTEND (no FAIL, some INCONCLUSIVE), STOP otherwise.
"""
import _common  # noqa: F401
import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from manifold_recovery.config import load
from manifold_recovery.scenario import SPIKE_START_POSE, sample_start, ZONES
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.score.obstacles import DockField
from manifold_recovery.certify.surrogate import certify_batch, alpha_bar_policy
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.data.proposal import ProposalSampler
from manifold_recovery.analysis.contraction import n_classes
from manifold_recovery.scenario import zone_of
from manifold_recovery.analysis.plots import plot_trajectories

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--ckpt", default=str(_common.ROOT / "runs/ckpt.pt"))
p.add_argument("--skip-planner", action="store_true",
               help="skip G3b optimality vs B1 and G4 (fast smoke)")
p.add_argument("--n-env", type=int, default=3, help="force draws per gate cell")
p.add_argument("--n-starts", type=int, default=3, help="random starts for the G2 generalisation row")
p.add_argument("--min-cert", type=int, default=10)
p.add_argument("--min-lift", type=float, default=1.3)
p.add_argument("--mono-tol", type=float, default=0.05)
p.add_argument("--verbose-b1", action="store_true")
a = p.parse_args()

from manifold_recovery.model.train import load as load_ckpt  # noqa: E402
from manifold_recovery.model.decode import Decoder  # noqa: E402

cfg = load(a.config)
model, meta = load_ckpt(a.ckpt)
if meta.get("config_hash") != cfg.hash():
    print(f"WARNING: checkpoint config hash {meta.get('config_hash')} != current {cfg.hash()}; "
          f"regenerate from 01 if the config changed.")
dec = Decoder(model, meta, cfg)
rtp = RTP(cfg.trajectory)
field = DockField()
T_h, dt = rtp.horizon()
rng = np.random.default_rng(1)
sampler = SyntheticSampler()
prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng, mix=cfg.data.prop_mix)
x0c = SPIKE_START_POSE
PASS, FAIL, INC = "PASS", "FAIL", "INCONCLUSIVE"
report = ["# Spike report (gates rev 4)", f"config hash: {cfg.hash()}",
          f"T_h = {T_h:.1f} s, z_mode = {cfg.online.z_mode}, latent = {dec.latent}", ""]
summary = {}


def draw_forces():
    w, sig = sampler.sample(cfg.data.sea_state, cfg.data.direction, T_h, dt, rng)
    return w[:cfg.trajectory.N], sig


def cert_of(om, x0, h1, w, sig):
    wb = np.broadcast_to(w, (len(om),) + w.shape)
    return certify_batch(om, x0, h1, 1.0, wb, sig, rtp, field, cfg)


def eval_cell(h1, x0, K, n_env, per_zone=False):
    """Mean over force draws of (V, V_proposal, breakdown); returns last decode too.
    With per_zone, also returns {zone: (V_z, V_prop_z)} using the target zone of
    each decode (g) and the terminal zone of each proposal sample."""
    Vs, Vps, bds, pz = [], [], [], {z.id: [[], []] for z in ZONES}
    last = None
    for _ in range(n_env):
        w, sig = draw_forces()
        om, g = dec.decode_zones(h1, x0, K, rng)
        cert = cert_of(om, x0, h1, w, sig)
        omp, gp = prop.sample_with_mode(np.broadcast_to(np.asarray(x0, float), (K, 3)), rng)
        certp = cert_of(omp, x0, h1, w, sig)
        Vs.append(cert.mask.mean()); Vps.append(certp.mask.mean())
        bds.append(cert.failure_breakdown())
        for z in ZONES:
            if (g == z.id).any():
                pz[z.id][0].append(cert.mask[g == z.id].mean())
            if (gp == z.id).any():
                pz[z.id][1].append(certp.mask[gp == z.id].mean())
        last = (w, sig, om, cert)
    bd = {k: float(np.mean([b[k] for b in bds])) for k in bds[0]}
    out = (float(np.mean(Vs)), float(np.mean(Vps)), bd, last)
    if per_zone:
        return out + ({k: (float(np.mean(v[0])) if v[0] else np.nan,
                           float(np.mean(v[1])) if v[1] else np.nan) for k, v in pz.items()},)
    return out


# ---- G1 ----------------------------------------------------------------
hist = meta["history"]
kl_f = float(np.mean(hist["kl"][-10:]))
kl_lo, kl_hi = 0.3, 1.2 * cfg.model.Cz_max
rec_early = float(np.mean(hist["recon"][:max(3, len(hist["recon"]) // 5)]))
rec_late = float(np.mean(hist["recon"][-10:]))
g1 = PASS if (kl_lo <= kl_f <= kl_hi) and (rec_late < rec_early) else FAIL
report += [f"## G1 training: {g1}",
           f"final KL {kl_f:.2f} (band [{kl_lo}, {kl_hi:.1f}]); recon {rec_early:.3f} -> {rec_late:.3f}", ""]
summary["G1"] = g1

# ---- G2 ----------------------------------------------------------------
h1_grid = np.array([0.9, 0.75, 0.5, 0.35, 0.15, 0.05])
V, Vp, BD = [], [], []
for h1 in h1_grid:
    v, vp, bd, _ = eval_cell(h1, x0c, cfg.online.K, a.n_env)
    V.append(v); Vp.append(vp); BD.append(bd)
V, Vp = np.array(V), np.array(Vp)
lift = np.where(Vp > 0, V / np.maximum(Vp, 1e-9), np.nan)
mean_lift = float(np.nanmean(lift))
# monotone: cert_frac must not rise with degradation beyond tolerance
mono_ok = bool(np.all(np.diff(V) <= a.mono_tol))
rho_c, _ = spearmanr(h1_grid, V)
g2 = PASS if (mean_lift >= a.min_lift and mono_ok and V.max() >= 0.30) else FAIL
report += [f"## G2 conditioning: {g2}",
           f"mean lift V/V_proposal over h1 grid = {mean_lift:.2f} (>= {a.min_lift}); "
           f"non-increasing with degradation: {mono_ok} (tol {a.mono_tol}); Spearman(h1, V) = {rho_c:.2f}",
           "h1     V     V_prop  lift   | thrust  sway  collision  terminal  (canonical start)"]
for h, v, vp, l, bd in zip(h1_grid, V, Vp, lift, BD):
    report.append(f"{h:.2f}  {v:.2f}   {vp:.2f}   {l:5.2f}  | {bd['thrust_fail']:.2f}   "
                  f"{bd['sway_fail']:.2f}   {bd['collision']:.2f}      {bd['terminal_fail']:.2f}")
# per-zone lift at h1 = 0.5 (which zone the model helps with)
_, _, _, _, pz = eval_cell(0.5, x0c, cfg.online.K, a.n_env, per_zone=True)
report.append("per-zone @ h1=0.5 (V / V_prop): " + ", ".join(
    f"{ZONES[k].name} {v[0]:.2f}/{v[1]:.2f}" for k, v in pz.items()))
# generalisation over the start arc at h1 = 0.5
starts = sample_start(rng, a.n_starts, cfg.data.start_d_range, cfg.data.start_bearing_deg,
                      cfg.data.start_heading_jitter_deg)
gen = [eval_cell(0.5, s, cfg.online.K, 1)[:2] for s in starts]
report.append("random starts @ h1=0.5 (V / V_prop): " +
              ", ".join(f"{v:.2f}/{vp:.2f}" for v, vp in gen))
report.append("")
summary["G2"] = g2

# ---- G3 ----------------------------------------------------------------
h1_ho = 0.25
g3a_val, _, _, last = eval_cell(h1_ho, x0c, 100, a.n_env)
g3a = PASS if g3a_val >= 0.30 else FAIL
report += ["## G3 held-out h1=0.25", f"(a) certified {g3a_val:.2f} over {a.n_env} draws (>= 0.30) -> {g3a}"]
summary["G3a"] = g3a
if not a.skip_planner:
    from manifold_recovery.baselines.restarts import run_b1, PlannerBank
    planners = PlannerBank(cfg)
    x0f = np.array([x0c[0], x0c[1], x0c[2], 0, 0, 0])
    w, sig, om, cert = last
    ok = np.flatnonzero(cert.mask)
    ab = float(alpha_bar_policy(sig, h1_ho, 1.0, cfg))
    b1 = run_b1(x0f, np.array([h1_ho, 1.0]), ab, T_h, w, planners, verbose=a.verbose_b1)
    line = (f"B1: {b1['n_feasible']}/{b1['n_seeds']} feasible ({b1['n_converged']} converged), "
            f"classes {b1['classes']}, best feasible effort "
            f"{b1['best']['cost'] if b1['best'] else None} "
            f"(zone {ZONES[b1['best']['zone']].name if b1['best'] else '-'})")
    if len(ok) == 0 or b1["best"] is None:
        g3b = INC
        report += [f"(b) {g3b}: " + ("no certified decode" if len(ok) == 0 else "no feasible B1"), line]
    else:
        best_i = ok[np.argmin(cert.max_d2[ok])]
        kin = rtp.kinematics(om[best_i][None], x0c)
        zid = int(zone_of(kin.pos[0, -1]))
        ft = planners(ZONES[zid]).solve(x0f, np.array([h1_ho, 1.0]), ab, w, T_h, warm_xi=kin.pos[0])
        ratio = ft.cost / b1["best"]["cost"]
        same = b1["best_by_zone"].get(zid)
        same_ratio = ft.cost / same["cost"] if same else float("nan")
        if not ft.feasible:
            g3b = INC
            report += [f"(b) {g3b}: fine-tune {ft.status}, slack {ft.slack_total:.3f} "
                       f"(effort {ft.cost:.1f}, ratio {ratio:.2f} not comparable)", line]
        else:
            g3b = PASS if ratio <= 1.10 else FAIL
            report += [f"(b) fine-tuned effort {ft.cost:.1f} ({ZONES[zid].name}) vs B1 best "
                       f"{b1['best']['cost']:.1f} = {ratio:.2f}x (<= 1.10) -> {g3b}; "
                       f"same-zone ratio {same_ratio:.2f}  [fine-tune {ft.status}, "
                       f"{ft.n_attempts} attempt(s)]", line]
    report.append("")
    summary["G3b"] = g3b

    # ---- G4 ------------------------------------------------------------
    parts = []
    report.append("## G4 class coverage (zone, winding)")
    for h1c in (0.9, 0.5):
        n_all, n_cert, n_ok, n_b1, b1_cls = [], [], 0, [], set()
        for _ in range(a.n_env):
            w2, sg2 = draw_forces()
            om2, _ = dec.decode_zones(h1c, x0c, cfg.online.K, rng)
            c2 = cert_of(om2, x0c, h1c, w2, sg2)
            kin2 = rtp.kinematics(om2, x0c)
            okk = np.flatnonzero(c2.mask); n_ok += len(okk)
            n_all.append(n_classes(kin2.pos)); n_cert.append(n_classes(kin2.pos[okk]))
            ab2 = float(alpha_bar_policy(sg2, h1c, 1.0, cfg))
            b1c = run_b1(x0f, np.array([h1c, 1.0]), ab2, T_h, w2, planners, verbose=a.verbose_b1)
            n_b1.append(len(b1c["classes"])); b1_cls |= set(b1c["classes"])
        nc, nb = int(np.max(n_cert)), int(np.max(n_b1))
        res = INC if (n_ok < a.min_cert or nb == 0) else (PASS if nc >= nb else FAIL)
        parts.append(res)
        report.append(f"h1={h1c}: classes among all decodes {int(np.max(n_all))}, among certified {nc} "
                      f"({n_ok} certified / {a.n_env} draws), B1 feasible classes {nb} "
                      f"{sorted(b1_cls)} -> {res}")
    g4 = FAIL if FAIL in parts else (INC if INC in parts else PASS)
    report.append("")
    summary["G4"] = g4

# ---- figure ---------------------------------------------------------------
om_s, _ = dec.decode_zones(0.5, x0c, 60, rng)
kin = rtp.kinematics(om_s, x0c)
cert_s = cert_of(om_s, x0c, 0.5, *draw_forces())
plot_trajectories(list(kin.pos), cert_s.mask.astype(float),
                  "Decoded candidates, h1=0.5 (colour = certified)",
                  _common.ROOT / "runs/z_sweep.png", x0=x0c, cbar_label="certified")

results = list(summary.values())
verdict = "STOP" if FAIL in results else ("EXTEND" if INC in results else "GO")
report += [f"## Verdict: {verdict}", "(" + ", ".join(f"{k}={v}" for k, v in summary.items()) + ")"]
out = _common.ROOT / "runs/spike_report.md"
out.write_text("\n".join(report))
(_common.ROOT / "runs/spike_summary.json").write_text(json.dumps(
    {"verdict": verdict, "gates": summary, "config_hash": cfg.hash(),
     "V": dict(zip(map(str, h1_grid), map(float, V))),
     "V_proposal": dict(zip(map(str, h1_grid), map(float, Vp))),
     "mean_lift": mean_lift, "g3a": g3a_val}, indent=2))
print("\n".join(report))
print("\nreport ->", out)