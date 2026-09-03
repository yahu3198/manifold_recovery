"""Baseline B1: energy OCP with class-seeded restarts (also the G3/G4 oracle; rev 2).

Seed set: straight line; lateral bulges at +-{10, 20} m mid-path; the shared
north/south detour polylines from ``scenario.DETOUR_POLYLINES`` (identical to
the proposal-mixture bases, so B1 and the manifold explore the same classes).

Rev 2 hardening (spike run 1: 0/7 seeds converged in 05, a "best cost" from a
lucky seed in 03):
- every seed's IPOPT status, slack, and effort are RECORDED, not just kept
  if converged;
- ``best`` and ``classes`` are taken only over FEASIBLE solutions
  (converged AND slack_total <= slack_tol), so a margin-violating point is
  never quoted as the optimum;
- ``n_feasible``/``n_converged`` are returned so gates can declare a
  comparison INCONCLUSIVE instead of failing against a broken oracle.
"""
from __future__ import annotations

import time

import numpy as np

from ..scenario import Zone, DETOUR_POLYLINES, detour_waypoints
from ..planner.energy_ocp import EnergyOCP
from ..certify.cluster import side_signature


def _seeds(x0, zone: Zone, n_pts: int = 41):
    p0, pg = np.asarray(x0[:2], float), zone.center
    t = np.linspace(0, 1, n_pts)[:, None]
    base = p0 + t * (pg - p0)
    d = pg - p0
    perp = np.array([-d[1], d[0]])
    perp = perp / np.linalg.norm(perp)
    bump = np.sin(np.pi * t[:, 0])[:, None]
    seeds = [("straight", base)]
    for off in (10.0, 20.0, -10.0, -20.0):
        seeds.append((f"bulge{off:+.0f}", base + off * bump * perp))
    for name in DETOUR_POLYLINES:
        seeds.append((name, detour_waypoints(name, x0, zone, n_pts)))
    return seeds


def run_b1(x0_full, zone: Zone, h, alpha_bar, T_h, w_plan, planner: EnergyOCP,
           verbose: bool = False):
    t0 = time.perf_counter()
    sols, attempts = [], []
    for name, xi0 in _seeds(x0_full, zone):
        r = planner.solve(x0_full, h, alpha_bar, w_plan, T_h, warm_xi=xi0)
        sig = tuple(int(v) for v in side_signature(r.xi[None])[0])
        rec = {"seed": name, "xi": r.xi, "cost": r.cost, "signature": sig,
               "wall": r.wall_time, "converged": r.converged,
               "feasible": r.feasible, "slack_total": r.slack_total,
               "status": r.status, "n_attempts": r.n_attempts}
        attempts.append(rec)
        if r.feasible:
            sols.append(rec)
        if verbose:
            print(f"    B1 seed {name:10s} {r.status:28s} slack={r.slack_total:7.3f} "
                  f"cost={r.cost:10.1f} sig={sig} tries={r.n_attempts}")
    wall = time.perf_counter() - t0
    classes = sorted({s["signature"] for s in sols})
    best = min(sols, key=lambda s: s["cost"]) if sols else None
    return {"solutions": sols, "attempts": attempts, "classes": classes,
            "best": best, "wall_time": wall,
            "n_converged": int(sum(a["converged"] for a in attempts)),
            "n_feasible": len(sols), "n_seeds": len(attempts)}


def enumerate_classes(x0_full, zone, h, alpha_bar, T_h, w_plan, planner):
    return run_b1(x0_full, zone, h, alpha_bar, T_h, w_plan, planner)["classes"]