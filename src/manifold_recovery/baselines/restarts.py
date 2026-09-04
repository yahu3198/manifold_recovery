"""Baseline B1: energy OCP with class-seeded restarts across ALL zones (rev 3).

Also the G3/G4 oracle. For every zone: the entry-via route (the same
``scenario.route_waypoints`` geometry the proposal's ``base="via"`` variant
uses), the straight chord, and lateral bulges at +-12 m. Goal points are the
zone centroids; the OCP terminal constraint is "inside the zone polygon".

``best``/``classes`` use FEASIBLE solutions only (converged AND obstacle
slack <= slack_tol). Every attempt is recorded with its IPOPT status so gates
can declare a comparison INCONCLUSIVE rather than fail against a broken
oracle. Signature = (zone, w_D1, w_D2).
"""
from __future__ import annotations

import time

import numpy as np

from ..scenario import ZONES, Zone, route_waypoints
from ..planner.energy_ocp import EnergyOCP
from ..certify.cluster import side_signature


def _seeds(x0, zone: Zone, n_pts: int = 41):
    p0, pg = np.asarray(x0[:2], float), zone.center
    t = np.linspace(0, 1, n_pts)[:, None]
    chord = p0 + t * (pg - p0)
    d = pg - p0
    perp = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    bump = np.sin(np.pi * t[:, 0])[:, None]
    seeds = [("via", route_waypoints(zone, x0, pg, n_pts)), ("chord", chord)]
    for off in (12.0, -12.0):
        seeds.append((f"bulge{off:+.0f}", chord + off * bump * perp))
    return seeds


class PlannerBank:
    """Lazily built EnergyOCP per zone (construction is the expensive part)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._p: dict[int, EnergyOCP] = {}

    def __call__(self, zone: Zone) -> EnergyOCP:
        if zone.id not in self._p:
            self._p[zone.id] = EnergyOCP(zone, self.cfg)
        return self._p[zone.id]


def run_b1(x0_full, h, alpha_bar, T_h, w_plan, planners: PlannerBank,
           zones=ZONES, verbose: bool = False):
    t0 = time.perf_counter()
    sols, attempts = [], []
    for zone in zones:
        planner = planners(zone)
        for name, xi0 in _seeds(x0_full, zone):
            r = planner.solve(x0_full, h, alpha_bar, w_plan, T_h, warm_xi=xi0)
            sig = tuple(int(v) for v in side_signature(r.xi[None])[0])
            rec = {"zone": zone.id, "seed": name, "xi": r.xi, "cost": r.cost,
                   "signature": sig, "wall": r.wall_time, "converged": r.converged,
                   "feasible": r.feasible, "slack_total": r.slack_total,
                   "status": r.status, "n_attempts": r.n_attempts}
            attempts.append(rec)
            if r.feasible:
                sols.append(rec)
            if verbose:
                print(f"    B1 {zone.name} {name:9s} {r.status:28s} slack={r.slack_total:7.3f} "
                      f"effort={r.cost:9.1f} sig={sig} tries={r.n_attempts}")
    wall = time.perf_counter() - t0
    classes = sorted({s["signature"] for s in sols})
    best = min(sols, key=lambda s: s["cost"]) if sols else None
    best_by_zone = {z.id: min((s for s in sols if s["zone"] == z.id),
                              key=lambda s: s["cost"], default=None) for z in zones}
    return {"solutions": sols, "attempts": attempts, "classes": classes,
            "best": best, "best_by_zone": best_by_zone, "wall_time": wall,
            "n_converged": int(sum(a["converged"] for a in attempts)),
            "n_feasible": len(sols), "n_seeds": len(attempts)}