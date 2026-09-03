"""Baseline B1: energy OCP with side-seeded restarts (also the G3/G4 oracle).

Seed set: straight line; lateral bulges at +-{10, 20} m mid-path (both sides);
explicit around-north and around-south waypoints clearing the dock block.
Returns every converged solution with its side signature, the best cost, and
the set of distinct classes found (``enumerate_classes`` for gate G4 and
scripts/00_precheck_geometry.py).
"""
from __future__ import annotations

import time

import numpy as np

from ..scenario import Zone
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
    seeds = [base]
    for off in (10.0, 20.0, -10.0, -20.0):
        seeds.append(base + off * bump * perp)
    # explicit detours around the dock block (north of Dock 1, south of Dock 2)
    for via in (np.array([-585.0, 250.0]), np.array([-583.0, 186.0])):
        half = int(n_pts * 0.55)
        t1 = np.linspace(0, 1, half)[:, None]
        t2 = np.linspace(0, 1, n_pts - half)[:, None]
        seeds.append(np.vstack([p0 + t1 * (via - p0), via + t2 * (pg - via)]))
    return seeds


def run_b1(x0_full, zone: Zone, h, alpha_bar, T_h, w_plan, planner: EnergyOCP):
    t0 = time.perf_counter()
    sols = []
    for xi0 in _seeds(x0_full, zone):
        r = planner.solve(x0_full, h, alpha_bar, w_plan, T_h, warm_xi=xi0)
        if r.converged:
            vel = np.gradient(r.xi, axis=0)
            sig = tuple(side_signature(r.xi[None], vel[None])[0])
            sols.append({"xi": r.xi, "cost": r.cost, "signature": sig,
                         "wall": r.wall_time})
    wall = time.perf_counter() - t0
    classes = sorted({s["signature"] for s in sols})
    best = min(sols, key=lambda s: s["cost"]) if sols else None
    return {"solutions": sols, "classes": classes, "best": best,
            "wall_time": wall}


def enumerate_classes(x0_full, zone, h, alpha_bar, T_h, w_plan, planner):
    return run_b1(x0_full, zone, h, alpha_bar, T_h, w_plan, planner)["classes"]
