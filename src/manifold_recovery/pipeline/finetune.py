"""Warm-started refinement of class representatives + exact re-certification.

For each representative: solve the energy OCP warm-started from the decoded
trajectory, then run the closed-loop exact check against a FRESH force
realisation (never the one used at generation). Returns ranked Candidates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..scenario import Zone
from ..certify.exact import rollout_certify, ExactResult
from ..score.obstacles import DockField
from ..planner.energy_ocp import EnergyOCP


@dataclass
class Candidate:
    xi: np.ndarray
    omega: np.ndarray | None
    zone_id: int
    signature: tuple
    plan_cost: float
    cert: ExactResult
    timings: dict = field(default_factory=dict)


def finetune(reps: dict, decoded_xi: np.ndarray, decoded_psi: np.ndarray,
             omega: np.ndarray, x0: np.ndarray, zone: Zone, h: np.ndarray,
             alpha_bar: float, T_h: float, w_plan: np.ndarray,
             w_true: np.ndarray, w_dt: float, planner: EnergyOCP,
             field_: DockField, cfg: Config) -> list[Candidate]:
    out = []
    x0_full = np.array([x0[0], x0[1], x0[2], 0.0, 0.0, 0.0])
    for sig, idx in reps.items():
        t0 = time.perf_counter()
        plan = planner.solve(x0_full, h, alpha_bar, w_plan, T_h,
                             warm_xi=decoded_xi[idx])
        t_plan = time.perf_counter() - t0
        psi_ref = np.unwrap(np.arctan2(*np.gradient(plan.xi, axis=0).T[::-1]))
        t0 = time.perf_counter()
        cert = rollout_certify(plan.xi, psi_ref, T_h, x0_full, h, alpha_bar,
                               w_true, w_dt, zone, field_, cfg)
        t_exact = time.perf_counter() - t0
        out.append(Candidate(xi=plan.xi, omega=omega[idx],
                             zone_id=zone.id, signature=sig,
                             plan_cost=plan.cost, cert=cert,
                             timings={"plan": t_plan, "exact": t_exact,
                                      "plan_converged": plan.converged}))
    out.sort(key=lambda c: (not c.cert.passed, c.plan_cost))
    return out
