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
from ..scenario import ZONES, zone_of
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
    plan: object = None          # PlanResult (X 6xN+1, U 2xN) for the 8-column reference


def finetune(reps: dict, decoded_xi: np.ndarray, decoded_psi: np.ndarray,
             omega: np.ndarray, x0: np.ndarray, planners, h: np.ndarray,
             alpha_bar: float, T_h: float, w_plan: np.ndarray,
             w_true: np.ndarray, w_dt: float,
             field_: DockField, cfg: Config) -> list[Candidate]:
    """``planners`` is a callable Zone -> EnergyOCP (baselines.restarts.PlannerBank).
    Each representative is refined in the zone its terminal point lies in."""
    out = []
    x0 = np.asarray(x0, float)
    # deployment passes the full fault-time state [x, y, psi, u, v, r]; the
    # offline scripts pass a pose and start from rest
    x0_full = x0.copy() if len(x0) >= 6 else np.array([x0[0], x0[1], x0[2], 0.0, 0.0, 0.0])
    for sig, idx in reps.items():
        zid = int(zone_of(decoded_xi[idx][-1]))
        if zid < 0:
            continue
        zone = ZONES[zid]
        planner = planners(zone)
        t0 = time.perf_counter()
        plan = planner.solve(x0_full, h, alpha_bar, w_plan, T_h,
                             warm_xi=decoded_xi[idx])
        t_plan = time.perf_counter() - t0
        # rev 4.2: heading reference is the plan's own psi (X[2], continuous),
        # not the path tangent: the OCP solution crabs against the environment
        # and the tangent heading is not what the thrusters were solved for
        psi_ref = np.asarray(plan.X[2], float)
        t0 = time.perf_counter()
        cert = rollout_certify(plan.xi, psi_ref, T_h, x0_full, h, alpha_bar,
                               w_true, w_dt, None, field_, cfg)
        t_exact = time.perf_counter() - t0
        out.append(Candidate(xi=plan.xi, omega=omega[idx],
                             zone_id=zone.id, signature=sig,
                             plan_cost=plan.cost, cert=cert,
                             timings={"plan": t_plan, "exact": t_exact,
                                      "plan_converged": plan.converged,
                                      "plan_feasible": plan.feasible,
                                      "plan_slack": plan.slack_total},
                             plan=plan))
    out.sort(key=lambda c: c.plan_cost)
    return out
