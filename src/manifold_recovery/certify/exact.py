"""Tier-2 exact certification: closed-loop rollout on the nominal model.

The candidate reference is tracked by a simple allocation controller at
dt_sim = 0.1 s: a PD law in position/heading produces a desired body-frame
acceleration, the corresponding wrench is computed from the model, and the
per-step least-effort allocation (the SAME wrench-set solve as everywhere
else) maps it to thruster commands and environmental utilisation, against
the REALISED force w_true (a fresh draw, deliberately different from the
w_hat used at generation). The plant integrates with the full w_true.

Pass iff the planar tracking error stays below e_max, the actual path never
enters a dock, and the vessel reaches the zone. In the full build VRX
replaces this tier; the interface is identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

from ..config import Config
from ..dynamics.params import VesselParams, PARAMS
from ..dynamics import fossen
from ..scenario import Zone, in_any_zone
from ..score.obstacles import DockField
from ..score.wrench_set import WrenchBoxes, distance_batch


@dataclass
class ExactResult:
    passed: bool
    max_track_err: float
    min_clearance: float
    arrival_time: float          # nan if never arrived
    energy: float                # sum ||u||^2_{R(H)} dt (effort proxy)
    reason: str


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rollout_certify(xi_ref: np.ndarray, psi_ref: np.ndarray, T_h: float,
                    x0: np.ndarray, h: np.ndarray, alpha_bar: float,
                    w_true: np.ndarray, w_dt: float, zone: Zone | None,
                    field: DockField, cfg: Config,
                    params: VesselParams = PARAMS) -> ExactResult:
    ec, wc = cfg.exact, cfg.wrench
    dt = ec.dt_sim
    n_steps = int(T_h * 1.3 / dt)
    t_ref = np.linspace(0.0, T_h, len(xi_ref))

    def ref_at(t):
        t = min(t, T_h)
        p = np.array([np.interp(t, t_ref, xi_ref[:, 0]),
                      np.interp(t, t_ref, xi_ref[:, 1])])
        v = np.array([np.interp(t, t_ref, np.gradient(xi_ref[:, 0], t_ref)),
                      np.interp(t, t_ref, np.gradient(xi_ref[:, 1], t_ref))])
        ps = np.interp(t, t_ref, np.unwrap(psi_ref))
        r = np.interp(t, t_ref, np.gradient(np.unwrap(psi_ref), t_ref))
        return p, v, ps, r

    t_w = np.arange(len(w_true)) * w_dt

    def w_at(t):
        return np.array([np.interp(t, t_w, w_true[:, j]) for j in range(3)])

    x = np.array([x0[0], x0[1], x0[2], 0.0, 0.0, 0.0], float)
    # rev 3: arrival = inside ANY zone (zone=None) or inside the given zone
    arrived = (lambda p: bool(in_any_zone(p))) if zone is None else zone.contains
    max_err = 0.0
    min_clear = np.inf
    energy = 0.0
    arrival = np.nan
    boxes = WrenchBoxes.unshrunk(wc.u_max, alpha_bar)
    w1 = wc.R0 + wc.gamma_R * (1.0 - h[0])
    w2 = wc.R0 + wc.gamma_R * (1.0 - h[1])

    for k in range(n_steps):
        t = k * dt
        p_r, v_r, psi_r, r_r = ref_at(t)
        pos = x[0:2]
        err = pos - p_r
        max_err = max(max_err, float(np.linalg.norm(err)))
        min_clear = min(min_clear, float(field.signed_distance(pos[None])[0]))
        if max_err > ec.e_max:
            return ExactResult(False, max_err, min_clear, arrival, energy,
                               "tracking_error")
        if min_clear < 0.0:
            return ExactResult(False, max_err, min_clear, arrival, energy,
                               "collision")
        if np.isnan(arrival) and arrived(pos):
            arrival = t
        # PD acceleration command in the plan frame -> body frame
        acc_cmd = ec.kp * (p_r - pos) + ec.kd * (v_r - _world_vel(x))
        psi = x[2]
        c, s = np.cos(psi), np.sin(psi)
        uv_dot_cmd = np.array([c * acc_cmd[0] + s * acc_cmd[1],
                               -s * acc_cmd[0] + c * acc_cmd[1]])
        r_cmd_dot = ec.kpsi * _wrap(psi_r - psi) + ec.kr * (r_r - x[5])
        nudot_cmd = np.array([uv_dot_cmd[0], uv_dot_cmd[1], r_cmd_dot])
        tau_des = fossen.required_wrench(x[3:6][None], nudot_cmd[None], params)[0]
        wk = w_at(t)
        d2, u_star, _ = distance_batch(tau_des[None, None], h[None],
                                       wk[None, None], boxes, params, iters=120)
        u = np.clip(u_star[0, 0], 0.0, wc.u_max)
        energy += float(w1 * u[0] ** 2 + w2 * u[1] ** 2) * dt
        x = fossen.rk4_step(x, u, wk, h, dt, params)

    passed = not np.isnan(arrival)
    reason = "ok" if passed else "no_arrival"
    return ExactResult(passed and max_err <= ec.e_max and min_clear >= 0.0,
                       max_err, min_clear, arrival, energy, reason)


def _world_vel(x):
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([c * x[3] - s * x[4], s * x[3] + c * x[4]])