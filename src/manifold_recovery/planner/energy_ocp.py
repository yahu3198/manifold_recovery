"""Energy-optimal reference OCP (ICRA Eq. 13), CasADi + IPOPT.

    min_{x, u, alpha}  int ||u||^2_{R(H)} dt
    s.t.  nominal dynamics with tau = B H u + diag(alpha) w_hat_k
          u in [0, u_max],  alpha in [0, alpha_bar]
          dock separation >= dock_margin (soft, slack-penalised)
          x_N inside the target zone (halfspace form, convex)

The repo exposes NO callable for this (Phase 0 finding R3), so this module is
the single optimizer used by the fine-tuner, baseline B1, and the G3 optimum
reference. Built once per (zone, N_ocp); x0, H, alpha_bar, dt, and the force
trajectory are Opti parameters, so repeated warm-started solves are cheap.

Dock avoidance uses the convex-polygon signed distance surrogate
sd(p) ~ max_i (a_i p - b_i) (negative inside), smoothed with a numerically
stable log-sum-exp so IPOPT sees C^1 constraints.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..config import Config
from ..dynamics.params import VesselParams, PARAMS
from ..scenario import Zone, DOCK_VERTICES, _poly_halfspaces


@dataclass
class PlanResult:
    xi: np.ndarray          # (N_ocp+1, 2) positions
    X: np.ndarray           # (6, N_ocp+1) full states
    U: np.ndarray           # (2, N_ocp)
    alpha: np.ndarray       # (3, N_ocp)
    cost: float             # effort integral ONLY (slack and regularisers excluded)
    converged: bool         # IPOPT Solve_Succeeded or Solved_To_Acceptable_Level
    wall_time: float
    slack_total: float = 0.0    # sum of dock slacks (m); > slack_tol = margin violated
    slack_max: float = 0.0
    obj_total: float = float("nan")   # effort + reg + w_slack * slack (what IPOPT minimised)
    status: str = ""            # IPOPT return_status of the final attempt
    n_attempts: int = 1

    @property
    def feasible(self) -> bool:
        """Converged AND dock margins honoured. Only feasible solutions may be
        quoted as an optimum (G3b) or counted as a class (G4)."""
        return bool(self.converged and self._slack_ok)


def _lse_max(vals, k: float = 1.0):
    """Stable smooth max over a CasADi column vector."""
    m = ca.mmax(vals)
    return m + ca.log(ca.sum1(ca.exp(k * (vals - m)))) / k


class EnergyOCP:
    def __init__(self, zone: Zone, cfg: Config, params: VesselParams = PARAMS):
        self.cfg, self.params, self.zone = cfg, params, zone
        pc, wc = cfg.planner, cfg.wrench
        N = pc.N_ocp
        p = params
        opti = ca.Opti()

        X = opti.variable(6, N + 1)
        U = opti.variable(2, N)
        A = opti.variable(3, N)
        S = opti.variable(len(DOCK_VERTICES), N + 1)   # dock slacks >= 0

        P_x0 = opti.parameter(6)
        P_h = opti.parameter(2)
        P_ab = opti.parameter(1)                        # alpha_bar
        P_dt = opti.parameter(1)
        P_w = opti.parameter(3, N)

        Bm = ca.DM([[1.0, 1.0], [0.0, 0.0], [-p.l / 2, p.l / 2]])
        Mi = ca.DM([1.0 / p.m, 1.0 / p.m, 1.0 / p.Izz])

        def f(x, u, w, a):
            psi = x[2]
            nu = x[3:6]
            tau = Bm @ (P_h * u) + a * w
            cvec = ca.vertcat(-p.m * nu[1] * nu[2], p.m * nu[0] * nu[2], 0.0)
            dvec = ca.vertcat(-(p.xu + p.xuu * ca.fabs(nu[0])) * nu[0],
                              -(p.yv + p.yvv * ca.fabs(nu[1])) * nu[1],
                              -(p.nr + p.nrr * ca.fabs(nu[2])) * nu[2])
            nudot = Mi * (tau - cvec - dvec)
            return ca.vertcat(ca.cos(psi) * nu[0] - ca.sin(psi) * nu[1],
                              ca.sin(psi) * nu[0] + ca.cos(psi) * nu[1],
                              nu[2], nudot)

        # dynamics (RK4), bounds
        for k in range(N):
            xk, uk, wk, ak = X[:, k], U[:, k], P_w[:, k], A[:, k]
            k1 = f(xk, uk, wk, ak)
            k2 = f(xk + P_dt / 2 * k1, uk, wk, ak)
            k3 = f(xk + P_dt / 2 * k2, uk, wk, ak)
            k4 = f(xk + P_dt * k3, uk, wk, ak)
            opti.subject_to(X[:, k + 1] == xk + P_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        opti.subject_to(opti.bounded(0.0, ca.vec(U), p.u_max))
        for k in range(N):
            opti.subject_to(A[:, k] >= 0.0)
            opti.subject_to(A[:, k] <= P_ab)
        opti.subject_to(ca.vec(S) >= 0.0)
        opti.subject_to(X[:, 0] == P_x0)

        # dock separation (soft) via stable smooth max of halfspace margins
        self._dock_hs = [_poly_halfspaces(np.asarray(v, float)) for v in DOCK_VERTICES]
        for k in range(N + 1):
            pt = X[0:2, k]
            for j, (Ah, bh) in enumerate(self._dock_hs):
                vals = ca.DM(Ah) @ pt - ca.DM(bh)
                sd = _lse_max(vals, k=1.0)
                opti.subject_to(sd >= pc.dock_margin - S[j, k])

        # terminal inside zone
        Az, bz = zone.halfspaces()
        opti.subject_to(ca.DM(Az) @ X[0:2, N] <= ca.DM(bz))
        # arrive gently
        opti.subject_to(opti.bounded(-0.6, X[3, N], 1.2))
        opti.subject_to(opti.bounded(-0.6, X[4, N], 0.6))

        w1 = wc.R0 + wc.gamma_R * (1.0 - P_h[0])
        w2 = wc.R0 + wc.gamma_R * (1.0 - P_h[1])
        effort = ca.sum2(w1 * U[0, :] ** 2 + w2 * U[1, :] ** 2) * P_dt
        reg = 1e-4 * ca.sumsqr(A) + 1e-6 * ca.sumsqr(U[:, 1:] - U[:, :-1])
        obj = effort + reg + pc.w_slack * ca.sum1(ca.sum2(S))
        opti.minimize(obj)
        self._obj = obj
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "sb": "yes",
                     "max_iter": pc.ipopt_max_iter,
                     "acceptable_tol": 1e-4, "tol": 1e-6,
                     "acceptable_iter": 5})
        self.opti = opti
        self._vars = (X, U, A, S)
        self._pars = (P_x0, P_h, P_ab, P_dt, P_w)
        self._effort = effort

    # ------------------------------------------------------------------
    def _warm_from_xi(self, xi: np.ndarray, x0: np.ndarray, dt: float):
        """Resample waypoints (Nw, 2) to N_ocp+1 and build a full state guess."""
        N = self.cfg.planner.N_ocp
        t_src = np.linspace(0.0, 1.0, len(xi))
        t_tgt = np.linspace(0.0, 1.0, N + 1)
        P = np.stack([np.interp(t_tgt, t_src, xi[:, j]) for j in range(2)], axis=1)
        V = np.gradient(P, dt, axis=0)
        psi = np.unwrap(np.arctan2(V[:, 1], V[:, 0]))
        psi[0] = x0[2]
        Xg = np.zeros((6, N + 1))
        Xg[0:2, :] = P.T
        Xg[2, :] = psi
        sp = np.linalg.norm(V, axis=1)
        Xg[3, :] = sp
        Xg[5, :] = np.gradient(psi, dt)
        return Xg

    def solve(self, x0: np.ndarray, h: np.ndarray, alpha_bar: float,
              w_traj: np.ndarray, T_h: float,
              warm_xi: np.ndarray | None = None) -> PlanResult:
        """w_traj: (>=N_ocp, 3) force realisation on the plan horizon.

        Rev 2: (i) thrust initial guess from drag balance at the seed speed,
        split by health (0.2 u_max = 940 N total against ~250 N drag at 1 m/s
        started IPOPT far from feasibility); (ii) on non-convergence, up to
        cfg.planner.ipopt_retries warm re-solves from the last iterate;
        (iii) slack and IPOPT status are reported so callers can distinguish
        "converged to a margin-violating point" from a true optimum.
        """
        N = self.cfg.planner.N_ocp
        pc, p = self.cfg.planner, self.params
        dt = T_h / N
        X, U, A, S = self._vars
        P_x0, P_h, P_ab, P_dt, P_w = self._pars
        o = self.opti
        o.set_value(P_x0, np.asarray(x0, float))
        o.set_value(P_h, np.asarray(h, float))
        o.set_value(P_ab, float(alpha_bar))
        o.set_value(P_dt, dt)
        idx = np.linspace(0, len(w_traj) - 1, N).round().astype(int)
        o.set_value(P_w, np.asarray(w_traj, float)[idx].T)

        if warm_xi is not None:
            Xg = self._warm_from_xi(np.asarray(warm_xi, float), x0, dt)
        else:
            Xg = self._warm_from_xi(
                np.linspace(x0[:2], self.zone.center, N + 1), x0, dt)
        # Drag-balance thrust guess: total = (|xu| + |xuu| sp) sp, shared in
        # proportion to health so the degraded thruster is not over-seeded.
        sp = np.clip(Xg[3, :N], 0.05, None)
        drag = (abs(p.xu) + abs(p.xuu) * sp) * sp
        hh = np.clip(np.asarray(h, float), 1e-3, 1.0)
        share = hh / hh.sum()
        Ug = np.clip(np.outer(share, drag) / hh[:, None], 0.0, p.u_max)
        o.set_initial(X, Xg)
        o.set_initial(U, Ug)
        o.set_initial(A, np.full((3, N), 0.5 * max(alpha_bar, 1e-3)))
        o.set_initial(S, np.zeros((len(DOCK_VERTICES), N + 1)))

        t0 = time.perf_counter()
        ok, status, attempts = False, "", 0
        sol = None
        for attempt in range(1 + max(int(pc.ipopt_retries), 0)):
            attempts += 1
            try:
                sol = o.solve()
                status = str(sol.stats().get("return_status", ""))
                ok = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
            except RuntimeError:
                sol = o.debug
                status = str(o.stats().get("return_status", "exception"))
                ok = False
            if ok:
                break
            # warm the next attempt from the last iterate
            o.set_initial(X, np.asarray(sol.value(X)))
            o.set_initial(U, np.asarray(sol.value(U)))
            o.set_initial(A, np.asarray(sol.value(A)))
            o.set_initial(S, np.asarray(sol.value(S)))
        wall = time.perf_counter() - t0
        Xv = np.asarray(sol.value(X))
        Sv = np.asarray(sol.value(S))
        res = PlanResult(xi=Xv[0:2, :].T, X=Xv,
                         U=np.asarray(sol.value(U)),
                         alpha=np.asarray(sol.value(A)),
                         cost=float(sol.value(self._effort)),
                         converged=ok, wall_time=wall,
                         slack_total=float(np.clip(Sv, 0.0, None).sum()),
                         slack_max=float(np.clip(Sv, 0.0, None).max()),
                         obj_total=float(sol.value(self._obj)),
                         status=status, n_attempts=attempts)
        res._slack_ok = res.slack_total <= pc.slack_tol
        return res