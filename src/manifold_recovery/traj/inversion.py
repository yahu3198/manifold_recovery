"""Differential inversion: candidate trajectory -> required wrench sequence.

R1 fix: tau_req = M nudot + C(nu) nu + D(nu) nu, with NO -w_hat term. The
environment appears only inside the feasible wrench set W (score/wrench_set).

R2 fix: instead of pure tangent heading (which forces v ~ 0 and demands sway
force m*u*r on every turn that a twin-thruster USV cannot produce), the
default mode solves a per-waypoint crab angle beta so the body sideslips.
With course angle chi = atan2(ydot, xdot):

    psi = chi - beta,   u = V cos(beta),   v = V sin(beta),   V = |pdot|

beta_k is chosen on a grid in [-beta_max, beta_max] to minimise the distance
of the quasi-static sway requirement

    tau_y(beta) ~= m u r - (yv + yvv |v|) v          (m vdot dropped)

to the admissible sway interval [0, alpha_bar_y * w_hat_y] (one-sided in the
sign of w_hat_y). This is a heuristic that shapes (psi, nu); the final
feasibility of all three axes is judged by the wrench-set solve regardless.
Yaw rate and nudot are then finite-differenced from the smooth psi/nu
sequences (acceptable at this surrogate level; spec 4.2 step 4).

mode="tangent" (beta = 0) is retained for the ablation and regression test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics.params import VesselParams, PARAMS
from ..dynamics.fossen import required_wrench
from .rtp import Kinematics


@dataclass
class InversionResult:
    psi: np.ndarray      # (..., N)
    beta: np.ndarray     # (..., N) crab angle actually used
    nu: np.ndarray       # (..., N, 3)
    nudot: np.ndarray    # (..., N, 3)
    tau_req: np.ndarray  # (..., N, 3)


def _moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[..., :1], pad, axis=-1), x,
                         np.repeat(x[..., -1:], pad, axis=-1)], axis=-1)
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="valid"), -1, xp)


def invert(kin: Kinematics, w_hat: np.ndarray, alpha_bar_y: float,
           params: VesselParams = PARAMS, mode: str = "crab",
           beta_max_deg: float = 25.0, n_beta: int = 41) -> InversionResult:
    """w_hat: (..., N, 3) body-frame environmental force realisation."""
    vel, acc, dt = kin.vel, kin.acc, kin.dt
    V = np.linalg.norm(vel, axis=-1)                      # (..., N)
    V_safe = np.maximum(V, 1e-6)
    chi = np.unwrap(np.arctan2(vel[..., 1], vel[..., 0]), axis=-1)
    chidot = (vel[..., 0] * acc[..., 1] - vel[..., 1] * acc[..., 0]) / V_safe**2

    if mode == "tangent":
        beta = np.zeros_like(V)
    elif mode == "crab":
        beta_max = np.deg2rad(beta_max_deg)
        grid = np.linspace(-beta_max, beta_max, n_beta)   # (G,)
        u_g = V_safe[..., None] * np.cos(grid)            # (..., N, G)
        v_g = V_safe[..., None] * np.sin(grid)
        r0 = chidot                                        # pass-1 yaw rate (beta=0)
        tau_y = params.m * u_g * r0[..., None] \
            - (params.yv + params.yvv * np.abs(v_g)) * v_g
        wy = w_hat[..., 1]
        lo = np.minimum(0.0, alpha_bar_y * wy)[..., None]
        hi = np.maximum(0.0, alpha_bar_y * wy)[..., None]
        resid = np.maximum(0.0, lo - tau_y) + np.maximum(0.0, tau_y - hi)
        # Tie-break toward small |beta| for smoothness.
        resid = resid + 1e-3 * np.abs(grid) * np.maximum(np.abs(tau_y).max(-1, keepdims=True), 1.0)
        beta = grid[np.argmin(resid, axis=-1)]
        beta = _moving_average(beta, 5)
    else:
        raise ValueError(f"unknown inversion mode: {mode}")

    psi = chi - beta
    u_b = V * np.cos(beta)
    v_b = V * np.sin(beta)
    r = np.gradient(psi, dt, axis=-1)
    nu = np.stack([u_b, v_b, r], axis=-1)
    nudot = np.gradient(nu, dt, axis=-2)
    tau_req = required_wrench(nu, nudot, params)          # R1: no -w_hat
    return InversionResult(psi=psi, beta=beta, nu=nu, nudot=nudot, tau_req=tau_req)
