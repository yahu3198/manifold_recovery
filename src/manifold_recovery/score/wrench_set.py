"""Feasible wrench set W(H, w_hat) and the batched distance/allocation solve.

    W = { B H u + diag(alpha) w_hat : u in [0, u_hi], alpha in [0, a_hi] }

For each (tau_req_k, H, w_hat_k) we solve the box-constrained least squares

    min_{u, alpha} || B H u + diag(alpha) w_hat_k - tau_req_k ||^2

with projected FISTA over z = [u1, u2, a1, a2, a3] (5 vars), fully vectorised
over (sample, waypoint). Outputs the squared distance d2_k, the least-effort
allocation u*_k, and alpha*_k. Validated against scipy.optimize.lsq_linear
in tests/test_wrench_set.py.

Margins (certificate only) shrink the UPPER bounds: u in [0, u_max - rho_u_k],
alpha in [0, alpha_bar - rho_a_k]; thrusters may always idle at zero
(design decision from the feasibility review).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics.params import VesselParams, PARAMS, B_matrix


@dataclass
class WrenchBoxes:
    """Upper bounds; lower bounds are 0. Scalars or per-waypoint arrays (N,)."""
    u_hi: np.ndarray | float
    a_hi: np.ndarray | float

    @staticmethod
    def unshrunk(u_max: float, alpha_bar: float) -> "WrenchBoxes":
        return WrenchBoxes(u_hi=u_max, a_hi=alpha_bar)


def _broadcast_bounds(bound, target_shape):
    b = np.asarray(bound, dtype=float)
    if b.ndim == 0:
        return np.broadcast_to(b, target_shape)
    # (N,) -> (..., N)
    return np.broadcast_to(b, target_shape)


def distance_batch(tau_req: np.ndarray, h: np.ndarray, w_hat: np.ndarray,
                   boxes: WrenchBoxes, params: VesselParams = PARAMS,
                   iters: int = 300, return_resid: bool = False):
    """tau_req (..., N, 3); h (..., 2); w_hat (..., N, 3).

    Returns d2 (..., N), u_star (..., N, 2), alpha_star (..., N, 3), and, if
    ``return_resid``, the signed per-axis residual (..., N, 3) in
    (surge N, sway N, yaw Nm) so the certificate can judge axes separately.
    """
    tau_req = np.asarray(tau_req, float)
    w_hat = np.asarray(w_hat, float)
    h = np.asarray(h, float)
    batch = tau_req.shape[:-2]
    N = tau_req.shape[-2]

    B = B_matrix(params.l)                                    # (3, 2)
    BH = B[None, :, :] * h.reshape(batch + (1, 2))            # (..., 3, 2)
    BH = np.broadcast_to(BH[..., None, :, :], batch + (N, 3, 2))
    A = np.zeros(batch + (N, 3, 5))
    A[..., :, 0:2] = BH
    A[..., 0, 2] = w_hat[..., 0]
    A[..., 1, 3] = w_hat[..., 1]
    A[..., 2, 4] = w_hat[..., 2]

    hi = np.empty(batch + (N, 5))
    u_hi = _broadcast_bounds(boxes.u_hi, batch + (N,))
    a_hi = _broadcast_bounds(boxes.a_hi, batch + (N,))
    hi[..., 0] = u_hi
    hi[..., 1] = u_hi
    hi[..., 2] = a_hi
    hi[..., 3] = a_hi
    hi[..., 4] = a_hi

    # Precondition: z = D zeta with D = hi, so the feasible box is [0, 1]^5.
    # Without this, thruster columns (O(1) x 2353) and environment columns
    # (O(300) x 1) leave A^T A with condition ~1e4-1e5 and FISTA crawls.
    D = np.where(hi > 0.0, hi, 1.0)
    hi_s = np.where(hi > 0.0, 1.0, 0.0)
    As = A * D[..., None, :]
    AtA = np.einsum("...ij,...ik->...jk", As, As)             # (..., N, 5, 5)
    Atb = np.einsum("...ij,...i->...j", As, tau_req)
    L = np.linalg.eigvalsh(AtA)[..., -1]
    L = np.maximum(L, 1e-9)[..., None]

    lo = np.zeros(batch + (N, 5))
    z = 0.5 * hi_s
    y = z.copy()
    t_m = 1.0
    for _ in range(iters):
        grad = np.einsum("...jk,...k->...j", AtA, y) - Atb
        z_new = np.clip(y - grad / L, lo, hi_s)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_m**2))
        y = z_new + ((t_m - 1.0) / t_new) * (z_new - z)
        y = np.clip(y, lo, hi_s)
        z, t_m = z_new, t_new

    z = D * z
    resid = np.einsum("...ij,...j->...i", A, z) - tau_req
    d2 = (resid ** 2).sum(-1)
    if return_resid:
        return d2, z[..., 0:2], z[..., 2:5], resid
    return d2, z[..., 0:2], z[..., 2:5]


def effort_R(u_star: np.ndarray, h: np.ndarray, R0: float, gamma_R: float):
    """||u*||^2_{R(H)} summed over waypoints, R(H) = R0 I + gamma_R diag(1-h)."""
    w1 = R0 + gamma_R * (1.0 - h[..., 0:1])
    w2 = R0 + gamma_R * (1.0 - h[..., 1:2])
    per_k = w1 * u_star[..., 0] ** 2 + w2 * u_star[..., 1] ** 2
    return per_k.sum(axis=-1)
