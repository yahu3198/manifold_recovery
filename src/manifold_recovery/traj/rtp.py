"""Residual trajectory primitives (RTP) with analytic derivatives (rev 3).

    xi(omega; x0) = xi_base(p_0, p_g) + F Phi w        (spec 4.1)

Rev 3 representation:  omega = [p_g (2), w (2 Bw)]  (flat, dim 2 + 2 Bw).
The goal point p_g is part of the generated object, so the manifold decides
WHICH zone to enter and WHERE; the base line runs from the sampled start p_0
(passed in as x0, batched) to p_g and the pinned residual F Phi w sits on top.

- Phi in R^{N x Bw}: logistic basis (Osa Eq. 23), centers uniform on [0, 1].
- F = diag(s(t)) with s(0) = s(1) = 0: endpoints pinned by construction.
- Horizon: rev 3 uses a FIXED T_h (trajectory.T_fixed) for every candidate
  so that x0 and p_g may vary per sample without per-sample time grids; the
  implied speed is L / T_fixed (0.6-1.2 m/s over the start arc).
- Time derivatives are analytic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TrajectoryCfg


def make_basis(N: int, Bw: int, alpha_b: float):
    t = np.linspace(0.0, 1.0, N)[:, None]
    c = np.linspace(0.0, 1.0, Bw)[None, :]
    b = 1.0 / (1.0 + np.exp(alpha_b * (t - c)))
    db = -alpha_b * b * (1.0 - b)
    ddb = alpha_b**2 * b * (1.0 - b) * (1.0 - 2.0 * b)
    return b, db, ddb, np.linalg.cond(b)


def make_scaling(N: int, eps: float):
    t = np.linspace(0.0, 1.0, N)
    s = np.minimum(1.0, np.minimum(t / eps, (1.0 - t) / eps))
    ds = np.zeros(N)
    ds[t < eps] = 1.0 / eps
    ds[t > 1.0 - eps] = -1.0 / eps
    return s, ds


@dataclass
class Kinematics:
    pos: np.ndarray   # (..., N, 2)
    vel: np.ndarray   # (..., N, 2)  physical time
    acc: np.ndarray   # (..., N, 2)
    T_h: float
    dt: float


class RTP:
    def __init__(self, cfg: TrajectoryCfg):
        self.cfg = cfg
        self.Phi, self.dPhi, self.ddPhi, self.cond = make_basis(cfg.N, cfg.Bw, cfg.alpha_basis)
        self.s, self.ds = make_scaling(cfg.N, cfg.ramp_eps)
        SPhi = self.s[:, None] * self.Phi
        self._SPhi = SPhi
        self._P = np.linalg.pinv(SPhi)               # (Bw, N)

    # -- representation helpers --------------------------------------------
    @property
    def dim(self) -> int:
        return 2 + 2 * self.cfg.Bw

    def split(self, omega: np.ndarray):
        """omega (..., 2+2Bw) -> p_g (..., 2), w (..., Bw, 2)."""
        om = np.asarray(omega, float)
        pg = om[..., :2]
        w = om[..., 2:].reshape(om.shape[:-1] + (self.cfg.Bw, 2))
        return pg, w

    def join(self, pg: np.ndarray, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, float)
        return np.concatenate([np.asarray(pg, float), w.reshape(w.shape[:-2] + (-1,))], axis=-1)

    def horizon(self, x0=None, zone=None):
        """Fixed horizon (rev 3). Arguments kept for legacy call sites."""
        if self.cfg.horizon_mode != "fixed":
            raise ValueError("rev 3 supports horizon_mode 'fixed' only")
        T_h = float(self.cfg.T_fixed)
        return T_h, T_h / (self.cfg.N - 1)

    # -- forward map --------------------------------------------------------
    def kinematics(self, omega: np.ndarray, x0: np.ndarray, zone=None) -> Kinematics:
        """omega (..., 2+2Bw); x0 (..., 3) or (3,) broadcastable to omega's batch."""
        cfg = self.cfg
        pg, w = self.split(omega)
        T_h, dt = self.horizon()
        p0 = np.broadcast_to(np.asarray(x0, float)[..., :2], pg.shape)
        t = np.linspace(0.0, 1.0, cfg.N)
        d = (pg - p0)[..., None, :]                                   # (..., 1, 2)
        base = p0[..., None, :] + t[:, None] * d

        res = np.einsum("nb,...bd->...nd", self.Phi, w)
        dres = np.einsum("nb,...bd->...nd", self.dPhi, w)
        ddres = np.einsum("nb,...bd->...nd", self.ddPhi, w)
        s = self.s[:, None]
        ds = self.ds[:, None]
        pos = base + s * res
        vel_n = d + ds * res + s * dres
        acc_n = 2.0 * ds * dres + s * ddres
        return Kinematics(pos=pos, vel=vel_n / T_h, acc=acc_n / T_h**2, T_h=T_h, dt=dt)

    # -- inverse map --------------------------------------------------------
    def project(self, waypoints: np.ndarray, x0: np.ndarray, zone=None) -> np.ndarray:
        """Least-squares omega for waypoints (..., N, 2); p_g = last waypoint."""
        wp = np.asarray(waypoints, float)
        pg = wp[..., -1, :]
        p0 = np.broadcast_to(np.asarray(x0, float)[..., :2], pg.shape)
        t = np.linspace(0.0, 1.0, self.cfg.N)
        base = p0[..., None, :] + t[:, None] * (pg - p0)[..., None, :]
        w = np.einsum("bn,...nd->...bd", self._P, wp - base)
        return self.join(pg, w)