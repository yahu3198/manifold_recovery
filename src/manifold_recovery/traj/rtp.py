"""Residual trajectory primitives (RTP) with analytic derivatives.

    xi(omega; x0, g) = xi_base(x0, p_g) + F Phi omega        (spec 4.1)

- xi_base: straight line from the fault-time position to the zone centroid.
- Phi in R^{N x Bw}: logistic basis b_i(t) = 1/(1+exp(alpha_b (t - c_i))),
  centers uniform on [0, 1] (Osa Eq. 23; low condition number, App. A).
- F = diag(s(t)) per coordinate with s(0) = s(1) = 0, so endpoints are pinned
  by construction. s(t) = min(1, t/eps, (1-t)/eps) (continuous ramp).
- Time derivatives are ANALYTIC (basis + ramp differentiated in closed form);
  no finite differences anywhere in training (R2-adjacent quality fix).

omega convention at module boundaries: flat (..., 2*Bw), reshaped internally
to (..., Bw, 2) with column order (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TrajectoryCfg
from ..scenario import Zone, DETOUR_POLYLINES, detour_length


def make_basis(N: int, Bw: int, alpha_b: float):
    """Return Phi, dPhi, ddPhi (each (N, Bw)) on normalized time, plus cond."""
    t = np.linspace(0.0, 1.0, N)[:, None]           # (N, 1)
    c = np.linspace(0.0, 1.0, Bw)[None, :]          # (1, Bw)
    b = 1.0 / (1.0 + np.exp(alpha_b * (t - c)))     # logistic
    db = -alpha_b * b * (1.0 - b)
    ddb = alpha_b**2 * b * (1.0 - b) * (1.0 - 2.0 * b)
    cond = np.linalg.cond(b)
    return b, db, ddb, cond


def make_scaling(N: int, eps: float):
    """s(t) = min(1, t/eps, (1-t)/eps) and its a.e. derivative."""
    t = np.linspace(0.0, 1.0, N)
    s = np.minimum(1.0, np.minimum(t / eps, (1.0 - t) / eps))
    ds = np.zeros(N)
    ds[t < eps] = 1.0 / eps
    ds[t > 1.0 - eps] = -1.0 / eps
    return s, ds


@dataclass
class Kinematics:
    pos: np.ndarray   # (..., N, 2)
    vel: np.ndarray   # (..., N, 2)  d/dtau (physical time)
    acc: np.ndarray   # (..., N, 2)
    T_h: float
    dt: float


class RTP:
    def __init__(self, cfg: TrajectoryCfg):
        self.cfg = cfg
        self.Phi, self.dPhi, self.ddPhi, self.cond = make_basis(cfg.N, cfg.Bw, cfg.alpha_basis)
        self.s, self.ds = make_scaling(cfg.N, cfg.ramp_eps)
        # Pseudo-inverse of the pinned operator (F Phi) for projection/proposal.
        SPhi = self.s[:, None] * self.Phi
        self._SPhi = SPhi
        self._P = np.linalg.pinv(SPhi)               # (Bw, N)

    # -- helpers ------------------------------------------------------------
    def horizon(self, x0: np.ndarray, zone: Zone):
        """Scalar horizon shared by every candidate for this (x0, zone).

        horizon_mode "straight" keeps the original d/v_nom. "detour" uses the
        longest admissible detour arc so around-dock classes are traversable at
        v_nom (the straight-line horizon forced ~2 m/s on a 123-145 m detour).
        """
        d = float(np.linalg.norm(zone.center - np.asarray(x0[:2], float)))
        if self.cfg.horizon_mode == "detour":
            d = max(d, max(detour_length(k, x0, zone) for k in DETOUR_POLYLINES))
        elif self.cfg.horizon_mode != "straight":
            raise ValueError(f"unknown horizon_mode: {self.cfg.horizon_mode}")
        T_h = float(np.clip(d / self.cfg.v_nom, self.cfg.T_min, self.cfg.T_max))
        return T_h, T_h / (self.cfg.N - 1)

    @staticmethod
    def _reshape(omega: np.ndarray, Bw: int) -> np.ndarray:
        return omega.reshape(omega.shape[:-1] + (Bw, 2))

    # -- forward map --------------------------------------------------------
    def kinematics(self, omega: np.ndarray, x0: np.ndarray, zone: Zone) -> Kinematics:
        """Positions and analytic first/second physical-time derivatives."""
        cfg = self.cfg
        om = self._reshape(np.asarray(omega, float), cfg.Bw)        # (..., Bw, 2)
        T_h, dt = self.horizon(x0, zone)
        p0 = np.asarray(x0[:2], float)
        pg = zone.center
        t = np.linspace(0.0, 1.0, cfg.N)
        base = p0[None, :] + t[:, None] * (pg - p0)[None, :]        # (N, 2)

        res = np.einsum("nb,...bd->...nd", self.Phi, om)
        dres = np.einsum("nb,...bd->...nd", self.dPhi, om)
        ddres = np.einsum("nb,...bd->...nd", self.ddPhi, om)
        s = self.s[:, None]
        ds = self.ds[:, None]

        pos = base + s * res
        # d/dt_norm, then scale by 1/T_h per physical-time derivative order.
        vel_n = (pg - p0)[None, :] + ds * res + s * dres
        acc_n = 2.0 * ds * dres + s * ddres                          # s'' = 0 a.e.
        vel = vel_n / T_h
        acc = acc_n / T_h**2
        return Kinematics(pos=pos, vel=vel, acc=acc, T_h=T_h, dt=dt)

    # -- inverse map --------------------------------------------------------
    def project(self, waypoints: np.ndarray, x0: np.ndarray, zone: Zone) -> np.ndarray:
        """Least-squares omega reproducing given waypoints (..., N, 2) -> flat."""
        cfg = self.cfg
        p0 = np.asarray(x0[:2], float)
        pg = zone.center
        t = np.linspace(0.0, 1.0, cfg.N)
        base = p0[None, :] + t[:, None] * (pg - p0)[None, :]
        resid = np.asarray(waypoints, float) - base
        om = np.einsum("bn,...nd->...bd", self._P, resid)
        return om.reshape(om.shape[:-2] + (2 * cfg.Bw,))