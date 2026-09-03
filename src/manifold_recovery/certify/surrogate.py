"""Tier-1 surrogate certificate (spec 4.3, R1+R5 corrected).

- alpha_bar comes from the EAMPC confidence policy (ICRA Eq. 8-9), NOT a
  constant: alpha_bar = min(1, kappa / (sigma_theta + eps)) * mu(H).
- Margins rho_k GROW with lookahead time (force-prediction error growth) and
  shrink only the UPPER box bounds (thrusters may idle at zero).
- Pass iff max_k d2_k <= eps_cert AND no dock collision AND terminal in zone.
The same ``certify_batch`` serves the gates, the contraction maps, baseline
B4, and the deployed pipeline; nothing else may implement certification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..dynamics.params import VesselParams, PARAMS
from ..scenario import Zone
from ..traj.rtp import RTP
from ..traj.inversion import invert
from ..score.obstacles import DockField
from ..score.wrench_set import WrenchBoxes, distance_batch


def mu_H(h1, h2):
    """ICRA Eq. 9 fault-severity function on combined health h1+h2."""
    s = np.asarray(h1, float) + np.asarray(h2, float)
    return np.where(s > 1.8, 0.0, np.where(s < 0.4, 1.0, 1.0 - s / 2.0))


def alpha_bar_policy(sigma_theta, h1, h2, cfg: Config):
    wc = cfg.wrench
    conf = np.minimum(1.0, wc.kappa / (np.asarray(sigma_theta, float) + wc.eps_conf))
    return conf * mu_H(h1, h2)


def margin_schedule(N: int, dt: float, cfg: Config):
    """Per-waypoint upper-bound shrinkage: linear growth, doubling at rho_growth_T."""
    wc = cfg.wrench
    t = np.arange(N) * dt
    growth = 1.0 + t / wc.rho_growth_T
    rho_u = wc.rho_u_frac * wc.u_max * growth
    rho_a = wc.rho_a * growth
    return rho_u, rho_a


@dataclass
class CertResult:
    mask: np.ndarray          # (...,) bool
    max_d2: np.ndarray
    min_clear: np.ndarray
    terminal_ok: np.ndarray
    alpha_bar: float


def certify_batch(omega: np.ndarray, x0: np.ndarray, zone: Zone,
                  h1: float, h2: float, w_seg: np.ndarray, sigma_theta: float,
                  rtp: RTP, field: DockField, cfg: Config,
                  params: VesselParams = PARAMS) -> CertResult:
    ab = float(alpha_bar_policy(sigma_theta, h1, h2, cfg))
    kin = rtp.kinematics(omega, x0, zone)
    inv = invert(kin, w_seg, alpha_bar_y=ab, params=params, mode="crab",
                 beta_max_deg=cfg.online.beta_max_deg)
    rho_u, rho_a = margin_schedule(cfg.trajectory.N, kin.dt, cfg)
    boxes = WrenchBoxes(u_hi=np.maximum(cfg.wrench.u_max - rho_u, 0.0),
                        a_hi=np.maximum(ab - rho_a, 0.0))
    h = np.broadcast_to(np.array([h1, h2], float), omega.shape[:-1] + (2,))
    d2, _, _ = distance_batch(inv.tau_req, h, w_seg, boxes, params,
                              iters=cfg.wrench.cert_iters)
    d = field.signed_distance(kin.pos)
    from matplotlib.path import Path as MplPath
    zp = MplPath(np.asarray(zone.vertices, float))
    term = zp.contains_points(kin.pos[..., -1, :].reshape(-1, 2)).reshape(
        omega.shape[:-1])
    max_d2 = d2.max(axis=-1)
    min_clear = d.min(axis=-1)
    mask = (max_d2 <= cfg.wrench.eps_cert) & (min_clear >= 0.0) & term
    return CertResult(mask=mask, max_d2=max_d2, min_clear=min_clear,
                      terminal_ok=term, alpha_bar=ab)
