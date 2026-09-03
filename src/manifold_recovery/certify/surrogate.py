"""Tier-1 surrogate certificate (spec 4.3, R1+R5 corrected; per-axis, rev 2).

- alpha_bar comes from the EAMPC confidence policy (ICRA Eq. 8-9), NOT a
  constant: alpha_bar = min(1, kappa / (sigma_theta + eps)) * mu(H).
- Margins rho_k GROW with lookahead time (force-prediction error growth) and
  shrink only the UPPER box bounds (thrusters may idle at zero).
- Pass iff
      max_k (r_x,k^2 + r_psi,k^2)        <= eps_cert                   (thruster axes, N^2)
  AND sum_k |r_y,k| / |Y_v| * dt         <= sway_drift_frac * e_max    (sway axis, m)
  AND no dock collision AND terminal in zone.
  The sway criterion is the lateral drift the residual would cause if never
  corrected (linear damping, conservative), so it is directly comparable to
  the tier-2 tracking tolerance. A peak bound was tried first and rejected:
  a healthy vessel's turning transients reach 20-40 N for a step or two while
  drifting ~1.3 m over the horizon.

Why per-axis (spike run 1 finding): the twin-thruster B matrix has a zero
sway row at EVERY health level, so the sway residual after crab inversion is
a sideslip demand the hull absorbs, not an actuator infeasibility. Folding it
into a single d2 made the certificate measure "can alpha_y w_y absorb sway"
(which mu(H) = 0 forbids for h1 > 0.8) instead of "can the degraded thrusters
execute this path", producing cert_frac(h1=0.9) = 0.07 < cert_frac(0.15) =
0.69. The sway bound is arbitrated downstream by the tier-2 rollout (e_max).

The same ``certify_batch`` serves the gates, the contraction maps, baseline
B4, and the deployed pipeline; nothing else may implement certification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

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
    mask: np.ndarray          # (...,) bool: full certificate
    max_d2: np.ndarray        # (...,) max_k (r_x^2 + r_psi^2): thruster-axis residual
    max_sway: np.ndarray      # (...,) max_k |r_y|: peak sway residual (N), diagnostic
    sway_drift: np.ndarray    # (...,) integrated uncorrected drift (m), the criterion
    max_d2_total: np.ndarray  # (...,) max_k sum over all three axes (legacy diagnostic)
    min_clear: np.ndarray
    terminal_ok: np.ndarray
    alpha_bar: float

    # Per-criterion masks so gates can report WHICH condition binds.
    @property
    def thrust_ok(self):
        return self._thrust_ok

    @property
    def sway_ok(self):
        return self._sway_ok

    @property
    def clear_ok(self):
        return self.min_clear >= 0.0

    def failure_breakdown(self) -> dict:
        n = max(int(np.prod(self.mask.shape)), 1)
        return {
            "thrust_fail": float((~self._thrust_ok).sum() / n),
            "sway_fail": float((~self._sway_ok).sum() / n),
            "collision": float((~self.clear_ok).sum() / n),
            "terminal_fail": float((~self.terminal_ok).sum() / n),
        }


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
    d2_tot, _, _, resid = distance_batch(inv.tau_req, h, w_seg, boxes, params,
                                         iters=cfg.wrench.cert_iters,
                                         return_resid=True)
    d2_thrust = resid[..., 0] ** 2 + resid[..., 2] ** 2       # surge^2 + yaw^2
    sway = np.abs(resid[..., 1])
    sway_drift = (sway / max(abs(params.yv), 1e-9) * kin.dt).sum(axis=-1)
    d = field.signed_distance(kin.pos)
    zp = MplPath(np.asarray(zone.vertices, float))
    term = zp.contains_points(kin.pos[..., -1, :].reshape(-1, 2)).reshape(
        omega.shape[:-1])
    max_d2 = d2_thrust.max(axis=-1)
    max_sway = sway.max(axis=-1)
    min_clear = d.min(axis=-1)
    thrust_ok = max_d2 <= cfg.wrench.eps_cert
    sway_ok = sway_drift <= cfg.wrench.sway_drift_frac * cfg.exact.e_max
    mask = thrust_ok & sway_ok & (min_clear >= 0.0) & term
    res = CertResult(mask=mask, max_d2=max_d2, max_sway=max_sway, sway_drift=sway_drift,
                     max_d2_total=d2_tot.max(axis=-1), min_clear=min_clear,
                     terminal_ok=term, alpha_bar=ab)
    res._thrust_ok = thrust_ok
    res._sway_ok = sway_ok
    return res