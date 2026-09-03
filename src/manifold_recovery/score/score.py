"""Training score R(xi; c, g) assembly (spec 4.4, R1-corrected).

R = -(lambda_o J_obs + lambda_s J_smooth + lambda_f J_feas + lambda_e J_effort)

J_feas uses UNSHRUNK boxes (soft signal); shrunk boxes are certificate-only.
Returns the score and a term dict for diagnostics, gates, and plots.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..dynamics.params import VesselParams, PARAMS
from ..scenario import Zone
from ..traj.rtp import RTP
from ..traj.inversion import invert
from .obstacles import DockField, j_obs
from .wrench_set import WrenchBoxes, distance_batch, effort_R


def score_batch(omega: np.ndarray, h: np.ndarray, w_seg: np.ndarray,
                x0: np.ndarray, zone: Zone, rtp: RTP, field: DockField,
                cfg: Config, params: VesselParams = PARAMS,
                inversion_mode: str = "crab"):
    """omega (..., 2Bw); h (..., 2); w_seg (..., N, 3) body-frame forces."""
    sc, wc = cfg.score, cfg.wrench
    kin = rtp.kinematics(omega, x0, zone)
    inv = invert(kin, w_seg, alpha_bar_y=wc.alpha_bar, params=params,
                 mode=inversion_mode, beta_max_deg=cfg.online.beta_max_deg)
    boxes = WrenchBoxes.unshrunk(wc.u_max, wc.alpha_bar)
    d2, u_star, alpha_star = distance_batch(inv.tau_req, h, w_seg, boxes,
                                            params, iters=wc.pgd_iters)
    J_obs_v, d = j_obs(field, kin.pos, kin.vel, sc.eps_obs)
    J_smooth = (kin.acc ** 2).sum(-1).sum(-1)
    J_feas = d2.sum(-1)
    J_effort = effort_R(u_star, h, wc.R0, wc.gamma_R)
    R = -(sc.lambda_o * J_obs_v + sc.lambda_s * J_smooth
          + sc.lambda_f * J_feas + sc.lambda_e * J_effort)
    terms = {
        "J_obs": J_obs_v, "J_smooth": J_smooth, "J_feas": J_feas,
        "J_effort": J_effort, "min_clear": d.min(axis=-1),
        "collides": (d.min(axis=-1) < 0.0),
        "d2_max": d2.max(axis=-1),
        "kin": kin, "inv": inv, "u_star": u_star, "alpha_star": alpha_star,
    }
    return R, terms
