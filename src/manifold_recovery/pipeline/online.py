"""The deployable fault-time pipeline (spec 4.7; rev 3); the ROS 2 node wraps this.

propose(): reachability pre-filter -> decode K candidates split across the
feasible zones (zone one-hot in c, z from the same-zone posterior bank) ->
surrogate-certify -> cluster survivors by (zone, winding) -> fine-tune +
exact-check representatives -> rank.
Per-stage wall times are recorded; ``timing_budget_ok`` implements the
Corollary 2 detection-delay check.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..traj.rtp import RTP
from ..scenario import feasible_zones
from ..score.obstacles import DockField
from ..certify.surrogate import certify_batch, alpha_bar_policy
from ..certify.cluster import cluster
from ..baselines.restarts import PlannerBank
from ..model.decode import Decoder
from .finetune import finetune, Candidate


@dataclass
class Proposal:
    candidates: list
    alpha_bar: float
    n_decoded: int
    n_certified: int
    timings: dict = field(default_factory=dict)


class RecoveryPipeline:
    def __init__(self, ckpt_path: str, cfg: Config, seed: int = 0):
        from ..model.train import load as load_ckpt      # torch import lazily
        model, meta = load_ckpt(ckpt_path)
        self.decoder = Decoder(model, meta, cfg)
        self.cfg = cfg
        self.rtp = RTP(cfg.trajectory)
        self.field = DockField()
        self.planners = PlannerBank(cfg)
        self.rng = np.random.default_rng(seed)

    def propose(self, h1: float, h2: float, w_seg: np.ndarray,
                sigma_theta: float, x0: np.ndarray, w_true: np.ndarray,
                w_dt: float) -> Proposal:
        cfg = self.cfg
        tms = {}
        t0 = time.perf_counter()
        w_bar = float(np.linalg.norm(w_seg[:, :2], axis=1).mean())
        zones = feasible_zones(x0, h1, h2, w_bar, cfg.trajectory.v_nom, cfg.trajectory.T_max)
        tms["condition"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        omega, _g = self.decoder.decode_zones(h1, np.asarray(x0[:3], float),
                                              cfg.online.K, self.rng, zones)
        tms["decode"] = time.perf_counter() - t0
        if len(omega) == 0:
            tms["total"] = sum(tms.values())
            return Proposal(candidates=[], alpha_bar=float(alpha_bar_policy(sigma_theta, h1, h2, cfg)),
                            n_decoded=0, n_certified=0, timings=tms)

        t0 = time.perf_counter()
        T_h, dt = self.rtp.horizon()
        w_grid = _resample_forces(w_seg, w_dt, cfg.trajectory.N, dt)
        ab = float(alpha_bar_policy(sigma_theta, h1, h2, cfg))
        cert = certify_batch(omega, np.asarray(x0[:3], float), h1, h2,
                             np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                             sigma_theta, self.rtp, self.field, cfg)
        tms["certify"] = time.perf_counter() - t0
        ok = np.flatnonzero(cert.mask)
        cands: list[Candidate] = []
        if len(ok):
            t0 = time.perf_counter()
            kin_all = self.rtp.kinematics(omega, np.asarray(x0[:3], float))
            reps_local = cluster(kin_all.pos[ok], kin_all.vel[ok], -cert.max_d2[ok])
            reps = {sig: int(ok[i]) for sig, i in reps_local.items()}
            tms["cluster"] = time.perf_counter() - t0

            from ..traj.inversion import invert
            inv = invert(kin_all, np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                         ab, mode="crab", beta_max_deg=cfg.online.beta_max_deg)
            cands = finetune(reps, kin_all.pos, inv.psi, omega, np.asarray(x0[:3], float),
                             self.planners, np.array([h1, h2]), ab, T_h, w_grid,
                             w_true, w_dt, self.field, cfg)
        cands.sort(key=lambda cd: (not cd.cert.passed, cd.plan_cost))
        tms["total"] = sum(v for k, v in tms.items() if k != "total")
        return Proposal(candidates=cands[:cfg.online.k_present], alpha_bar=ab,
                        n_decoded=len(omega), n_certified=int(len(ok)), timings=tms)

    @staticmethod
    def timing_budget_ok(t_detect: float, delta_op: float, tau_bar_d: float):
        return (t_detect + delta_op) < tau_bar_d


def _resample_forces(w_seg: np.ndarray, w_dt: float, n: int, dt: float):
    t_src = np.arange(len(w_seg)) * w_dt
    t_tgt = np.minimum(np.arange(n) * dt, t_src[-1])
    return np.stack([np.interp(t_tgt, t_src, w_seg[:, j]) for j in range(3)], axis=1)
