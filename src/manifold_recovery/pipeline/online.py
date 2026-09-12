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
from ..scenario import feasible_zones, ZONES
from ..score.obstacles import DockField
from ..certify.surrogate import certify_batch, alpha_bar_policy
from ..certify.cluster import cluster
from ..baselines.restarts import PlannerBank
from ..model.decode import Decoder
from ..data.proposal import ProposalSampler
from .finetune import finetune, Candidate


@dataclass
class Proposal:
    candidates: list
    alpha_bar: float
    n_decoded: int
    n_certified: int
    timings: dict = field(default_factory=dict)
    cert_breakdown: dict = field(default_factory=dict)


class RecoveryPipeline:
    def __init__(self, ckpt_path: str | None, cfg: Config, seed: int = 0):
        """ckpt_path=None builds a pipeline without a decoder (B4 arm only)."""
        self.decoder = None
        if ckpt_path is not None:
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
        """Manifold arm: decode across feasible zones, then the shared tail."""
        cfg = self.cfg
        tms = {}
        t0 = time.perf_counter()
        w_bar = float(np.linalg.norm(w_seg[:, :2], axis=1).mean())
        zones = feasible_zones(x0, h1, h2, w_bar, cfg.trajectory.v_nom, cfg.trajectory.T_max)
        tms["prefilter_empty"] = float(len(zones) == 0)
        if not zones:
            # The spike pre-filter discounts v_nom by thrust authority and is not
            # exercised offline (starts 100-140 m out, v_nom = 1). Decode for every
            # zone and let the certificate decide rather than publishing nothing.
            zones = list(ZONES)
        tms["condition"] = time.perf_counter() - t0
        if self.decoder is None:
            raise RuntimeError("propose() needs a checkpoint; this pipeline was built for the B4 arm")
        t0 = time.perf_counter()
        omega, _g = self.decoder.decode_zones(h1, np.asarray(x0[:3], float),
                                              cfg.online.K, self.rng, zones)
        tms["decode"] = time.perf_counter() - t0
        return self._finish(omega, h1, h2, w_seg, sigma_theta, x0, w_true, w_dt, tms)

    def propose_b4(self, h1: float, h2: float, w_seg: np.ndarray,
                   sigma_theta: float, x0: np.ndarray, w_true: np.ndarray,
                   w_dt: float, K: int | None = None) -> Proposal:
        """Rejection-sampling arm: identical tail, proposal samples instead of
        decodes. K defaults to online.K so the two arms certify the same number
        of candidates (latency-matched); the paper's B4 budget variant is in
        baselines/rejection.py."""
        cfg = self.cfg
        tms = {}
        t0 = time.perf_counter()
        prop = ProposalSampler(self.rtp, cfg.data.prop_mid_std_m, self.rng, mix=cfg.data.prop_mix)
        omega = prop.sample(np.asarray(x0[:3], float), self.rng, K or cfg.online.K)
        tms["decode"] = time.perf_counter() - t0
        return self._finish(omega, h1, h2, w_seg, sigma_theta, x0, w_true, w_dt, tms)

    def _finish(self, omega, h1, h2, w_seg, sigma_theta, x0, w_true, w_dt, tms) -> Proposal:
        cfg = self.cfg
        ab = float(alpha_bar_policy(sigma_theta, h1, h2, cfg))
        if len(omega) == 0:
            tms["total"] = sum(tms.values())
            return Proposal(candidates=[], alpha_bar=ab, n_decoded=0, n_certified=0, timings=tms)
        t0 = time.perf_counter()
        T_h, dt = self.rtp.horizon()
        w_grid = _resample_forces(w_seg, w_dt, cfg.trajectory.N, dt)
        x03 = np.asarray(x0[:3], float)
        cert = certify_batch(omega, x03, h1, h2,
                             np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                             sigma_theta, self.rtp, self.field, cfg)
        tms["certify"] = time.perf_counter() - t0
        ok = np.flatnonzero(cert.mask)
        cands: list[Candidate] = []
        if len(ok):
            t0 = time.perf_counter()
            kin_all = self.rtp.kinematics(omega, x03)
            reps_local = cluster(kin_all.pos[ok], kin_all.vel[ok], -cert.max_d2[ok])
            reps = {sig: int(ok[i]) for sig, i in reps_local.items()}
            tms["cluster"] = time.perf_counter() - t0
            from ..traj.inversion import invert
            inv = invert(kin_all, np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                         ab, mode="crab", beta_max_deg=cfg.online.beta_max_deg)
            t0 = time.perf_counter()
            cands = finetune(reps, kin_all.pos, inv.psi, omega, np.asarray(x0, float),
                             self.planners, np.array([h1, h2]), ab, T_h, w_grid,
                             w_true, w_dt, self.field, cfg)
            tms["finetune_and_rollout"] = time.perf_counter() - t0
        cands.sort(key=lambda cd: cd.plan_cost)
        tms["total"] = sum(v for k, v in tms.items() if k != "total")
        return Proposal(candidates=cands[:cfg.online.k_present], alpha_bar=ab,
                        n_decoded=len(omega), n_certified=int(len(ok)), timings=tms,
                        cert_breakdown=cert.failure_breakdown())

    @staticmethod
    def timing_budget_ok(t_detect: float, delta_op: float, tau_bar_d: float):
        return (t_detect + delta_op) < tau_bar_d


def _resample_forces(w_seg: np.ndarray, w_dt: float, n: int, dt: float):
    t_src = np.arange(len(w_seg)) * w_dt
    t_tgt = np.minimum(np.arange(n) * dt, t_src[-1])
    return np.stack([np.interp(t_tgt, t_src, w_seg[:, j]) for j in range(3)], axis=1)
