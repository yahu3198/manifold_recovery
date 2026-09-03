"""The deployable fault-time pipeline (spec 4.7); the ROS 2 node wraps this.

propose(): build c -> feasible zones -> decode a z grid -> surrogate-certify
-> cluster survivors -> fine-tune + exact-check representatives -> rank.
Per-stage wall times are recorded with every proposal, and
``timing_budget_ok`` implements the Corollary 2 detection-delay check.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..scenario import ZONES, feasible_zones
from ..traj.rtp import RTP
from ..features.condition import Standardizer, build_c
from ..score.obstacles import DockField
from ..certify.surrogate import certify_batch, alpha_bar_policy
from ..certify.cluster import cluster
from ..planner.energy_ocp import EnergyOCP
from .finetune import finetune, Candidate


@dataclass
class Proposal:
    candidates: list
    alpha_bar: float
    n_decoded: int
    n_certified: int
    timings: dict = field(default_factory=dict)


class RecoveryPipeline:
    def __init__(self, ckpt_path: str, cfg: Config):
        from ..model.train import load as load_ckpt      # torch import lazily
        import torch
        self.torch = torch
        self.model, meta = load_ckpt(ckpt_path)
        self.std_om = Standardizer.from_dict(meta["std_omega"])
        self.std_c = Standardizer.from_dict(meta["std_c"])
        self.cfg = cfg
        self.rtp = RTP(cfg.trajectory)
        self.field = DockField()
        self._planners: dict[int, EnergyOCP] = {}

    def _planner(self, zone_id: int) -> EnergyOCP:
        if zone_id not in self._planners:
            self._planners[zone_id] = EnergyOCP(ZONES[zone_id], self.cfg)
        return self._planners[zone_id]

    def propose(self, h1: float, h2: float, w_seg: np.ndarray,
                sigma_theta: float, x0: np.ndarray, w_true: np.ndarray,
                w_dt: float, zone_id: int | None = None) -> Proposal:
        cfg = self.cfg
        tms = {}
        t0 = time.perf_counter()
        c = build_c(h1, h2, w_seg, w_dt, spike=True).reshape(1, -1)
        c_std = self.std_c.transform(c.astype(np.float32))
        w_bar = float(np.linalg.norm(w_seg[:, :2], axis=1).mean())
        zones = ([ZONES[zone_id]] if zone_id is not None
                 else feasible_zones(x0, h1, h2, w_bar,
                                     cfg.trajectory.v_nom, cfg.trajectory.T_max))
        tms["condition"] = time.perf_counter() - t0

        ab = float(alpha_bar_policy(sigma_theta, h1, h2, cfg))
        all_cands: list[Candidate] = []
        n_dec = n_cert = 0
        for zone in zones:
            t0 = time.perf_counter()
            z = np.linspace(cfg.online.z_lo, cfg.online.z_hi, cfg.online.K)
            zt = self.torch.tensor(z[:, None], dtype=self.torch.float32)
            ct = self.torch.tensor(np.repeat(c_std, cfg.online.K, axis=0))
            om_std = self.model.decode(zt, ct).numpy()
            omega = self.std_om.inverse(om_std)
            tms.setdefault("decode", 0.0)
            tms["decode"] += time.perf_counter() - t0
            n_dec += len(omega)

            t0 = time.perf_counter()
            T_h, dt = self.rtp.horizon(x0, zone)
            n_need = cfg.trajectory.N
            w_grid = _resample_forces(w_seg, w_dt, n_need, dt)
            cert = certify_batch(omega, x0, zone, h1, h2,
                                 np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                                 sigma_theta, self.rtp, self.field, cfg)
            tms.setdefault("certify", 0.0)
            tms["certify"] += time.perf_counter() - t0
            ok = np.flatnonzero(cert.mask)
            n_cert += len(ok)
            if len(ok) == 0:
                continue

            t0 = time.perf_counter()
            kin = self.rtp.kinematics(omega[ok], x0, zone)
            reps_local = cluster(kin.pos, kin.vel, -cert.max_d2[ok])
            reps = {sig: ok[i] for sig, i in reps_local.items()}
            tms.setdefault("cluster", 0.0)
            tms["cluster"] += time.perf_counter() - t0

            kin_all = self.rtp.kinematics(omega, x0, zone)
            from ..traj.inversion import invert
            inv = invert(kin_all, np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                         ab, mode="crab", beta_max_deg=cfg.online.beta_max_deg)
            cands = finetune(reps, kin_all.pos, inv.psi, omega, x0, zone,
                             np.array([h1, h2]), ab, T_h, w_grid,
                             w_true, w_dt, self._planner(zone.id),
                             self.field, cfg)
            all_cands.extend(cands)

        all_cands.sort(key=lambda cd: (not cd.cert.passed, cd.plan_cost))
        tms["total"] = sum(v for k, v in tms.items() if k != "total")
        return Proposal(candidates=all_cands[:cfg.online.k_present],
                        alpha_bar=ab, n_decoded=n_dec, n_certified=n_cert,
                        timings=tms)

    @staticmethod
    def timing_budget_ok(t_detect: float, delta_op: float, tau_bar_d: float):
        return (t_detect + delta_op) < tau_bar_d


def _resample_forces(w_seg: np.ndarray, w_dt: float, n: int, dt: float):
    t_src = np.arange(len(w_seg)) * w_dt
    t_tgt = np.arange(n) * dt
    t_tgt = np.minimum(t_tgt, t_src[-1])
    return np.stack([np.interp(t_tgt, t_src, w_seg[:, j]) for j in range(3)], axis=1)
