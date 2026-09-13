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
from ..scenario import feasible_zones, ZONES, zone_of
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


# rev 4.5: two-stage generation. Stage 1 (decode/sample + screen + cluster +
# inversion) yields one certified DECODED representative per class in about a
# second and is publishable as a reference on its own; stage 2 (OCP refinement)
# may replace it later. The offline scripts call propose(), which runs both.
@dataclass
class DecodedCandidate:
    idx: int                  # index into the candidate set
    zone_id: int
    signature: tuple
    pos: np.ndarray           # (N, 2) world positions over [0, T_h]
    vel: np.ndarray           # (N, 2) world velocities, physical time
    psi: np.ndarray           # (N,) heading from the inversion (with sideslip)
    max_d2: float             # thruster-axis residual, screen margin (lower = better)
    min_clear: float


@dataclass
class Stage1:
    omega: np.ndarray
    x0: np.ndarray
    h: np.ndarray
    alpha_bar: float
    T_h: float
    w_grid: np.ndarray
    w_true: np.ndarray
    w_dt: float
    cert: object
    ok: np.ndarray
    kin_all: object
    inv: object
    reps: dict
    decoded: list             # DecodedCandidate, ranked by max_d2
    timings: dict
    n_decoded: int
    n_certified: int


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
            # rev 4.3: the spike pre-filter discounts v_nom by thrust authority and is
            # not exercised offline; decode for every zone and let the screen decide.
            zones = list(ZONES)
        tms["condition"] = time.perf_counter() - t0
        if self.decoder is None:
            raise RuntimeError("propose() needs a checkpoint; this pipeline was built for the B4 arm")
        t0 = time.perf_counter()
        omega, _g = self.decoder.decode_zones(h1, np.asarray(x0[:3], float),
                                              cfg.online.K, self.rng, zones)
        tms["decode"] = time.perf_counter() - t0
        return self._finish(omega, h1, h2, w_seg, sigma_theta, x0, w_true, w_dt, tms)

    # ---- rev 4.5: two-stage interface used by the sidecar --------------------
    def propose_stage1(self, h1: float, h2: float, w_seg: np.ndarray,
                       sigma_theta: float, x0: np.ndarray, w_true: np.ndarray,
                       w_dt: float, arm: str = "manifold") -> Stage1:
        """Decode (or sample, arm="b4"), screen, cluster and invert. Returns the
        certified decoded representative per class, ranked by screen margin."""
        cfg = self.cfg
        tms = {}
        t0 = time.perf_counter()
        if arm == "manifold":
            if self.decoder is None:
                raise RuntimeError("manifold arm needs a checkpoint")
            w_bar = float(np.linalg.norm(w_seg[:, :2], axis=1).mean())
            zones = feasible_zones(x0, h1, h2, w_bar, cfg.trajectory.v_nom, cfg.trajectory.T_max)
            tms["prefilter_empty"] = float(len(zones) == 0)
            if not zones:
                zones = list(ZONES)
            omega, _g = self.decoder.decode_zones(h1, np.asarray(x0[:3], float),
                                                  cfg.online.K, self.rng, zones)
        else:
            prop = ProposalSampler(self.rtp, cfg.data.prop_mid_std_m, self.rng, mix=cfg.data.prop_mix)
            omega = prop.sample(np.asarray(x0[:3], float), self.rng, cfg.online.K)
        tms["decode"] = time.perf_counter() - t0

        ab = float(alpha_bar_policy(sigma_theta, h1, h2, cfg))
        T_h, dt = self.rtp.horizon()
        w_grid = _resample_forces(w_seg, w_dt, cfg.trajectory.N, dt)
        x03 = np.asarray(x0[:3], float)
        t0 = time.perf_counter()
        cert = certify_batch(omega, x03, h1, h2,
                             np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                             sigma_theta, self.rtp, self.field, cfg)
        tms["certify"] = time.perf_counter() - t0
        ok = np.flatnonzero(cert.mask)
        reps, decoded, kin_all, inv = {}, [], None, None
        if len(ok):
            t0 = time.perf_counter()
            kin_all = self.rtp.kinematics(omega, x03)
            reps_local = cluster(kin_all.pos[ok], kin_all.vel[ok], -cert.max_d2[ok])
            reps = {sig: int(ok[i]) for sig, i in reps_local.items()}
            from ..traj.inversion import invert
            inv = invert(kin_all, np.broadcast_to(w_grid, (len(omega),) + w_grid.shape),
                         ab, mode="crab", beta_max_deg=cfg.online.beta_max_deg)
            for sig, idx in reps.items():
                zid = int(zone_of(kin_all.pos[idx][-1]))
                if zid < 0:
                    continue
                decoded.append(DecodedCandidate(idx=idx, zone_id=zid, signature=sig,
                                                pos=np.asarray(kin_all.pos[idx], float),
                                                vel=np.asarray(kin_all.vel[idx], float),
                                                psi=np.asarray(inv.psi[idx], float),
                                                max_d2=float(cert.max_d2[idx]),
                                                min_clear=float(cert.min_clear[idx])))
            decoded.sort(key=lambda d: d.max_d2)
            tms["cluster"] = time.perf_counter() - t0
        tms["stage1"] = sum(v for k, v in tms.items() if k not in ("prefilter_empty",))
        return Stage1(omega=omega, x0=np.asarray(x0, float), h=np.array([h1, h2]),
                      alpha_bar=ab, T_h=T_h, w_grid=w_grid, w_true=w_true, w_dt=w_dt,
                      cert=cert, ok=ok, kin_all=kin_all, inv=inv, reps=reps,
                      decoded=decoded, timings=tms,
                      n_decoded=len(omega), n_certified=int(len(ok)))

    def refine_stage2(self, s1: Stage1) -> Proposal:
        """OCP refinement of the class representatives of a Stage1 result."""
        cfg = self.cfg
        tms = dict(s1.timings)
        cands: list[Candidate] = []
        if len(s1.ok):
            t0 = time.perf_counter()
            cands = finetune(s1.reps, s1.kin_all.pos, s1.inv.psi, s1.omega, s1.x0,
                             self.planners, s1.h, s1.alpha_bar, s1.T_h, s1.w_grid,
                             s1.w_true, s1.w_dt, self.field, cfg)
            tms["finetune_and_rollout"] = time.perf_counter() - t0
        cands.sort(key=lambda cd: cd.plan_cost)   # rev 4.3: rollout advisory
        tms["total"] = sum(v for k, v in tms.items() if k not in ("total", "stage1", "prefilter_empty"))
        return Proposal(candidates=cands[:cfg.online.k_present], alpha_bar=s1.alpha_bar,
                        n_decoded=s1.n_decoded, n_certified=s1.n_certified, timings=tms,
                        cert_breakdown=s1.cert.failure_breakdown())

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
        cands.sort(key=lambda cd: cd.plan_cost)   # rev 4.3: rollout advisory
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
