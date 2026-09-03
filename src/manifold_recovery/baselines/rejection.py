"""Baseline B4: proposal + certification, latency-matched to the pipeline.

Identical code path to the pipeline with ONE swap: proposal samples instead of
decoder samples. Its acceptance rate is V_proposal, the normaliser separating
physical recoverability from what the manifold added (feasibility review R4).
"""
from __future__ import annotations

import time

import numpy as np

from ..certify.surrogate import certify_batch
from ..certify.cluster import cluster
from ..data.proposal import ProposalSampler


def run_b4(x0, zone, h1, h2, w_seg, sigma_theta, rtp, field, cfg, rng,
           wall_budget_s: float, batch: int = 200):
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng, x0=x0, zone=zone,
                           mix=cfg.data.prop_mix)
    t0 = time.perf_counter()
    n_tried = n_ok = 0
    reps = {}
    while time.perf_counter() - t0 < wall_budget_s:
        om = prop.sample(batch, rng)
        w = np.broadcast_to(w_seg, (batch,) + w_seg.shape)
        cert = certify_batch(om, x0, zone, h1, h2, w, sigma_theta, rtp,
                             field, cfg)
        n_tried += batch
        ok = np.flatnonzero(cert.mask)
        n_ok += len(ok)
        if len(ok):
            kin = rtp.kinematics(om[ok], x0, zone)
            local = cluster(kin.pos, kin.vel, -cert.max_d2[ok])
            for sig, i in local.items():
                cand = (float(cert.max_d2[ok][i]), om[ok][i])
                if sig not in reps or cand[0] < reps[sig][0]:
                    reps[sig] = cand
    return {"acceptance": n_ok / max(n_tried, 1), "n_tried": n_tried,
            "classes": sorted(reps.keys()),
            "reps": {k: v[1] for k, v in reps.items()},
            "wall_time": time.perf_counter() - t0}
