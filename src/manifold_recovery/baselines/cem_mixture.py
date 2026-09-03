"""Baseline B2: per-instance multimodal CEM over omega (Osa's CEM comparison).

20 Gaussian components in weight space, initialised from the proposal;
per-component elites update means, variances shrink geometrically. Reports
best score, classes found, and wall time (no amortisation: pays full cost
at fault time)."""
from __future__ import annotations

import time

import numpy as np

from ..score.score import score_batch
from ..certify.cluster import side_signature


def run_b2(x0, zone, h, w_seg_fn, rtp, field, cfg, rng,
           n_comp: int = 20, n_per: int = 20, iters: int = 12,
           sigma0: float = 1.0, shrink: float = 0.85):
    from ..data.proposal import ProposalSampler
    t0 = time.perf_counter()
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng)
    means = prop.sample(n_comp, rng)
    sig = sigma0 * np.abs(means).mean() + 1e-3
    best = {"R": -np.inf, "omega": None}
    for _ in range(iters):
        oms = means[:, None, :] + sig * rng.standard_normal(
            (n_comp, n_per, means.shape[1]))
        flat = oms.reshape(-1, means.shape[1])
        hM = np.tile(np.asarray(h, float), (len(flat), 1))
        w = w_seg_fn(len(flat))
        R, terms = score_batch(flat, hM, w, x0, zone, rtp, field, cfg)
        Rc = R.reshape(n_comp, n_per)
        elite = Rc.argmax(axis=1)
        means = oms[np.arange(n_comp), elite]
        i = int(R.argmax())
        if R[i] > best["R"]:
            best = {"R": float(R[i]), "omega": flat[i],
                    "kin": None}
        sig *= shrink
    kin = rtp.kinematics(means, x0, zone)
    sigs = side_signature(kin.pos, kin.vel)
    classes = sorted({tuple(s) for s in sigs})
    return {"best_R": best["R"], "best_omega": best["omega"],
            "classes": classes, "wall_time": time.perf_counter() - t0}
