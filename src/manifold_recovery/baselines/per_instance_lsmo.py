"""Baseline B3: unconditioned per-instance LSMO (train a fresh VAE for one
condition at fault time). Measures the amortisation gap: identical model and
loss with dim_c = 0, small per-instance dataset, wall clock reported."""
from __future__ import annotations

import time

import numpy as np


def run_b3(x0, zone, h1, w_seg_fn, rtp, field, cfg, rng,
           n_data: int = 1500, epochs: int = 150):
    import torch
    from ..score.score import score_batch
    from ..data.proposal import ProposalSampler
    from ..data.dataset import shape_weights
    from ..model.cvae import CVAE
    from ..model.losses import weighted_elbo
    from ..features.condition import Standardizer

    t0 = time.perf_counter()
    prop = ProposalSampler(rtp, cfg.data.prop_mid_std_m, rng, x0=x0, zone=zone,
                           mix=cfg.data.prop_mix)
    om = prop.sample(n_data, rng)
    h = np.tile([h1, 1.0], (n_data, 1))
    R, _ = score_batch(om, h, w_seg_fn(n_data), x0, zone, rtp, field, cfg)
    f, _ = shape_weights(R, np.full(n_data, h1), cfg.model.a_shaping, n_bins=1)
    std = Standardizer.fit(om)
    omt = torch.tensor(std.transform(om), dtype=torch.float32)
    ft = torch.tensor(f, dtype=torch.float32)
    keep = ft > 0
    omt, ft = omt[keep], ft[keep]
    c0 = torch.zeros(len(omt), 0)
    model = CVAE(omt.shape[1], 0, cfg.model.latent_dim, tuple(cfg.model.hidden))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    for ep in range(epochs):
        Cz = cfg.model.Cz_max * min(1.0, ep / max(1, int(0.6 * epochs)))
        oh, mu, lv = model(omt, c0)
        loss, _ = weighted_elbo(oh, omt, mu, lv, ft, cfg.model.gamma, Cz,
                                cfg.model.recon_sigma)
        opt.zero_grad(); loss.backward(); opt.step()
    wall = time.perf_counter() - t0
    return {"model": model, "std_omega": std, "wall_time": wall,
            "n_train": int(keep.sum())}