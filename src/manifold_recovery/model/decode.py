"""Fault-time latent sampling and decoding (rev 3), shared by scripts 03/04,
the online pipeline, and the contraction maps. One code path.

z_mode "bank"  : resample stored posterior means of training samples whose
                 raw condition is closest to the query (nearest 20% in
                 standardised c-space, f-weighted), plus Gaussian jitter.
z_mode "prior" : z ~ N(0, I).
z_mode "grid"  : linspace on [z_lo, z_hi] (1-D latents only; rev 1/2).
"""
from __future__ import annotations

import numpy as np

from ..features.condition import Standardizer


class Decoder:
    def __init__(self, model, meta: dict, cfg):
        import torch
        self.torch = torch
        self.model = model
        self.cfg = cfg
        self.std_om = Standardizer.from_dict(meta["std_omega"])
        self.std_c = Standardizer.from_dict(meta["std_c"])
        self.latent = int(meta["latent"])
        self.z_bank = np.asarray(meta.get("z_bank", np.zeros((0, self.latent))), np.float32)
        self.f_bank = np.asarray(meta.get("f_bank", np.zeros(0)), np.float32)
        self.c_bank = np.asarray(meta.get("c_bank", np.zeros((0, 1))), np.float32)

    def sample_z(self, c_raw: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
        oc = self.cfg.online
        mode = oc.z_mode
        if mode == "grid":
            if self.latent != 1:
                raise ValueError("z_mode 'grid' needs latent_dim = 1")
            return np.linspace(oc.z_lo, oc.z_hi, K)[:, None]
        if mode == "prior" or len(self.z_bank) == 0:
            return rng.standard_normal((K, self.latent))
        if mode != "bank":
            raise ValueError(f"unknown z_mode {mode}")
        cs = self.std_c.transform(self.c_bank)
        cq = self.std_c.transform(np.asarray(c_raw, np.float32).reshape(1, -1))
        d = np.linalg.norm(cs - cq, axis=1)
        n_near = max(50, int(0.2 * len(d)))
        near = np.argsort(d)[:n_near]
        w = self.f_bank[near]
        w = w / w.sum() if w.sum() > 0 else np.full(len(near), 1.0 / len(near))
        pick = rng.choice(near, size=K, p=w)
        return self.z_bank[pick] + oc.z_jitter * rng.standard_normal((K, self.latent))

    def decode(self, c_raw: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
        """c_raw (dim_c,) -> omega (K, 2+2Bw) in physical units."""
        z = self.sample_z(c_raw, K, rng).astype(np.float32)
        c = self.std_c.transform(np.asarray(c_raw, np.float32).reshape(1, -1))
        c = np.repeat(c, K, axis=0).astype(np.float32)
        om = self.model.decode(self.torch.tensor(z), self.torch.tensor(c)).numpy()
        return self.std_om.inverse(om)