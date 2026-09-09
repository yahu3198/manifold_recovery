"""Fault-time latent sampling and decoding (rev 4), shared by scripts 03/04,
the online pipeline, and the contraction maps. One code path.

The model is conditional on the target zone g (one-hot in c) and emits a
zone-relative endpoint; ``decode`` handles one zone, ``decode_zones`` splits
K across a set of zones and returns absolute-frame omegas plus the zone of
each candidate. Latent sampling modes:
  "bank"  : resample stored posterior means of SAME-ZONE training samples
            nearest in standardised c (f-weighted) plus Gaussian jitter.
  "prior" : z ~ N(0, I).
  "grid"  : linspace on [z_lo, z_hi] (1-D latents only).
"""
from __future__ import annotations

import numpy as np

from ..features.condition import Standardizer, build_c, zone_from_c
from ..scenario import ZONES
from .omega_transform import from_model


class Decoder:
    def __init__(self, model, meta: dict, cfg):
        """``model``: a CVAE (torch) or any callable (z, c) -> omega_std as numpy,
        the latter so the sampling logic is unit-testable without torch."""
        if callable(model) and not hasattr(model, "decode"):
            self._fn = model
        else:
            import torch
            self._fn = lambda z, c: model.decode(torch.tensor(z), torch.tensor(c)).numpy()
        self.cfg = cfg
        self.std_om = Standardizer.from_dict(meta["std_omega"])
        self.std_c = Standardizer.from_dict(meta["std_c"])
        self.latent = int(meta["latent"])
        if meta.get("omega_repr") != "zone_relative_v4":
            raise ValueError("checkpoint predates rev 4 (no zone-relative endpoint); retrain with 02")
        self.z_bank = np.asarray(meta["z_bank"], np.float32)
        self.f_bank = np.asarray(meta["f_bank"], np.float32)
        self.c_bank = np.asarray(meta["c_bank"], np.float32)
        self.g_bank = np.asarray(meta["g_bank"], int)

    def sample_z(self, c_raw: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
        oc = self.cfg.online
        if oc.z_mode == "grid":
            if self.latent != 1:
                raise ValueError("z_mode 'grid' needs latent_dim = 1")
            return np.linspace(oc.z_lo, oc.z_hi, K)[:, None]
        if oc.z_mode == "prior":
            return rng.standard_normal((K, self.latent))
        if oc.z_mode != "bank":
            raise ValueError(f"unknown z_mode {oc.z_mode}")
        g = int(zone_from_c(c_raw))
        same = np.flatnonzero(self.g_bank == g)
        if len(same) == 0:
            return rng.standard_normal((K, self.latent))
        cs = self.std_c.transform(self.c_bank[same])
        cq = self.std_c.transform(np.asarray(c_raw, np.float32).reshape(1, -1))
        d = np.linalg.norm(cs - cq, axis=1)
        n_near = max(50, int(0.2 * len(d)))
        near = same[np.argsort(d)[:n_near]]
        w = self.f_bank[near]
        w = w / w.sum() if w.sum() > 0 else np.full(len(near), 1.0 / len(near))
        pick = rng.choice(near, size=K, p=w)
        return self.z_bank[pick] + oc.z_jitter * rng.standard_normal((K, self.latent))

    def decode(self, c_raw: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
        """c_raw (dim_c,) with the zone one-hot set -> omega (K, 2+2Bw), absolute frame."""
        z = self.sample_z(c_raw, K, rng).astype(np.float32)
        c = np.repeat(self.std_c.transform(np.asarray(c_raw, np.float32).reshape(1, -1)),
                      K, axis=0).astype(np.float32)
        om_m = self.std_om.inverse(np.asarray(self._fn(z, c), np.float32))
        g = int(zone_from_c(c_raw))
        return from_model(om_m, np.full(K, g))

    def decode_zones(self, h1: float, x0: np.ndarray, K: int, rng: np.random.Generator,
                     zones=ZONES):
        """Split K across zones; returns (omega (K', dim), g (K',))."""
        zones = list(zones)
        if not zones:
            return np.zeros((0, 2 + 2 * self.cfg.trajectory.Bw)), np.zeros(0, int)
        per = [K // len(zones) + (1 if i < K % len(zones) else 0) for i in range(len(zones))]
        oms, gs = [], []
        for z, k in zip(zones, per):
            if k == 0:
                continue
            c = build_c(h1, np.asarray(x0, float)[:3], z.id, spike=True)
            oms.append(self.decode(c, k, rng))
            gs.append(np.full(k, z.id))
        return np.concatenate(oms), np.concatenate(gs)