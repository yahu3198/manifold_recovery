"""Training loop with G1 diagnostics and a self-contained checkpoint.

Checkpoint contents: state_dict, config dict + hash, omega/c Standardizers,
training history (loss, recon, KL per epoch), dataset sidecar, and (rev 3)
a z-bank: the posterior means of the kept training samples with their
shaping weights and raw conditions, so fault-time decoding can resample
latents where the aggregate posterior actually has mass instead of walking a
grid through inter-cluster gaps. ``load`` restores everything the online
pipeline needs; nothing else may be required at fault time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..features.condition import Standardizer
from .cvae import CVAE
from .losses import weighted_elbo


def train(cfg: Config, dataset, out_path: str | Path, verbose: bool = True):
    torch.manual_seed(cfg.model.seed)
    std_om = Standardizer.fit(dataset.omega)
    std_c = Standardizer.fit(dataset.c)
    om = torch.tensor(std_om.transform(dataset.omega), dtype=torch.float32)
    c = torch.tensor(std_c.transform(dataset.c), dtype=torch.float32)
    f = torch.tensor(dataset.f, dtype=torch.float32)
    keep = f > 0
    om, c, f = om[keep], c[keep], f[keep]

    model = CVAE(om.shape[1], c.shape[1], cfg.model.latent_dim,
                 tuple(cfg.model.hidden))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    n = len(om)
    hist = {"loss": [], "recon": [], "kl": [], "Cz": []}
    for ep in range(cfg.model.epochs):
        Cz = cfg.model.Cz_max * min(1.0, ep / max(1, int(0.6 * cfg.model.epochs)))
        perm = torch.randperm(n)
        ep_loss = ep_rec = ep_kl = 0.0
        nb = 0
        for s in range(0, n, cfg.model.batch):
            idx = perm[s:s + cfg.model.batch]
            oh, mu, logvar = model(om[idx], c[idx])
            loss, parts = weighted_elbo(oh, om[idx], mu, logvar, f[idx],
                                        cfg.model.gamma, Cz,
                                        cfg.model.recon_sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); ep_rec += parts["recon"]; ep_kl += parts["kl"]
            nb += 1
        hist["loss"].append(ep_loss / nb)
        hist["recon"].append(ep_rec / nb)
        hist["kl"].append(ep_kl / nb)
        hist["Cz"].append(Cz)
        if verbose and (ep % max(1, cfg.model.epochs // 10) == 0 or ep == cfg.model.epochs - 1):
            print(f"  ep {ep:4d}  loss {hist['loss'][-1]:.3f}  "
                  f"recon {hist['recon'][-1]:.3f}  KL {hist['kl'][-1]:.3f}  Cz {Cz:.2f}",
                  flush=True)

    with torch.no_grad():
        mu_bank, _ = model.encoder(om, c)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "z_bank": mu_bank.numpy().astype(np.float32),
        "f_bank": f.numpy().astype(np.float32),
        "c_bank": std_c.inverse(c.numpy()).astype(np.float32),
        "state_dict": model.state_dict(),
        "dim_omega": om.shape[1], "dim_c": c.shape[1],
        "latent": cfg.model.latent_dim, "hidden": tuple(cfg.model.hidden),
        "std_omega": std_om.to_dict(), "std_c": std_c.to_dict(),
        "config": cfg.to_dict(), "config_hash": cfg.hash(),
        "history": hist, "dataset_meta": dataset.meta,
    }, out_path)
    return out_path, hist


def load(ckpt_path: str | Path):
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CVAE(d["dim_omega"], d["dim_c"], d["latent"], tuple(d["hidden"]))
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model, d
