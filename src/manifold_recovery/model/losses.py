"""Importance-weighted conditional ELBO with capacity annealing (spec 4.6).

    L = E_f[ 0.5 ||omega_hat - omega||^2 / sigma^2 ]  +  gamma | KL_f - Cz |

where E_f is the f(R)-weighted batch mean (weights normalised per batch) and
KL_f the weighted mean KL. Cz anneals linearly 0 -> Cz_max (Dupont-style),
which prevents early posterior collapse (gate G1's KL band [0.3, 5]).
"""
from __future__ import annotations

import torch


def weighted_elbo(omega_hat, omega, mu, logvar, f, gamma: float, Cz: float,
                  recon_sigma: float = 1.0):
    w = f / f.sum().clamp_min(1e-8)
    recon_per = 0.5 * ((omega_hat - omega) ** 2).sum(dim=-1) / recon_sigma ** 2
    recon = (w * recon_per).sum()
    kl_per = 0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    kl = (w * kl_per).sum()
    loss = recon + gamma * (kl - Cz).abs()
    return loss, {"recon": float(recon.detach()), "kl": float(kl.detach())}
