"""Conditional VAE over RTP weights (spec 4.6).

Encoder: [omega_std, c_std] -> 256 -> 256 -> (mu, logvar), latent dim 1 (spike).
Decoder: [z, c_std] -> 256 -> 256 -> omega_std_hat.
Zone input is dropped in the spike (single zone); the full build appends a
one-hot g to both. ``decode`` is the only method the online pipeline calls.
Inputs/outputs are STANDARDIZED; the checkpoint stores both Standardizers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])          # drop trailing ReLU


class Encoder(nn.Module):
    def __init__(self, dim_omega: int, dim_c: int, latent: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp([dim_omega + dim_c, *hidden])
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)

    def forward(self, omega, c):
        h = torch.relu(self.net(torch.cat([omega, c], dim=-1)))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, dim_omega: int, dim_c: int, latent: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp([latent + dim_c, *hidden])
        self.out = nn.Linear(hidden[-1], dim_omega)

    def forward(self, z, c):
        h = torch.relu(self.net(torch.cat([z, c], dim=-1)))
        return self.out(h)


class CVAE(nn.Module):
    def __init__(self, dim_omega: int, dim_c: int, latent: int = 1,
                 hidden=(256, 256)):
        super().__init__()
        self.encoder = Encoder(dim_omega, dim_c, latent, hidden)
        self.decoder = Decoder(dim_omega, dim_c, latent, hidden)
        self.latent = latent

    def forward(self, omega, c):
        mu, logvar = self.encoder(omega, c)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z, c), mu, logvar

    @torch.no_grad()
    def decode(self, z, c):
        return self.decoder(z, c)
