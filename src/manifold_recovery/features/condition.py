"""Condition vector c: the SINGLE builder used offline and at fault time.

Rev 4 spike:  c = [h1, dx/100, dy/100, dpsi, onehot(g)]   (dim 4 + n_zones)
  dx, dy = start position relative to the harbor opening centre (m),
  dpsi   = start heading relative to the bearing toward the opening (rad),
  g      = target zone (the planner's decision variable; the manifold is
           conditional on it and decodes per zone at fault time).
Full build appends h2 and the environment features. One code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scenario import OPENING_CENTER, ZONES

N_ZONES = len(ZONES)
DIM_C = 4 + N_ZONES


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def env_features(w_seg: np.ndarray, dt: float):
    wxy = w_seg[:, :2]
    mag = np.linalg.norm(wxy, axis=1)
    w_bar = float(mag.mean())
    mean_vec = wxy.mean(axis=0)
    th = float(np.arctan2(mean_vec[1], mean_vec[0]))
    sigma_w = float(mag.std())
    x = mag - mag.mean()
    if len(x) >= 8:
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=dt)
        f_dom = float(freqs[1:][np.argmax(spec[1:])]) if len(spec) > 1 else 0.0
    else:
        f_dom = 0.0
    return w_bar, np.cos(th), np.sin(th), sigma_w, f_dom


def start_features(x0: np.ndarray) -> np.ndarray:
    """x0 (..., 3) -> (..., 3): [dx/100, dy/100, dpsi]."""
    x0 = np.asarray(x0, float)
    d = x0[..., :2] - OPENING_CENTER
    bearing = np.arctan2(-d[..., 1], -d[..., 0])
    dpsi = _wrap(x0[..., 2] - bearing)
    return np.stack([d[..., 0] / 100.0, d[..., 1] / 100.0, dpsi], axis=-1)


def build_c(h1, x0: np.ndarray, g, spike: bool = True) -> np.ndarray:
    """h1 scalar or (n,); x0 (3,) or (n, 3); g zone index scalar or (n,).
    Returns (dim_c,) for scalar inputs, else (n, dim_c)."""
    if not spike:
        raise NotImplementedError("full-build condition vector not part of the spike")
    h1a = np.asarray(h1, float).reshape(-1)
    sf = start_features(x0).reshape(-1, 3)
    ga = np.asarray(g, int).reshape(-1)
    n = max(len(h1a), len(sf), len(ga))
    h1a = np.broadcast_to(h1a, (n,))
    sf = np.broadcast_to(sf, (n, 3))
    ga = np.broadcast_to(ga, (n,))
    onehot = np.eye(N_ZONES)[ga]
    c = np.column_stack([h1a, sf, onehot])
    scalar = np.ndim(h1) == 0 and np.asarray(x0).ndim == 1 and np.ndim(g) == 0
    return c[0] if scalar else c


def zone_from_c(c: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(c)[..., 4:4 + N_ZONES], axis=-1)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        return cls(mean=X.mean(axis=0), std=np.maximum(X.std(axis=0), 1e-8))

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse(self, X):
        return X * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]))