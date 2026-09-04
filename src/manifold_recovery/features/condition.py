"""Condition vector c: the SINGLE builder used offline and at fault time.

Rev 3 spike:  c = [h1, dx/100, dy/100, dpsi]
  dx, dy = start position relative to the harbor opening centre (m),
  dpsi   = start heading relative to the bearing toward the opening (rad).
The start now varies per sample, so the manifold must be told where it is.
Full build appends h2 and the environment features (w_bar, cos/sin th_w,
sigma_w, f_dom). One code path for training and deployment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scenario import OPENING_CENTER


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


def build_c(h1, x0: np.ndarray, h2=1.0, w_seg: np.ndarray | None = None,
            dt: float = 1.0, spike: bool = True) -> np.ndarray:
    """h1 scalar or (n,); x0 (3,) or (n, 3). Returns (n, dim_c) or (dim_c,)."""
    h1a = np.asarray(h1, float)
    sf = start_features(x0)
    if h1a.ndim == 0 and sf.ndim == 1:
        c = np.concatenate([[float(h1a)], sf])
        if not spike:
            c = np.concatenate([c, [float(h2)], env_features(w_seg, dt)])
        return c
    h1a = np.broadcast_to(h1a.reshape(-1), (len(sf.reshape(-1, 3)),))
    c = np.column_stack([h1a, sf.reshape(-1, 3)])
    if not spike:
        raise NotImplementedError("full-build condition vector is per-sample; use the spike path")
    return c


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