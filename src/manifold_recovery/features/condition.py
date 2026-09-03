"""Condition vector c: the SINGLE builder used offline and at fault time.

Offline (data/dataset.py) and online (pipeline/online.py) both call
``build_c``; keeping one code path prevents train/deploy conditioning drift.
Spike: c = [h1] (h2 = 1, one environment family, one zone; spec 4.7).
Full:  c = [h1, h2, w_bar, cos th_w, sin th_w, sigma_w, f_dom] + zone one-hot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def env_features(w_seg: np.ndarray, dt: float):
    """w_seg (T, 3) body-frame -> (w_bar, cos_th, sin_th, sigma_w, f_dom)."""
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


def build_c(h1, h2, w_seg: np.ndarray | None, dt: float, spike: bool = True):
    if spike:
        return np.atleast_1d(np.asarray(h1, dtype=float)).reshape(-1, 1) \
            if np.ndim(h1) else np.array([float(h1)])
    w_bar, c_th, s_th, sig, f_dom = env_features(w_seg, dt)
    return np.array([float(h1), float(h2), w_bar, c_th, s_th, sig, f_dom])


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
