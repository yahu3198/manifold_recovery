"""Fault-time environmental force forecast from the recent disturbance history.

Used identically by the manifold arm and the rejection-sampling arm of the
deployment so neither is favoured. Per body axis, on the last ``window_s`` of
EKF disturbance estimates (resampled at ``dt``):

    w_j(t) = m_j + A_j cos(2 pi f_j t + phi_j),   t in [0, horizon_s]

with m_j the mean, f_j the FFT peak in [f_lo, f_hi] of the detrended series
(wave band), and (A_j, phi_j) by least squares at f_j. This is the same
"mean + dominant oscillation" structure the synthetic sampler and the MPC's
RLS predictor assume, so the certificate sees the environment the same way
offline and online. A "true" realisation for the tier-2 rollout is the
forecast plus block-bootstrapped residuals (the vessel's own Gazebo run is
the real tier 2; the rollout is a pre-screen).
"""
from __future__ import annotations

import numpy as np


def _fit_axis(t, x, f_lo, f_hi):
    m = float(np.mean(x))
    r = x - m
    n = len(r)
    if n < 16:
        return m, 0.0, 0.0, 0.0, r
    dt = float(t[1] - t[0])
    spec = np.abs(np.fft.rfft(r * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not band.any():
        return m, 0.0, 0.0, 0.0, r
    f0 = float(freqs[band][np.argmax(spec[band])])
    # refine within +-1 FFT bin: a 60 s window has 0.017 Hz resolution, and a
    # bin's worth of error drifts the phase by a full cycle over a 150 s horizon
    df = freqs[1] - freqs[0]
    best = None
    for f in np.linspace(max(f0 - df, f_lo), min(f0 + df, f_hi), 41):
        C = np.column_stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)])
        (a, b), res, *_ = np.linalg.lstsq(C, r, rcond=None)
        sse = float(res[0]) if len(res) else float(np.sum((r - C @ np.array([a, b])) ** 2))
        if best is None or sse < best[0]:
            best = (sse, f, a, b, C)
    _, f, a, b, C = best
    A, phi = float(np.hypot(a, b)), float(np.arctan2(-b, a))
    resid = r - (a * C[:, 0] + b * C[:, 1])
    return m, A, float(f), phi, resid


class ForceForecaster:
    def __init__(self, window_s: float = 60.0, dt: float = 0.1,
                 f_lo: float = 0.05, f_hi: float = 1.0):
        self.window_s, self.dt, self.f_lo, self.f_hi = window_s, dt, f_lo, f_hi
        self._t, self._w = [], []

    def push(self, t: float, w_body) -> None:
        self._t.append(float(t))
        self._w.append(np.asarray(w_body, float)[:3])
        while self._t and self._t[0] < t - self.window_s - 1.0:
            self._t.pop(0); self._w.pop(0)

    def ready(self, min_s: float = 20.0) -> bool:
        return len(self._t) > 2 and (self._t[-1] - self._t[0]) >= min_s

    def forecast(self, horizon_s: float, rng: np.random.Generator | None = None,
                 n_true: int = 1):
        """Returns (w_hat [n, 3], w_true [n, 3] or [n_true, n, 3], dt, fit)."""
        t = np.asarray(self._t); W = np.stack(self._w)
        tt = np.arange(t[0], t[-1], self.dt)
        Wr = np.stack([np.interp(tt, t, W[:, j]) for j in range(3)], axis=1)
        tr = tt - tt[-1]                           # forecast origin = now
        th = np.arange(0.0, horizon_s + self.dt, self.dt)
        w_hat = np.zeros((len(th), 3)); fit = []; resid = []
        for j in range(3):
            m, A, f, phi, r = _fit_axis(tr, Wr[:, j], self.f_lo, self.f_hi)
            w_hat[:, j] = m + A * np.cos(2 * np.pi * f * th + phi)
            fit.append({"mean": m, "amp": A, "freq": f, "phase": phi,
                        "resid_std": float(np.std(r))})
            resid.append(r)
        rng = rng or np.random.default_rng(0)
        block = max(int(5.0 / self.dt), 2)
        trues = []
        for _ in range(n_true):
            wt = w_hat.copy()
            for j in range(3):
                r = resid[j]
                if len(r) < block:
                    continue
                n_blocks = len(th) // block + 1
                starts = rng.integers(0, len(r) - block + 1, size=n_blocks)
                boot = np.concatenate([r[s:s + block] for s in starts])[:len(th)]
                wt[:, j] += boot
            trues.append(wt)
        w_true = trues[0] if n_true == 1 else np.stack(trues)
        return w_hat, w_true, self.dt, fit


def sigma_theta_from_confidence(conf: float, kappa: float = 1.5, eps: float = 0.01) -> float:
    """Map the MPC's prediction confidence in [0, 1] to the certificate's
    sigma_theta so that min(1, kappa/(sigma+eps)) = max(0.3, sqrt(conf)),
    the trust floor the deployed node uses."""
    c = max(np.sqrt(max(conf, 0.0)), 0.3)
    return float(kappa / c - eps)
