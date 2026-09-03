"""STOMP-covariance proposal over omega (spec 4.5, Osa Eq. 25-26).

Waypoint-space noise n ~ N(0, a * Sigma) with Sigma = (A^T A)^{-1}
(A = second-difference matrix) is projected into weight space through the
pinned operator pinv(F Phi), so sampled trajectories are smooth and
endpoint-pinned by construction. ``calibrate`` sets the scale so the
mid-trajectory positional std matches cfg.data.prop_mid_std_m.
"""
from __future__ import annotations

import numpy as np

from ..traj.rtp import RTP


def second_difference(N: int) -> np.ndarray:
    A = 2.0 * np.eye(N)
    idx = np.arange(N - 1)
    A[idx, idx + 1] = -1.0
    A[idx + 1, idx] = -1.0
    return A


class ProposalSampler:
    def __init__(self, rtp: RTP, target_mid_std: float, rng: np.random.Generator):
        self.rtp = rtp
        N = rtp.cfg.N
        A = second_difference(N)
        Sigma = np.linalg.inv(A.T @ A)
        self._chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(N))
        self.scale = 1.0
        self._calibrate(target_mid_std, rng)

    def _raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((n, self.rtp.cfg.N, 2))
        wp_noise = np.einsum("nm,kmd->knd", self._chol, z) * self.scale
        om = np.einsum("bn,knd->kbd", self.rtp._P, wp_noise)
        return om.reshape(n, -1)

    def _calibrate(self, target_mid_std: float, rng: np.random.Generator,
                   n_cal: int = 2000):
        om = self._raw(n_cal, rng)
        # measure the achieved mid-trajectory std through the pinned map
        res = np.einsum("nb,kbd->knd", self.rtp._SPhi,
                        om.reshape(n_cal, self.rtp.cfg.Bw, 2))
        mid = res[:, self.rtp.cfg.N // 2, :]
        achieved = float(np.sqrt((mid ** 2).mean()))
        if achieved > 1e-9:
            self.scale *= target_mid_std / achieved

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._raw(n, rng)
