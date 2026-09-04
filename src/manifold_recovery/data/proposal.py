"""Zone-seeded STOMP-covariance proposal over omega = [p_g, w] (rev 3).

Per sample: pick a target zone (mode) with prior cfg.data.prop_mix, draw the
goal point p_g uniformly inside that zone (>= 2 m from its boundary), and add
STOMP-covariance noise n ~ N(0, a Sigma), Sigma = (A^T A)^{-1}, to the
STRAIGHT chord start -> p_g, scaled so the mid-trajectory positional std is
cfg.data.prop_mid_std_m.

The proposal is deliberately NAIVE about the harbor: it knows the zones but
not the docks. With ``base="via"`` it instead seeds the entry-via route
(``scenario.route_waypoints``); that variant certified 100% of samples below
95% degradation in the rev-3 smoke test, i.e. it is already a planner and
leaves the manifold nothing to learn. The via route is reserved for the B1
oracle seeds. Learning the bend around Dock 1 / Dock 2 is the manifold's job
and what B4 (naive proposal + certificate) is compared against.

Because the start pose is sampled too, ``sample_with_mode`` takes a batch of
starts x0 (n, 3) and returns omega (n, 2+2Bw) and mode (n,) = zone index.
"""
from __future__ import annotations

import numpy as np

from ..traj.rtp import RTP
from ..scenario import ZONES, route_waypoints

MODE_NAMES = tuple(z.name for z in ZONES)


def second_difference(N: int) -> np.ndarray:
    A = 2.0 * np.eye(N)
    idx = np.arange(N - 1)
    A[idx, idx + 1] = -1.0
    A[idx + 1, idx] = -1.0
    return A


class ProposalSampler:
    def __init__(self, rtp: RTP, target_mid_std: float, rng: np.random.Generator,
                 mix: tuple | None = None, goal_margin: float = 2.0,
                 base: str = "chord"):
        if base not in ("chord", "via"):
            raise ValueError("base must be 'chord' or 'via'")
        self.base = base
        self.rtp = rtp
        N = rtp.cfg.N
        A = second_difference(N)
        Sigma = np.linalg.inv(A.T @ A)
        self._chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(N))
        self.scale = 1.0
        self._calibrate(target_mid_std, rng)
        mix = (1.0,) * len(ZONES) if mix is None else tuple(float(m) for m in mix)
        if len(mix) != len(ZONES):
            raise ValueError(f"prop_mix has {len(mix)} entries; expected {len(ZONES)}")
        p = np.asarray(mix, float)
        if p.sum() <= 0 or (p < 0).any():
            raise ValueError("prop_mix must be non-negative with positive sum")
        self.mix = p / p.sum()
        self.goal_margin = goal_margin

    # -- noise on the residual weights ---------------------------------------
    def _raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((n, self.rtp.cfg.N, 2))
        wp_noise = np.einsum("nm,kmd->knd", self._chol, z) * self.scale
        return np.einsum("bn,knd->kbd", self.rtp._P, wp_noise)          # (n, Bw, 2)

    def _calibrate(self, target_mid_std: float, rng: np.random.Generator,
                   n_cal: int = 2000):
        w = self._raw(n_cal, rng)
        res = np.einsum("nb,kbd->knd", self.rtp._SPhi, w)
        mid = res[:, self.rtp.cfg.N // 2, :]
        achieved = float(np.sqrt((mid ** 2).mean()))
        if achieved > 1e-9:
            self.scale *= target_mid_std / achieved

    # -- public -------------------------------------------------------------
    def base_omega(self, x0: np.ndarray, mode: np.ndarray, p_g: np.ndarray) -> np.ndarray:
        """Noise-free base omegas for starts x0 (n, 3), zones mode (n,), goals p_g (n, 2)."""
        if self.base == "chord":
            return self.rtp.join(p_g, np.zeros((len(p_g), self.rtp.cfg.Bw, 2)))
        N = self.rtp.cfg.N
        wps = np.stack([route_waypoints(ZONES[int(m)], x0[i], p_g[i], N)
                        for i, m in enumerate(mode)], axis=0)
        return self.rtp.project(wps, x0)

    def sample_with_mode(self, x0: np.ndarray, rng: np.random.Generator):
        """x0 (n, 3) -> (omega (n, 2+2Bw), mode (n,) zone index)."""
        x0 = np.asarray(x0, float)
        n = len(x0)
        mode = rng.choice(len(ZONES), size=n, p=self.mix)
        p_g = np.empty((n, 2))
        for z in ZONES:
            m = mode == z.id
            if m.any():
                p_g[m] = z.sample_inside(rng, int(m.sum()), self.goal_margin)
        om = self.base_omega(x0, mode, p_g)
        pg, w = self.rtp.split(om)
        return self.rtp.join(pg, w + self._raw(n, rng)), mode

    def sample(self, x0: np.ndarray, rng: np.random.Generator, n: int | None = None):
        """Convenience: x0 (3,) with n, or x0 (n, 3)."""
        x0 = np.asarray(x0, float)
        if x0.ndim == 1:
            x0 = np.broadcast_to(x0, (n, 3))
        return self.sample_with_mode(x0, rng)[0]