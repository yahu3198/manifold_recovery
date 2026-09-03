"""Mode-seeded STOMP-covariance proposal over omega (spec 4.5, Osa Eq. 25-26; rev 2).

Waypoint-space noise n ~ N(0, a * Sigma) with Sigma = (A^T A)^{-1}
(A = second-difference matrix) is projected into weight space through the
pinned operator pinv(F Phi), so sampled trajectories are smooth and
endpoint-pinned by construction. ``_calibrate`` sets the scale so the
mid-trajectory positional std matches cfg.data.prop_mid_std_m.

Rev 2 (spike run 1 finding, G4): with a 3 m mid-std around the straight base
the proposal essentially never reaches the around-dock corridors (the docks
are ~25 m across and the detours run 120-145 m), so the training set held a
single homotopy class and the decoder faithfully reproduced N_H = 1. The
proposal is now a MIXTURE: the same noise is added to one of three base
paths, straight / north detour / south detour, drawn with cfg.data.prop_mix.
Detour bases are the shared ``scenario.DETOUR_POLYLINES`` projected through
``RTP.project`` (so B1's seeds and the proposal explore identical classes).
``sample_with_mode`` exposes the component index for diagnostics.
"""
from __future__ import annotations

import numpy as np

from ..traj.rtp import RTP
from ..scenario import (Zone, ZONES, SPIKE_START_POSE, SPIKE_ZONE_ID,
                        DETOUR_POLYLINES, detour_waypoints)

MODE_NAMES = ("straight",) + tuple(DETOUR_POLYLINES.keys())   # index -> name


def second_difference(N: int) -> np.ndarray:
    A = 2.0 * np.eye(N)
    idx = np.arange(N - 1)
    A[idx, idx + 1] = -1.0
    A[idx + 1, idx] = -1.0
    return A


class ProposalSampler:
    def __init__(self, rtp: RTP, target_mid_std: float, rng: np.random.Generator,
                 x0: np.ndarray | None = None, zone: Zone | None = None,
                 mix: tuple | None = None):
        self.rtp = rtp
        N = rtp.cfg.N
        A = second_difference(N)
        Sigma = np.linalg.inv(A.T @ A)
        self._chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(N))
        self.scale = 1.0
        self._calibrate(target_mid_std, rng)

        # Mixture bases in omega-space: straight base is omega = 0 by construction.
        x0 = SPIKE_START_POSE if x0 is None else np.asarray(x0, float)
        zone = ZONES[SPIKE_ZONE_ID] if zone is None else zone
        self.x0, self.zone = x0, zone
        Bw2 = 2 * rtp.cfg.Bw
        means = [np.zeros(Bw2)]
        for name in DETOUR_POLYLINES:
            wp = detour_waypoints(name, x0, zone, N)
            means.append(rtp.project(wp, x0, zone))
        self.means = np.stack(means, axis=0)                   # (n_modes, 2Bw)
        mix = (1.0, 0.0, 0.0) if mix is None else tuple(float(m) for m in mix)
        if len(mix) != len(self.means):
            raise ValueError(f"prop_mix has {len(mix)} entries; expected {len(self.means)}")
        p = np.asarray(mix, float)
        if p.sum() <= 0 or (p < 0).any():
            raise ValueError("prop_mix must be non-negative with positive sum")
        self.mix = p / p.sum()

    # -- noise --------------------------------------------------------------
    def _raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((n, self.rtp.cfg.N, 2))
        wp_noise = np.einsum("nm,kmd->knd", self._chol, z) * self.scale
        om = np.einsum("bn,knd->kbd", self.rtp._P, wp_noise)
        return om.reshape(n, -1)

    def _calibrate(self, target_mid_std: float, rng: np.random.Generator,
                   n_cal: int = 2000):
        om = self._raw(n_cal, rng)
        res = np.einsum("nb,kbd->knd", self.rtp._SPhi,
                        om.reshape(n_cal, self.rtp.cfg.Bw, 2))
        mid = res[:, self.rtp.cfg.N // 2, :]
        achieved = float(np.sqrt((mid ** 2).mean()))
        if achieved > 1e-9:
            self.scale *= target_mid_std / achieved

    # -- public -------------------------------------------------------------
    def sample_with_mode(self, n: int, rng: np.random.Generator):
        """Returns (omega (n, 2Bw), mode (n,) int) with mode indexing MODE_NAMES."""
        mode = rng.choice(len(self.mix), size=n, p=self.mix)
        return self.means[mode] + self._raw(n, rng), mode

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sample_with_mode(n, rng)[0]

    def base_paths(self) -> dict:
        """Mode name -> (N, 2) base positions, for plots and sanity checks."""
        kin = self.rtp.kinematics(self.means, self.x0, self.zone)
        return {name: kin.pos[i] for i, name in enumerate(MODE_NAMES)}