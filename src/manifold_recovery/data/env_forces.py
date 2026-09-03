"""Environmental force realisations behind one interface (spec 5.3).

SyntheticSampler: mean + wave-band oscillation + OU drift, matched loosely to
the ICRA Table I sea-state families. All magnitude constants are CALIBRATION
PLACEHOLDERS (marked FROM_LOGS) to be replaced by the force-logging campaign;
they are chosen so sea state 3 produces |w_xy| of order 100-200 N, consistent
with the disturbance plots in the RO-MAN/ICRA papers.

LoggedSampler: draws tagged segments from the store written by
io_bridge/bag_reader.py and resamples them to the planning dt.

Both return (w_body [n, 3], sigma_theta) so the certificate's confidence
policy has a consistent input in either mode.
"""
from __future__ import annotations

import numpy as np

# (wind m/s, wave height m, wave period s) per ICRA sea-state families. FROM_LOGS.
SEA_STATE_TABLE = {
    2: (3.5, 0.30, 3.5),
    3: (6.0, 0.65, 4.5),
    4: (9.0, 1.10, 5.5),
    5: (12.0, 1.70, 6.5),
}
# Force direction bands (deg, direction the force pushes TOWARD in world/plan
# frame). 'beneficial' roughly toward the harbor (-x); 'nominal' oblique. FROM_LOGS.
DIRECTION_BANDS = {"beneficial": (160.0, 200.0), "nominal": (110.0, 145.0)}
C_WIND = 3.0          # N per (m/s)^2, FROM_LOGS placeholder
C_WAVE = 90.0         # N per m wave height, FROM_LOGS placeholder
YAW_FRACTION = 0.33   # matches the [0.5, 0.5, 0.33] directional coefficients


class SyntheticSampler:
    def sample(self, sea_state: int, direction: str, T: float, dt: float,
               rng: np.random.Generator):
        wind, hs, period = SEA_STATE_TABLE[int(sea_state)]
        n = int(round(T / dt)) + 1
        t = np.arange(n) * dt
        lo, hi = DIRECTION_BANDS[direction]
        th = np.deg2rad(rng.uniform(lo, hi))
        mean_mag = C_WIND * wind ** 2
        amp = C_WAVE * hs
        phase = rng.uniform(0, 2 * np.pi)
        osc = amp * np.sin(2 * np.pi * t / period + phase)
        # slow OU drift on magnitude and direction
        drift = np.zeros(n)
        dth = np.zeros(n)
        s_m, s_th = 0.08 * mean_mag, np.deg2rad(4.0)
        a = np.exp(-dt / 20.0)
        for k in range(1, n):
            drift[k] = a * drift[k - 1] + np.sqrt(1 - a * a) * s_m * rng.standard_normal()
            dth[k] = a * dth[k - 1] + np.sqrt(1 - a * a) * s_th * rng.standard_normal()
        mag = np.maximum(mean_mag + osc + drift, 0.0)
        ang = th + dth
        w = np.zeros((n, 3))
        w[:, 0] = mag * np.cos(ang)
        w[:, 1] = mag * np.sin(ang)
        w[:, 2] = YAW_FRACTION * mag * np.sin(2 * np.pi * t / (2.5 * period) + phase) \
            * np.sign(np.sin(th))
        # heuristic RLS parameter uncertainty: better excitation -> lower sigma.
        sigma_theta = float(np.clip(60.0 / (mag.std() + mean_mag * 0.2 + 1.0), 0.15, 3.0))
        return w, sigma_theta


class LoggedSampler:
    """Segment store: npz with w_body [Ns, T, 3], sigma_theta [Ns], dt,
    sea_state [Ns], direction [Ns] (bytes)."""

    def __init__(self, store_path: str):
        d = np.load(store_path, allow_pickle=True)
        self.w = d["w_body"]
        self.sigma = d["sigma_theta"]
        self.dt = float(d["dt"])
        self.sea = d["sea_state"]
        self.dirn = np.asarray([s.decode() if isinstance(s, bytes) else str(s)
                                for s in d["direction"]])

    def sample(self, sea_state: int, direction: str, T: float, dt: float,
               rng: np.random.Generator):
        mask = (self.sea == int(sea_state)) & (self.dirn == direction)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"no logged segments for sea_state={sea_state}, "
                             f"direction={direction}")
        i = int(rng.choice(idx))
        seg = self.w[i]
        t_src = np.arange(seg.shape[0]) * self.dt
        n = int(round(T / dt)) + 1
        t_tgt = np.arange(n) * dt
        if t_tgt[-1] > t_src[-1]:            # tile if the segment is short
            reps = int(np.ceil(t_tgt[-1] / t_src[-1])) + 1
            seg = np.concatenate([seg] * reps, axis=0)
            t_src = np.arange(seg.shape[0]) * self.dt
        start_max = t_src[-1] - t_tgt[-1]
        t0 = rng.uniform(0.0, max(start_max, 0.0))
        w = np.stack([np.interp(t_tgt + t0, t_src, seg[:, j]) for j in range(3)], axis=1)
        return w, float(self.sigma[i])
