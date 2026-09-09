"""Model-side representation of omega (rev 4).

Physics side (RTP, score, certificate):  omega = [p_g (absolute, m), w].
Model side (encoder/decoder):            omega_m = [p_g - centre(g), w].

Given the target zone g, the endpoint the model has to produce is a small
within-zone offset (a few metres) instead of an absolute position whose
variance is dominated by WHICH zone. Combined with the one-hot g in c, the
discrete zone choice is removed from the continuous latent entirely, so the
decoder has nothing left to average over.
"""
from __future__ import annotations

import numpy as np

from ..scenario import ZONES

_CENTRES = np.stack([z.center for z in ZONES])          # (n_zones, 2)


def to_model(omega: np.ndarray, g: np.ndarray) -> np.ndarray:
    om = np.array(omega, float, copy=True)
    om[..., :2] -= _CENTRES[np.asarray(g, int)]
    return om


def from_model(omega_m: np.ndarray, g: np.ndarray) -> np.ndarray:
    om = np.array(omega_m, float, copy=True)
    om[..., :2] += _CENTRES[np.asarray(g, int)]
    return om