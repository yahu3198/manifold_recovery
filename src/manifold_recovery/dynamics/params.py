"""Vessel constants ported from the ICRA EAMPC codebase.

Provenance (single source of truth = the deployed ACADOS model, NOT the paper's
general Fossen form):
  - vrx_control/scripts/wamv.py           : m, Izz, l ('B'), damping, added mass = 0
  - vrx_control/scripts/generate_c_code.py: u bounds [0, 2353] N, W_u = diag(0.001)
  - vrx_control/include/vrx_control/wamv_mpc.h duplicates the same numbers.
The code model has diagonal M, zero added mass, and no x_g coupling; keep it so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VesselParams:
    m: float = 180.0
    Izz: float = 446.0
    l: float = 2.05427          # thruster separation (called B in wamv.py)
    xu: float = -100.0
    xuu: float = -150.0
    yv: float = -100.0
    yvv: float = -100.0
    nr: float = -300.0
    nrr: float = -300.0
    u_min: float = 0.0
    u_max: float = 2353.0


PARAMS = VesselParams()


def B_matrix(l: float = PARAMS.l) -> np.ndarray:
    """Thrust configuration matrix (3x2). Rows: surge, sway, yaw.

    Matches wamv.py: Tx = Tp + Ts; Ty = 0; Mz = -l/2*Tp + l/2*Ts.
    """
    return np.array([[1.0, 1.0], [0.0, 0.0], [-l / 2.0, l / 2.0]])
