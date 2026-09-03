import pathlib
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.certify.surrogate import (mu_H, alpha_bar_policy,
                                                 margin_schedule)

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def test_mu_H_matches_icra_eq9():
    assert mu_H(1.0, 1.0) == 0.0
    assert mu_H(0.05, 0.05) == 1.0
    assert abs(mu_H(0.5, 1.0) - (1.0 - 1.5 / 2.0)) < 1e-12


def test_alpha_bar_monotone_in_confidence():
    a_good = alpha_bar_policy(0.2, 0.1, 1.0, CFG)
    a_bad = alpha_bar_policy(3.0, 0.1, 1.0, CFG)
    assert a_good >= a_bad >= 0.0
    assert a_good <= mu_H(0.1, 1.0) + 1e-12


def test_margins_grow_with_time():
    rho_u, rho_a = margin_schedule(CFG.trajectory.N, 1.5, CFG)
    assert rho_u[-1] > rho_u[0] and rho_a[-1] > rho_a[0]
    assert rho_u[0] >= CFG.wrench.rho_u_frac * CFG.wrench.u_max * 0.999
