import pathlib
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import ZONES, SPIKE_START_POSE, sample_start
from manifold_recovery.traj.rtp import RTP

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def _random_omega(rtp, rng, n):
    pg = np.stack([ZONES[i % 3].center for i in range(n)])
    w = rng.standard_normal((n, rtp.cfg.Bw, 2))
    return rtp.join(pg, w), pg


def test_endpoints_pinned_to_start_and_goal():
    rtp = RTP(CFG.trajectory)
    assert rtp.cond < 1e4
    rng = np.random.default_rng(0)
    om, pg = _random_omega(rtp, rng, 8)
    kin = rtp.kinematics(om, SPIKE_START_POSE)
    assert np.abs(kin.pos[:, 0] - SPIKE_START_POSE[:2]).max() < 1e-9
    assert np.abs(kin.pos[:, -1] - pg).max() < 1e-9


def test_batched_starts():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(3)
    om, pg = _random_omega(rtp, rng, 6)
    x0 = sample_start(rng, 6)
    kin = rtp.kinematics(om, x0)
    assert np.abs(kin.pos[:, 0] - x0[:, :2]).max() < 1e-9
    assert np.abs(kin.pos[:, -1] - pg).max() < 1e-9


def test_analytic_derivatives_match_fd():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(1)
    om, _ = _random_omega(rtp, rng, 4)
    kin = rtp.kinematics(om, SPIKE_START_POSE)
    vel_fd = np.gradient(kin.pos, kin.dt, axis=1)
    err = np.abs(kin.vel[:, 2:-2] - vel_fd[:, 2:-2]).max()
    assert err < 0.3 * np.abs(kin.vel).max()


def test_projection_roundtrip():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(2)
    om, _ = _random_omega(rtp, rng, 4)
    kin = rtp.kinematics(om, SPIKE_START_POSE)
    om2 = rtp.project(kin.pos, SPIKE_START_POSE)
    kin2 = rtp.kinematics(om2, SPIKE_START_POSE)
    assert np.abs(kin.pos - kin2.pos).max() < 1e-6