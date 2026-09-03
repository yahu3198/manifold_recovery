import pathlib
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import ZONES, SPIKE_START_POSE, SPIKE_ZONE_ID
from manifold_recovery.traj.rtp import RTP

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def test_endpoints_and_condition():
    rtp = RTP(CFG.trajectory)
    assert rtp.cond < 1e4
    rng = np.random.default_rng(0)
    om = rng.standard_normal((8, 2 * CFG.trajectory.Bw))
    kin = rtp.kinematics(om, SPIKE_START_POSE, ZONES[SPIKE_ZONE_ID])
    assert np.abs(kin.pos[:, 0] - SPIKE_START_POSE[:2]).max() < 1e-9
    assert np.abs(kin.pos[:, -1] - ZONES[SPIKE_ZONE_ID].center).max() < 1e-9


def test_analytic_derivatives_match_fd():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(1)
    om = rng.standard_normal((4, 2 * CFG.trajectory.Bw))
    kin = rtp.kinematics(om, SPIKE_START_POSE, ZONES[SPIKE_ZONE_ID])
    vel_fd = np.gradient(kin.pos, kin.dt, axis=1)
    err = np.abs(kin.vel[:, 2:-2] - vel_fd[:, 2:-2]).max()
    assert err < 0.3 * np.abs(kin.vel).max()   # ramp kinks excluded, coarse fd


def test_projection_roundtrip():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(2)
    om = rng.standard_normal((4, 2 * CFG.trajectory.Bw))
    kin = rtp.kinematics(om, SPIKE_START_POSE, ZONES[SPIKE_ZONE_ID])
    om2 = rtp.project(kin.pos, SPIKE_START_POSE, ZONES[SPIKE_ZONE_ID])
    kin2 = rtp.kinematics(om2, SPIKE_START_POSE, ZONES[SPIKE_ZONE_ID])
    assert np.abs(kin.pos - kin2.pos).max() < 1e-6
