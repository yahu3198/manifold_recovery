import pathlib
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import ZONES, DOCKS, SPIKE_START_POSE, SPIKE_ZONE_ID
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.data.proposal import ProposalSampler

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def test_zone_constants_verbatim():
    assert ZONES[0].vertices[0] == (-580.0, 258.0)
    assert ZONES[1].vertices[2] == (-595.0, 208.0)
    assert len(DOCKS) == 2 and all(p.is_valid for p in DOCKS)


def test_halfspaces_contain_centroid():
    for z in ZONES:
        A, b = z.halfspaces()
        assert np.all(A @ z.center - b <= 1e-9)


def test_proposal_pinned_and_calibrated():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(0)
    prop = ProposalSampler(rtp, CFG.data.prop_mid_std_m, rng)
    om = prop.sample(400, rng)
    kin = rtp.kinematics(om, SPIKE_START_POSE, ZONES[SPIKE_ZONE_ID])
    assert np.abs(kin.pos[:, 0] - SPIKE_START_POSE[:2]).max() < 1e-8
    mid = kin.pos[:, CFG.trajectory.N // 2] - kin.pos[:, CFG.trajectory.N // 2].mean(0)
    s = float(np.sqrt((mid ** 2).mean()))
    assert 2.0 < s < 5.0, s
