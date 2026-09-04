import pathlib
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import (ZONES, DOCKS, LAND_VERTICES, ENTRY_VIAS, SPIKE_START_POSE,
                                        sample_start, zone_of, route_waypoints)
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.score.obstacles import DockField
from manifold_recovery.data.proposal import ProposalSampler
from manifold_recovery.certify.cluster import side_signature

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def test_zone_constants_verbatim():
    assert ZONES[0].vertices[0] == (-580.0, 258.0)
    assert ZONES[1].vertices[2] == (-595.0, 208.0)
    assert len(DOCKS) == 2 and all(p.is_valid for p in DOCKS)


def test_halfspaces_contain_centroid():
    for z in ZONES:
        A, b = z.halfspaces()
        assert np.all(A @ z.center - b <= 1e-9)


def test_shoreline_blocks_west_and_leaves_zones_open():
    field = DockField()
    # points on land (west of the quay, behind the moles) are inside an obstacle
    assert (field.signed_distance(np.array([[-605.0, 230.0], [-590.0, 254.0], [-586.0, 182.5]])) < 0).all()
    # zone interiors and entry vias are free water
    for z in ZONES:
        assert (field.signed_distance(z.sample_inside(np.random.default_rng(0), 50)) > 0).all()
    assert (field.signed_distance(np.array(list(ENTRY_VIAS.values()), float)) > 0).all()
    assert len(LAND_VERTICES) == 7


def test_via_routes_clear_obstacles():
    field = DockField()
    for z in ZONES:
        wp = route_waypoints(z, SPIKE_START_POSE, z.center, 200)
        assert field.signed_distance(wp).min() > 1.0, z.name


def test_proposal_modes_map_to_zone_classes():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(0)
    prop = ProposalSampler(rtp, CFG.data.prop_mid_std_m, rng)
    x0 = sample_start(rng, 300)
    om, mode = prop.sample_with_mode(x0, rng)
    kin = rtp.kinematics(om, x0)
    assert np.abs(kin.pos[:, 0] - x0[:, :2]).max() < 1e-8
    assert (zone_of(kin.pos[:, -1]) == mode).all()          # goal inside the chosen zone
    sig = side_signature(kin.pos)
    assert (sig[:, 0] == mode).all()
    # residual noise calibrated to prop_mid_std_m at mid-path
    w = prop._raw(400, rng)
    res = np.einsum("nb,kbd->knd", rtp._SPhi, w)
    s = float(np.sqrt((res[:, CFG.trajectory.N // 2] ** 2).mean()))
    assert 2.0 < s < 5.0, s