"""Rev 4: zone-conditioned condition vector, zone-relative omega, decode path."""
import pathlib
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.scenario import ZONES, SPIKE_START_POSE, sample_start, zone_of
from manifold_recovery.traj.rtp import RTP
from manifold_recovery.features.condition import build_c, zone_from_c, DIM_C
from manifold_recovery.model.omega_transform import to_model, from_model
from manifold_recovery.model.decode import Decoder

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def test_build_c_shapes_and_zone():
    c1 = build_c(0.5, SPIKE_START_POSE, 2)
    assert c1.shape == (DIM_C,) and int(zone_from_c(c1)) == 2 and c1[0] == 0.5
    rng = np.random.default_rng(0)
    x0 = sample_start(rng, 5)
    g = np.array([0, 1, 2, 1, 0])
    c = build_c(rng.uniform(0, 1, 5), x0, g)
    assert c.shape == (5, DIM_C)
    assert (zone_from_c(c) == g).all()


def test_omega_transform_roundtrip():
    rtp = RTP(CFG.trajectory)
    rng = np.random.default_rng(1)
    g = rng.integers(0, 3, 40)
    pg = np.stack([ZONES[k].center for k in g]) + rng.normal(0, 2, (40, 2))
    om = rtp.join(pg, rng.standard_normal((40, rtp.cfg.Bw, 2)))
    om_m = to_model(om, g)
    assert np.abs(om_m[:, :2]).max() < 10.0          # zone-relative, a few metres
    assert np.allclose(from_model(om_m, g), om)


def test_decoder_zone_split_and_absolute_frame():
    """A stub model that returns a zero zone-relative endpoint must decode to
    the zone centroid of the requested zone, for every zone, in the absolute frame."""
    rtp = RTP(CFG.trajectory)
    dim = rtp.dim
    meta = {
        "std_omega": {"mean": [0.0] * dim, "std": [1.0] * dim},
        "std_c": {"mean": [0.0] * DIM_C, "std": [1.0] * DIM_C},
        "latent": CFG.model.latent_dim, "omega_repr": "zone_relative_v4",
        "z_bank": np.zeros((30, CFG.model.latent_dim), np.float32),
        "f_bank": np.ones(30, np.float32),
        "c_bank": np.stack([build_c(0.5, SPIKE_START_POSE, k % 3) for k in range(30)]).astype(np.float32),
        "g_bank": np.array([k % 3 for k in range(30)]),
    }
    stub = lambda z, c: np.zeros((len(z), dim), np.float32)
    dec = Decoder(stub, meta, CFG)
    om, g = dec.decode_zones(0.5, SPIKE_START_POSE, 9, np.random.default_rng(0))
    assert om.shape == (9, dim) and sorted(set(g.tolist())) == [0, 1, 2]
    kin = rtp.kinematics(om, SPIKE_START_POSE)
    assert (zone_of(kin.pos[:, -1]) == g).all()
    for k in range(3):
        assert np.allclose(om[g == k, :2], ZONES[k].center)