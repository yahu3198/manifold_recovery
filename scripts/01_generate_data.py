import _common  # noqa: F401
import argparse
import numpy as np  # noqa: F401

from manifold_recovery.config import load
from manifold_recovery.data.dataset import generate
from manifold_recovery.data.env_forces import SyntheticSampler, LoggedSampler

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--out", default=str(_common.ROOT / "data_out/spike_dataset.npz"))
p.add_argument("--n", type=int, default=None)
p.add_argument("--logged-store", default=None,
               help="npz from io_bridge.bag_reader; default = synthetic")
a = p.parse_args()
cfg = load(a.config)
if a.n is not None:
    from dataclasses import replace
    cfg = replace(cfg, data=replace(cfg.data, n_samples=a.n))
sampler = LoggedSampler(a.logged_store) if a.logged_store else SyntheticSampler()
generate(cfg, sampler, a.out)
