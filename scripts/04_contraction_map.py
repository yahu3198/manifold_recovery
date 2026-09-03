import _common  # noqa: F401
import argparse
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.analysis.contraction import (compute_maps, h1_test,
                                                    h2_test, h3_test)
from manifold_recovery.analysis.plots import plot_V_heatmap

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--ckpt", default=str(_common.ROOT / "runs/ckpt.pt"))
p.add_argument("--proposal-only", action="store_true")
a = p.parse_args()
cfg = load(a.config)
rng = np.random.default_rng(2)
decode_fn = None
if not a.proposal_only:
    import torch
    from manifold_recovery.model.train import load as load_ckpt
    from manifold_recovery.features.condition import Standardizer
    model, meta = load_ckpt(a.ckpt)
    std_om = Standardizer.from_dict(meta["std_omega"])
    std_c = Standardizer.from_dict(meta["std_c"])

    def decode_fn(h1, K):
        z = np.linspace(cfg.online.z_lo, cfg.online.z_hi, K)[:, None]
        c = std_c.transform(np.full((K, 1), h1, dtype=np.float32))
        om = model.decode(torch.tensor(z, dtype=torch.float32),
                          torch.tensor(c, dtype=torch.float32)).numpy()
        return std_om.inverse(om)

df = compute_maps(decode_fn, cfg, SyntheticSampler(), rng)
(_common.ROOT / "runs").mkdir(exist_ok=True)
df.to_csv(_common.ROOT / "runs/maps.csv", index=False)
val = "V_proposal" if a.proposal_only else "V"
plot_V_heatmap(df, val, _common.ROOT / f"runs/heatmap_{val}.png",
               title=f"{val} over severity x sea state")
print(df.to_string(index=False))
if not a.proposal_only:
    print("H1:", h1_test(df)); print("H2:", h2_test(df)); print("H3:", h3_test(df))
