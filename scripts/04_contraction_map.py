import _common  # noqa: F401
import argparse
import numpy as np

from manifold_recovery.config import load
from manifold_recovery.data.env_forces import SyntheticSampler
from manifold_recovery.analysis.contraction import (compute_maps, h1_test,
                                                    h2_test, h3_test, lift_test)
from manifold_recovery.analysis.plots import plot_V_heatmap

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--ckpt", default=str(_common.ROOT / "runs/ckpt.pt"))
p.add_argument("--proposal-only", action="store_true")
a = p.parse_args()
cfg = load(a.config)
rng = np.random.default_rng(2)
decoder = None
if not a.proposal_only:
    from manifold_recovery.model.train import load as load_ckpt
    from manifold_recovery.model.decode import Decoder
    model, meta = load_ckpt(a.ckpt)
    decoder = Decoder(model, meta, cfg)

df = compute_maps(decoder, cfg, SyntheticSampler(), rng)
(_common.ROOT / "runs").mkdir(exist_ok=True)
df.to_csv(_common.ROOT / "runs/maps.csv", index=False)
plot_V_heatmap(df, "V_proposal", _common.ROOT / "runs/heatmap_V_proposal.png",
               title="V_proposal over severity x sea state")
if not a.proposal_only:
    plot_V_heatmap(df, "V", _common.ROOT / "runs/heatmap_V.png",
                   title="V over severity x sea state")
    plot_V_heatmap(df, "lift", _common.ROOT / "runs/heatmap_lift.png",
                   title="lift = V / V_proposal")
print(df.to_string(index=False))
if not a.proposal_only:
    print("H1:", h1_test(df)); print("H2:", h2_test(df)); print("H3:", h3_test(df))
    print("LIFT (pre-registered):", lift_test(df))
