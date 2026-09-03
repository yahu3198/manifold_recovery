import _common  # noqa: F401
import argparse

from manifold_recovery.config import load
from manifold_recovery.data.dataset import ManifoldDataset
from manifold_recovery.model.train import train
from manifold_recovery.analysis.plots import plot_history

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(_common.ROOT / "configs/spike.yaml"))
p.add_argument("--data", default=str(_common.ROOT / "data_out/spike_dataset.npz"))
p.add_argument("--out", default=str(_common.ROOT / "runs/ckpt.pt"))
p.add_argument("--epochs", type=int, default=None)
a = p.parse_args()
cfg = load(a.config)
if a.epochs is not None:
    from dataclasses import replace
    cfg = replace(cfg, model=replace(cfg.model, epochs=a.epochs))
ds = ManifoldDataset(a.data)
path, hist = train(cfg, ds, a.out)
plot_history(hist, _common.ROOT / "runs/training_history.png")
print("checkpoint ->", path)
