#!/usr/bin/env python3
"""
Export the real data Figure 1 draws: three screened recovery routes and the
single-reference baseline path that ends in pier contact.

Routes
------
Decodes candidates zone by zone at one severity, screens them with
``certify_batch`` and keeps, per zone, the screened candidate with the best
thruster-axis margin (min ``max_d2``), which is the same rule the sidecar uses
to pick what it publishes at stage 1. The omega vectors are saved alongside the
positions so the exact curves can be regenerated without rerunning the decoder.

Zone 3 is scarce at moderate severity (the decoder hugs Dock 2), so K is per
zone and several force draws are tried before giving up. If a zone yields
nothing screened, the script stops rather than quietly substituting an
unscreened curve; --allow-unscreened overrides that and records the fact in the
metadata so the caption can be corrected.

Baseline
--------
Reads an internal-arm bag, takes ground-truth odometry from fault onset, and
truncates at the first pier contact (signed distance < 0 in the obstacle
field), which is the event the figure is claiming.

Usage
-----
    python scripts/export_fig1_data.py --h1 0.5 --seed 1 --k 150 --draws 3
    python scripts/export_fig1_data.py --bag ~/usv_ws/experiments/bags/internal_d50_ss3_beneficial_seed0300
    python scripts/fig1_motivation.py --routes fig1_data/routes.npz \
                                      --baseline fig1_data/baseline.npy --debug-png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _bootstrap():
    try:
        import manifold_recovery  # noqa: F401
        return Path(__file__).resolve().parents[1]
    except ModuleNotFoundError:
        pass
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "manifold_recovery").is_dir():
            sys.path.insert(0, str(parent / "src"))
            return parent
    raise ModuleNotFoundError("manifold_recovery not importable")


ROOT = _bootstrap()

from manifold_recovery.scenario import SPIKE_START_POSE, ZONES, zone_of  # noqa: E402

ROUTE_KEYS = {0: "Z1", 1: "Z2", 2: "Z3"}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def export_routes(a) -> dict:
    from manifold_recovery.config import load
    from manifold_recovery.model.train import load as load_ckpt
    from manifold_recovery.model.decode import Decoder
    from manifold_recovery.traj.rtp import RTP
    from manifold_recovery.score.obstacles import DockField
    from manifold_recovery.certify.surrogate import certify_batch
    from manifold_recovery.data.env_forces import SyntheticSampler

    cfg = load(a.config)
    model, meta = load_ckpt(a.ckpt)
    ck_hash, cfg_hash = meta.get("config_hash"), cfg.hash()
    if ck_hash != cfg_hash:
        msg = (f"checkpoint config hash {ck_hash} != config {cfg_hash}: these "
               f"routes would not be the paper's model")
        if not a.allow_hash_mismatch:
            raise SystemExit("ERROR: " + msg + "\n(pass --allow-hash-mismatch "
                             "only if you know why they differ)")
        print("WARNING: " + msg)

    dec = Decoder(model, meta, cfg)
    rtp = RTP(cfg.trajectory)
    field = DockField()
    T_h, dt = rtp.horizon()
    sampler = SyntheticSampler()
    x0 = SPIKE_START_POSE
    sea = a.sea_state if a.sea_state is not None else cfg.data.sea_state
    dirn = a.direction or cfg.data.direction

    print(f"config hash {cfg_hash}   h1 {a.h1}   seed {a.seed}   "
          f"K {a.k}/zone/draw   draws {a.draws}   SS{sea} {dirn}")

    best: dict[int, dict] = {}
    counts = {z.id: [0, 0] for z in ZONES}          # [decoded, screened]

    for d in range(a.draws):
        rng = np.random.default_rng(a.seed + 1000 * d)   # recorded, per draw
        w, sig = sampler.sample(sea, dirn, T_h, dt, rng)
        w = w[:cfg.trajectory.N]
        for z in ZONES:
            om, _ = dec.decode_zones(a.h1, x0, a.k, rng, zones=[z])
            wb = np.broadcast_to(w, (len(om),) + w.shape)
            cert = certify_batch(om, x0, a.h1, 1.0, wb, sig, rtp, field, cfg)
            pos = rtp.kinematics(om, x0).pos                      # (K, N, 2)
            term_ok = zone_of(pos[:, -1, :]) == z.id
            counts[z.id][0] += len(om)
            ok = np.flatnonzero(cert.mask & term_ok)
            counts[z.id][1] += len(ok)
            pool = ok if len(ok) else (np.flatnonzero(term_ok)
                                       if a.allow_unscreened else np.array([], int))
            if len(pool) == 0:
                continue
            i = int(pool[np.argmin(cert.max_d2[pool])])
            cand = {"draw": d, "index": i, "screened": bool(cert.mask[i]),
                    "max_d2": float(cert.max_d2[i]),
                    "min_clear_m": float(cert.min_clear[i]),
                    "alpha_bar": float(cert.alpha_bar),
                    "pos": pos[i], "omega": om[i]}
            prev = best.get(z.id)
            # prefer screened, then the better thruster-axis margin
            better = (prev is None
                      or (cand["screened"] and not prev["screened"])
                      or (cand["screened"] == prev["screened"]
                          and cand["max_d2"] < prev["max_d2"]))
            if better:
                best[z.id] = cand

    for z in ZONES:
        n_dec, n_scr = counts[z.id]
        got = best.get(z.id)
        state = ("none" if got is None
                 else ("screened" if got["screened"] else "UNSCREENED"))
        print(f"  {z.name}: {n_scr}/{n_dec} screened over {a.draws} draw(s) "
              f"-> {state}"
              + (f", max_d2 {got['max_d2']:.3g}, clearance {got['min_clear_m']:.2f} m"
                 if got else ""))

    missing = [ZONES[i].name for i in (0, 1, 2) if i not in best]
    if missing:
        raise SystemExit(
            f"ERROR: no route for {missing}. Raise --k or --draws, try another "
            f"--seed, or lower --h1 (the decoder's Zone 3 share is 0.04 at "
            f"h1 = 0.5). --allow-unscreened takes the best terminal-correct "
            f"candidate instead and records it in the metadata.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = {ROUTE_KEYS[i]: best[i]["pos"] for i in (0, 1, 2)}
    npz.update({f"omega_{ROUTE_KEYS[i]}": best[i]["omega"] for i in (0, 1, 2)})
    np.savez(out / "routes.npz", **npz)

    all_screened = all(best[i]["screened"] for i in (0, 1, 2))
    meta_out = {
        "h1": a.h1, "seed": a.seed, "k_per_zone_per_draw": a.k,
        "draws": a.draws, "sea_state": sea, "direction": dirn,
        "config_hash": cfg_hash, "checkpoint_hash": ck_hash,
        "ckpt": str(a.ckpt), "x0": list(map(float, x0)),
        "T_h": T_h, "N": int(cfg.trajectory.N),
        "all_screened": all_screened,
        "selection_rule": "screened and terminal-in-zone, min max_d2",
        "routes": {ROUTE_KEYS[i]: {k: v for k, v in best[i].items()
                                   if k not in ("pos", "omega")}
                   for i in (0, 1, 2)},
    }
    (out / "routes.meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"wrote {out / 'routes.npz'} and {out / 'routes.meta.json'}"
          + ("" if all_screened else "  [contains an UNSCREENED route]"))
    return meta_out


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------
def export_baseline(a):
    from manifold_recovery.io_bridge.bag_reader import read_bag
    from manifold_recovery.score.obstacles import DockField

    bag = Path(a.bag).expanduser()
    d = read_bag(bag)
    if len(d["t_h"]) == 0 or len(d["t_o"]) == 0:
        raise SystemExit(f"{bag.name}: missing health or odometry")
    fault = np.flatnonzero(d["h"].min(axis=1) < 99.9)
    if len(fault) == 0:
        raise SystemExit(f"{bag.name}: no fault in bag")
    t_f = float(d["t_h"][fault[0]])
    sel = (d["t_o"] >= t_f) & (d["t_o"] <= t_f + a.post_fault_s)
    pos = d["pos"][sel]
    if len(pos) < 2:
        raise SystemExit(f"{bag.name}: no odometry after the fault")

    sd = DockField().signed_distance(pos)
    hit = np.flatnonzero(sd < 0)
    if len(hit):
        pos = pos[:hit[0] + 1]
        print(f"  contact at t+{d['t_o'][sel][hit[0]] - t_f:.1f} s, "
              f"({pos[-1, 0]:.1f}, {pos[-1, 1]:.1f}), path truncated there")
    else:
        print(f"  WARNING: no pier contact in this bag (min clearance "
              f"{sd.min():.2f} m). The figure's contact marker would be "
              f"unsupported; pick an aligned internal-arm trial that collided.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "baseline.npy", pos)
    (out / "baseline.meta.json").write_text(json.dumps(
        {"bag": str(bag), "t_fault": t_f, "post_fault_s": a.post_fault_s,
         "n_points": int(len(pos)), "contact": bool(len(hit)),
         "min_clearance_m": float(sd.min()),
         "end": [float(pos[-1, 0]), float(pos[-1, 1])]}, indent=2))
    print(f"wrote {out / 'baseline.npy'} ({len(pos)} points)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/spike.yaml"))
    p.add_argument("--ckpt", default=str(ROOT / "runs/ckpt.pt"))
    p.add_argument("--out", default="fig1_data")
    p.add_argument("--h1", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--k", type=int, default=150, help="decodes per zone per draw")
    p.add_argument("--draws", type=int, default=3, help="force draws")
    p.add_argument("--sea-state", type=int, default=None)
    p.add_argument("--direction", default=None,
                   choices=["beneficial", "nominal"])
    p.add_argument("--allow-hash-mismatch", action="store_true")
    p.add_argument("--allow-unscreened", action="store_true",
                   help="fall back to the best terminal-correct candidate for a "
                        "zone with no screened decode, and record it")
    p.add_argument("--skip-routes", action="store_true")
    p.add_argument("--bag", default=None,
                   help="internal-arm bag directory for the baseline path")
    p.add_argument("--post-fault-s", type=float, default=400.0)
    a = p.parse_args()

    if not a.skip_routes:
        export_routes(a)
    if a.bag:
        print("baseline:")
        export_baseline(a)


if __name__ == "__main__":
    main()
