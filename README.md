# manifold_recovery

Spike codebase for *Learning Conditional Recovery-Trajectory Manifolds for
Fault-Degraded USVs with Certified Feasibility* (RAL, target Nov 2026).
Companion documents: the implementation specification and the code-structure
design doc; this repository implements Phases 0-5 (gates G1-G4). ROS 2 / VRX
integration is the full build and lives outside the spike (see `ros2/`).

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[model]"        # torch needed only for training/gates
pip install -e ".[logs]"         # rosbags, only for bag import
pip install pytest
```

## Spike workflow (in order)

```bash
source .venv/bin/activate
pytest tests/ -q                          # Phase 0-2 invariants
python scripts/00_precheck_geometry.py    # G4 prerequisite: >= 2 classes exist
python scripts/01_generate_data.py        # 8k samples, ~20 s, synthetic w_hat
python scripts/02_train.py                # CVAE, ~700 epochs CPU
python scripts/03_eval_gates.py           # G1-G4 -> runs/spike_report.md
python scripts/04_contraction_map.py      # V heatmaps, H1-H3 tests
python scripts/05_baselines.py            # B1/B2/B4 (+B3 with torch)
```

Switching from synthetic to logged forces once the force-logging campaign has
run: `python -c "from manifold_recovery.io_bridge.bag_reader import
extract_force_segments; extract_force_segments([...bags...], 'data_out/forces.npz')"`
then pass `--logged-store data_out/forces.npz` to `01_generate_data.py`.
Nothing else changes (single `EnvForceSampler` interface).

## Layout (see the design doc for per-file APIs)

```
src/manifold_recovery/
  config.py scenario.py       typed config + verbatim harbor geometry
  dynamics/                   deployed ACADOS model, batched (consistency-tested)
  traj/                       RTP basis + crab-angle inversion (R1, R2 fixes)
  score/                      dock costs, wrench-set distance, score assembly
  features/                   ONE condition builder for offline and fault time
  data/                       proposal, env forces (synthetic|logged), dataset
  io_bridge/                  rosbag2 -> force-segment store (no ROS install)
  model/                      CVAE, weighted ELBO + capacity anneal, training
  planner/                    energy OCP (Eq. 13), CasADi+IPOPT, warm-startable
  certify/                    surrogate (alpha_bar policy, growing margins),
                              side-signature clustering, exact rollout check
  pipeline/                   RecoveryPipeline.propose with per-stage timings
  analysis/                   V/N_H maps, H1-H3 (embedded ICRA Fig. 5 grid)
  baselines/                  B1 restarts, B2 CEM, B3 per-instance, B4 rejection
  external/wamv_reference.py  vendored wamv.py, ground truth for tests
```

Structural invariants worth defending in review: one optimizer
(`planner/energy_ocp.py`) serves fine-tuning, B1, and the G3 reference; one
certificate (`certify_batch`) serves gates, maps, B4, and deployment; one
condition builder serves training and fault time; `tests/test_dynamics_
consistency.py` pins the physics to the deployed model at 1e-10.

## Validation status (container, Aug 2026)

- 13/14 tests green; `test_model_overfit` auto-skips without torch.
- `00_precheck_geometry.py`: PASS, two homotopy classes at h1 = 0.9
  (signatures (-1,+1)/(+1,-1)), B1 best cost 1345.9, 26 s for 7 seeds.
- Dataset generation: 600 samples in 1.4 s; ESS/n ~= 0.13 (=> ~1000 at 8k).
- Wrench solver vs scipy `lsq_linear`: worst relative d2 error 7e-5 at 1500
  iterations (preconditioned FISTA); production default 300 iterations.
- Not yet executed here: training (torch), hence G1-G4 numbers and the
  V heatmaps. `04_contraction_map.py --proposal-only` runs without torch.

## Calibration markers to resolve before the paper

- `FROM_LOGS` in `data/env_forces.py`: force-magnitude constants and the
  sigma_theta heuristic; replaced by the ~8-run force-logging campaign.
- `FROM_REPO` fallback `gamma_R` in the config (paper's R(H) health penalty;
  value not present in the repo).
- `eps_cert` and `dock_margin` are Phase 1 tuning constants.
- B1 seed convergence: 2/7 seeds converge cold; acceptable for class
  enumeration, but tighten (more IPOPT iterations, better velocity seeds)
  before quoting B1 as the optimum in G3.

### Phase-1 calibration note (in-container, 28 Aug 2026)

`eps_cert` was calibrated from 1e-3 to 25.0 N^2. Diagnostic: even near-feasible
proposal trajectories carry an irreducible worst-waypoint wrench residual of
4-6 N (17-30 N^2) from the crab-angle grid quantisation (41 points over
+-25 deg) and the dropped m*vdot term in the quasi-static sway balance; 1e-3
N^2 (0.03 N) sat four orders of magnitude below that floor and rejected
everything. A 5 N unmodeled force shifts equilibrium speed ~2 cm/s against
250 N damping, well inside the tier-2 rollout tolerance (e_max = 5 m), which
remains the exact arbiter. Certification now uses `cert_iters = 1500` FISTA
iterations (score keeps 300). With this calibration the proposal-only
contraction map (`runs/maps.csv`) is non-degenerate: V_proposal rises with sea
state at 50-75% degradation (0.01 -> 0.32) and is 0 at 95-100%, exactly the
headroom H1/G3 ask the trained manifold to fill.
