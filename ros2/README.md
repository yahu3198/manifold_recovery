# Deployment bridge (rev 4.2): closed-loop trials in VRX

Pairs with the step-1 changes in `vrx/vrx_control` (parameters, `MANIFOLD_RETURN`
mode, position trigger, `/wamv/manifold_ref`, `/wamv/manifold_status`).

## Pieces

| file | role |
|---|---|
| `ros2/vrx_manifold_node.py` | sidecar: detects the fault from `/wamv/thruster_health`, forecasts the environment from the last 60 s of `/wamv/disturbance`, runs the arm, publishes the 8-column reference, logs a JSON per trial |
| `src/.../io_bridge/forecast.py` | mean + dominant-oscillation forecast (frequency refined within one FFT bin), bootstrapped "true" realisation for the tier-2 pre-screen, `sigma_theta_from_confidence` |
| `src/.../io_bridge/reference.py` | plan (X, U) -> `[t0, dt, rows of 8]` at 0.05 s with a 30 s terminal hold |
| `src/.../pipeline/online.py` | `propose` (manifold arm) and `propose_b4` (rejection arm) share one tail: certify -> cluster -> fine-tune -> rollout -> rank |
| `ros2/run_trials.py` | campaign runner: per (arm, cell, seed) launches Gazebo (trial world + canonical spawn), bag record, sidecar, MPC; stops on completion or 180 s after the fault; manifests |
| `ros2/score_trials.py` | offline scorer from bags (same geometry as the certificate) + force-store builder |
| `src/.../io_bridge/bag_reader.py` | heading-filtered, pre-fault-only windows; sigma_theta from prediction confidence |

## Environment

The sidecar imports rclpy and the trained model. Create the venv with
`python3 -m venv --system-site-packages .venv` (or add the ROS Python path)
and source ROS before running. `pip install rosbags pandas` for the scorer.

## Order

1. Build vrx_control with the step-1 changes. Manual check without the sidecar
   (position trigger, hold, fallback).
2. `python3 vrx_control/scripts/make_trial_worlds.py --template <sydney_regatta.sdf> --out ~/usv_ws/worlds --n-trials 20`
3. Sign check in a beneficial world: heading west, `/wamv/disturbance` w_x > 0.
4. One manual trial with the sidecar:
   `python ros2/vrx_manifold_node.py --arm manifold --ckpt runs/ckpt.pt --trial-id manual --viz`
   then launch the MPC with `ref_source:=manifold`. Expect "published N rows"
   in the sidecar log and "reference received" in the MPC log, no 2*pi yaw jump
   in `/wamv/error_pose`.
5. Campaign: `python ros2/run_trials.py --worlds ~/usv_ws/worlds --out ~/usv_ws/experiments --arms manifold b4 internal --degrade 0.5 0.95 --n-trials 20`
   (about 12-15 h wall time for 240 trials; resumable, finished trials are skipped).
6. `python ros2/score_trials.py --bags ~/usv_ws/experiments/bags --out ~/usv_ws/experiments --build-store data_out/force_segments.npz`
7. Logged-force offline rerun: `python scripts/01_generate_data.py --logged-store data_out/force_segments.npz`, then 02-05.

## Known limitation to carry into the paper

The OCP and the certificate admit only the trusted fraction alpha_bar of the
predicted force into the plant model (the ICRA virtual-actuator semantics).
The tier-2 pre-screen, which applies the FULL predicted force plus residuals,
passes in beneficial conditions and fails on tracking error under strong
quartering or head forces (70 N sway / 110 N head at 0.7 m/s in the synthetic
test). The sidecar therefore publishes the best certified candidate regardless
of the pre-screen result and logs `rollout_passed`; the Gazebo MPC's feedback
is the real tier 2. A rev-5 certificate should put the full predicted force in
the dynamics and use alpha_bar only to bound reliance on the helpful component.
