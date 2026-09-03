# ROS 2 sidecar (full build, not part of the spike)

The offline pipeline in `src/manifold_recovery` is deliberately ROS-free.
Deployment follows the sidecar plan agreed in the design document:

- New package `vrx_manifold` (Python, ament) hosting a single node that wraps
  `pipeline.online.RecoveryPipeline`. It subscribes to `/wamv/odom`,
  `/wamv/thruster_health` (or the fault-diagnosis output), `/wamv/disturbance`
  and `/wamv/prediction_metrics`, and on a fault event publishes the selected
  candidate as `/wamv/manifold_ref` using the existing 8-column reference
  schema `[x, y, psi, u, v, r, Tp, Ts]` that `vrx_control` already consumes.
- The only diff inside `vrx_control` is a `ref_source` parameter and one
  additional subscription; the MPC node remains untouched otherwise.
- Force logging for `io_bridge/bag_reader.py` records
  `/wamv/disturbance`, `/wamv/disturbance_world`, `/wamv/learned_features`,
  `/wamv/prediction_metrics` with bag names `env_log_ss<k>_<direction>_run<n>`.

Nothing in this directory is required for gates G1-G4.
