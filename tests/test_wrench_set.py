import numpy as np
from scipy.optimize import lsq_linear

from manifold_recovery.dynamics.params import PARAMS, B_matrix
from manifold_recovery.score.wrench_set import WrenchBoxes, distance_batch


def test_matches_scipy_box_ls():
    rng = np.random.default_rng(0)
    n = 200
    tau = rng.uniform(-2000, 2000, (n, 1, 3))
    h = rng.uniform(0, 1, (n, 2))
    w = rng.uniform(-300, 300, (n, 1, 3))
    boxes = WrenchBoxes.unshrunk(PARAMS.u_max, 1.0)
    d2, u_star, a_star = distance_batch(tau, h, w, boxes, iters=1500)
    B = B_matrix()
    for i in range(0, n, 7):
        A = np.zeros((3, 5))
        A[:, :2] = B * h[i]
        A[0, 2], A[1, 3], A[2, 4] = w[i, 0]
        res = lsq_linear(A, tau[i, 0], bounds=(np.zeros(5),
                         np.array([PARAMS.u_max, PARAMS.u_max, 1, 1, 1])))
        d2_ref = float(np.sum((A @ res.x - tau[i, 0]) ** 2))
        assert abs(d2[i, 0] - d2_ref) <= 1e-3 * max(d2_ref, 1.0), (d2[i, 0], d2_ref)


def test_zero_distance_for_interior_wrench():
    rng = np.random.default_rng(1)
    h = np.array([[0.7, 1.0]])
    u = np.array([500.0, 800.0])
    w = rng.uniform(-200, 200, (1, 1, 3))
    alpha = np.array([0.4, 0.4, 0.4])
    B = B_matrix()
    tau = (B @ (h[0] * u) + alpha * w[0, 0])[None, None]
    boxes = WrenchBoxes.unshrunk(PARAMS.u_max, 1.0)
    d2, _, _ = distance_batch(tau, h, w, boxes, iters=600)
    assert d2[0, 0] < 1e-3   # N^2; residual force < 0.03 N
