"""Rev 4.2: forecaster, reference conversion, sigma mapping (no ROS needed)."""
import numpy as np

from manifold_recovery.io_bridge.forecast import ForceForecaster, sigma_theta_from_confidence
from manifold_recovery.io_bridge.reference import plan_to_rows, rows_to_message_data, wrap_pi


def test_forecaster_recovers_mean_and_oscillation():
    fc = ForceForecaster(60.0, 0.1)
    rng = np.random.default_rng(0)
    for t in np.arange(0.0, 70.0, 0.02):
        fc.push(t, [-120 + 40 * np.cos(2 * np.pi * t / 4.5 + 0.3) + rng.normal(0, 5),
                    60 + 25 * np.cos(2 * np.pi * t / 4.5 + 1.2) + rng.normal(0, 5),
                    5 + 3 * np.sin(2 * np.pi * t / 11)])
    assert fc.ready()
    w_hat, w_true, dt, fit = fc.forecast(150.0, rng)
    assert w_hat.shape == w_true.shape == (1501, 3) and dt == 0.1
    assert abs(fit[0]["mean"] + 120) < 3 and abs(fit[0]["amp"] - 40) < 4 and abs(fit[0]["freq"] - 1 / 4.5) < 0.01
    assert abs(fit[1]["freq"] - 1 / 4.5) < 0.01
    assert not np.allclose(w_hat, w_true)          # bootstrapped residual added


def test_sigma_mapping_matches_mpc_trust_floor():
    for c in (0.0, 0.05, 0.5, 0.8, 1.0):
        s = sigma_theta_from_confidence(c, 1.5, 0.01)
        conf = min(1.0, 1.5 / (s + 0.01))
        assert abs(conf - max(np.sqrt(c), 0.3)) < 1e-9


def test_plan_to_rows_layout():
    class P:
        X = np.vstack([np.linspace(-460, -580, 81), np.full(81, 221.0), np.linspace(3.1, 3.4, 81),
                       np.full(81, 0.8), np.zeros(81), np.zeros(81)])
        U = np.full((2, 80), 100.0)
    rows = plan_to_rows(P, 150.0, 0.05, feedforward=True, hold_s=10.0)
    assert rows.shape == (3001 + 200, 8)
    assert np.all(np.abs(rows[:, 2]) <= np.pi)           # psi wrapped
    assert rows[-1, 3] == 0.0 and rows[-1, 0] == -580     # hold at the terminal pose, zero velocity
    assert rows[10, 6] == 100.0
    d = rows_to_message_data(rows, 123.4, 0.05)
    assert d[0] == 123.4 and d[1] == 0.05 and (len(d) - 2) % 8 == 0
    assert abs(wrap_pi(3.4) - (3.4 - 2 * np.pi)) < 1e-12
