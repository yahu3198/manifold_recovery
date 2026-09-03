"""Pin fossen.py to the vendored ACADOS model (wamv.py) to 1e-10.

acados_template is shimmed with an attribute bag so the vendored file stays
byte-verbatim; only its CasADi expressions are exercised here.
"""
import sys
import types

import numpy as np
import casadi as ca

if "acados_template" not in sys.modules:            # shim, no ACADOS install
    _fake = types.ModuleType("acados_template")

    class _AcadosModel:                              # attribute bag
        pass

    _fake.AcadosModel = _AcadosModel
    sys.modules["acados_template"] = _fake

from manifold_recovery.dynamics import fossen
from manifold_recovery.external import wamv_reference


def _reference_fn():
    model = wamv_reference.export_wamv_model()
    return ca.Function("f", [model.x, model.u, model.p], [model.f_expl_expr])


def test_matches_vendored_model():
    f_ref = _reference_fn()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        x = rng.uniform(-5, 5, 6)
        u = rng.uniform(0, 2353, 2)
        w = rng.uniform(-300, 300, 3)
        h = rng.uniform(0, 1, 2)
        # wamv.py parameter vector: [Tp_prev, Ts_prev, wx, wy, wpsi, hTp, hTs]
        p = np.concatenate([[0.0, 0.0], w, h])
        ours = fossen.f_continuous(x, u, w, h)
        ref = np.asarray(f_ref(x, u, p)).ravel()
        assert np.allclose(ours, ref, atol=1e-10), (ours, ref)
