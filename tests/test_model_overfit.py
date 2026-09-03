import pathlib
import pytest

torch = pytest.importorskip("torch")

from manifold_recovery.config import load
from manifold_recovery.model.cvae import CVAE
from manifold_recovery.model.losses import weighted_elbo

CFG = load(pathlib.Path(__file__).resolve().parents[1] / "configs/spike.yaml")


def test_overfit_small():
    """The CVAE must exploit BOTH pathways: condition c and latent z.

    Data are constructed so a 1-D latent suffices: omega = 2*c + 3*z_true*u
    + 0.1*noise. The reconstruction floor is ~0.5*40*0.01 = 0.2, so large
    relative and absolute improvements are achievable (unlike the previous
    version of this test, whose i.i.d. 40-D noise had a floor of ~20 that no
    1-D latent model could beat; observed convergence to 19.2 confirmed the
    model was already correct).
    """
    torch.manual_seed(0)
    n, d = 200, 2 * CFG.trajectory.Bw
    c = torch.rand(n, 1)
    z_true = torch.randn(n, 1)
    u = torch.randn(d)
    u = u / u.norm()
    om = 2.0 * c + 3.0 * z_true * u + 0.1 * torch.randn(n, d)
    f = torch.ones(n)
    model = CVAE(d, 1, 1, (64, 64))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = last = None
    for ep in range(600):
        oh, mu, lv = model(om, c)
        loss, parts = weighted_elbo(oh, om, mu, lv, f, gamma=10.0,
                                    Cz=min(5.0, 5.0 * ep / 240),
                                    recon_sigma=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep == 0:
            first = parts["recon"]
        last = parts["recon"]
    # 10x relative drop AND within reach of the ~0.2 floor.
    assert last < first / 10.0, (first, last)
    assert last < 2.0, last
