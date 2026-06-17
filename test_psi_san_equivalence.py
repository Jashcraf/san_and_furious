"""SAN is the static least-squares PSI estimator on one symmetric 5-frame burst.

These tests pin that equivalence so a future refactor of the phase-shifting
machinery can't silently change the SAN iteration.

Per dark-zone pixel the focal intensity is exactly quadratic in the command,
``I(w) = |E0|^2 + 2 Re(E0* k' w) + |k'|^2 |w|^2``, i.e. linear in the four
unknowns ``[|E0|^2, 2 Re(E0* k'), -2 Im(E0* k'), |k'|^2]``.  SAN evaluates this
at the 5 commands ``{w0, w0+/-1, w0+/-1j}`` and takes symmetric differences --
which are precisely the central-difference (least-squares) solution of that
linear system.  The only thing separating SAN from the pooled static-PSI solve
(:meth:`SpeckleNuller._furious_step`) is SAN's per-quadrature ``mod_cos``/
``mod_sin`` normalization vs. PSI's single pooled ``|k'|^2``; in the ideal linear
model they coincide, and they diverge only by the measured cos/sin quadrature
asymmetry (a few %, a forward-model nonlinearity at finite probe amplitude).

Runs under pytest, or standalone:  ``python test_psi_san_equivalence.py``
"""

import numpy as np

from san import CoronagraphModel
from san.algorithms import SpeckleNuller


def _san_burst(model):
    """The 5 dark-zone SAN frames at commands {0, +1, -1, +1j, -1j} from w0 = 0."""
    a = np.zeros(model.num_actuators)
    cosp, sinp = model.cos_probe, model.sin_probe
    dz = np.asarray(model.dark_zone.shaped, dtype=bool)
    I0 = model.image(a)[dz]
    Icp = model.image(a + cosp)[dz]
    Icm = model.image(a - cosp)[dz]
    Isp = model.image(a + sinp)[dz]
    Ism = model.image(a - sinp)[dz]
    return I0, Icp, Icm, Isp, Ism


def _san_estimate(I0, Icp, Icm, Isp, Ism):
    """SAN's per-quadrature kappa^2 and the correction it applies (w0 = 0)."""
    mod_cos = (Icp + Icm - 2 * I0) / 2.0
    mod_sin = (Isp + Ism - 2 * I0) / 2.0
    san_cos = (Icp - Icm) / (2 * mod_cos)
    san_sin = (Isp - Ism) / (2 * mod_sin)
    dw_san = -0.5 * (san_cos + 1j * san_sin)
    return mod_cos, mod_sin, dw_san


def test_psi_lstsq_recovers_san_step():
    """The 4-parameter static-PSI least-squares fit on the SAN burst reproduces
    SAN's kappa^2 and (with per-quadrature normalization) its correction exactly."""
    model = CoronagraphModel(iwa_ld=3, owa_ld=12, seed=1)
    I0, Icp, Icm, Isp, Ism = _san_burst(model)
    mod_cos, mod_sin, dw_san = _san_estimate(I0, Icp, Icm, Isp, Ism)

    # Static PSI: fit I = a0 + a1 c + a2 s + a3 |w|^2 over the 5 frames.
    W = np.array([0, 1, -1, 1j, -1j], dtype=complex)
    Iframes = np.stack([I0, Icp, Icm, Isp, Ism], axis=0)
    X = np.column_stack([np.ones(5), W.real, W.imag, np.abs(W) ** 2])
    a0, a1, a2, a3 = np.linalg.lstsq(X, Iframes, rcond=None)[0]

    # (1) the |w|^2 regressor IS SAN's pooled second difference -> kappa^2.
    np.testing.assert_allclose(a3, 0.5 * (mod_cos + mod_sin), rtol=1e-9)

    # (2) PSI with per-quadrature kappa == SAN, to machine precision.
    dw_psi_perquad = -(a1 / (2 * mod_cos) + 1j * a2 / (2 * mod_sin))
    np.testing.assert_allclose(dw_psi_perquad, dw_san, atol=1e-10)

    # (3) PSI with a single pooled kappa matches SAN to within the measured
    #     cos/sin quadrature asymmetry (the only source of disagreement).
    dw_psi_pooled = -(a1 + 1j * a2) / (2 * a3)
    asym = np.max(np.abs(mod_cos - mod_sin)) / np.max(np.abs(mod_cos))
    rel = np.max(np.abs(dw_psi_pooled - dw_san)) / np.max(np.abs(dw_san))
    assert rel <= asym + 1e-6, f"pooled-kappa mismatch {rel:.2e} exceeds asymmetry {asym:.2e}"


def test_furious_step_reproduces_san_step_on_one_burst():
    """The real ``_furious_step`` method, fed exactly one SAN burst plus the
    SAN-calibrated kappa^2, applies the same correction as ``san_step`` -- within
    the pooled-kappa quadrature asymmetry (it uses a single pooled kappa^2)."""
    model = CoronagraphModel(iwa_ld=3, owa_ld=12, seed=1)

    # Reference: an actual san_step from zero command; w ends at the correction.
    ref = SpeckleNuller(model)
    ref.san_step()
    dw_san = ref.coefficients.copy()

    # Drive _furious_step over the same burst frames + SAN-pooled kappa^2.
    fnf = SpeckleNuller(model)
    a = fnf.actuators
    cosp, sinp = model.cos_probe, model.sin_probe
    dz = fnf.dz
    I0_full = fnf.last_image
    Icp, Icm = model.image(a + cosp), model.image(a - cosp)
    Isp, Ism = model.image(a + sinp), model.image(a - sinp)

    mod_cos = (Icp[dz] + Icm[dz] - 2 * I0_full[dz]) / 2.0
    mod_sin = (Isp[dz] + Ism[dz] - 2 * I0_full[dz]) / 2.0
    fnf.kappa2 = 0.5 * (mod_cos + mod_sin)

    # Start with current actuators
    w0 = fnf.coefficients                           

    # Apply offsets for each image
    for offset, img in [(1 + 0j, Icp), (-1 + 0j, Icm), (1j, Isp), (-1j, Ism)]:
        fnf._record(w0 + offset, img)

    fnf.psi_step()                  # applies (correction - w0), w0 = 0
    dw_fnf = fnf.coefficients.copy()

    asym = np.max(np.abs(mod_cos - mod_sin)) / np.max(np.abs(mod_cos))
    rel = np.max(np.abs(dw_fnf - dw_san)) / np.max(np.abs(dw_san))
    assert rel <= asym + 1e-6, f"_furious_step vs san_step {rel:.2e} exceeds asymmetry {asym:.2e}"


if __name__ == "__main__":
    test_psi_lstsq_recovers_san_step()
    print("ok  test_psi_lstsq_recovers_san_step")
    test_furious_step_reproduces_san_step_on_one_burst()
    print("ok  test_furious_step_reproduces_san_step_on_one_burst")
    print("all passed")
