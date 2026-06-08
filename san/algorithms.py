"""Speckle-nulling algorithms as composable, single-``step()`` classes.

Every algorithm subclasses :class:`SpeckleNuller`, which holds the state common to
all of them — the current DM command, the per-pixel modal command ``w``, the
recorded phase-shifting history, the per-pixel gain ``|k'|^2`` and the exposure
tally — and drives a :class:`san.models.CoronagraphModel` forward model.  Each
algorithm only overrides :meth:`step`, which performs one iteration:

    model = CoronagraphModel()
    nuller = SpeckleAreaNulling(model)
    for _ in range(10):
        nuller.step()
        print(nuller.contrast)

Available algorithms
--------------------
- :class:`SpeckleAreaNulling`     -- classic 5-frame symmetric-difference SAN.
- :class:`SANAndFurious`          -- SAN seed + per-pixel least-squares history reuse.
- :class:`MinStepNulling`         -- non-SAN 3-frame (2 fresh exposures) downward step.
- :class:`FastAndFurious`         -- one fresh probe + forgetting-weighted history.
- :class:`FastAndFuriousNoProbe`  -- correction-history-only step (no fresh probe).
"""

import numpy as np


class SpeckleNuller:
    """Base class holding the DM state and the shared phase-shifting machinery.

    Concrete algorithms override :meth:`step`.  The reusable primitives
    :meth:`san_step`, :meth:`furious_step` and :meth:`_solve_residual_differential`
    live here so the algorithms can compose them.

    Notes on the model
    ------------------
    Per dark-zone pixel the focal field is linear in command, ``E_tot = E0 + k' w``,
    where ``w`` is the complex modal command in units of ``a_probe`` (``Re`` -> cosine
    quadrature, ``Im`` -> sine quadrature) and ``k'`` is the field-per-command.  The
    per-pixel gain ``kappa2 = |k'|^2`` is calibrated from a symmetric SAN probe burst.

    Parameters
    ----------
    model : CoronagraphModel
        The forward model to drive (provides ``image``, ``command``, ``cos_probe``,
        ``sin_probe``, ``dark_zone``, ``a_probe`` and ``num_*``).
    gain : float
        Loop gain applied to every correction.
    forget : float in (0, 1]
        Exponential forgetting factor for the least-squares history.  ``forget=1``
        pools every frame equally (most exposure-efficient, but biased toward the
        wrong null by inter-mode cross-talk); ``forget<1`` down-weights stale frames
        so the fit re-linearizes around the current state.
    """

    def __init__(self, model, gain=1.0, forget=1.0):
        self.model = model
        self.gain = gain
        self.forget = forget

        self.dz = np.asarray(model.dark_zone.shaped, dtype=bool)   # 2D boolean mask
        self.a_probe = model.a_probe
        self.eps_floor = model.eps
        self.n_modes = model.num_frequencies                        # 1:1 with dark-zone pixels

        # current DM state
        self.actuators = np.zeros(model.num_actuators)              # actuator commands
        self.w = np.zeros(self.n_modes, dtype=complex)              # modal command (a_probe units)
        self.kappa2 = None                                          # per-pixel |k'|^2

        # phase-shifting "frames" reused for the least-squares inversions
        self.prior_corrections = []   # complex modal command per frame (a_probe units)
        self.prior_images = []        # measured dark-zone intensity per frame
        self.n_exposures = 0          # total forward-model evaluations (efficiency metric)
        self._iter = 0                # step counter
        self.last_image = self._image(self.actuators)   # always the current-state image

    # -- convenience ----------------------------------------------------------
    @property
    def contrast(self):
        """Current dark-zone mean contrast."""
        return self.last_image[self.dz].mean()

    def _resolve_gain(self, gain):
        return self.gain if gain is None else gain

    # -- exposure / state helpers --------------------------------------------
    def _image(self, actuators):
        """Take one exposure (forward-model evaluation) and tally it."""
        self.n_exposures += 1
        return self.model.image(actuators)

    def _command(self, w_modal):
        """DM actuator vector for a complex modal command (a_probe units)."""
        return self.model.command(w_modal)

    def _record(self, w_modal, image_full):
        """Append one phase-shifting frame: the applied command and the image it made."""
        self.prior_corrections.append(np.asarray(w_modal, dtype=complex).copy())
        self.prior_images.append(image_full[self.dz].copy())

    def _apply(self, dw):
        """Add an incremental modal command ``dw`` (a_probe units) to the DM state, and
        refresh the current-state image so it can be reused as the next frame without
        re-exposing."""
        self.actuators = self.actuators + self._command(dw)
        self.w = self.w + dw
        self.last_image = self._image(self.actuators)

    # -- reusable algorithmic primitives -------------------------------------
    def san_step(self, gain=None):
        """Traditional SAN: 5 exposures around the current state.  Records all five
        frames, (re)calibrates the per-pixel gain ``|k'|^2``, and applies the SAN
        correction.  With ``w`` in ``a_probe`` units, the +/-cos probe is ``w0 +/- 1``
        and the +/-sin probe is ``w0 +/- 1j``."""
        gain = self._resolve_gain(gain)
        a = self.actuators
        cos_probe = self.model.cos_probe
        sin_probe = self.model.sin_probe

        I0 = self.last_image                                 # reuse current-state image (free)
        Icp = self._image(a + cos_probe); Icm = self._image(a - cos_probe)
        Isp = self._image(a + sin_probe); Ism = self._image(a - sin_probe)

        w0 = self.w
        self._record(w0,       I0)
        self._record(w0 + 1,   Icp)
        self._record(w0 - 1,   Icm)
        self._record(w0 + 1j,  Isp)
        self._record(w0 - 1j,  Ism)

        dz = self.dz
        mod_cos = (Icp[dz] + Icm[dz] - 2 * I0[dz]) / 2.0     # = |k'|^2 (cos quadrature)
        mod_sin = (Isp[dz] + Ism[dz] - 2 * I0[dz]) / 2.0
        self.kappa2 = np.maximum(0.5 * (mod_cos + mod_sin), self.eps_floor)

        san_cos = (Icp[dz] - Icm[dz]) / (2 * mod_cos)
        san_sin = (Isp[dz] - Ism[dz]) / (2 * mod_sin)
        self._apply(-0.5 * gain * (san_cos + 1j * san_sin))

    def furious_step(self, gain=None):
        """Per-pixel least-squares phase-shifting over the recorded history, with
        exponential forgetting so stale frames don't bias the static-linear fit.

        Moving the known ``|k'|^2 |w_n|^2`` to the left (``y_n = I_n - |k'|^2 |w_n|^2``)
        gives the standard phase-shifting normal equations, solved per pixel; then
        ``E0 k'^* = (a1 + i a2)/2`` and the nulling command is
        ``w_target = -(a1 + i a2) / (2 |k'|^2)``."""
        gain = self._resolve_gain(gain)
        # reuse the current-state image as a fresh frame (no new exposure), then solve.
        self._record(self.w, self.last_image)

        W = np.asarray(self.prior_corrections)               # (Nframes, Npix) complex
        I = np.asarray(self.prior_images)                    # (Nframes, Npix)
        nframes, npix = W.shape
        wt = (self.forget ** np.arange(nframes)[::-1])[:, None]   # newest frame -> weight 1

        c = W.real
        s = W.imag
        y = I - self.kappa2[None, :] * np.abs(W) ** 2

        M = np.empty((npix, 3, 3))
        M[:, 0, 0] = wt.sum()
        M[:, 0, 1] = M[:, 1, 0] = (wt * c).sum(0)
        M[:, 0, 2] = M[:, 2, 0] = (wt * s).sum(0)
        M[:, 1, 1] = (wt * c * c).sum(0)
        M[:, 1, 2] = M[:, 2, 1] = (wt * c * s).sum(0)
        M[:, 2, 2] = (wt * s * s).sum(0)
        b = np.stack([(wt * y).sum(0), (wt * y * c).sum(0), (wt * y * s).sum(0)], axis=-1)

        a = np.linalg.solve(M, b)                            # (Npix, 3): a0, a1, a2
        Z = 0.5 * (a[:, 1] + 1j * a[:, 2])                   # = E0 * conj(k')
        w_target = -Z / self.kappa2                          # nulling command (a_probe units)
        self._apply(gain * (w_target - self.w))

    def _solve_residual_differential(self):
        """Per-pixel weighted 2x2 least squares for the residual ``Z = R_now k'^*`` from
        the differential model over the forgetting-weighted history.  For each recorded
        frame k (command ``w_k``, image ``I_k``), with ``delta_k = w_now - w_k``::

            Re(Z delta_k^*) = (|k'|^2 |delta_k|^2 - (I_k - I0)) / 2 =: g_k.

        Clustered frames (``delta_k -> 0``) contribute ~0 and self-cancel.  Returns
        complex ``Z`` per pixel."""
        I0 = self.last_image[self.dz]
        W = np.asarray(self.prior_corrections)              # (Nframes, Npix) complex
        I = np.asarray(self.prior_images)                   # (Nframes, Npix)
        nframes, npix = W.shape
        wt = (self.forget ** np.arange(nframes)[::-1])[:, None]

        delta = self.w[None, :] - W                         # command change to now
        p = delta.real
        q = delta.imag
        g = 0.5 * (self.kappa2[None, :] * np.abs(delta) ** 2 - (I - I0[None, :]))

        M = np.empty((npix, 2, 2))
        M[:, 0, 0] = (wt * p * p).sum(0)
        M[:, 0, 1] = M[:, 1, 0] = (wt * p * q).sum(0)
        M[:, 1, 1] = (wt * q * q).sum(0)
        b = np.stack([(wt * p * g).sum(0), (wt * q * g).sum(0)], axis=-1)

        # tiny Tikhonov ridge so pixels that have only seen one quadrature don't blow up
        ridge = 1e-6 * (M[:, 0, 0] + M[:, 1, 1]) + 1e-30
        M[:, 0, 0] += ridge
        M[:, 1, 1] += ridge

        Z = np.linalg.solve(M, b)                           # (Npix, 2): Re Z, Im Z
        return Z[:, 0] + 1j * Z[:, 1]

    # -- subclass interface ---------------------------------------------------
    def step(self, gain=None):
        """Perform one iteration of the algorithm.  Overridden by each subclass."""
        raise NotImplementedError


class SpeckleAreaNulling(SpeckleNuller):
    """Classic Speckle Area Nulling (Nishikawa 2022 / Oya 2017).

    At each dark-zone point the unknown speckle field ``E`` is probed by a known DM
    field ``P`` at +/- amplitude.  The 3-bucket phase-shifting estimator gives
    ``san = (I+ - I-)/(I+ + I- - 2 I0) = 2 Re(E P*)/|P|^2``; probing both a cosine and
    a quadrature sine reconstructs the full complex field, and the DM command that
    nulls it is ``-(san/2) * a_probe`` per quadrature.

    One :meth:`step` is a single symmetric 5-frame SAN burst (4 fresh exposures).
    """

    def step(self, gain=None):
        self.san_step(gain)
        self._iter += 1


class SANAndFurious(SpeckleNuller):
    """SAN and Furious (SAF): SAN seeding + per-pixel least-squares history reuse.

    Classical phase-shifting interferometry needs >=3 temporal frames of the *whole*
    interferogram because it can only piston the reference phase.  A coronagraph DM
    imposes a phase shift that varies across the focal plane, so every dark-zone pixel
    runs its own 3-bucket interferometer.  No probe data is thrown away: each applied
    command is recorded with the image it produced, and once a pixel has seen >=3
    phase-diverse commands its static speckle field is estimated by least squares
    (:meth:`furious_step`).

    Parameters
    ----------
    probe_every_n : int
        How often :meth:`step` spends a 5-frame SAN burst to inject fresh near-state
        phase diversity and recalibrate ``|k'|^2``.  ``1`` probes every iteration
        (deepest, most exposures); ``n>1`` reuses the command history (furious-only) in
        between, trading depth-per-iteration for far fewer exposures.
    forget : float in (0, 1]
        Forgetting factor (see :class:`SpeckleNuller`).  Defaults to ``0.5``.
    """

    def __init__(self, model, gain=1.0, forget=0.5, probe_every_n=1):
        super().__init__(model, gain=gain, forget=forget)
        self.probe_every_n = probe_every_n

    def step(self, gain=None):
        gain = self._resolve_gain(gain)
        if self.kappa2 is None or self._iter % self.probe_every_n == 0:
            self.san_step(gain)
        self.furious_step(gain)
        self._iter += 1


class MinStepNulling(SpeckleNuller):
    """Non-SAN 3-frame downward iteration (2 fresh exposures per step).

    Solving the per-pixel speckle field needs only 3 frames; SAN spends 5 because its
    *symmetric* probes cancel the probe self-term ``|k'|^2`` without knowing it.  Here
    the current corrected image is reused as the reference (free), and two small
    *one-sided* probes ``+eps`` and ``+i*eps`` are applied.  With residual field
    ``R = E0 + k' w`` (``|R|^2 = I0``)::

        I0   = |R|^2                                  (reference, reused)
        I_re = |R|^2 + |k'|^2 eps^2 + 2 eps Re(R k'^*)
        I_im = |R|^2 + |k'|^2 eps^2 + 2 eps Im(R k'^*)

    so ``Re/Im(R k'^*) = (I_re/I_im - I0 - |k'|^2 eps^2)/(2 eps)`` and the null step is
    ``dw = -(R k'^*)/|k'|^2``.  Three one-sided frames can't also solve for ``|k'|^2``,
    so the gain is calibrated by auto-seeding one :meth:`san_step` on the first call.

    Parameters
    ----------
    eps : float
        One-sided probe amplitude (in ``a_probe`` units).
    """

    def __init__(self, model, gain=1.0, forget=1.0, eps=0.5):
        super().__init__(model, gain=gain, forget=forget)
        self.eps = eps

    def step(self, gain=None):
        gain = self._resolve_gain(gain)
        if self.kappa2 is None:
            self.san_step(gain)             # one symmetric burst to calibrate |k'|^2
            self._iter += 1
            return

        a = self.actuators
        dz = self.dz
        eps = self.eps
        cos_probe = self.model.cos_probe
        sin_probe = self.model.sin_probe

        I0 = self.last_image[dz]                             # reference (free, reused)
        Ire = self._image(a + eps * cos_probe)[dz]
        Iim = self._image(a + eps * sin_probe)[dz]

        self_term = self.kappa2 * eps ** 2                   # one-sided probe self-intensity
        re = (Ire - I0 - self_term) / (2 * eps)              # Re(R k'^*)
        im = (Iim - I0 - self_term) / (2 * eps)              # Im(R k'^*)
        self._apply(-gain * (re + 1j * im) / self.kappa2)    # dw = -(R k'^*)/|k'|^2
        self._iter += 1


class FastAndFurious(SpeckleNuller):
    """Fast & Furious-style step: ONE fresh probe per iteration.

    The missing quadrature is supplied by the forgetting-weighted history of prior
    command changes (the "Furious" temporal diversity, via
    :meth:`_solve_residual_differential`).  The fresh probe alternates real/imag each
    iteration so both quadratures stay sampled even as the corrections (hence their
    diversity) shrink.  Cost is 1 fresh exposure/iteration; the first call auto-seeds
    one :meth:`san_step` to calibrate ``|k'|^2``.

    Parameters
    ----------
    eps : float
        Fresh-probe amplitude (in ``a_probe`` units).
    """

    def __init__(self, model, gain=1.0, forget=0.5, eps=0.5):
        super().__init__(model, gain=gain, forget=forget)
        self.eps = eps

    def step(self, gain=None):
        gain = self._resolve_gain(gain)
        if self.kappa2 is None:
            self.san_step(gain)             # one symmetric burst to calibrate |k'|^2
            self._iter += 1
            return

        a = self.actuators
        eps = self.eps
        if self._iter % 2 == 0:
            probe, dw_fresh = self.model.cos_probe, eps + 0j      # +eps   (real quadrature)
        else:
            probe, dw_fresh = self.model.sin_probe, 1j * eps      # +i*eps (imag quadrature)
        I_fresh_full = self._image(a + eps * probe)
        self._record(self.w, self.last_image)               # reference frame (a prior correction)
        self._record(self.w + dw_fresh, I_fresh_full)       # fresh-probe frame

        Zc = self._solve_residual_differential()
        self._apply(-gain * Zc / self.kappa2)
        self._iter += 1


class FastAndFuriousNoProbe(SpeckleNuller):
    """Correction-history-only step ("Fast aNd Furious"): NO fresh probe.

    After a single 5-frame SAN seed (which calibrates ``|k'|^2`` and injects both
    quadratures), every update reuses ONLY the history of correction images: the prior
    corrections themselves play the role of :class:`FastAndFurious`'s fresh quadrature
    probe, since each contributes ``delta_k = w_now - w_k`` to the same differential
    least squares.  Cost: 1 exposure/iteration (just the post-correction image).

    Because the seed's +/- cos/sin frames sit at fixed, well-spread commands, their
    ``delta_k`` to the drifting current state stay large and span both quadratures,
    keeping the per-pixel 2x2 solve conditioned.  With ``forget=1`` they are never
    discarded; a strong forgetting factor would throw away the only diversity, so the
    default here is ``forget=1.0``.
    """

    def __init__(self, model, gain=1.0, forget=1.0):
        super().__init__(model, gain=gain, forget=forget)

    def step(self, gain=None):
        gain = self._resolve_gain(gain)
        if self.kappa2 is None:
            self.san_step(gain)             # 5-frame seed: calibrate |k'|^2 + both quadratures
            self._iter += 1
            return

        self._record(self.w, self.last_image)               # current corrected state (no probe)
        Zc = self._solve_residual_differential()
        self._apply(-gain * Zc / self.kappa2)
        self._iter += 1
