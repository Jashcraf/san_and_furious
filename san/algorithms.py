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
- :class:`MinStepNulling`         -- non-SAN 3-frame (2 fresh exposures) downward step.
- :class:`LagAndFurious`          -- one fresh probe + forgetting-weighted history.
- :class:`FastAndFuriousNoProbe`  -- correction-history-only step (no fresh probe).
"""

import numpy as np
import ipdb
from .san_math import mat_inv_2x2, mat_inv_3x3

class SpeckleNuller:
    """Base class holding the DM state and the shared phase-shifting machinery.

    Concrete algorithms override :meth:`step`.  The reusable primitives
    :meth:`san_step`, :meth:`psi_step` and :meth:`_solve_residual_differential`
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
        self.actuators = np.zeros(model.num_actuators)              # current actuator commands

        # self.w is complex, where the real part is the cosine coefficient, and the imag is the sine
        self.coefficients = np.zeros(self.n_modes, dtype=complex)   # current complex modal command (a_probe units)
        self.kappa2 = None                                          # per-pixel |k'|^2

        # phase-shifting "frames" reused for the least-squares inversions
        self.prior_corrections = []   # complex modal command per frame (a_probe units)
        self.prior_images = []        # measured dark-zone intensity per frame
        self.prior_kappa2 = []
        self.prior_field_est = []
        self.n_exposures = 0          # total forward-model evaluations (efficiency metric)
        self._iter = 0                # step counter
        self.last_image = self._image(self.actuators)   # always the current-state image

    # convenience methods
    @property
    def contrast(self):
        """Current dark-zone mean contrast."""
        return self.last_image[self.dz].mean()

    def _resolve_gain(self, gain):
        return self.gain if gain is None else gain

    # exposure / state helpers 
    def _image(self, actuators):
        """Take one exposure (forward-model evaluation) and tally it."""
        self.n_exposures += 1
        return self.model.image(actuators)

    # was _command
    def _coefficients_to_command(self, coefficients):
        """DM actuator vector for a complex modal command (a_probe units)."""
        return self.model.command(coefficients)

    def _record(self, coefficients, image_full):
        """Append one phase-shifting frame: the applied command and the image it made."""
        self.prior_corrections.append(np.asarray(coefficients, dtype=complex).copy())
        self.prior_images.append(image_full[self.dz].copy())

    def _add_command(self, dw):
        """Add an incremental modal command ``dw`` (a_probe units) to the DM state, and
        refresh the current-state image so it can be reused as the next frame without
        re-exposing."""
        # Updates actuator state
        self.actuators = self.actuators + self._coefficients_to_command(dw)
        
        # Updates coefficient state
        self.coefficients += dw
        self.last_image = self._image(self.actuators)

    # -- reusable algorithmic primitives -------------------------------------
    def san_step(self, gain=None):
        """Traditional SAN: 5 exposures around the current state.  Records all five
        frames, (re)calibrates the per-pixel gain ``|k'|^2``, and applies the SAN
        correction.  With ``w`` in ``a_probe`` units, the +/-cos probe is ``w0 +/- 1``
        and the +/-sin probe is ``w0 +/- 1j``."""

        # Store current instrument state (after all prior corrections) 
        a = self.actuators
        w0 = self.coefficients

        # Get correction complex coefficients
        correction = self._solve_san_coefficients(gain)
        
        # Convert to DM commands
        command = self._coefficients_to_command(correction)
        
        # Get an updated image
        I_cor = self._image(a + command)

        # Append correction to history with "probed" intensity
        # Because we add to w0, this stores the total phase shift
        # applied to the N-1 correction
        self._record(w0 + correction, I_cor)

        # Update N-1 state to N - updates last image
        self._add_command(correction)

    def psi_step(self, gain=None):
        
        # Store reference values
        a = self.actuators
        w0 = self.coefficients
        
        # Get correction complex coefficients
        # correction = self._solve_psi_coefficients(gain)
        correction = self._solve_residual_differential(gain)
        
        # Convert to DM commands
        command = self._coefficients_to_command(correction)
        
        # Get an updated image
        I_cor = self._image(a + command)

        # Append correction to history with "probed" intensity
        # Because we add to w0, this stores the total phase shift
        # applied to the original correction
        self._record(w0 + correction, I_cor)

        # Actually apply the correction - updates last image
        self._add_command(correction)

    def furious_step(self, gain=None):
        # Store reference values
        a = self.actuators
        w0 = self.coefficients
        
        # Get correction complex coefficients
        # correction = self._solve_psi_coefficients(gain)
        correction = self._solve_furious_coefficients(gain)
        
        # Convert to DM commands
        command = self._coefficients_to_command(correction)
        
        # Get an updated image
        I_cor = self._image(a + command)

        # Append correction to history with "probed" intensity
        # Because we add to w0, this stores the total phase shift
        # applied to the original correction
        self._record(w0 + correction, I_cor)

        # Actually apply the correction - updates last image
        self._add_command(correction)


    def _solve_san_coefficients(self, gain):
        """Solve for the complex coefficients that correct a DH
        using the Speckle Area Nulling method by Oya et al. 2017

        Parameters
        ----------
        gain : float or NoneType
            gain to apply to correction estimation, if None,
            defaults to 1 via self._resolve_gain(gain)

        Returns
        -------
        complex ndarray
           array of coefficients corresponding to each fourier mode, 
           where the real part is the cosine and the imag part is the sine 
        """
        
        gain = self._resolve_gain(gain)
        cos_probe = self.model.cos_probe
        sin_probe = self.model.sin_probe
        
        # Grabs current command state
        w0 = self.coefficients
        a = self.actuators

        # Returns phase-shifted coefficients with images 
        I0 = self.last_image                                 # reuse current-state image (free)
        Icp = self._image(a + cos_probe)
        Icm = self._image(a - cos_probe)
        Isp = self._image(a + sin_probe)
        Ism = self._image(a - sin_probe)

        # Records applied phase shifts
        # self._record(w0, I0) 
        # self._record(w0 + 1,   Icp)
        # self._record(w0 - 1,   Icm)
        # self._record(w0 + 1j,  Isp)
        # self._record(w0 - 1j,  Ism)

        # Gets dark zone boolean
        dz = self.dz
        mod_cos = (Icp[dz] + Icm[dz] - 2 * I0[dz]) / 2.0     # = |k'|^2 (cos quadrature)
        mod_sin = (Isp[dz] + Ism[dz] - 2 * I0[dz]) / 2.0     # = |k'|^2 (sin quadrature)
        
        # store quadrature-specific kappa for logging
        self.kappa_cos = mod_cos
        self.kappa_sin = mod_sin
        
        # Doesn't let response go below epsilon, otherwise averages
        self.kappa2 = np.maximum(0.5 * (mod_cos + mod_sin), self.eps_floor)
        self.prior_kappa2.append(self.kappa2)

        # Actual SAN coefficients
        san_cos = (Icp[dz] - Icm[dz]) / (2 * mod_cos)
        san_sin = (Isp[dz] - Ism[dz]) / (2 * mod_sin)
        field_est = san_cos + 1j * san_sin

        # Record the field estimation
        self.prior_field_est.append(field_est)

        # -0.5 for converting OPD to mirror surface
        return -0.5 * gain * field_est 

    def _solve_psi_coefficients(self, gain):
        """Solves Grievenkamp 1984 least-squares PSI matrix
        to get phase of speckle correction. Requires history
        of phase shifted speckles.

        Parameters
        ----------
        gain : float or NoneType
            gain to apply to correction estimation, if None,
            defaults to 1 via self._resolve_gain(gain)

        Returns
        -------
        complex ndarray
           array of coefficients corresponding to each fourier mode, 
           where the real part is the cosine and the imag part is the sine 
        """
        
        gain = self._resolve_gain(gain)
        cos_probe = self.model.cos_probe
        sin_probe = self.model.sin_probe
        
        # Grabs current command state
        w0 = self.coefficients
        a = self.actuators

        # reuse the current-state image as a fresh frame (no new exposure), then solve.
        W = np.asarray(self.prior_corrections)               # (Nframes, Npix) complex
        I = np.asarray(self.prior_images)                    # (Nframes, Npix) real
        nframes, npix = W.shape

        # Unpack cosine (c) and sine (s) coefficients
        c = W.real
        s = W.imag
        
        # Subtract probe energy from prior images - requires kappa2, which is computed by SAN
        y = I - self.kappa2[None, :] * np.abs(W) ** 2

        # pre-define matrix
        M = np.empty((npix, 3, 3))
        M[:, 0, 0] = nframes 
        M[:, 0, 1] = M[:, 1, 0] = c.sum(axis=0)
        M[:, 0, 2] = M[:, 2, 0] = s.sum(axis=0)
        M[:, 1, 1] = (c * c).sum(axis=0)
        M[:, 1, 2] = M[:, 2, 1] = (c * s).sum(axis=0)
        M[:, 2, 2] = (s * s).sum(axis=0)
        b = np.stack([y.sum(axis=0),
                      (y * c).sum(axis=0),
                      (y * s).sum(axis=0)], axis=-1)

        # tiny Tikhonov ridge so pixels that have only seen one quadrature don't blow up
        ridge = 1e-6 * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]) + 1e-30
        M[:, 0, 0] += ridge
        M[:, 1, 1] += ridge
        M[:, 2, 2] += ridge

        # Solve the matrix inversion with analytic inverse
        Minv = mat_inv_3x3(M)
        a = Minv @ b[..., None] # append axis for matrix mult
        a = a[..., 0]           # take it away

        # get the correction
        a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]

        # a1 = 2 Re(E0* k'), a2 = -2 Im(E0* k'); the ABSOLUTE nulling command is
        # w_target = -(a1 + i a2) / (2 |k'|^2).  Unlike _solve_san_coefficients (which
        # probes around the current state and so returns a step), this static fit yields
        # the absolute target, so return the INCREMENT from w0 -- otherwise psi_step
        # lands at w0 + w_target and the command explodes once w0 != 0.
        w_target = -0.5 * (a1 + 1j * a2) / self.kappa2
        return gain * (w_target - w0)

    def _solve_furious_coefficients(self, gain):
        """Estimate the current residual field from the two most recent corrections
        and return the modal command that nulls it.

        Direct port of ``solve_prev_field`` + propagation in ``iterative_san.py``.
        Work in modal-residual units ``G = R / k'`` (so ``I = |k'|^2 |G|^2`` and the
        command that nulls a residual ``G`` is ``dw = -gain * G``).  The SAN-convention
        field estimate stored in ``prior_field_est`` is ``F = 2 G`` (see
        :meth:`_solve_san_coefficients`), so ``Ghat = F / 2``.

        With ``g`` the loop gain and the three most recent *distinct* frames
        ``i-2, i-1, i`` -- the current state is the newest recorded frame
        ``prior_images[-1]`` (``last_image`` is the SAME state, so it is NOT a separate
        frame), and ``F[-1]``/``F[-2]`` are the estimates whose ``-g Ghat`` corrections
        produced frames ``i`` and ``i-1`` --::

            P_newest = (I_{i-1} - I_i   + g^2 |k'|^2 |Ghat_{i-1}|^2) / (2 g)   [= |k'|^2 Re(Ghat_{i-1}^* G_{i-1})]
            P_slid   = (I_{i-2} - I_{i-1} - g^2 |k'|^2 |Ghat_{i-2}|^2) / (2 g) [= |k'|^2 Re(Ghat_{i-2}^* G_{i-1})]

        A 2x2 solve over the two prior estimate directions gives the residual
        ``G_{i-1}``; propagating the one applied correction forward yields the current
        residual estimate ``Ghat_i = G_{i-1} - g Ghat_{i-1}``, nulled by ``dw = -g Ghat_i``.

        Parameters
        ----------
        gain : float or NoneType
            loop gain; defaults to 1 via :meth:`_resolve_gain`.

        Returns
        -------
        complex ndarray
            modal command increment (a_probe units) per Fourier mode.
        """
        gain = self._resolve_gain(gain)

        I = np.asarray(self.prior_images)                    # (Nframes, Npix) real
        nframes, npix = I.shape
        kappa2 = self.kappa2                                  # (Npix,) = |k'|^2

        # Before three distinct frames exist, the 2x2 solve has too little temporal
        # diversity; fall back to the single-correction bootstrap (iterative_san.bootstrap)
        # to seed it.  The next furious step has 3 frames and takes the 2x2 path.
        if nframes < 2:
            raise ValueError(
                "furious_step needs >= 2 recorded frames to bootstrap; "
                "seed with >= 2 san_step calls")
        if nframes < 3:
            return self._bootstrap_furious(I, kappa2, gain)

        # Actual applied modal strokes, recovered by differencing the recorded total
        # commands.  This is the real DM move between frames and is gain-agnostic, so
        # the solve does NOT assume the seeding gain equals the current furious gain --
        # the (previously implicit) assumption that broke convergence when the SAN seed
        # ran at a different gain than the furious loop.
        W = np.asarray(self.prior_corrections)              # (Nframes, Npix) complex
        s_slid = W[-2] - W[-3]                               # stroke i-2 -> i-1 (made I[-2])
        s_new  = W[-1] - W[-2]                               # stroke i-1 -> i   (made I[-1])

        # three consecutive, distinct intensities in modal units |G|^2 = I / |k'|^2
        i_im2, i_im1, i_i = I[-3] / kappa2, I[-2] / kappa2, I[-1] / kappa2

        # projections of the unknown residual G_{i-1} (at frame i-1) onto each stroke:
        #   |G_{i-1} - s_slid|^2 = i_im2  ->  Re(s_slid^* G) = (i_im1 - i_im2 + |s_slid|^2)/2
        #   |G_{i-1} + s_new |^2 = i_i    ->  Re(s_new^*  G) = (i_i  - i_im1 - |s_new|^2)/2
        P_slid = (i_im1 - i_im2 + np.abs(s_slid) ** 2) / 2.0
        P_new  = (i_i   - i_im1 - np.abs(s_new) ** 2) / 2.0
        b0 = np.stack([P_slid, P_new], axis=-1)             # (Npix, 2)

        # M rows are the two applied stroke directions (modal units)
        M = np.empty((npix, 2, 2))
        M[:, 0, 0] = s_slid.real
        M[:, 0, 1] = s_slid.imag
        M[:, 1, 0] = s_new.real
        M[:, 1, 1] = s_new.imag

        # normal equations with a collinearity ridge: when the two strokes align
        # (residual collapsing along its own line) the ill-determined axis is pulled
        # toward zero -- the correct limit -- while the parallel axis solves cleanly.
        Mt = np.transpose(M, (0, 2, 1))
        N = Mt @ M                                           # (Npix, 2, 2)
        rhs = (Mt @ b0[..., None])[..., 0]                   # (Npix, 2)

        lam = np.std(N[:, 0, 0])
        ridge = 1e-2 * lam + 1e-10
        N[:, 0, 0] += ridge
        N[:, 1, 1] += ridge
        G_prev = np.linalg.solve(N, rhs[..., None])[..., 0]  # (Npix, 2) = residual G_{i-1}
        G_prev = G_prev[:, 0] + 1j * G_prev[:, 1]

        # propagate the most recent applied stroke to the current frame i
        Ghat_i = G_prev + s_new

        # store in SAN convention (2 * Ghat) for the next furious step; null the current
        # residual with dw = -gain * Ghat_i (matches SAN's -0.5 * gain * field_est).
        self.prior_field_est.append(2.0 * Ghat_i)
        return -gain * Ghat_i

    def _bootstrap_furious(self, I, kappa2, gain, s0=+1):
        """Single-correction bootstrap that seeds the furious 2x2 solve.

        Port of ``bootstrap`` in ``iterative_san.py``, vectorized over pixels and
        worked in modal-residual units ``G = R / k'`` (so ``|G|^2 = I / |k'|^2``).
        With only one applied correction recorded there is no second stroke direction
        for the 2x2 solve, so the residual is reconstructed from the two intensity
        circles plus a *guessed* perpendicular sign ``s0``.  That guess is harmless:
        the next step has three frames and the unambiguous 2x2 solve overwrites it.

        The single applied stroke ``s`` (recovered from the recorded commands, hence
        gain-agnostic) took the field from frame ``I[-2]`` (before) to ``I[-1]``
        (after), with ``G_after = G_before + s``::

            |G_before + s|^2 = i_after  ->  Re(s^* G_before) = (i_after - i_before - |s|^2)/2
            |G_before|^2     = i_before                       (the pre-correction circle)

        decomposing ``G_before`` along ``s`` (well-determined) and its perpendicular
        (sign-guessed), then propagating the stroke to the current frame::

            Ghat_i = G_before + s,   nulled by  dw = -gain * Ghat_i.

        Parameters
        ----------
        I : ndarray, shape (Nframes, Npix)
            Recorded dark-zone intensities (newest last); only the last two are used.
        kappa2 : ndarray, shape (Npix,)
            Per-pixel gain ``|k'|^2``.
        gain : float
            Loop gain applied when nulling the reconstructed current residual.
        s0 : int
            Guessed sign of the perpendicular component (overwritten next step).

        Returns
        -------
        complex ndarray
            modal command increment (a_probe units) per Fourier mode.
        """
        W = np.asarray(self.prior_corrections)               # (Nframes, Npix) complex
        s = W[-1] - W[-2]                                     # the single applied stroke (made I[-1])
        i_before = I[-2] / kappa2                             # modal |G_before|^2
        i_after  = I[-1] / kappa2

        proj = (i_after - i_before - np.abs(s) ** 2) / 2.0   # = Re(s^* G_before)
        mag = np.abs(s)
        safe = np.where(mag > 0, mag, 1.0)                   # avoid 0/0 at un-probed pixels
        unit = s / safe
        perp = 1j * unit                                     # 90-deg rotation
        G_par = proj / safe                                  # component of G_before along s
        G_perp = np.sqrt(np.maximum(0.0, i_before - G_par ** 2))
        G_before = G_par * unit + s0 * G_perp * perp

        # propagate the applied stroke to the current frame
        Ghat_i = G_before + s

        # store in SAN convention (2 * Ghat) so the next (2x2) step can read it back
        self.prior_field_est.append(2.0 * Ghat_i)
        return -gain * Ghat_i

    def _solve_residual_differential(self, gain=None):
        """Per-pixel weighted 2x2 least squares for the residual speckle field over the
        forgetting-weighted history, returned as a ready-to-apply command increment
        (consistent with :meth:`_solve_san_coefficients` and
        :meth:`_solve_psi_coefficients`).

        For each recorded frame k (command ``w_k``, image ``I_k``), with
        ``delta_k = w_now - w_k`` and residual ``Z = R_now k'^*``::

            Re(Z delta_k^*) = (|k'|^2 |delta_k|^2 - (I_k - I0)) / 2 =: g_k.

        Clustered frames (``delta_k -> 0``) contribute ~0 and self-cancel.  The 2x2 solve
        gives ``Z = R_now k'^*``; the nulling command is ``dw = -gain * Z / |k'|^2`` (no
        spurious 0.5 -- the solve returns ``Z`` directly, so damping belongs in ``gain``).

        Parameters
        ----------
        gain : float or NoneType
            gain to apply to the correction, defaults to 1 via self._resolve_gain(gain)

        Returns
        -------
        complex ndarray
           array of coefficients corresponding to each fourier mode,
           where the real part is the cosine and the imag part is the sine
        """
        gain = self._resolve_gain(gain)
        I0 = self.last_image[self.dz]
        W = np.asarray(self.prior_corrections)              # (Nframes, Npix) complex
        I = np.asarray(self.prior_images)                   # (Nframes, Npix)
        nframes, npix = W.shape
        wt = (self.forget ** np.arange(nframes)[::-1])[:, None]

        delta = self.coefficients[None, :] - W                         # command change to now
        c = delta.real
        s = delta.imag
        g = 0.5 * (self.kappa2[None, :] * np.abs(delta) ** 2 - (I - I0[None, :]))

        M = np.empty((npix, 2, 2))
        M[:, 0, 0] = (wt * c ** 2).sum(0)
        M[:, 0, 1] = M[:, 1, 0] = (wt * c * s).sum(0)
        M[:, 1, 1] = (wt * s ** 2).sum(0)
        b = np.stack([(wt * c * g).sum(0), (wt * s * g).sum(0)], axis=-1)
        b = b[..., None]

        # tiny Tikhonov ridge so pixels that have only seen one quadrature don't blow up
        ridge = 1e-6 * (M[:, 0, 0] + M[:, 1, 1]) + 1e-30
        M[:, 0, 0] += ridge
        M[:, 1, 1] += ridge
        Z = np.linalg.solve(M, b)
        Z = Z[..., 0]
        return -gain * (Z[:, 0] + 1j * Z[:, 1]) / self.kappa2

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

    def _solve_minstep_coefficients(self, gain=None):
        
        gain = self._resolve_gain(gain)
        cos_probe = self.model.cos_probe
        sin_probe = self.model.sin_probe
        
        # Grabs current command state
        w0 = self.coefficients
        a = self.actuators

        # Returns phase-shifted coefficients with images 
        I0 = self.last_image[self.dz]                                 # reuse current-state image (free)
        Ire = self._image(a + self.eps * cos_probe)[self.dz]
        Iim = self._image(a + self.eps * sin_probe)[self.dz]
        
        self_term = self.kappa2 * self.eps ** 2                   # one-sided probe self-intensity
        re = (Ire - I0 - self_term) / (2 * self.eps)              # Re(R k'^*)
        im = (Iim - I0 - self_term) / (2 * self.eps)              # Im(R k'^*)
        self._add_command(-gain * (re + 1j * im) / self.kappa2)    # dw = -(R k'^*)/|k'|^2
        return -gain * (re + 1j * im) / self.kappa2

    def recalibrate(self, gain=None):
        self.san_step(gain)
        self._iter += 1

    def min_step(self, gain=None):

        # Requires SAN step to calibrate |k'|^2        
        if self.kappa2 is None:
            self.recalibrate(gain=gain)
            return
        
        # Store reference values
        a = self.actuators
        w0 = self.coefficients
        
        # Get correction complex coefficients
        # correction = self._solve_psi_coefficients(gain)
        correction = self._solve_minstep_coefficients(gain)
        
        # Convert to DM commands
        command = self._coefficients_to_command(correction)
        
        # Get an updated image
        I_cor = self._image(a + command)

        # Append correction to history with "probed" intensity
        # Because we add to w0, this stores the total phase shift
        # applied to the original correction
        self._record(w0 + correction, I_cor)

        # Actually apply the correction
        self._add_command(correction)


class LagStepNulling(SpeckleNuller):
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
    
    def _solve_lagstep_coefficients(self, gain=None):
        
        gain = self._resolve_gain(gain)
        cos_probe = self.model.cos_probe
        sin_probe = self.model.sin_probe
        
        # Grabs current command state
        w0 = self.coefficients
        a = self.actuators

        # Returns phase-shifted coefficients with images 
        I0 = self.last_image[self.dz]                                 # reuse current-state image (free)
        Ire = self._image(a + self.eps * cos_probe)[self.dz]
        Iim = self._image(a + self.eps * sin_probe)[self.dz]
        
        self_term = self.kappa2 * self.eps ** 2                   # one-sided probe self-intensity
        re = (Ire - I0 - self_term) / (2 * self.eps)              # Re(R k'^*)
        im = (Iim - I0 - self_term) / (2 * self.eps)              # Im(R k'^*)
        self._add_command(-gain * (re + 1j * im) / self.kappa2)    # dw = -(R k'^*)/|k'|^2
        return -gain * (re + 1j * im) / self.kappa2

    def recalibrate(self, gain=None):
        self.san_step(gain)
        self._iter += 1

    def lag_step(self, gain=None):

        # Requires SAN step to calibrate |k'|^2        
        if self.kappa2 is None:
            self.recalibrate(gain=gain)
            return
        
        # Store reference values
        a = self.actuators
        w0 = self.coefficients
        
        # Alternate cosine / sine probes
        if self._iter % 2 == 0:
            probe, dw_fresh = self.model.cos_probe, self.eps + 0j      # +eps   (real quadrature)
        else:
            probe, dw_fresh = self.model.sin_probe, 1j * self.eps      # +i*eps (imag quadrature)
        
        # Take and record new fresh image
        I_fresh_full = self._image(a + self.eps * probe)
        self._record(self.coefficients, self.last_image)               # reference frame (a prior correction)
        self._record(self.coefficients + dw_fresh, I_fresh_full)       # fresh-probe frame
        
        # Get correction complex coefficients
        correction = self._solve_residual_differential(gain)
        
        # Convert to DM commands
        command = self._coefficients_to_command(correction)
        
        # Get an updated image
        I_cor = self._image(a + command)

        # Append correction to history with "probed" intensity
        # Because we add to w0, this stores the total phase shift
        # applied to the original correction
        self._record(w0 + correction, I_cor)

        # Actually apply the correction
        self._add_command(correction)

        # Advance the step counter so the fresh probe alternates cos/sin quadratures
        # next iteration; without this only one quadrature is ever freshly probed and
        # the other axis goes singular as the corrections shrink (the loop plateaus).
        self._iter += 1

    # def step(self, gain=None):
    #     gain = self._resolve_gain(gain)
    #     if self.kappa2 is None:
    #         self.san_step(gain)             # one symmetric burst to calibrate |k'|^2
    #         self._iter += 1
    #         return

    #     a = self.actuators
    #     eps = self.eps
        
    #     if self._iter % 2 == 0:
    #         probe, dw_fresh = self.model.cos_probe, eps + 0j      # +eps   (real quadrature)
    #     else:
    #         probe, dw_fresh = self.model.sin_probe, 1j * eps      # +i*eps (imag quadrature)
    #     I_fresh_full = self._image(a + eps * probe)
    #     self._record(self.coefficients, self.last_image)               # reference frame (a prior correction)
    #     self._record(self.coefficients + dw_fresh, I_fresh_full)       # fresh-probe frame

    #     Zc = self._solve_residual_differential()
    #     self._add_command(Zc)
    #     self._iter += 1
