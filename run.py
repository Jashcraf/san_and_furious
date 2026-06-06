from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import ipdb

# Input parameters from hcipy EFC demo
pupil_diameter = 7e-3 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m
npix = 256
iwa_ld = 6
owa_ld = 12
offset_ld = 6
a_probe = wavelength / 50          # probe amplitude (m of DM surface)

# Some HAKA-like deformable mirror settings, and aberration
num_actuators_across = 32
actuator_spacing = 1.05 / 32 * pupil_diameter # m
aberration_ptv = 0.02 * wavelength # m
eps = 1e-9

# Save the SAN convergence as an animated .gif (one frame per iteration).
SAVE_MOVIE = True
MOVIE_FILENAME = 'san_convergence.gif'
MOVIE_FPS = 2

# Digest inputs into hcipy simulation
spatial_resolution = focal_length * wavelength / pupil_diameter # microns
iwa = iwa_ld * spatial_resolution # m
owa = owa_ld * spatial_resolution # m
offset = offset_ld * spatial_resolution # m

# Create grids
pupil_grid = make_pupil_grid(npix, pupil_diameter * 1.2)
focal_grid = make_focal_grid(4, 16, spatial_resolution=spatial_resolution)
prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# Create aperture and dark zone - aperture is super-gaussian so we can run at low-res
aperture = Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

dark_zone = circular_aperture(2 * owa)(focal_grid)
dark_zone -= circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
coronagraph = PerfectCoronagraph(aperture, order=6)

# Assume tip-tilt correction is happening with QACITS
tip_tilt = make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
aberration = SurfaceAberration(pupil_grid,
                               aberration_ptv,
                               pupil_diameter,
                               remove_modes=tip_tilt,
                               exponent=-3)

influence_functions = make_gaussian_influence_functions(pupil_grid,
                                                        num_actuators_across,
                                                        actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)

# The Fourier modes must be evaluated on the DM's *actual* actuator positions,
# which span num_actuators_across * actuator_spacing (= 1.05 * pupil_diameter),
# NOT the pupil grid extent (1.2 * pupil_diameter).
actuator_extent = num_actuators_across * actuator_spacing
actuator_grid = make_uniform_grid(
    dims=[num_actuators_across, num_actuators_across],
    extent=[actuator_extent, actuator_extent]
)

# Build up Fourier grid.
# hcipy's make_fourier_basis builds modes as cos(p . x) / sin(p . x) with NO 2*pi,
# so the fourier_grid must hold ANGULAR frequencies (rad/m), i.e. 2*pi * (x / (lambda*F)).
fx = 2 * np.pi * focal_grid.x[dark_zone] / (wavelength * focal_length)
fy = 2 * np.pi * focal_grid.y[dark_zone] / (wavelength * focal_length)
fourier_grid = CartesianGrid(UnstructuredCoords([fx, fy]))

# Convert fourier_grid (angular) frequencies back to focal plane positions.
# Inverse of the construction above: x = p / (2*pi) * lambda * F.
fx = fourier_grid.x
fy = fourier_grid.y

focal_x_expected = fx / (2 * np.pi) * wavelength * focal_length
focal_y_expected = fy / (2 * np.pi) * wavelength * focal_length

# Plot to compare with dark zone. imshow uses pixel coordinates, so convert the
# expected focal positions (meters) to pixel indices on BOTH axes consistently.
dx = focal_grid.delta[0]
dy = focal_grid.delta[1]
px = (focal_x_expected - focal_grid.x.min()) / dx
py = (focal_y_expected - focal_grid.y.min()) / dy


# sort_by_energy MUST be False: the SAN correction indexes san_correction in
# dark-zone-pixel order (= fourier_grid point order) and feeds it back as mode
# coefficients, so mode i must keep driving dark-zone pixel i. Energy-sorting
# permutes the modes and applies each correction to the wrong spatial frequency.
fourier_modes = make_fourier_basis(
    grid=actuator_grid,
    fourier_grid=fourier_grid,
    sort_by_energy=False
)

num_frequencies = len(fourier_modes) // 2

def make_phased_probe(phase_shifts_cos, amplitudes_cos, phase_shifts_sin, amplitudes_sin):
    """Make a sinusoidal deformable mirror probe composed of both sines and cosines

    Parameters
    ----------
    phase_shifts_cos : ndarray
        phase shifts `delta` to apply to A * cos(x + delta)
    amplitudes_cos : ndarray
        Amplitudes `A` to apply to A * cos(x + delta)
    phase_shifts_sin : ndarray
        phase shifts `delta` to apply to sin(x + delta)
    amplitudes_sin : ndarray
        Amplitudes `A` to apply to A * sin(x + delta)

    Returns
    -------
    ndarray
        array of actuator commands for deformable mirror
    """

    coefficients = np.zeros(len(fourier_modes))
    for i, (phic, ampc, phis, amps) in enumerate(zip(phase_shifts_cos,
                                                     amplitudes_cos,
                                                     phase_shifts_sin,
                                                     amplitudes_sin)):
        coefficients[2*i] = ampc * np.cos(phic)
        coefficients[2*i + 1] = amps * np.sin(phis)

    # normalize by the number of modes to maintain surface RMS
    coefficients /= len(fourier_modes)

    return fourier_modes.transformation_matrix @ coefficients


def get_image(actuators=None, include_aberration=True):
    """Simulate an image given actuator commands

    Parameters
    ----------
    actuators : ndarray, optional
        Actuator commands from `make_phased_probe`, by default None
    include_aberration : bool, optional
        whether to include the pre-computed SurfaceAberration map, by default True

    Returns
    -------
    Wavefront
        wavefront containing the simulated electric field
    """

    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = Wavefront(aperture, wavelength)

    if include_aberration:
        wf = aberration(wf)

    img = prop(coronagraph(deformable_mirror(wf)))

    return img


# Get reference intensity
img_ref = prop(Wavefront(aperture, wavelength)).intensity

# Get nominal image
img = get_image().intensity.shaped / img_ref.max() # * dark_zone.shaped


# ---------------------------------------------------------------------------
# Speckle Area Nulling (Nishikawa 2022 / Oya 2017) summary by Claude Code
#
# At each dark-zone point the (unknown) speckle field is E and a probe adds a
# known field P. Probing at +/- amplitude and unprobed gives, per the 3-step
# phase-shifting algorithm:
#       I+ - I-          = 4 Re(E P*)          (numerator)
#       I+ + I- - 2 I0   = 2 |P|^2             (denominator)
# so   san = (I+ - I-)/(I+ + I- - 2 I0) = 2 Re(E P*)/|P|^2   (dimensionless).
# A single quadrature (cos OR sin) only measures one component of E, so we probe
# BOTH a cosine and a sine (quadrature) pattern to reconstruct the full complex
# field, then drive each mode's cos/sin coefficient to cancel it. The DM amplitude
# that nulls a quadrature is -(san/2) * a_probe (minus subtracts the measured
# field; 1/2 undoes the factor of 2 in the numerator).
# ---------------------------------------------------------------------------

# SAN Settings
dz_mask = dark_zone.shaped == 1

# make_phased_probe builds cos coeff = ampc*cos(phic) and sin coeff = amps*sin(phis).
# Use phic=0 (cos(0)=1) and phis=pi/2 (sin(pi/2)=1) so the amplitudes pass through cleanly.
cos_phase = np.zeros(num_frequencies)
sin_phase = np.full(num_frequencies, np.pi / 2)
zero_amp = np.zeros(num_frequencies)
unit_amp = np.full(num_frequencies, a_probe)

# Pure cosine probe and pure sine (quadrature) probe, as DM actuator vectors.
cos_probe = make_phased_probe(cos_phase, unit_amp, sin_phase, zero_amp)
sin_probe = make_phased_probe(cos_phase, zero_amp, sin_phase, unit_amp)


def image_norm(actuators):
    """Generate normalized-intensity image

    Parameters
    ----------
    actuators : ndarray
        actuator commands for deformable mirror

    Returns
    -------
    ndarray
        normalized intensity from `get_image`
    """
    return get_image(actuators=actuators).intensity.shaped / img_ref.max()


def darkzone_contrast(actuators):
    """Helper function to get mean value in image norm"""
    return image_norm(actuators)[dz_mask].mean()


def san_correction_step(total_actuators):
    """One SAN iteration: probe both quadratures around the current DM state and
    return the incremental correction actuator vector."""
    I0 = image_norm(total_actuators)

    Icp = image_norm(total_actuators + cos_probe)
    Icm = image_norm(total_actuators - cos_probe)
    Isp = image_norm(total_actuators + sin_probe)
    Ism = image_norm(total_actuators - sin_probe)

    san_cos = (Icp - Icm) / (Icp + Icm - 2 * I0)
    san_sin = (Isp - Ism) / (Isp + Ism - 2 * I0)

    correction = make_phased_probe(cos_phase, san_cos[dz_mask],
                                   sin_phase, san_sin[dz_mask])
    correction *= -0.5 * a_probe
    return correction


# SAN and Furious (SAF): reuse the *history* of applied DM commands as a per-pixel
# phase-shifting interferogram, and solve each dark-zone pixel's field by least squares.
class SANAndFurious:
    """Speckle Area Nulling with per-pixel least-squares phase-shifting ("...and Furious").

    Classical phase-shifting interferometry can only piston the reference phase, so it
    needs >=3 temporal frames of the *whole* interferogram. A coronagraph DM can impose a
    phase shift that varies across the focal plane, delta(x, y), so every dark-zone pixel
    runs its own 3-bucket interferometer. We never throw probe data away: every command we
    apply (a traditional SAN probe or a furious correction) is recorded with the image it
    produced, and once a pixel has seen >=3 phase-diverse commands we estimate its (static)
    speckle field E0 by least squares.

    Per dark-zone pixel j, the focal field is linear in command, E_tot = E0 + k' w, where
    w is the modal command coefficient (in units of a_probe) and k' the field-per-command:

        I_n = |E0|^2 + |k'|^2 |w_n|^2 + 2 Re(E0 k'^*) Re(w_n) + 2 Im(E0 k'^*) Im(w_n)

    Moving the known |k'|^2 |w_n|^2 to the left (y_n = I_n - |k'|^2 |w_n|^2) gives the
    standard least-squares phase-shifting normal equations (sums over recorded frames):

        [ a0 ]     [  N      Sc       Ss     ]^-1 [ Sy   ]
        [ a1 ]  =  [  Sc     Scc      Scs    ]    [ Syc  ]
        [ a2 ]     [  Ss     Scs      Sss    ]    [ Sys  ]

    with c = Re(w_n), s = Im(w_n). Then E0 k'^* = (a1 + i a2)/2, so the command that nulls
    the pixel is w_target = -(a1 + i a2) / (2 |k'|^2). The per-pixel gain |k'|^2 is
    calibrated from the +/- probe modulation of a traditional SAN step.
    """

    def __init__(self, forward_model=image_norm, dark_zone=dark_zone, a_probe=a_probe,
                 forget=0.5, probe_every_n=1):
        """interface with the SAN and Furious (SAF) speckle nulling technique

        Parameters
        ----------
        forward_model : callable
            callable function of DM actuators that returns a shaped intensity array
        dark_zone : Field of bool
            area on the focal plane to correct
        a_probe : float
            probe amplitude (m of DM surface); also the unit of the modal command w
        forget : float in (0, 1]
            exponential forgetting factor for the least-squares history. forget=1 pools
            every frame equally (most exposure-efficient, but biased toward the wrong null
            by inter-mode cross-talk); forget<1 down-weights stale frames so the fit
            re-linearizes around the current state and converges like residual SAN.
        probe_every_n : int
            how often step() spends a 4-exposure SAN probe burst. 1 probes every iteration
            (deepest, most exposures); n>1 probes once every n iterations and reuses the
            command history (furious-only) in between, trading depth-per-iteration for far
            fewer exposures.
        """
        self.fwd = forward_model
        self.dz = np.asarray(dark_zone.shaped, dtype=bool)   # 2D boolean mask
        self.a_probe = a_probe
        self.forget = forget
        self.probe_every_n = probe_every_n
        self.n_modes = num_frequencies                       # 1:1 with dark-zone pixels

        # current DM state
        self.actuators = np.zeros(len(influence_functions))  # actuator commands
        self.w = np.zeros(self.n_modes, dtype=complex)       # modal command (a_probe units)
        self.kappa2 = None                                   # per-pixel |k'|^2

        # phase-shifting "frames" reused for the least-squares inversion
        self.prior_corrections = []   # complex modal command per frame (a_probe units)
        self.prior_images = []        # measured dark-zone intensity per frame
        self.n_exposures = 0          # total forward-model evaluations (efficiency metric)
        self._iter = 0                # step counter (drives the probe_every_n schedule)
        self.last_image = self._image(self.actuators)   # always the current-state image

    # -- helpers --------------------------------------------------------------
    def _image(self, actuators):
        """Take one exposure (forward-model evaluation) and tally it."""
        self.n_exposures += 1
        return self.fwd(actuators)

    def _command(self, w_modal):
        """DM actuator vector for a complex modal command (a_probe units)."""
        return make_phased_probe(cos_phase, w_modal.real * self.a_probe,
                                 sin_phase, w_modal.imag * self.a_probe)

    def _record(self, w_modal, image_full):
        """Append one phase-shifting frame: the applied command and the image it made."""
        self.prior_corrections.append(np.asarray(w_modal, dtype=complex).copy())
        self.prior_images.append(image_full[self.dz].copy())

    def _apply(self, dw):
        """Add an incremental modal command dw (a_probe units) to the DM state, and refresh
        the current-state image so it can be reused as the next frame without re-exposing."""
        self.actuators = self.actuators + self._command(dw)
        self.w = self.w + dw
        self.last_image = self._image(self.actuators)

    # -- the two ways to grow the history -------------------------------------
    def san_step(self, gain=1.0):
        """Traditional SAN: 5 exposures around the current state. Records all five frames,
        (re)calibrates the per-pixel gain |k'|^2, and applies the SAN correction. With w in
        a_probe units, the +/-cos probe is w0 +/- 1 and the +/-sin probe is w0 +/- 1j."""
        a = self.actuators
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
        self.kappa2 = np.maximum(0.5 * (mod_cos + mod_sin), eps)

        san_cos = (Icp[dz] - Icm[dz]) / (2 * mod_cos)
        san_sin = (Isp[dz] - Ism[dz]) / (2 * mod_sin)
        self._apply(-0.5 * gain * (san_cos + 1j * san_sin))

    def furious_step(self, gain=1.0):
        """Per-pixel least-squares phase-shifting over the recorded history, with exponential
        forgetting so stale frames don't bias the static-linear fit (see `forget`)."""
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

    def min_step(self, gain=1.0, eps=0.5):
        """Non-SAN 3-frame downward iteration.

        Solving the per-pixel speckle field needs only 3 frames; SAN spends 5 (I0, +/-cos,
        +/-sin) because its *symmetric* probes cancel the probe self-term |k'|^2 without
        knowing it. Here we instead reuse the current corrected image as the reference
        (frame 1, free) and apply two small *one-sided* probes +eps and +i*eps (frames 2-3),
        so each iteration costs only 2 fresh exposures. Per pixel, with residual field
        R = E0 + k' w (|R|^2 = I0):

            I0   = |R|^2                                  (reference, reused)
            I_re = |R|^2 + |k'|^2 eps^2 + 2 eps Re(R k'^*)
            I_im = |R|^2 + |k'|^2 eps^2 + 2 eps Im(R k'^*)

        => Re/Im(R k'^*) = (I_re/I_im - I0 - |k'|^2 eps^2)/(2 eps), and the null step is
        dw = -R/k' = -(R k'^*)/|k'|^2. Three one-sided frames cannot also solve for |k'|^2
        (4 unknowns), so the per-pixel gain is taken from an earlier SAN burst; this method
        auto-seeds one san_step the first time to calibrate it."""
        if self.kappa2 is None:
            self.san_step(gain=gain)        # one symmetric burst to calibrate |k'|^2
            return

        a = self.actuators
        dz = self.dz
        I0 = self.last_image[dz]                             # reference (free, reused)
        Ire = self._image(a + eps * cos_probe)[dz]
        Iim = self._image(a + eps * sin_probe)[dz]

        self_term = self.kappa2 * eps ** 2                   # one-sided probe self-intensity
        re = (Ire - I0 - self_term) / (2 * eps)              # Re(R k'^*)
        im = (Iim - I0 - self_term) / (2 * eps)              # Im(R k'^*)
        self._apply(-gain * (re + 1j * im) / self.kappa2)    # dw = -(R k'^*)/|k'|^2

    def _solve_residual_differential(self):
        """Per-pixel weighted 2x2 least squares for the residual Z = R_now k'^*, from the
        differential model over the forgetting-weighted history. For each recorded frame k
        (command w_k, image I_k), with delta_k = w_now - w_k:
            Re(Z delta_k^*) = Re(Z) Re(delta_k) + Im(Z) Im(delta_k)
                            = (|k'|^2 |delta_k|^2 - (I_k - I0)) / 2 =: g_k.
        Clustered frames (delta_k -> 0) contribute ~0 and self-cancel. Returns complex Z."""
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

    def ff_step(self, gain=1.0, eps=0.5):
        """Fast & Furious-style step: ONE fresh probe per iteration, with the missing
        quadrature supplied by the forgetting-weighted history of prior command changes
        (the "Furious" temporal diversity). The fresh probe alternates real/imag each
        iteration so both quadratures stay sampled even as the corrections (hence their
        diversity) shrink. Cost is 1 fresh exposure/iteration; auto-seeds one san_step for
        |k'|^2."""
        if self.kappa2 is None:
            self.san_step(gain=gain)        # one symmetric burst to calibrate |k'|^2
            return

        a = self.actuators
        if self._iter % 2 == 0:
            probe, dw_fresh = cos_probe, eps + 0j           # +eps   (real quadrature)
        else:
            probe, dw_fresh = sin_probe, 1j * eps           # +i*eps (imag quadrature)
        I_fresh_full = self._image(a + eps * probe)
        self._record(self.w, self.last_image)               # reference frame (a prior correction)
        self._record(self.w + dw_fresh, I_fresh_full)       # fresh-probe frame

        Zc = self._solve_residual_differential()
        self._apply(-gain * Zc / self.kappa2)
        self._iter += 1

    def fnf_step(self, gain=1.0):
        """Correction-history-only step ("Fast aNd Furious"): NO fresh probe.

        A departure from ff_step inspired by Fast & Furious' reuse of prior DM commands as
        phase diversity. After a single 5-frame SAN seed (which calibrates |k'|^2 and injects
        both quadratures), every update reuses ONLY the history of correction images: the
        previous corrections themselves play the role of ff_step's fresh quadrature probe,
        since each contributes delta_k = w_now - w_k to the same differential least squares.
        Cost: 1 exposure/iteration (just the post-correction image), no dedicated probes.

        Because the seed's +/- cos/sin frames sit at fixed, well-spread commands, their
        delta_k to the (drifting) current state stay large and span both quadratures, so they
        keep the per-pixel 2x2 solve conditioned. With forget=1 they are never discarded; a
        strong forgetting factor would throw away the only diversity and should be avoided."""
        if self.kappa2 is None:
            self.san_step(gain=gain)        # 5-frame seed: calibrate |k'|^2 + both quadratures
            return

        self._record(self.w, self.last_image)               # current corrected state (no probe)
        Zc = self._solve_residual_differential()
        self._apply(-gain * Zc / self.kappa2)
        self._iter += 1

    def step(self, gain=1.0):
        """One SAF iteration. Every probe_every_n-th iteration spends a 4-exposure SAN probe
        burst to inject fresh near-state phase diversity and re-calibrate the per-pixel gain
        |k'|^2; every iteration then refines with the least-squares solve that reuses the
        (forgotten) command history. With probe_every_n=1 this probes every step (deepest);
        with probe_every_n>1 the in-between iterations are furious-only and cost a single
        exposure each, which is where the exposure savings come from."""
        if self.kappa2 is None or self._iter % self.probe_every_n == 0:
            self.san_step(gain=gain)
        self.furious_step(gain=gain)
        self._iter += 1


# ---- Iterate ----
# USE_SAF=True runs the least-squares "SAN and Furious" estimator (reuses command
# history); USE_SAF=False runs the original symmetric-difference SAN step.
USE_SAF = True
num_iterations = 10
gain = 1.0 # Generally adjustable
total_actuators = np.zeros(len(influence_functions))
img_initial = image_norm(total_actuators)
saf = SANAndFurious(forward_model=image_norm, dark_zone=dark_zone, a_probe=a_probe)


def render_frame(iteration, img_now, history):
    """Render one movie frame (initial PSF | current PSF | convergence curve).

    Fixed figsize, color scale and axis limits keep every frame the same pixel
    size and layout, which the .gif encoder requires. Returns an RGB array.
    """
    fig = plt.figure(figsize=[15, 5], dpi=90)

    ax1 = fig.add_subplot(131)
    ax1.set_title('Initial')
    ax1.imshow(np.log10(img_initial), cmap="inferno", vmax=-2, vmin=-10)

    ax2 = fig.add_subplot(132)
    ax2.set_title(f'SAN iteration {iteration}')
    ax2.imshow(np.log10(img_now), cmap="inferno", vmax=-2, vmin=-10)

    ax3 = fig.add_subplot(133)
    ax3.set_title('Dark-zone mean contrast')
    ax3.semilogy(range(len(history)), history, 'o-')
    ax3.set_xlim(-0.5, num_iterations + 0.5)
    ax3.set_ylim(10**np.floor(np.log10(min(history)) - 0.5),
                 10**np.ceil(np.log10(max(history)) + 0.5))
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('mean contrast')
    ax3.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


contrast_history = [darkzone_contrast(total_actuators)]
print(f"iter  0: dark-zone mean contrast = {contrast_history[0]:.3e}")

frames = []
if SAVE_MOVIE:
    frames.append(render_frame(0, img_initial, contrast_history))

for k in range(num_iterations):
    if USE_SAF:
        saf.step(gain=gain)
        total_actuators = saf.actuators
    else:
        total_actuators = total_actuators + gain * san_correction_step(total_actuators)
    img_now = image_norm(total_actuators)
    contrast_history.append(darkzone_contrast(total_actuators))
    print(f"iter {k + 1:2d}: dark-zone mean contrast = {contrast_history[-1]:.3e}")
    if SAVE_MOVIE:
        frames.append(render_frame(k + 1, img_now, contrast_history))

img_corrected = image_norm(total_actuators)

if SAVE_MOVIE:
    import imageio.v2 as imageio
    # hold the final frame a little longer so the converged state is visible.
    durations = [1.0 / MOVIE_FPS] * len(frames)
    durations[-1] = 2.0
    imageio.mimsave(MOVIE_FILENAME, frames, duration=durations, loop=0)
    print(f"saved movie: {os.path.abspath(MOVIE_FILENAME)} ({len(frames)} frames)")

plt.figure(figsize=[15, 5])
plt.subplot(131)
plt.title('Initial')
plt.imshow(np.log10(img_initial), cmap="inferno", vmax=-2, vmin=-10)
plt.colorbar()
plt.subplot(132)
plt.title(f'After {num_iterations} SAN iterations')
plt.imshow(np.log10(img_corrected), cmap="inferno", vmax=-2, vmin=-10)
plt.colorbar()
plt.subplot(133)
plt.title('Dark-zone mean contrast')
plt.semilogy(range(len(contrast_history)), contrast_history, 'o-')
plt.xlabel('iteration')
plt.ylabel('mean contrast')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()


# ---- Efficiency comparison: contrast vs *cumulative exposures* ----
# The fair currency for a real instrument is exposures (probe images), not iterations.
# This reruns each estimator from scratch on the SAME static aberration and plots how
# deep the dark zone gets per exposure spent. In a noiseless sim plain SAN is already
# very exposure-efficient; the least-squares history reuse is expected to pull ahead
# mainly once measurement noise is present (it averages frames down).
COMPARE_EFFICIENCY = True
if COMPARE_EFFICIENCY:
    def efficiency_curve(make_estimator, stepper, n_iter):
        est = make_estimator()
        exposures = [est.n_exposures]
        contrast = [est.last_image[dz_mask].mean()]
        for _ in range(n_iter):
            stepper(est)
            exposures.append(est.n_exposures)
            contrast.append(est.last_image[dz_mask].mean())
        return np.array(exposures), np.array(contrast)

    curves = [
        ('SAN (5-frame, probe every iter)', efficiency_curve(lambda: SANAndFurious(), lambda e: e.san_step(gain=gain), 14)),
        ('SAF LSQ (probe_every_n=1)',       efficiency_curve(lambda: SANAndFurious(probe_every_n=1), lambda e: e.step(gain=gain), 14)),
        ('non-SAN 3-frame (min_step)',      efficiency_curve(lambda: SANAndFurious(), lambda e: e.min_step(gain=gain), 22)),
        ('Fast & Furious 1-probe (ff_step)', efficiency_curve(lambda: SANAndFurious(), lambda e: e.ff_step(gain=gain), 34)),
        ('FnF correction-only (fnf_step)',  efficiency_curve(lambda: SANAndFurious(), lambda e: e.fnf_step(gain=gain), 20)),
    ]

    plt.figure(figsize=[8, 6])
    for label, (exposures, contrast) in curves:
        plt.semilogy(exposures, contrast, 'o-', markersize=4, label=label)
    plt.title('Speckle nulling efficiency: contrast vs exposures')
    plt.xlabel('cumulative exposures (forward-model evaluations)')
    plt.ylabel('dark-zone mean contrast')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---- Maintenance demo: hold a dark hole against drift with few probe frames ----
# The science goal is not the deepest null but holding moderate contrast while spending as
# few *dedicated probe* frames as possible (every probe frame is lost science time). A
# low-order quasi-static drift is injected with a second "disturbance" DM, and a single
# drift realization is replayed for every strategy:
#   - open loop          : freeze the seed command, take only science frames (0 probes).
#   - SAN every N steps   : re-probe every N steps, hold the command in between (4 probes
#                           per probe event). N is the probe-duty-cycle knob.
#   - science-frame+dither: never take a dedicated probe; put a tiny known dither on each
#                           science frame and run the differential least squares over the
#                           (forgetting-weighted) history (0 probe frames).
MAINTENANCE_DEMO = True
if MAINTENANCE_DEMO:
    rng = np.random.default_rng(3)
    T_maint = 36
    K_drift = 6            # quasi-static drift lives in a low-dim mode subspace
    drift_sigma = 0.015    # per-step random-walk amplitude (a_probe units)
    dither_amp = 0.05      # micro-dither for the science-frame estimator (a_probe units)

    n_act = len(influence_functions)
    disturbance_dm = DeformableMirror(influence_functions)   # injects the drift ("truth")
    _drift = {'cmd': np.zeros(n_act)}

    def fwd_drift(actuators):
        deformable_mirror.actuators = actuators
        disturbance_dm.actuators = _drift['cmd']
        wf = aberration(Wavefront(aperture, wavelength))
        wf = disturbance_dm(wf)
        wf = deformable_mirror(wf)
        return prop(coronagraph(wf)).intensity.shaped / img_ref.max()

    # one quasi-static drift realization (low-dim random walk), replayed for each strategy
    drift_dirs = (rng.standard_normal((K_drift, num_frequencies))
                  + 1j * rng.standard_normal((K_drift, num_frequencies)))
    _amp = np.zeros(K_drift)
    drift_seq = []
    for _ in range(T_maint):
        _amp = _amp + drift_sigma * rng.standard_normal(K_drift)
        dwm = (_amp[:, None] * drift_dirs).sum(0)
        drift_seq.append(make_phased_probe(cos_phase, dwm.real * a_probe,
                                           sin_phase, dwm.imag * a_probe).copy())

    # seed a dark hole at zero drift (shared starting point for every strategy)
    _drift['cmd'] = np.zeros(n_act)
    seed = SANAndFurious(forward_model=fwd_drift, dark_zone=dark_zone, a_probe=a_probe)
    for _ in range(6):
        seed.san_step(1.0)
    act_seed = seed.actuators.copy()
    w_seed = seed.w.copy()
    kap = seed.kappa2.copy()
    histW0 = [w.copy() for w in seed.prior_corrections]   # seed frames bootstrap diversity
    histI0 = [im.copy() for im in seed.prior_images]
    dzc = dark_zone.shaped == 1

    def maint_open():
        c = []
        for t in range(T_maint):
            _drift['cmd'] = drift_seq[t]
            c.append(fwd_drift(act_seed)[dzc].mean())
        return np.array(c), 0

    def maint_sparse(N):
        act = act_seed.copy(); c = []; probes = 0
        for t in range(T_maint):
            _drift['cmd'] = drift_seq[t]
            if t % N == 0:
                I0 = fwd_drift(act)[dzc]
                Icp = fwd_drift(act + cos_probe)[dzc]; Icm = fwd_drift(act - cos_probe)[dzc]
                Isp = fwd_drift(act + sin_probe)[dzc]; Ism = fwd_drift(act - sin_probe)[dzc]
                probes += 4
                mc = (Icp + Icm - 2 * I0) / 2; ms = (Isp + Ism - 2 * I0) / 2
                dw = -0.5 * ((Icp - Icm) / (2 * mc) + 1j * (Isp - Ism) / (2 * ms))
                act = act + make_phased_probe(cos_phase, dw.real * a_probe,
                                              sin_phase, dw.imag * a_probe)
            c.append(fwd_drift(act)[dzc].mean())
        return np.array(c), probes

    def maint_science(forget=0.7, gain=0.7, dither=dither_amp):
        act = act_seed.copy(); w = w_seed.copy()
        Wh = [x.copy() for x in histW0]; Ih = [x.copy() for x in histI0]; c = []
        for t in range(T_maint):
            _drift['cmd'] = drift_seq[t]
            # tiny known dither on the science frame, alternating quadrature for diversity
            dd = dither if t % 2 == 0 else 1j * dither
            actd = act + (dither * cos_probe if t % 2 == 0 else dither * sin_probe)
            I0 = fwd_drift(actd)[dzc]
            Wh.append((w + dd).copy()); Ih.append(I0.copy())

            W = np.asarray(Wh); I = np.asarray(Ih); nfr = W.shape[0]
            wt = (forget ** np.arange(nfr)[::-1])[:, None]
            delta = (w + dd)[None, :] - W; p = delta.real; q = delta.imag
            gg = 0.5 * (kap[None, :] * np.abs(delta) ** 2 - (I - I0[None, :]))
            M = np.empty((num_frequencies, 2, 2))
            M[:, 0, 0] = (wt * p * p).sum(0)
            M[:, 0, 1] = M[:, 1, 0] = (wt * p * q).sum(0)
            M[:, 1, 1] = (wt * q * q).sum(0)
            b = np.stack([(wt * p * gg).sum(0), (wt * q * gg).sum(0)], axis=-1)
            r = 1e-6 * (M[:, 0, 0] + M[:, 1, 1]) + 1e-30
            M[:, 0, 0] += r; M[:, 1, 1] += r
            Z = np.linalg.solve(M, b)
            # null the UNDITHERED command: R(w) k'* = Z - kap*dd  (subtract the known dither)
            dw = -gain * ((Z[:, 0] + 1j * Z[:, 1]) - kap * dd) / kap
            w = w + dw
            act = act + make_phased_probe(cos_phase, dw.real * a_probe,
                                          sin_phase, dw.imag * a_probe)
            c.append(fwd_drift(act)[dzc].mean())   # contrast at the (undithered) command
        return np.array(c), 0

    maint = [
        ('open loop (freeze)',     *maint_open()),
        ('SAN every step',         *maint_sparse(1)),
        ('SAN every 4 steps',      *maint_sparse(4)),
        ('SAN every 8 steps',      *maint_sparse(8)),
        ('science-frame + dither', *maint_science()),
    ]

    plt.figure(figsize=[13, 5])
    ax1 = plt.subplot(121)
    for label, c, probes in maint:
        ax1.semilogy(range(T_maint), c, 'o-', markersize=3, label=f'{label} [{probes} probes]')
    ax1.set_title('Maintaining contrast against drift')
    ax1.set_xlabel('maintenance step (science frame)')
    ax1.set_ylabel('dark-zone mean contrast')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(122)
    for label, c, probes in maint:
        ax2.loglog(max(probes, 0.5), c.mean(), 'o', markersize=9)
        ax2.annotate(label, (max(probes, 0.5), c.mean()),
                     textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax2.set_title('Cost of maintenance: mean contrast vs probe frames')
    ax2.set_xlabel('total dedicated probe frames (less = more science time)')
    ax2.set_ylabel('mean maintained contrast')
    ax2.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
