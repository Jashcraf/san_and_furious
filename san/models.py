"""Simple coronagraph forward models with a deformable mirror.

A :class:`CoronagraphModel` bundles the hcipy simulation that used to live at the
top of ``run.py``: the pupil/focal grids, aperture, dark zone, a perfect
coronagraph, a static surface aberration and a continuous deformable mirror,
together with the Fourier-mode basis that maps modal commands onto dark-zone
spatial frequencies.

The model is the *forward model* the speckle-nulling algorithms in
:mod:`san.algorithms` drive.  Its core interface is:

    model.set_dm(actuator_commands)   # update the DM's internal state
    model.image(actuators=None)       # normalized-intensity image (forward model)

plus a few helpers the algorithms need to build sinusoidal DM probes
(:meth:`make_phased_probe`, :meth:`command`, :attr:`cos_probe`, :attr:`sin_probe`).
"""

import numpy as np
from hcipy import *


class CoronagraphModel:
    """A coronagraph + deformable-mirror forward model.

    Parameters mirror the inputs at the top of the original ``run.py`` (taken from
    the hcipy EFC demo).  All defaults reproduce that configuration.

    Parameters
    ----------
    pupil_diameter : float
        Pupil diameter (m).
    wavelength : float
        Observing wavelength (m).
    focal_length : float
        Focal length feeding the Fraunhofer propagator (m).
    npix : int
        Number of pixels across the pupil grid.
    iwa_ld, owa_ld : float
        Inner / outer working angle of the dark zone (lambda/D).
    offset_ld : float
        One-sided (x > offset) dark-zone offset (lambda/D).
    num_actuators_across : int
        Deformable-mirror actuators per side.
    actuator_spacing : float, optional
        Physical actuator pitch (m).  Defaults to ``1.05 / 32 * pupil_diameter``
        (a HAKA-like geometry).
    aberration_ptv : float, optional
        Peak-to-valley of the static surface aberration (m).  Defaults to
        ``0.02 * wavelength``.
    a_probe : float, optional
        Probe amplitude / modal-command unit (m of DM surface).  Defaults to
        ``wavelength / 50``.
    coronagraph_order : int
        Order of the :class:`PerfectCoronagraph`.
    aberration_exponent : float
        Power-law exponent of the surface aberration PSD.
    eps : float
        Small floor used to keep per-pixel gains positive.
    seed : int or None
        If given, seeds ``numpy`` before drawing the random surface aberration so
        the model is reproducible.
    """

    def __init__(self,
                 pupil_diameter=7e-3,
                 wavelength=700e-9,
                 focal_length=500e-3,
                 npix=256,
                 iwa_ld=6,
                 owa_ld=12,
                 offset_ld=6,
                 num_actuators_across=32,
                 actuator_spacing=None,
                 aberration_ptv=None,
                 a_probe=None,
                 coronagraph_order=6,
                 aberration_exponent=-3,
                 eps=1e-9,
                 seed=None):

        # ---- digest inputs -------------------------------------------------
        self.pupil_diameter = pupil_diameter
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.npix = npix
        self.num_actuators_across = num_actuators_across
        self.eps = eps

        if actuator_spacing is None:
            actuator_spacing = 1.05 / 32 * pupil_diameter
        if aberration_ptv is None:
            aberration_ptv = 0.02 * wavelength
        if a_probe is None:
            a_probe = wavelength / 50
        self.actuator_spacing = actuator_spacing
        self.a_probe = a_probe

        # microns; angular sizes converted to physical focal-plane distances
        self.spatial_resolution = focal_length * wavelength / pupil_diameter
        iwa = iwa_ld * self.spatial_resolution
        owa = owa_ld * self.spatial_resolution
        offset = offset_ld * self.spatial_resolution

        # ---- grids and propagator -----------------------------------------
        self.pupil_grid = make_pupil_grid(npix, pupil_diameter * 1.2)
        self.focal_grid = make_focal_grid(4, 16, spatial_resolution=self.spatial_resolution)
        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length)

        # super-gaussian aperture so we can run at low resolution
        self.aperture = Field(
            np.exp(-(self.pupil_grid.as_('polar').r / (0.5 * pupil_diameter)) ** 30),
            self.pupil_grid)
        self.lyot_stop = circular_aperture(0.9 * self.pupil_diameter)(self.pupil_grid)

        # ---- dark zone -----------------------------------------------------
        dark_zone = circular_aperture(2 * owa)(self.focal_grid)
        dark_zone -= circular_aperture(2 * iwa)(self.focal_grid)
        dark_zone *= self.focal_grid.x > offset
        self.dark_zone = dark_zone.astype(bool)
        self.dz_mask = self.dark_zone.shaped == 1   # 2D boolean mask

        # ---- optical elements ---------------------------------------------
        #self.coronagraph = PerfectCoronagraph(self.aperture, order=coronagraph_order)
        self.coronagraph = VortexCoronagraph(self.pupil_grid, coronagraph_order, lyot_stop=self.lyot_stop)

        # tip-tilt is assumed handled elsewhere (e.g. QACITS), so remove it
        tip_tilt = make_zernike_basis(3, pupil_diameter, self.pupil_grid, starting_mode=2)
        if seed is not None:
            np.random.seed(seed)
        self.aberration = SurfaceAberration(self.pupil_grid,
                                            aberration_ptv,
                                            pupil_diameter,
                                            remove_modes=tip_tilt,
                                            exponent=aberration_exponent)

        self.influence_functions = make_gaussian_influence_functions(
            self.pupil_grid, num_actuators_across, actuator_spacing)
        self.deformable_mirror = DeformableMirror(self.influence_functions)
        self.num_actuators = len(self.influence_functions)

        # A second "disturbance" DM upstream of the correction DM, used to inject a
        # known quasi-static drift ("truth") for maintenance experiments.  With zero
        # commands it is an identity, so it never affects the nominal forward model.
        self.disturbance_dm = DeformableMirror(self.influence_functions)

        # ---- Fourier-mode basis -------------------------------------------
        self._build_fourier_modes()

        # ---- probes and normalization -------------------------------------
        # make_phased_probe builds cos coeff = ampc*cos(phic), sin coeff = amps*sin(phis).
        # phic=0 (cos 0 = 1) and phis=pi/2 (sin pi/2 = 1) pass amplitudes through cleanly.
        self.cos_phase = np.zeros(self.num_frequencies)
        self.sin_phase = np.full(self.num_frequencies, np.pi / 2)
        zero_amp = np.zeros(self.num_frequencies)
        unit_amp = np.full(self.num_frequencies, a_probe)
        self.cos_probe = self.make_phased_probe(self.cos_phase, unit_amp, self.sin_phase, zero_amp)
        self.sin_probe = self.make_phased_probe(self.cos_phase, zero_amp, self.sin_phase, unit_amp)

        # reference (unaberrated, no coronagraph) intensity peak for normalization
        self.img_ref = self.prop(Wavefront(self.aperture, wavelength)).intensity

    # -- Fourier-mode construction -------------------------------------------
    def _build_fourier_modes(self):
        """Build the Fourier-mode basis on the DM's actual actuator positions.

        The modes must be evaluated on the actuator grid, which spans
        ``num_actuators_across * actuator_spacing`` (= 1.05 * pupil_diameter), NOT
        the 1.2 * pupil_diameter pupil-grid extent.
        """
        actuator_extent = self.num_actuators_across * self.actuator_spacing
        actuator_grid = make_uniform_grid(
            dims=[self.num_actuators_across, self.num_actuators_across],
            extent=[actuator_extent, actuator_extent])

        # hcipy's make_fourier_basis builds cos(p . x) / sin(p . x) with NO 2*pi, so
        # the fourier_grid holds ANGULAR frequencies (rad/m) = 2*pi * x / (lambda*F).
        fx = 2 * np.pi * self.focal_grid.x[self.dark_zone] / (self.wavelength * self.focal_length)
        fy = 2 * np.pi * self.focal_grid.y[self.dark_zone] / (self.wavelength * self.focal_length)
        fourier_grid = CartesianGrid(UnstructuredCoords([fx, fy]))

        # sort_by_energy MUST be False: corrections index modes in dark-zone-pixel
        # order, so mode i must keep driving dark-zone pixel i.  Energy-sorting would
        # permute the modes and apply each correction to the wrong spatial frequency.
        self.fourier_modes = make_fourier_basis(
            grid=actuator_grid, fourier_grid=fourier_grid, sort_by_energy=False)
        self.num_frequencies = len(self.fourier_modes) // 2

    # -- probe construction ---------------------------------------------------
    def make_phased_probe(self, phase_shifts_cos, amplitudes_cos, phase_shifts_sin, amplitudes_sin):
        """Build a sinusoidal DM probe composed of both cosines and sines.

        Parameters
        ----------
        phase_shifts_cos : ndarray
            Phase shifts ``delta`` applied to ``A * cos(x + delta)``.
        amplitudes_cos : ndarray
            Amplitudes ``A`` applied to ``A * cos(x + delta)``.
        phase_shifts_sin : ndarray
            Phase shifts ``delta`` applied to ``A * sin(x + delta)``.
        amplitudes_sin : ndarray
            Amplitudes ``A`` applied to ``A * sin(x + delta)``.

        Returns
        -------
        ndarray
            Actuator commands for the deformable mirror.
        """
        coefficients = np.zeros(len(self.fourier_modes))
        for i, (phic, ampc, phis, amps) in enumerate(zip(phase_shifts_cos,
                                                         amplitudes_cos,
                                                         phase_shifts_sin,
                                                         amplitudes_sin)):
            coefficients[2 * i] = ampc * np.cos(phic)
            coefficients[2 * i + 1] = amps * np.sin(phis)

        # normalize by the number of modes to maintain surface RMS
        coefficients /= len(self.fourier_modes)
        return self.fourier_modes.transformation_matrix @ coefficients

    def command(self, w_modal):
        """Actuator vector for a complex modal command ``w`` (in ``a_probe`` units).

        ``Re(w)`` drives the cosine quadrature and ``Im(w)`` the sine quadrature.
        """
        w_modal = np.asarray(w_modal, dtype=complex)
        return self.make_phased_probe(self.cos_phase, w_modal.real * self.a_probe,
                                      self.sin_phase, w_modal.imag * self.a_probe)

    # -- the deformable mirror ------------------------------------------------
    def set_dm(self, actuator_commands):
        """Update the deformable mirror's internal state.

        Parameters
        ----------
        actuator_commands : ndarray
            Actuator commands (length ``num_actuators``).
        """
        self.deformable_mirror.actuators = actuator_commands

    def set_disturbance(self, actuator_commands):
        """Update the upstream disturbance DM (the injected drift "truth").

        Parameters
        ----------
        actuator_commands : ndarray
            Actuator commands (length ``num_actuators``).  All-zero is an identity.
        """
        self.disturbance_dm.actuators = actuator_commands

    # -- the forward model ----------------------------------------------------
    def forward(self, actuators=None, include_aberration=True):
        """Propagate to the focal plane and return the post-coronagraph wavefront.

        Parameters
        ----------
        actuators : ndarray, optional
            Actuator commands.  If given, the DM state is updated first via
            :meth:`set_dm`.
        include_aberration : bool
            Whether to apply the static surface aberration.

        Returns
        -------
        Wavefront
            The focal-plane wavefront (electric field).
        """
        if actuators is not None:
            self.set_dm(actuators)

        wf = Wavefront(self.aperture, self.wavelength)
        if include_aberration:
            wf = self.aberration(wf)
        # disturbance DM is upstream of the correction DM (identity when un-driven)
        wf = self.disturbance_dm(wf)
        return self.prop(self.coronagraph(self.deformable_mirror(wf)))

    def image(self, actuators=None, include_aberration=True):
        """Normalized-intensity image: the forward model the algorithms drive.

        Parameters
        ----------
        actuators : ndarray, optional
            Actuator commands; updates the DM state first if given.
        include_aberration : bool
            Whether to include the static surface aberration.

        Returns
        -------
        ndarray
            Shaped intensity image, normalized by the reference PSF peak.
        """
        return self.forward(actuators, include_aberration).intensity.shaped / self.img_ref.max()

    def darkzone_contrast(self, actuators=None):
        """Mean normalized intensity inside the dark zone."""
        return self.image(actuators)[self.dz_mask].mean()
