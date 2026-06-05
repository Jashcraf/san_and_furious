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




# ---- Iterate ----
num_iterations = 10
gain = 1.0 # Generally adjustable
total_actuators = np.zeros(len(influence_functions))
img_initial = image_norm(total_actuators)


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
