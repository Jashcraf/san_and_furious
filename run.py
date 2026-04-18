from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Input parameters from hcipy EFC demo
pupil_diameter = 7e-3 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m
npix = 256
iwa_ld = 3
owa_ld = 12
offset_ld = 1

num_actuators_across = 32
actuator_spacing = 1.05 / 32 * pupil_diameter # m
aberration_ptv = 0.02 * wavelength # m
eps = 1e-9

spatial_resolution = focal_length * wavelength / pupil_diameter # microns 
iwa = iwa_ld * spatial_resolution # m
owa = owa_ld * spatial_resolution # m
offset = offset_ld * spatial_resolution # m

# Create grids
pupil_grid = make_pupil_grid(npix, pupil_diameter * 1.2)
focal_grid = make_focal_grid(4, 16, spatial_resolution=spatial_resolution)
prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# Create aperture and dark zone
aperture = Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

dark_zone = circular_aperture(2 * owa)(focal_grid)
dark_zone -= circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
coronagraph = PerfectCoronagraph(aperture, order=4)

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
actuator_grid = make_uniform_grid(
    dims=[num_actuators_across, num_actuators_across],
    extent=[num_actuators_across * actuator_spacing, num_actuators_across * actuator_spacing]
) 

# Build up Fourier grid
fx = focal_grid.x[dark_zone==1] / (wavelength * focal_length) 
fy = focal_grid.y[dark_zone==1] / (wavelength * focal_length) 
print("rmax_focal = ",np.sqrt(fx**2 + fy**2).max())
fourier_grid = CartesianGrid(UnstructuredCoords([fx, fy]))
print(fourier_grid.x.max())


fourier_modes = make_fourier_basis(
    grid=actuator_grid,
    fourier_grid=fourier_grid,
    sort_by_energy=True
)

num_frequencies = len(fourier_modes) // 2
print("frequency range (cycles/m):", np.sqrt(fx**2 + fy**2).min(), np.sqrt(fx**2 + fy**2).max())
print("expected iwa/owa (cycles/m):", iwa_ld / pupil_diameter, owa_ld / pupil_diameter)


def make_phased_probe(phase_shifts, amplitudes):

    coefficients = np.zeros(len(fourier_modes))
    for i, (phi, amp) in enumerate(zip(phase_shifts, amplitudes)):
        coefficients[2*i] = amp * np.cos(phi)
        coefficients[2*i + 1] = -1 * amp * np.sin(phi)
    
    # normalize by the number of modes
    coefficients /= len(fourier_modes)

    return fourier_modes.transformation_matrix @ coefficients


def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror.actuators = actuators 

    wf = Wavefront(aperture, wavelength)

    if include_aberration:
        wf = aberration(wf)

    img = prop(coronagraph(deformable_mirror(wf)))

    return img

img_ref = prop(Wavefront(aperture, wavelength)).intensity

img = get_image().intensity.shaped / img_ref.max() * dark_zone.shaped

probe_coeffs = make_phased_probe(phase_shifts=np.full(num_frequencies, 0),
                                 amplitudes=-np.full(num_frequencies, wavelength / 20))
img_probed = get_image(actuators=probe_coeffs).intensity.shaped / img_ref.max() * dark_zone.shaped

plt.figure(figsize=[10,5])
plt.subplot(121)
plt.imshow(np.log10(img), cmap="inferno")
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log10(img_probed), cmap="inferno")
plt.colorbar()
plt.show()



