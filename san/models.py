class Model:

    def __init__(self,
        self,
        pupil_diameter,
        wavelength,
        focal_length,
        npix,
        iwa_ld,
        owa_ld,
        offset_ld,
        num_actuators_across,
        actuator_sampling,
    ):

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



