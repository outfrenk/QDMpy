"""Comprehensive measurement management for Quantum Diamond Microscopy.

This module provides the central `Measurement` class that serves as the primary interface
for working with Quantum Diamond Microscope (QDM) experiments. Key capabilities include:

- Data integration: Combines ODMR spectral data with optical images
- Spatial analysis: Maps spectral properties across the spatial dimensions
- Image processing: Handles light and laser reference images
- Metadata tracking: Maintains experiment parameters and processing history
- Output management: Organizes results in a structured directory hierarchy
- Statistical analysis: Identifies outliers and performs data quality assessment

The Measurement class integrates data from the ODMR module with optical images and
provides a unified interface for analysis and visualization of QDM experiments.
"""
from __future__ import annotations

import logging
import os, sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from os import PathLike

if not __package__:
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go one level up to the package root
    package_root = os.path.abspath(os.path.join(current_dir, '..'))
    # Add to path if not already there
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


# Following import must be after setup_package_paths
import QDMpy.constants as qcon
from QDMpy.odmr.odmr import ODMR  # noqa: E402
from QDMpy import guess
try:
    import pygpufit.gpufit as gf
    GPUFIT_PRESENT = True
except ModuleNotFoundError:
    GPUFIT_PRESENT = False
LOG = logging.getLogger(__name__)


class Measurement:
    """The Measurement class encapsulates all data and processing related to a single QDM
    (Quantum Diamond Microscope) measurement.

    It manages:
        - Raw and processed ODMR data using the ODMR instance.
        - Associated images (light and laser).
        - Fitting operations via external fitting instances.

    Attributes:
        odmr (ODMR): Instance managing ODMR data and processing.
        light_image (NDArray): Light image array with shape (height, width).
        laser_image (NDArray): Laser image array with shape (height, width).
        output_directory (Path): Path to the output directory.
        pixel_spacing (float): Spacing between pixels in meters.
        _outliers (Optional[NDArray]): Boolean mask for outlier pixels.
        _B111 (Optional[NDArray]): B111 field array, populated after fitting.
        _fit_model (str): Name of the model used for fitting ODMR spectra.
        metadata (Dict[str, Any]): Additional metadata for the measurement.
    """

    def __init__(
        self,
        odmr: ODMR,
        light_image: NDArray,
        laser_image: NDArray,
        output_directory: str | Path | PathLike,
        pixel_spacing: float = 1e-6,
        fit_model: str = 'auto',
    ) -> None:
        """Initialize the Measurement object.

        Args:
            odmr (ODMR): An initialized ODMR instance containing ODMR data.
            light_image (NDArray): Light image array with shape (height, width).
            laser_image (NDArray): Laser image array with shape (height, width).
            output_directory (Union[str, Path, PathLike]): Path to the output directory.
            pixel_spacing (float): Spacing between pixels in meters (pixel size).
                Default is 1 µm (1e-6).
            fit_model (str): Name of the model used for fitting ODMR spectra. Default is "auto".
                            If "auto", the model is chosen based on the mean ODMR data.

        Raises:
            ValueError: If the ODMR instance is not properly initialized or if image shapes
                       don't match the ODMR data.
        """
        LOG.info('Initializing Measurement object.')
        LOG.info('Output directory: "%s"', output_directory)

        self.output_directory = Path(output_directory)
        self.pixel_spacing = pixel_spacing
        self.metadata: dict[str, Any] = {}

        # Store the ODMR instance
        LOG.debug('Setting ODMR data.')
        self.odmr = odmr

        # Validate ODMR data availability
        try:
            # Use public property instead of accessing protected member
            _ = self.odmr.raw_data
        except ValueError:
            raise ValueError('ODMR instance has no raw data')

        # Validate ODMR instance data
        LOG.debug('ODMR raw data shape: %s', self.odmr.raw_data.shape)

        # Check if data has been processed
        if self.odmr.is_processed:
            LOG.debug('ODMR processed data shape: %s', self.odmr.processed_data.shape)
        else:
            LOG.warning('ODMR data has not been processed yet. Some functionality may be limited.')

        LOG.debug('ODMR frequencies shape: %s', self.odmr.raw_data.frequencies.shape)

        # Initialize outlier mask
        LOG.debug('Initializing outlier mask.')
        self._outliers: NDArray | None = np.ones(self.odmr.raw_data.shape, dtype=bool)

        # Store light and laser images
        LOG.debug('Storing light and laser images.')
        self.light_image = light_image
        self.laser_image = laser_image

        # Initialize B111 field, fit model, and initial parameters
        LOG.debug('Initializing B111 field and fit model.')
        self._B111: NDArray | None = None
        # Placeholder for future fit integration
        self._fit_model = fit_model
        self._initial_parameters: NDArray | None = None

    def fit_odmr(self, fit_method: str = 'auto'):
        """Fit the Measurement object according to the Diamond model."""
        # Select data to use
        # first we need the initial guesses before making a call to pygpufit.gpufit.fit_constrained
        if self.odmr.is_processed:
            odmrdata = self.odmr.processed_data
        else:
            odmrdata = self.odmr.raw_data
        # find diamond model if auto
        if self._fit_model == 'auto':
            self._fit_model = guess.guess_model(odmrdata.data)
        # make guesses
        LOG.info("Starting initial guess for ODMR data using %s diamond model", self._fit_model)
        self._initial_parameters = guess.guess_initial_fit_parameters(
            odmrdata.data,
            odmrdata.frequencies,
            self._fit_model
        )
        LOG.debug(f"Initial parameters have shape: {self._initial_parameters.shape}")
        if fit_method == 'auto':
            if GPUFIT_PRESENT:
                self.gpufit(odmrdata)
            else:
                # TODO: add option for CPU
                self.cpufit(odmrdata)
        elif fit_method != 'gpufit' or fit_method != 'cpufit':
            raise Exception(f"Fit method {fit_method} is not supported.")
        else:
            {'gpufit': self.gpufit(odmrdata), 'cpufit':self.cpufit(odmrdata)}[fit_method]()

    def gpufit(self, odmrdata: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Calculates the fit parameters for the GPU model for diamond model."""
        LOG.info(f"Using gpufit method")
        # fit it
        n_pol, f_range, n_pix, n_freqs = odmrdata.shape
        constraint_list = self._fit_model.get_constraint_array
        constraint_types = np.array(constraint_list[:, 2].flatten()).astype(np.int32)
        constraint_pixel = np.tile(np.array(constraint_list[:, :2].flatten()).astype(np.float32),
                                   (n_pix * n_pol, 1))
        # initialize empty parameter variable
        self.parameters = np.zeros((n_pol, f_range, n_pix, len(constraint_types)))
        states = np.zeros_like(self.parameters)
        chi_squares = np.zeros_like(self.parameters)
        iterations = np.zeros_like(self.parameters)

        for i in np.arange(0, f_range):
            frange = odmrdata.frequencies.reshape(f_range, n_freqs)
            frange_typ = frange / 1e9
            LOG.info(f"Fitting frange {i} from {frange_typ[i].min():.3f}-{frange_typ[i].max():.3f} GHz")
            data_input = odmrdata.data[:, i].reshape((-1, n_freqs))

            parameters, state, chi_square, iteration, comp_time = gf.fit_constrained(
                data=np.ascontiguousarray(data_input, dtype=np.float32),
                user_info=np.ascontiguousarray(frange, dtype=np.float32),
                constraints=np.ascontiguousarray(constraint_pixel, dtype=np.float32),
                constraint_types=constraint_types,
                initial_parameters=np.ascontiguousarray(
                    self._initial_parameters[:, i].reshape(-1, len(self._fit_model.parameters_unique)), dtype=np.float32),
                weights=None,
                model_id=self._fit_model.model_id,
                max_number_iterations=1000,
                tolerance=1e-10
            )
            self.parameters[:, i] = parameters.reshape((n_pol, n_pix, -1))
            states[:, i] = state.reshape((n_pol, n_pix, -1))
            chi_squares[:, i] = chi_square.reshape((n_pol, n_pix, -1))
            iterations[:, i] = iteration.reshape((n_pol, n_pix, -1))

            LOG.info(f"Fitting finished in {comp_time:2.1f} seconds.")
            LOG.debug(f"Nr of iterations: {iterations[:, i]}")
        return states, chi_squares, iterations, iterations

    def calculate_b_field(self, bz: bool = True):
        """Calculate the B111 field and optionally the Bz-field."""
        # resonance
        mean_resonance = (self.parameters[:, 1, :, 0] - self.parameters[:, 0, :, 0]) / 2
        b111_remanence = ((mean_resonance[1] - mean_resonance[0]) / qcon.GAMMA / 2).reshape(self.light_image.shape)
        b111_induced = ((mean_resonance[1] + mean_resonance[0]) / qcon.GAMMA / 2).reshape(self.light_image.shape)

        if bz:
            return self.b111_to_bxyz(b111_remanence)
        else:
            return b111_remanence, b111_induced

    def b111_to_bxyz(self, bmap: NDArray, pixel_size: float = 1.2e-06,
                     rotation_angle: float = 0, direction_vector = None
                     ) -> NDArray:
        """
        Convert a map measured along the direction u to a Bz map of the sample.

        Args:
            bmap: 2D array
                The map to be converted
            pixel_size: float
                The size of the pixel in the map in m
            rotation_angle: float
                The rotation of the diamond lattice axes around z-axis
            direction_vector:
                The direction of the 111 axis with respect to the QDM measurement frame. Default (180°, 35.3°)


        Returns:
            2D array
                The converted map
        """

        unit_vector = self.get_unit_vector(rotation_angle, direction_vector)

        ypix, xpix = bmap.shape
        step_size = 1 / pixel_size

        # these freq. coordinates match the fft algorithm
        fx = (
                np.concatenate([np.arange(0, xpix / 2, 1), np.arange(-xpix / 2, 0, 1)])
                * step_size
                / xpix
        )
        fy = (
                np.concatenate([np.arange(0, ypix / 2, 1), np.arange(-ypix / 2, 0, 1)])
                * step_size
                / ypix
        )

        fgrid_x, fgrid_y = np.meshgrid(fx + 1e-30, fy + 1e-30)

        kx = 2 * np.pi * fgrid_x
        ky = 2 * np.pi * fgrid_y
        k = np.sqrt(kx ** 2 + ky ** 2)

        e = np.fft.fft2(bmap)

        x_filter = -1j * kx / k
        y_filter = -1j * ky / k
        z_filter = k / (
                unit_vector[2] * k - unit_vector[1] * 1j * ky - unit_vector[0] * 1j * kx
        )  # calculate the filter frequency response associated with the x component

        map_z = np.fft.ifft2(e * z_filter)
        map_x = np.fft.ifft2(e * z_filter * x_filter)
        map_y = np.fft.ifft2(e * z_filter * y_filter)

        return np.stack([map_x.real, map_y.real, map_z.real])

    def get_unit_vector(self,
            rotation_angle: float, direction_vector = None
                        ) -> NDArray:
        """
        Get the unit vector of the sample in the lab frame.

        Args:
            rotation_angle: float
                The rotation of the diamond lattice axes around z-axis
            direction_vector:
                The direction of the 111 axis with respect to the QDM measurement frame. Default (180°, 35.3°)

        Returns:
            A normalized unit vector in the instrument frame
        """

        if direction_vector is None:
            direction_vector = np.array([0, np.sqrt(2 / 3), np.sqrt(1 / 3)])

        LOG.info(
            f"Getting unit vector from rotation angle {rotation_angle} along direction vector {direction_vector}"
        )

        alpha = np.rad2deg(rotation_angle)
        rotation_matrix = [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]

        unit_vector = np.matmul(rotation_matrix, direction_vector)
        unit_vector /= np.linalg.norm(unit_vector)
        return unit_vector

    def __str__(self) -> str:
        """Return a string representation of the Measurement object.

        Returns:
            str: A human-readable string representation of the Measurement.
        """
        return (f"Measurement(odmr={self.odmr}, "
                f"output_directory='{self.output_directory}', "
                f"pixel_spacing={self.pixel_spacing} m)")

    def __repr__(self) -> str:
        """Return a developer string representation of the Measurement object.

        Returns:
            str: A detailed string representation for debugging and development.
        """
        return (f"Measurement(odmr={self.odmr!r}, "
                f"light_image.shape={self.light_image.shape}, "
                f"laser_image.shape={self.laser_image.shape}, "
                f"output_directory='{self.output_directory}', "
                f"pixel_spacing={self.pixel_spacing})")


if __name__ == '__main__':
    import numpy as np

    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor, FluorescenceCorrectionProcessor

    LOG.setLevel(logging.DEBUG)
    # User-friendly initialization with proper paths
    data_folder = "/media/mike/data/Dropbox/FOV18x"
    loader = MatlabLoader(data_folder=data_folder)
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.processor_manager.add_processor(FluorescenceCorrectionProcessor())
    odmr.process_data()

    # Create dummy image data for testing
    dummy_light = np.ones((10, 10))
    dummy_laser = np.ones((10, 10))

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'output')
    os.makedirs(output_dir, exist_ok=True)

    measure = Measurement(
        odmr,
        dummy_light,
        dummy_laser,
        output_dir,
    )

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('QtAgg')
    plt.imshow(measure.odmr.processed_data.data[0,0,:,0].reshape(measure.odmr.processed_data.scan_dimensions))
    plt.show()