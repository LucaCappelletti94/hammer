"""Submodule providing a scikit-learn-like Scaler for matchms Spectrum objects."""

from typing import List, Optional
from multiprocessing import cpu_count, Pool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from matchms import Spectrum
from tqdm.auto import tqdm
import numpy as np
import numpy.typing as npt
from hammer.constants import LOG2


class TransposedSpectraScaler(BaseEstimator, TransformerMixin):
    """Scaler for matchms Spectrum objects robust to outlier peaks."""

    def __init__(
        self,
        bins: int = 2048,
        include_losses: bool = False,
        normalize: bool = True,
        normalize_by_parent_mass: bool = False,
        log_intensity: bool = True,
        log_mz: bool = True,
        verbose: bool = True,
        dtype: npt.DTypeLike = np.float32,
        n_jobs: Optional[int] = 1,
    ):
        """Initialize the scaler.

        Parameters
        ----------
        bins : int = 128
            Number of equipopulated bins to use for the m/z axis.
        include_losses : bool = True
            Whether to include losses in the transformation.
        normalize : bool = True
            Whether to normalize the MZ.
        normalize_by_parent_mass : bool = False
            Whether to normalize the MZ by the parent mass.
        log_intensity : bool = True
            Whether to log-transform the intensities.
        log_mz : bool = True
            Whether to log-transform the m/z values.
        verbose : bool = True
            Whether to display progress bars.
        dtype : npt.DTypeLike = np.float32
            Data type to use for the transformed data.
        """
        self.bins: int = bins
        self.dtype: npt.DTypeLike = dtype
        self.include_losses: bool = include_losses
        self.verbose: bool = verbose
        self.normalize: bool = normalize
        self.normalize_by_parent_mass: bool = normalize_by_parent_mass
        self.log_intensity: bool = log_intensity
        self.log_mz: bool = log_mz
        self.peaks_scaler: MinMaxScaler = MinMaxScaler()
        self.losses_scaler: MinMaxScaler = MinMaxScaler()
        if n_jobs is None:
            n_jobs = cpu_count()
        self._n_jobs: int = n_jobs

    # pylint: disable=unused-argument, invalid-name
    def fit(self, X: List[Spectrum], y=None):
        """Fit the scaler on the provided data.

        Parameters
        ----------
        X
            The data to fit the scaler on.
        y
            Included for compatibility with scikit-learn API.
        """
        if self.normalize_by_parent_mass:
            assert all(
                spectrum.get("parent_mass") is not None for spectrum in X
            ), "Parent mass is required for normalizing by parent mass."

        peaks = np.concatenate(
            [
                (
                    spectrum.mz / spectrum.get("parent_mass")
                    if self.normalize_by_parent_mass
                    else spectrum.mz
                )
                for spectrum in tqdm(
                    X,
                    desc="Fitting peaks scaler",
                    unit="spectrum",
                    disable=not self.verbose,
                    leave=False,
                    dynamic_ncols=True,
                )
            ]
        )

        # We fit the scaler on the m/z values.
        self.peaks_scaler.fit(peaks.reshape(-1, 1))

        if self.include_losses:
            losses_mzs = []

            for spectrum in tqdm(
                X,
                desc="Fitting losses scaler",
                unit="spectrum",
                disable=not self.verbose,
                leave=False,
                dynamic_ncols=True,
            ):
                spectrum_losses = spectrum.losses
                if spectrum_losses is None or spectrum_losses.mz.size == 0:
                    continue
                if self.normalize_by_parent_mass:
                    losses_mzs.extend(spectrum_losses.mz / spectrum.get("parent_mass"))
                else:
                    losses_mzs.extend(spectrum_losses.mz)

            # We fit the scaler on the m/z values.
            self.losses_scaler.fit(np.array(losses_mzs).reshape(-1, 1))

    def _transform_spectrum(self, spectrum: Spectrum) -> np.ndarray:
        """Transform the provided spectrum.

        Parameters
        ----------
        spectrum
            The spectrum to transform.

        Returns
        -------
        np.ndarray
            The transformed spectrum.
        """
        # We allocate the two arrays to store the binned and averaged
        # m/z values and the bin averaged intensities.
        if self.include_losses:
            features = 4
        else:
            features = 2

        binned = np.zeros((self.bins, features), dtype=self.dtype)
        population = np.zeros((self.bins, features), dtype=self.dtype)

        # We compute the number of peaks for each bin. In some
        # instances, the number of peaks for each bin may be less
        # than one, and in such cases we spread the intensity of the
        # peak over the number of bins necessary to cover the full
        # width of the peak.

        if self.include_losses:
            spectrum_losses = spectrum.losses
            if spectrum_losses is not None and spectrum_losses.mz.size > 0:
                if self.normalize_by_parent_mass:
                    normalized_loss_mzs = self.losses_scaler.transform(
                        spectrum_losses.mz.reshape(-1, 1) / spectrum.get("parent_mass")
                    ).flatten()
                else:
                    normalized_loss_mzs = self.losses_scaler.transform(
                        spectrum_losses.mz.reshape(-1, 1)
                    ).flatten()

                intensities = spectrum_losses.intensities

                if self.log_intensity:
                    intensities = np.log1p(intensities) / LOG2
                
                if self.log_mz:
                    normalized_loss_mzs = np.log1p(normalized_loss_mzs) / LOG2

                for mz, intensity in zip(
                    normalized_loss_mzs, spectrum_losses.intensities
                ):
                    assert intensity >= 0
                    assert intensity <= 1

                    bin_index = int(np.round(intensity * (self.bins - 1)))
                    binned[bin_index, 2] += mz
                    binned[bin_index, 3] += mz
                    population[bin_index, 1] += 1

        if self.normalize_by_parent_mass:
            normalized_mzs = self.peaks_scaler.transform(
                spectrum.mz.reshape(-1, 1) / spectrum.get("parent_mass")
            ).flatten()
        else:
            normalized_mzs = self.peaks_scaler.transform(
                spectrum.mz.reshape(-1, 1)
            ).flatten()

        intensities = spectrum.intensities

        if self.log_intensity:
            intensities = np.log1p(intensities) / LOG2

        if  self.log_mz:
            normalized_mzs = np.log1p(normalized_mzs) / LOG2

        for mz, intensity in zip(normalized_mzs, spectrum.intensities):
            bin_index = int(np.round(intensity * (self.bins - 1)))
            binned[bin_index, 0] += mz
            binned[bin_index, 1] += intensity
            population[bin_index, 0] += 1

        if self.normalize:
            for i in range(self.bins):
                if population[i, 0] > 0:
                    binned[i, 0] /= population[i, 0]
                    binned[i, 1] /= population[i, 0]
                if self.include_losses and population[i, 1] > 0:
                    binned[i, 2] /= population[i, 1]
                    binned[i, 3] /= population[i, 1]

        return binned

    # pylint: disable=unused-argument, invalid-name
    def transform(self, X: List[Spectrum], y=None) -> np.ndarray:
        """Transform the provided data.

        Parameters
        ----------
        X
            The data to transform.
        y
            Included for compatibility with scikit-learn API.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        # We allocate the two arrays to store the binned and averaged
        # m/z values and the bin averaged intensities.
        if self.include_losses:
            features = 4
        else:
            features = 2

        # We allocate the two arrays to store the binned and averaged
        # m/z values and the bin averaged intensities.
        binned = np.zeros((len(X), self.bins, features), dtype=self.dtype)

        if len(X) < self._n_jobs:
            n_jobs = 1
        else:
            n_jobs = self._n_jobs

        if n_jobs == 1:
            for i, spectrum in enumerate(
                tqdm(
                    X,
                    desc="Transforming spectra",
                    unit="spectrum",
                    disable=not self.verbose,
                    leave=False,
                    dynamic_ncols=True,
                )
            ):
                binned[i] = self._transform_spectrum(spectrum)
        else:
            with Pool(n_jobs) as pool:
                for i, transformed_spectrum in enumerate(
                    tqdm(
                        pool.imap(self._transform_spectrum, X),
                        desc="Transforming spectra",
                        unit="spectrum",
                        total=len(X),
                        disable=not self.verbose,
                        leave=False,
                        dynamic_ncols=True,
                    )
                ):
                    binned[i] = transformed_spectrum

        return binned

    # pylint: disable=unused-argument
    def fit_transform(self, X: List[Spectrum], y=None, **fit_params) -> np.ndarray:
        """Fit the scaler on the provided data and transform it.

        Parameters
        ----------
        X
            The data to fit the scaler on and transform.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        self.fit(X)
        return self.transform(X)
