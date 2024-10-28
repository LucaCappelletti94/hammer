"""Submodule providing a scikit-learn-like Scaler for matchms Spectrum objects."""

from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from matchms import Spectrum
from matchms.filtering import normalize_intensities, default_filters
from tqdm.auto import tqdm
import numpy as np
import numpy.typing as npt


class RobustSpectraScaler(BaseEstimator, TransformerMixin):
    """Scaler for matchms Spectrum objects robust to outlier peaks."""

    def __init__(
        self, bins: int = 1000, verbose: bool = True, dtype: npt.DTypeLike = np.float32
    ):
        """Initialize the scaler.

        Parameters
        ----------
        bins : int = 1000
            Number of equipopulated bins to use for the m/z axis.
        verbose : bool = True
            Whether to display progress bars.
        dtype : npt.DTypeLike = np.float32
            Data type to use for the transformed data.
        """
        self.bins: int = bins
        self.dtype: npt.DTypeLike = dtype
        self.verbose: bool = verbose
        self.scaler = RobustScaler()

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
        # We collect all of the m/z values from the spectra,
        # so we can fit the RobustScaler on them so we may avoid
        # skewed buckets due to outliers.
        mz_values = []
        for spectrum in tqdm(
            X,
            desc="Fitting spectra scaler",
            unit="spectrum",
            disable=not self.verbose,
            leave=False,
            dynamic_ncols=True,
        ):
            spectrum = normalize_intensities(default_filters(spectrum))
            mz_values.extend(spectrum.peaks.mz)

        # We fit the RobustScaler on the m/z values.
        self.scaler.fit(np.array(mz_values).reshape(-1, 1))

    # pylint: disable=unused-argument, invalid-name
    def transform(
        self, X: List[Spectrum], y=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform the provided data.

        Parameters
        ----------
        X
            The data to transform.
        y
            Included for compatibility with scikit-learn API.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The transformed data, including the binned and averaged
            m/z values and the bin averaged intensities, which are
            also normalized using matchms default filters and normalizers.
            The third array contains the normalized populations of the bins.
        """

        # We allocate the two arrays to store the binned and averaged
        # m/z values and the bin averaged intensities.
        binned_mz = np.zeros((len(X), self.bins), dtype=self.dtype)
        binned_intensity = np.zeros((len(X), self.bins), dtype=self.dtype)
        normalized_populations = np.zeros((len(X), self.bins), dtype=self.dtype)

        for i, spectrum in enumerate(
            tqdm(
                X,
                desc="Transforming spectra",
                unit="spectrum",
                leave=False,
                dynamic_ncols=True,
                disable=not self.verbose,
            )
        ):
            spectrum = normalize_intensities(default_filters(spectrum))

            # We iterate the mz and intensity values of the spectrum
            # and bin them according to the fitted RobustScaler.
            normalized_mzs = self.scaler.transform(spectrum.peaks.mz.reshape(-1, 1))
            for mz, normalized_mz, intensity in zip(
                spectrum.peaks.mz, normalized_mzs, spectrum.peaks.intensities
            ):
                bin_index: int = int(np.floor(normalized_mz * self.bins))

                if bin_index < 0:
                    bin_index = 0
                if bin_index >= self.bins:
                    bin_index = self.bins - 1

                binned_mz[i, bin_index] += mz
                binned_intensity[i, bin_index] += intensity
                normalized_populations[i, bin_index] += 1

            # We normalize the values in the bins by the counts
            # in the bins, ignoring empty bins.
            for j in range(self.bins):
                if normalized_populations[i, j] == 0:
                    continue
                binned_mz[i, j] /= normalized_populations[i, j]
                binned_intensity[i, j] /= normalized_populations[i, j]

            # Next we normalize the intensities in the bins.
            normalized_populations[i] /= np.max(normalized_populations[i])

        return binned_mz, binned_intensity, normalized_populations

    # pylint: disable=unused-argument
    def fit_transform(
        self, X: List[Spectrum], y=None, **fit_params
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the scaler on the provided data and transform it.

        Parameters
        ----------
        X
            The data to fit the scaler on and transform.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The transformed data, including the binned and averaged
            m/z values and the bin averaged intensities, which are
            also normalized using matchms default filters and normalizers.
            The third array contains the normalized populations of the bins.
        """
        self.fit(X)
        return self.transform(X)
