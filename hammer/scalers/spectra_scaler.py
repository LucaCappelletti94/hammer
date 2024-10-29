"""Submodule providing a scikit-learn-like Scaler for matchms Spectrum objects."""

from typing import List, Optional
from multiprocessing import cpu_count, Pool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from matchms import Spectrum
from tqdm.auto import tqdm
import numpy as np
import numpy.typing as npt


class RobustSpectraScaler(BaseEstimator, TransformerMixin):
    """Scaler for matchms Spectrum objects robust to outlier peaks."""

    def __init__(
        self,
        bins: int = 1000,
        verbose: bool = True,
        dtype: npt.DTypeLike = np.float32,
        n_jobs: Optional[int] = 1,
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
        self.bin_ends: np.ndarray = np.empty(self.bins, dtype=dtype)
        self.dtype: npt.DTypeLike = dtype
        self.verbose: bool = verbose
        self.scaler = RobustScaler()
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
            mz_values.extend(spectrum.peaks.mz)

        # We fit the RobustScaler on the m/z values.
        normalized_mz_values = self.scaler.fit_transform(
            np.array(mz_values).reshape(-1, 1)
        ).flatten()

        # We sort the normalized m/z values to find the bin boundaries.
        sorted_normalized_mz_values = np.sort(normalized_mz_values)
        sorted_normalized_mz_values[sorted_normalized_mz_values < 0] = 0
        sorted_normalized_mz_values[sorted_normalized_mz_values > 1] = 1

        # We split the bins into chunks of equal size.
        for i in range(self.bins - 1):
            self.bin_ends[i] = sorted_normalized_mz_values[
                int((i + 1) * len(sorted_normalized_mz_values) / self.bins)
            ]

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
        binned = np.zeros((self.bins, 3), dtype=self.dtype)

        # We iterate the mz and intensity values of the spectrum
        # and bin them according to the fitted RobustScaler.
        normalized_mzs = self.scaler.transform(spectrum.peaks.mz.reshape(-1, 1))

        for normalized_mz, intensity in zip(normalized_mzs, spectrum.peaks.intensities):
            clipped_normalized_mz = np.clip(normalized_mz, 0, 1)

            bin_index = np.searchsorted(self.bin_ends, clipped_normalized_mz) - 1

            binned[bin_index, 0] += normalized_mz
            binned[bin_index, 1] += intensity
            binned[bin_index, 2] += 1

        # We normalize the values in the bins by the counts
        # in the bins, ignoring empty bins.
        for j in range(self.bins):
            if binned[j, 2] == 0:
                continue
            binned[j, 0] /= binned[j, 2]
            binned[j, 1] /= binned[j, 2]

        # Next we normalize the intensities in the bins.
        binned[:, 2] /= np.max(binned[:, 2])

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
        binned = np.zeros((len(X), self.bins, 3), dtype=self.dtype)

        with Pool(self._n_jobs) as pool:
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
