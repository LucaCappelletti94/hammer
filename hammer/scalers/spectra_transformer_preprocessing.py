"""Preprocessing to feed spectra into a transformer layer."""

from typing import List, Optional
from multiprocessing import Pool, cpu_count
from matchms import Spectrum
from matchms.filtering import reduce_to_number_of_peaks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import numpy as np


class SpectralTransformerPreprocessing(BaseEstimator, TransformerMixin):
    """Preprocessing to feed spectra into a transformer layer."""

    def __init__(
        self,
        number_of_peaks: int = 100,
        verbose: bool = True,
        n_jobs: Optional[int] = 1,
    ):
        """Initialize the preprocessor."""
        self.number_of_peaks = number_of_peaks
        self.mz_scaler = MinMaxScaler(feature_range=(0.000001, 1))
        self.intensity_scaler = MinMaxScaler(feature_range=(0.000001, 1))
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs is not None else cpu_count()

    # pylint: disable=unused-argument, invalid-name
    def fit(self, X: List[Spectrum], y=None):
        """Fit the preprocessor."""
        filtered_spectra = [
            reduce_to_number_of_peaks(s, n_max=self.number_of_peaks)
            for s in tqdm(
                X,
                desc="Filtering spectra",
                disable=not self.verbose,
                leave=False,
                dynamic_ncols=True,
            )
        ]

        mzs = np.concatenate(
            [
                s.peaks.mz
                for s in tqdm(
                    filtered_spectra,
                    desc="Fitting MZ scaler",
                    disable=not self.verbose,
                    leave=False,
                    dynamic_ncols=True,
                )
            ]
        )
        self.mz_scaler.fit(mzs.reshape(-1, 1))
        intensities = np.concatenate(
            [
                s.peaks.intensities
                for s in tqdm(
                    filtered_spectra,
                    desc="Fitting intensity scaler",
                    disable=not self.verbose,
                    leave=False,
                    dynamic_ncols=True,
                )
            ]
        )
        self.intensity_scaler.fit(intensities.reshape(-1, 1))

        return self

    def _transform_spectrum(self, spectrum: Spectrum) -> np.ndarray:
        """Transform a single spectrum."""
        reduced_spectrum = reduce_to_number_of_peaks(spectrum, n_max=self.number_of_peaks)
        mz_scaled = self.mz_scaler.transform(reduced_spectrum.peaks.mz.reshape(-1, 1))
        intensity_scaled = self.intensity_scaler.transform(
            reduced_spectrum.peaks.intensities.reshape(-1, 1)
        )

        # We pad the spectrum with zeros if it has less peaks than the desired number.
        if len(mz_scaled) < self.number_of_peaks:
            mz_scaled = np.pad(
                mz_scaled,
                ((0, self.number_of_peaks - len(mz_scaled)), (0, 0)),
                mode="constant",
            )
            intensity_scaled = np.pad(
                intensity_scaled,
                ((0, self.number_of_peaks - len(intensity_scaled)), (0, 0)),
                mode="constant",
            )

        return np.concatenate([mz_scaled, intensity_scaled], axis=1)

    def transform(self, X: List[Spectrum]) -> np.ndarray:
        """Transform the spectra."""
        transformed_spectra = np.zeros((len(X), self.number_of_peaks, 2))

        if len(X) < self.n_jobs:
            n_jobs: Optional[int] = 1
        else:
            n_jobs = self.n_jobs

        if n_jobs == 1:
            for i, spectrum in enumerate(
                tqdm(
                    X,
                    desc="Preprocessing spectra",
                    disable=not self.verbose,
                    leave=False,
                    dynamic_ncols=True,
                )
            ):
                transformed_spectra[i] = self._transform_spectrum(spectrum)
        else:
            with Pool(n_jobs) as pool:
                for i, transformed_spectrum in enumerate(
                    pool.imap(self._transform_spectrum, X)
                ):
                    transformed_spectra[i] = transformed_spectrum

        return transformed_spectra

    def fit_transform(self, X: List[Spectrum], y=None, **fit_params) -> np.ndarray:
        """Fit the preprocessor on the provided data and transform it."""
        self.fit(X)
        return self.transform(X)
