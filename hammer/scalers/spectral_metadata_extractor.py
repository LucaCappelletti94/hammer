"""Submodule providing a class for extracting metadata from spectral data files.

Features being extracted
------------------------

The following features are extracted from the spectral data:

- Charge (scalar): The charge of the precursor ion.
- Parent mass (scalar): The mass of the precursor ion.
- Precursor MZ (scalar): The m/z value of the precursor ion.
- Number of peaks (scalar): The number of peaks in the spectrum.
- Number of losses (scalar): The number of losses in the spectrum.
- Various statistics regarding MZ (vector).
- Mass Analyzer (categorical): The type of mass analyzer used to acquire the spectrum.
- Adduct (categorical + scalar): The adduct used to acquire the spectrum.
    For the time being, the adduct is represented as a one-hot encoded vector.
    TODO: Extract more meaningful features from the adduct.
- Ionization mode (categorical): The ionization mode used to acquire the spectrum.
- Mass spec Ionization mode (categorical)


All of the scalar features are normalized using a RobustScaler.
When a feature is missing, a zero value is used.
The categorical feature is one-hot encoded.
When the feature is missing, a zero vector is used.
"""

from typing import List, Optional, Union, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from matchms import Spectrum, Fragments
from tqdm.auto import tqdm
import numpy as np


def normalize_charge(charge: Optional[Union[int, float, str]]) -> float:
    """Normalize the charge of a precursor ion.

    Parameters
    ----------
    charge : str
        The charge of the precursor ion.

    Returns
    -------
    float
        The normalized charge of the precursor ion.
    """
    if charge is None:
        return 0.0

    if isinstance(charge, (int, float)):
        return float(charge)

    contains_minus: bool = "-" in charge

    normalized_charge: str = charge.replace("+", "").replace("-", "")

    if contains_minus:
        return -float(normalized_charge)

    return float(normalized_charge)


def normalize_adduct(adduct: Optional[str]) -> str:
    """Normalize the adduct of a precursor ion.

    Parameters
    ----------
    adduct : str
        The adduct of the precursor ion.

    Returns
    -------
    str
        The normalized adduct of the precursor ion.
    """
    if adduct is None:
        return "UNKNOWN"

    if adduct.endswith("]1+"):
        adduct = adduct[:-2] + "+"

    if adduct.endswith("]1-"):
        adduct = adduct[:-2] + "-"

    return adduct


def extract_mz_features(spectrum: Spectrum) -> np.ndarray:
    """Extract MZ features from a spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum to extract MZ features from.

    Returns
    -------
    np.ndarray
        The extracted MZ features.
    """
    return np.array(
        [
            spectrum.get("precursor_mz", spectrum.get("parent_mass", 0.0)),
            spectrum.get("parent_mass", spectrum.get("precursor_mz", 0.0)),
            spectrum.peaks.mz.mean(),
            np.median(spectrum.peaks.mz),
            spectrum.peaks.mz.std(),
            spectrum.peaks.mz.min(),
            spectrum.peaks.mz.max(),
        ]
    )


class SpectralMetadataExtractor(BaseEstimator, TransformerMixin):
    """A metadata extractor for matchms Spectrum objects."""

    def __init__(
        self,
        include_adducts: bool = False,
        verbose: bool = True,
        n_jobs: Optional[int] = 1,
    ):
        """Initialize the metadata extractor.

        Parameters
        ----------
        include_adducts : bool = True
            Whether to include adducts in the extracted metadata.
        verbose : bool = True
            Whether to display progress bars.
        n_jobs : Optional[int] = 1
            Number of jobs to run in parallel.
        """
        self.verbose: bool = verbose
        self.n_jobs: Optional[int] = n_jobs

        self._charge_scaler: RobustScaler = RobustScaler()
        self._mz_scaler: RobustScaler = RobustScaler()
        self._n_peaks_scaler: RobustScaler = RobustScaler()
        self._n_losses_scaler: RobustScaler = RobustScaler()
        self._mass_analyzer_encoder: OneHotEncoder = OneHotEncoder(
            dtype=np.uint8,
            handle_unknown="ignore",
        )
        self._mass_spec_ionization_mode_encoder: OneHotEncoder = OneHotEncoder(
            dtype=np.uint8,
            handle_unknown="ignore",
        )
        self._ionization_mode_encoder: OneHotEncoder = OneHotEncoder(
            dtype=np.uint8,
            handle_unknown="ignore",
        )
        if include_adducts:
            self._adduct_encoder: Optional[OneHotEncoder] = OneHotEncoder(
                dtype=np.uint8,
                handle_unknown="ignore",
            )
        else:
            self._adduct_encoder = None

    # pylint: disable=invalid-name
    # pylint: disable=unused-argument
    def fit(self, X: List[Spectrum], y=None):
        """Fit the metadata extractor.

        Parameters
        ----------
        X : List[Spectrum]
            The list of spectra to fit the metadata extractor to.
        y : None
            Ignored.

        Returns
        -------
        self
            The fitted metadata extractor.
        """
        property_values: np.ndarray = np.fromiter(
            (
                normalize_charge(spectrum.get("charge"))
                for spectrum in tqdm(
                    X,
                    desc="Extracting charge",
                    disable=not self.verbose,
                    total=len(X),
                    leave=False,
                    dynamic_ncols=True,
                )
                if spectrum.get("charge") is not None
            ),
            dtype=np.float32,
        )
        if property_values.size == 0:
            raise ValueError("No charge found in the spectra.")
        self._charge_scaler.fit(property_values.reshape(-1, 1))

        if self._adduct_encoder is not None:
            adducts: List[str] = [
                normalize_adduct(spectrum.get("adduct"))
                for spectrum in tqdm(
                    X,
                    desc="Extracting adducts",
                    disable=not self.verbose,
                    total=len(X),
                    leave=False,
                    dynamic_ncols=True,
                )
            ]

            if len(adducts) == 0:
                raise ValueError("No adducts found in the spectra.")

            self._adduct_encoder.fit(np.array(adducts).reshape(-1, 1))

        number_of_peaks: np.ndarray = np.zeros(len(X))
        number_of_losses: np.ndarray = np.zeros(len(X))
        mz_features: np.ndarray = np.zeros((len(X), 7))
        mass_analyzers: List[str] = []
        ionization_modes: List[str] = []
        mass_spec_ionization_modes: List[str] = []

        for i, spectrum in tqdm(
            enumerate(X),
            desc="Extract spectral features",
            disable=not self.verbose,
            total=len(X),
            leave=False,
            dynamic_ncols=True,
        ):
            spectrum_losses: Optional[Fragments] = spectrum.losses
            if spectrum_losses is not None:
                number_of_losses[i] = spectrum_losses.mz.size
            else:
                number_of_losses[i] = 0

            mass_analyzer: str = spectrum.get("ms_mass_analyzer", "UNKNOWN")
            mass_analyzers.append(mass_analyzer)
            ionization_mode: str = spectrum.get("ionmode", "UNKNOWN")
            ionization_modes.append(ionization_mode)
            mass_spec_ionization_mode: str = spectrum.get("ms_ionisation", "UNKNOWN")
            mass_spec_ionization_modes.append(mass_spec_ionization_mode)
            number_of_peaks[i] = spectrum.peaks.mz.size

            mz_features[i] = extract_mz_features(spectrum)

        self._n_peaks_scaler.fit(number_of_peaks.reshape(-1, 1))
        self._n_losses_scaler.fit(number_of_losses.reshape(-1, 1))
        self._mz_scaler.fit(mz_features)
        self._mass_analyzer_encoder.fit(np.array(mass_analyzers).reshape(-1, 1))
        self._ionization_mode_encoder.fit(np.array(ionization_modes).reshape(-1, 1))
        self._mass_spec_ionization_mode_encoder.fit(
            np.array(mass_spec_ionization_modes).reshape(-1, 1)
        )

    def transform(self, X: List[Spectrum]) -> Dict[str, np.ndarray]:
        """Transform the input spectra into metadata features.

        Parameters
        ----------
        X : List[Spectrum]
            The list of spectra to transform.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the extracted metadata features.
        """
        adducts: List[str] = []
        number_of_peaks: np.ndarray = np.zeros(len(X))
        number_of_losses: np.ndarray = np.zeros(len(X))
        charges: np.ndarray = np.zeros(len(X))
        mz_features: np.ndarray = np.zeros((len(X), 7))
        mass_analyzers: List[str] = []
        ionization_modes: List[str] = []
        mass_spec_ionization_modes: List[str] = []

        for i, spectrum in tqdm(
            enumerate(X),
            desc="Extracting metadata",
            disable=not self.verbose,
            total=len(X),
            leave=False,
            dynamic_ncols=True,
        ):
            spectrum_losses: Optional[Fragments] = spectrum.losses
            if spectrum_losses is not None:
                number_of_losses[i] = spectrum_losses.mz.size

            charges[i] = normalize_charge(spectrum.get("charge"))
            number_of_peaks[i] = spectrum.peaks.mz.size
            mz_features[i] = extract_mz_features(spectrum)
            mass_analyzer: str = spectrum.get("ms_mass_analyzer", "UNKNOWN")
            mass_analyzers.append(mass_analyzer)
            ionization_mode: str = spectrum.get("ionmode", "UNKNOWN")
            ionization_modes.append(ionization_mode)
            mass_spec_ionization_mode: str = spectrum.get("ms_ionisation", "UNKNOWN")
            mass_spec_ionization_modes.append(mass_spec_ionization_mode)
            if self._adduct_encoder is not None:
                adducts.append(normalize_adduct(spectrum.get("adduct")))

        metadata = {
            "charge": self._charge_scaler.transform(charges.reshape(-1, 1)),
            "mz_features": self._mz_scaler.transform(mz_features),
            "n_peaks": self._n_peaks_scaler.transform(number_of_peaks.reshape(-1, 1)),
            "n_losses": self._n_losses_scaler.transform(
                number_of_losses.reshape(-1, 1)
            ),
            "mass_analyzer": self._mass_analyzer_encoder.transform(
                np.array(mass_analyzers).reshape(-1, 1)
            ).toarray(),
            "ionization_mode": self._ionization_mode_encoder.transform(
                np.array(ionization_modes).reshape(-1, 1)
            ).toarray(),
            "mass_spec_ionization_mode": self._mass_spec_ionization_mode_encoder.transform(
                np.array(mass_spec_ionization_modes).reshape(-1, 1)
            ).toarray(),
        }

        if self._adduct_encoder is not None:
            metadata["adduct"] = self._adduct_encoder.transform(
                np.array(adducts).reshape(-1, 1)
            ).toarray()

        return metadata

    def fit_transform(
        self, X: List[Spectrum], y=None, **fit_params
    ) -> Dict[str, np.ndarray]:
        """Fit the metadata extractor and transform the input spectra into metadata features.

        Parameters
        ----------
        X : List[Spectrum]
            The list of spectra to fit the metadata extractor to and transform.
        y : None
            Ignored.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the extracted metadata features.
        """
        self.fit(X)
        return self.transform(X)
