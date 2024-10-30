"""Submodule providing the GNPS dataset labelled according to our latest NP model."""

from typing import Iterator, Optional, Tuple, Dict, List
import os
from glob import glob
import pickle
import compress_json
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering import (
    normalize_intensities,
    default_filters,
    select_by_mz,
    reduce_to_number_of_peaks,
    select_by_relative_intensity,
    add_parent_mass,
    require_minimum_number_of_peaks,
)
from matchms.importing import load_from_mgf
from downloaders import BaseDownloader
from tqdm.auto import tqdm
from hammer.dags import NPCDAG
from hammer.datasets.dataset import Dataset

RARE_TERMS: List[str] = [
    "Amphilectane diterpenoids",
    "Apotirucallane triterpenoids",
    "Bourbonane sesquiterpenoids",
    "Cedrane and Isocedrane sesquiterpenoids",
    "Copaane sesquiterpenoids",
    "Casbane diterpenoids",
    "Cheilanthane sesterterpenoids",
    "Guanacastane diterpenoids",
    "Laurane sesquiterpenoids",
    "Phenoxazine alkaloids",
    "Strobilurins and derivatives",
    "Valparane diterpenoids",
    "Acutumine alkaloids",
    "Glycosyldiacylglycerols",
    "Viscidane diterpenoids",
    "Carotenoids (C40, β-κ)",
    "Chamigrane sesquiterpenoids",
    "Carotenoids (C40, β-Ψ)",
    "Lactam bearing macrolide lactones",
    "Monocarbocyclic sesterterpenoids",
]


class GNPSDataset(Dataset):
    """Class defining the harmonized NPC dataset."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        maximal_number_of_molecules: Optional[int] = None,
        threshold: float = 0.9,
        ionization: str = "both",
        directory: str = "datasets/gnps",
        verbose: bool = True,
    ):
        """Initialize the NPC dataset."""
        super().__init__(
            random_state=random_state,
            maximal_number_of_molecules=maximal_number_of_molecules,
            verbose=verbose,
        )
        if ionization not in ["positive", "negative", "both"]:
            raise ValueError(
                f"Invalid value for ionization: {ionization}. "
                "Valid values are 'positive', 'negative' and 'both'."
            )

        self._layered_dag = NPCDAG()
        self._directory = directory
        self._number_of_samples = 0
        self._threshold = threshold
        self._ionization = ionization
        self._spectra: List[Spectrum] = []
        self._labels: List[np.ndarray] = []

    @staticmethod
    def name() -> str:
        """Return the name of the GNPS dataset."""
        return "GNPS"

    @staticmethod
    def description() -> str:
        """Return the description of the GNPS dataset."""
        return "The GNPS dataset labelled according to our latest NP model."

    def layered_dag(self) -> NPCDAG:
        """Return the Layered DAG for the NPC dataset."""
        return self._layered_dag

    def number_of_samples(self) -> int:
        """Return the number of labeled SMILES in the NPC dataset."""
        return self._number_of_samples

    def iter_samples(self) -> Iterator[Tuple[Spectrum, np.ndarray]]:
        """Return an iterator over the labeled SMILES in the NPC dataset."""

        if os.path.exists("gnps_spectra.pkl") and os.path.exists("gnps_labels.pkl"):
            with open("gnps_spectra.pkl", "rb") as f:
                self._spectra = pickle.load(f)

            with open("gnps_labels.pkl", "rb") as f:
                self._labels = pickle.load(f)

            self._number_of_samples = len(self._spectra)

        if self._spectra and self._labels:
            return zip(self._spectra, self._labels)

        # downloader = BaseDownloader(process_number=1, verbose=self._verbose)
        # downloader.download(
        #     [
        #         "https://external.gnps2.org/processed_gnps_data/matchms.mgf",
        #         "zenodo_url_to_the_labels.json",
        #     ],
        #     [
        #         os.path.join(self._directory, "matchms.mgf.gz"),
        #         os.path.join(self._directory, "labels.json.xz"),
        #     ],
        # )

        mgf_path = "matchms.mgf"
        spectra_labels: Dict[str, pd.DataFrame] = {
            os.path.basename(path).split(".")[0]: pd.read_csv(path, index_col=0)
            for path in glob("matchms_predictions/*.csv")
        }

        number_of_labels = spectra_labels["pathways"].shape[0]

        self._number_of_samples = 0

        self._spectra = []
        self._labels = []

        for i, spectrum in tqdm(
            enumerate(load_from_mgf(mgf_path)),
            desc="Loading GNPS data",
            total=number_of_labels,
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
        ):
            found_uncertain = False
            for label in spectra_labels.values():
                spectrum_labels = label.iloc[i]
                if (spectrum_labels < self._threshold).all():
                    found_uncertain = True
                    break
                if all(
                    label_name in RARE_TERMS
                    for label_name, prediction in spectrum_labels.items()
                    if prediction > self._threshold
                ):
                    found_uncertain = True
                    break
            if found_uncertain:
                continue

            if spectrum.peaks.mz.size == 0:
                continue

            labels = np.zeros(self.layered_dag().number_of_nodes())

            for label in spectra_labels.values():
                spectrum_labels = label.iloc[i]
                assert spectrum.get("smiles") == label.index[i]
                for label_name, prediction in spectrum_labels.items():
                    if label_name in RARE_TERMS:
                        continue
                    if prediction >= self._threshold:
                        labels[self.layered_dag().node_id(label_name)] = 1

            # If any of the intensities is negative we skip the spectrum
            # as I have no clue what to do with those.
            if np.any(spectrum.peaks.intensities < 0):
                continue

            if self._ionization != "both":
                if spectrum.get("ionmode") != self._ionization:
                    continue

            spectrum = default_filters(spectrum)
            spectrum = add_parent_mass(spectrum)
            normalized_spectrum: Optional[Spectrum] = normalize_intensities(spectrum)
            if normalized_spectrum is None:
                continue
            normalized_spectrum = select_by_mz(
                normalized_spectrum, mz_from=0, mz_to=2000
            )
            if (
                require_minimum_number_of_peaks(normalized_spectrum, n_required=5)
                is None
            ):
                continue

            self._number_of_samples += 1

            self._spectra.append(normalized_spectrum)
            self._labels.append(labels)

        with open("gnps_spectra.pkl", "wb") as f:
            pickle.dump(self._spectra, f)

        with open("gnps_labels.pkl", "wb") as f:
            pickle.dump(self._labels, f)

        return zip(self._spectra, self._labels)
