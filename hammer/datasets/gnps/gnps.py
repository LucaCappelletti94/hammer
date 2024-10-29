"""Submodule providing the GNPS dataset labelled according to our latest NP model."""

from typing import Iterator, Optional, Tuple, Dict, List
import os
from glob import glob
import pickle
import compress_json
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering import normalize_intensities, default_filters
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
    "Carotenoids (C40, β-κ)"
]


class GNPSDataset(Dataset):
    """Class defining the harmonized NPC dataset."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        maximal_number_of_molecules: Optional[int] = None,
        directory: str = "datasets/gnps",
        verbose: bool = True,
    ):
        """Initialize the NPC dataset."""
        super().__init__(
            random_state=random_state,
            maximal_number_of_molecules=maximal_number_of_molecules,
            verbose=verbose,
        )
        self._layered_dag = NPCDAG()
        self._directory = directory
        self._number_of_samples = 0
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
                if (spectrum_labels < 0.75).all():
                    found_uncertain = True
                    break
                if all(
                    label_name in RARE_TERMS
                    for label_name, prediction in spectrum_labels.items()
                    if prediction > 0.75
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
                    if prediction >= 0.75:
                        labels[self.layered_dag().node_id(label_name)] = 1

            spectrum = default_filters(spectrum)
            spectrum = normalize_intensities(spectrum)
            self._number_of_samples += 1

            self._spectra.append(spectrum)
            self._labels.append(labels)

        with open("gnps_spectra.pkl", "wb") as f:
            pickle.dump(self._spectra, f)

        with open("gnps_labels.pkl", "wb") as f:
            pickle.dump(self._labels, f)

        return zip(self._spectra, self._labels)
