"""Submodule providing the GNPS dataset labelled according to our latest NP model."""

from typing import Iterator, Optional, Tuple, Dict, List
import os
from collections import Counter
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
    add_parent_mass,
    require_minimum_number_of_peaks,
)
from matchms.importing import load_from_mgf
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import (  # pylint: disable=import-error, no-name-in-module
    MolFromSmiles,
    MolToSmiles,
)
from downloaders import BaseDownloader
from tqdm.auto import tqdm
from hammer.dags import NPCDAG
from hammer.datasets.dataset import Dataset

RARE_TERMS: List[str] = [
    "Chamigrane sesquiterpenoids",
    "RiPPs-Cyanobactins",
    "Aphidicolane diterpenoids",
    "Monocarbocyclic sesterterpenoids",
    "Lactam bearing macrolide lactones",
    "RiPPs-Lasso peptides",
    "Bourbonane sesquiterpenoids",
    "Laurane sesquiterpenoids",
    "Longifolane sesquiterpenoids",
    "Amphilectane diterpenoids",
    "Oxa-Bridged Macrolides",
    "Ladder polyethers",
    "Furanoabietane diterpenoids",
    "Cycloapotirucallane triterpenoids",
    "Fenchane monoterpenoids",
    "Simple oxindole alkaloids",
    "Prezizaane sesquiterpenoids",
    "Macrolide lactams",
    "Cyclopiazonic acid-tpye tetramate alkaloids",
    "Casbane diterpenoids",
    "Macrotetrolides",
    "Acutumine alkaloids",
    "Tricyclic guanidine alkaloids",
    "Spongiane diterpenoids",
    "Secogermacrane sesquiterpenoids",
    "Carbapenems",
    "Phenoxazine alkaloids",
    "Acorane sesquiterpenoids",
    "Minor lignans",
    "Coumaronochromones",
    "3-Decalinoyltetramic acids",
    "Phloroglucinol-terpene hybrids",
    "Phytane diterpenoids",
    "Isolactarane sesquiterpenoids",
    "Santalane sesquiterpenoids",
    "Segetane diterpenoids",
    "Valerenane sesquiterpenoids",
    "Carabrane sesquiterpenoids",
    "Tremulane sesquiterpenoids",
    "Platensimycin and Platencins",
    "Longipinane sesquiterpenoids",
    "Thiodiketopiperazine alkaloids",
    "Pepluane diterpenoids",
    "Noreremophilane sesquiterpenoids",
    "Iphionane sesquiterpenoids",
    "Zizaane sesquiterpenoids",
    "Campherenane sesquiterpenoids",
    "Guanacastane diterpenoids",
    "Icetexane diterpenoids",
    "Trachylobane diterpenoids",
    "Serratane triterpenoids",
    "Apotirucallane triterpenoids",
    "Cheilanthane sesterterpenoids",
    "Epoxy fatty acids",
    "Heterocyclic fatty acids",
    "Valparane diterpenoids",
    "Thujane monoterpenoids",
    "Tetracyclic diterpenoids",
    "Vitamin D2 and derivatives",
    "Strobilurins and derivatives",
    "Long-Chain Bicyclic Phosphotriester",
    "RiPPs-Lanthipeptides",
    "Carotenoids (C40, β-κ)",
    "Simple cyclic polyketides",
    "Friedelane triterpenoids",
    "Cedrane and Isocedrane sesquiterpenoids",
    "Aristolane sesquiterpenoids",
    "Fusicoccane diterpenoids",
    "Cycloeudesmane sesquiterpenoids",
    "Coumarinolignans",
    "Atisane diterpenoids",
    "Oplopane sesquiterpenoids",
    "Shionane triterpenoids",
    "Multiflorane triterpenoids",
    "Patchoulane sesquiterpenoids",
    "Tropolones and derivatives (PKS)",
    "Illudane sesquiterpenoids",
]


class GNPSDataset(Dataset):
    """Class defining the harmonized NPC dataset."""

    def __init__(
        self,
        random_state: int = 1_532_791_432,
        maximal_number_of_molecules: Optional[int] = None,
        threshold: float = 0.75,
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

    def _build_smiles_map(
        self, spectra: List[Spectrum]
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        """Build a map from SMILES to indices in the dataset."""
        unique_smiles_map: Dict[str, List[int]] = {}
        unique_smiles: List[str] = []

        for i, spectrum in enumerate(
            tqdm(
                spectra,
                desc="Building smiles map",
                leave=False,
                dynamic_ncols=True,
                disable=not self._verbose,
            )
        ):
            smiles = spectrum.get("smiles")

            if smiles not in unique_smiles_map:
                # We make sure that the smiles are canonicalized.
                mol: Mol = MolFromSmiles(smiles)

                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")

                smiles = MolToSmiles(mol)

            if smiles not in unique_smiles_map:
                unique_smiles_map[smiles] = []
                unique_smiles.append(smiles)

            unique_smiles_map[smiles].append(i)

        assert len(unique_smiles) == len(unique_smiles_map), (
            f"Length of unique SMILES ({len(unique_smiles)}) "
            f"does not match length of unique SMILES map ({len(unique_smiles_map)})"
        )

        return unique_smiles, unique_smiles_map

    def primary_split(
        self, test_size: float
    ) -> Tuple[Tuple[List[Spectrum], np.ndarray], Tuple[List[Spectrum], np.ndarray]]:
        """Split the dataset into training and test sets."""

        samples, labels = self.all_samples()

        # Having now identified the least common labels in the leaf layer, we
        # can use these to split the dataset into training and test sets in a
        # stratified manner.
        unique_smiles, unique_smiles_map = self._build_smiles_map(samples)

        counters: np.ndarray = np.zeros(self.layered_dag().number_of_nodes(), dtype=int)

        for smiles in unique_smiles:
            first_spectrum_index = unique_smiles_map[smiles][0]
            counters += labels[first_spectrum_index].astype(int)

        # We identify the least common labels.
        least_common_labels: np.ndarray = np.zeros(len(unique_smiles))
        number_of_pathways: int = len(self.layered_dag().nodes_in_layer("pathways"))
        number_of_superclasses: int = len(
            self.layered_dag().nodes_in_layer("superclasses")
        )
        for i, smiles in enumerate(unique_smiles):
            first_spectrum_index = unique_smiles_map[smiles][0]
            classes = []
            for k in np.argwhere(labels[first_spectrum_index] == 1):
                # We only consider the least common labels in the leaf layer.
                if k >= number_of_pathways + number_of_superclasses:
                    classes.append(k)
            least_common_labels[i] = min(
                classes,
                key=lambda k: counters[k],
            )

        # We make sure that all labels appear at least two times in the dataset.
        nodes = self.layered_dag().nodes()
        counter: Counter = Counter(
            [
                nodes[int(least_common_label)]
                for least_common_label in least_common_labels
            ]
        )

        if any(count < 3 for count in counter.values()):
            rare_terms = [
                (node_name, count) for node_name, count in counter.items() if count < 3
            ]
            raise ValueError(
                f"Terms appear less than two times in the dataset: {rare_terms}"
            )

        train_smiles_indices, test_smiles_indices = train_test_split(
            np.arange(len(unique_smiles)),
            stratify=least_common_labels,
            test_size=test_size,
            random_state=self._random_state,
        )

        return (
            (
                [
                    samples[spectra_index]
                    for smiles_index in train_smiles_indices
                    for spectra_index in unique_smiles_map[unique_smiles[smiles_index]]
                ],
                np.vstack(
                    [
                        labels[spectra_index]
                        for smiles_index in train_smiles_indices
                        for spectra_index in unique_smiles_map[
                            unique_smiles[smiles_index]
                        ]
                    ]
                ),
            ),
            (
                [
                    samples[spectra_index]
                    for smiles_index in test_smiles_indices
                    for spectra_index in unique_smiles_map[unique_smiles[smiles_index]]
                ],
                np.vstack(
                    [
                        labels[spectra_index]
                        for smiles_index in test_smiles_indices
                        for spectra_index in unique_smiles_map[
                            unique_smiles[smiles_index]
                        ]
                    ]
                ),
            ),
        )

    def train_split(
        self, number_of_holdouts: int, validation_size: float, test_size: float
    ) -> Iterator[
        Tuple[
            Tuple[List[Spectrum], np.ndarray],
            Tuple[List[Spectrum], np.ndarray],
        ]
    ]:
        """Split the dataset into training and test sets."""
        (train_samples, train_labels), (_test_samples, _test_labels) = (
            self.primary_split(
                test_size=test_size,
            )
        )
        unique_smiles, unique_smiles_map = self._build_smiles_map(train_samples)

        counters: np.ndarray = np.zeros(self.layered_dag().number_of_nodes(), dtype=int)

        for smiles in unique_smiles:
            first_spectrum_index = unique_smiles_map[smiles][0]
            counters += train_labels[first_spectrum_index].astype(int)

        least_common_labels: np.ndarray = np.zeros(len(unique_smiles))

        number_of_pathways: int = len(self.layered_dag().nodes_in_layer("pathways"))
        number_of_superclasses: int = len(
            self.layered_dag().nodes_in_layer("superclasses")
        )
        for i, smiles in enumerate(unique_smiles):
            first_spectrum_index = unique_smiles_map[smiles][0]
            classes = []
            for k in np.argwhere(train_labels[first_spectrum_index] == 1):
                # We only consider the least common labels in the leaf layer.
                if k >= number_of_pathways + number_of_superclasses:
                    classes.append(k)

            least_common_labels[i] = min(
                classes,
                key=lambda k: counters[k],
            )

        node = self.layered_dag().nodes()
        counter: Counter = Counter(
            [
                node[int(least_common_label)]
                for least_common_label in least_common_labels
            ]
        )

        if any(count < 2 for count in counter.values()):
            rare_terms = [
                (node_name, count) for node_name, count in counter.items() if count < 2
            ]
            raise ValueError(
                f"Terms appear less than two times in the dataset: {rare_terms}"
            )

        splitter: StratifiedShuffleSplit = StratifiedShuffleSplit(
            n_splits=number_of_holdouts,
            test_size=validation_size,
            random_state=self._random_state,
        )

        for train_indices, validation_indices in tqdm(
            splitter.split(np.arange(len(unique_smiles)), least_common_labels),
            total=number_of_holdouts,
            desc="Holdouts",
            leave=False,
            dynamic_ncols=True,
            disable=not self._verbose,
            unit="holdout",
        ):
            yield (
                (
                    [
                        train_samples[spectra_index]
                        for smiles_index in train_indices
                        for spectra_index in unique_smiles_map[
                            unique_smiles[smiles_index]
                        ]
                    ],
                    np.vstack(
                        [
                            train_labels[spectra_index]
                            for smiles_index in train_indices
                            for spectra_index in unique_smiles_map[
                                unique_smiles[smiles_index]
                            ]
                        ]
                    ),
                ),
                (
                    [
                        train_samples[spectra_index]
                        for smiles_index in validation_indices
                        for spectra_index in unique_smiles_map[
                            unique_smiles[smiles_index]
                        ]
                    ],
                    np.vstack(
                        [
                            train_labels[spectra_index]
                            for smiles_index in validation_indices
                            for spectra_index in unique_smiles_map[
                                unique_smiles[smiles_index]
                            ]
                        ]
                    ),
                ),
            )
