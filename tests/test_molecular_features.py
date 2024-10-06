"""Test module for the computation of molecular features."""

from typing import Dict, Union
from time import time
from multiprocessing import Pool
from tqdm.auto import tqdm
import pandas as pd
from np_classifier.training.molecular_features import compute_features

PARAMETERS = [
    "include_morgan_fingerprint",
    "include_rdkit_fingerprint",
    "include_atom_pair_fingerprint",
    "include_topological_torsion_fingerprint",
    "include_feature_morgan_fingerprint",
    "include_avalon_fingerprint",
    "include_maccs_fingerprint",
    "include_map4_fingerprint",
    "include_descriptors",
    "include_skfp_autocorr_fingerprint",
    "include_skfp_avalon_fingerprint",
    "include_skfp_ecfp_fingerprint",
    "include_skfp_erg_fingerprint",
    "include_skfp_estate_fingerprint",
    "include_skfp_functional_groups_fingerprint",
    "include_skfp_ghose_crippen_fingerprint",
    "include_skfp_klekota_roth_fingerprint",
    "include_skfp_laggner_fingerprint",
    "include_skfp_layered_fingerprint",
    "include_skfp_lingo_fingerprint",
    "include_skfp_maccs_fingerprint",
    "include_skfp_map_fingerprint",
    "include_skfp_mhfp_fingerprint",
    "include_skfp_mqns_fingerprint",
    "include_skfp_pattern_fingerprint",
    "include_skfp_pubchem_fingerprint",
    "include_skfp_rdkit_fingerprint",
    "include_skfp_secfp_fingerprint",
    "include_skfp_topological_torsion_fingerprint",
    "include_skfp_vsa_fingerprint",
]


def test_compute_features():
    """Test the computation of molecular features."""
    for parameter in PARAMETERS:
        try:
            compute_features(
                "CN1[C@H]2CC[C@@H]1[C@@H](C(OC)=O)[C@@H](OC(C3=CC=CC=C3)=O)C2",
                **{parameter: True},
            )
        except Exception as e:
            raise AssertionError(f"Error for parameter {parameter}") from e


def molecular_features_time(d: (str, str)) -> Dict[str, Union[str, float]]:
    """Compute the time for the computation of molecular features."""
    (smile, parameter) = d

    start = time()
    _features = compute_features(
        smile,
        **{other: False for other in PARAMETERS if other != parameter},
        **{parameter: True},
    )
    end = time()
    return {"parameter": parameter, "time": end - start}


def test_molecular_features_time():
    """Test to measure the time for the computation of molecular features."""
    smiles = pd.read_csv("np_classifier/training/categorical.csv.gz").smiles[:1000]

    with Pool() as pool:
        performance = list(
            tqdm(
                pool.imap(
                    molecular_features_time,
                    ((s, p) for s in smiles for p in PARAMETERS),
                ),
                total=len(smiles) * len(PARAMETERS),
                desc="Computing molecular features",
                unit="smile",
            )
        )
        pool.close()
        pool.join()

    performance = pd.DataFrame(performance)
    performance.to_csv("features_time_requirements.csv", index=False)
