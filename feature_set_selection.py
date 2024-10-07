"""Script to compare quality of different features, primarily fingerprints."""

import os
from typing import Optional
import silence_tensorflow.auto  # pylint: disable=unused-import
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import compress_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score
from np_classifier.training import Dataset


def compare_feature_sets():
    """Compare the quality of different feature sets."""
    dataset = Dataset(
        # include_atom_pair_fingerprint=True,
        # include_maccs_fingerprint=True,
        include_morgan_fingerprint=True,
        # include_rdkit_fingerprint=True,
        # include_avalon_fingerprint=True,
        # include_descriptors=True,
        # include_topological_torsion_fingerprint=True,
        # include_skfp_autocorr_fingerprint=True,
        include_skfp_ecfp_fingerprint=True,
        # include_skfp_estate_fingerprint=True,
        # include_skfp_ghose_crippen_fingerprint=True,
        # include_skfp_klekota_roth_fingerprint=True,
        # include_skfp_laggner_fingerprint=True,
        # include_skfp_layered_fingerprint=True,
        # include_skfp_lingo_fingerprint=True,
        # include_skfp_mqns_fingerprint=True,
        # include_skfp_pattern_fingerprint=True,
        # include_skfp_pubchem_fingerprint=True,
        include_skfp_secfp_fingerprint=True,
        # include_skfp_vsa_fingerprint=True,
        radius=2,
    )

    performance = []

    stored_performance: Optional[pd.DataFrame] = None

    if os.path.exists("feature_sets_performance_radius2.csv"):
        stored_performance = pd.read_csv("feature_sets_performance_radius2.csv")

    for _, (train_x, train_y), (valid_x, valid_y) in dataset.train_split(augment=False):
        concatenated_train_y = np.hstack(list(train_y.values()))
        concatenated_valid_y = np.hstack(list(valid_y.values()))
        
        # We check that at least one value is true in the concatenated labels
        assert np.any(concatenated_train_y, axis=0).all()
        assert np.any(concatenated_valid_y, axis=0).all()

        for feature_name in tqdm(
            train_x.keys(),
            desc="Comparing feature sets",
            total=len(train_x.keys()),
            unit="feature set",
            leave=False,
        ):
            if stored_performance is not None:
                if (
                    stored_performance["feature_set"] == feature_name
                ).any():
                    continue

            if np.isnan(train_x[feature_name]).any():
                print(f"Skipping {feature_name} due to NaN values")
                continue

            # For each feature set, we train a Random Forest to predict
            # all of the multi-class labels.
            classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                n_jobs=-1,
                verbose=0,
            )
            classifier.fit(train_x[feature_name], concatenated_train_y)
            valid_predictions = classifier.predict(valid_x[feature_name])
            train_predictions = classifier.predict(train_x[feature_name])

            # We calculate the average precision, Matthews correlation coefficient,
            # and accuracy for the valid set.
            for prediction, ground_truth, prediction_set_name in [
                (train_predictions, concatenated_train_y, "train"),
                (valid_predictions, concatenated_valid_y, "valid"),
            ]:
                average_precision = average_precision_score(
                    ground_truth, prediction, average="weighted"
                )
                accuracy = accuracy_score(ground_truth, prediction > 0.5)

                performance.append(
                    {
                        "feature_set": feature_name,
                        "set": prediction_set_name,
                        "average_precision": average_precision,
                        "accuracy": accuracy,
                    }
                )
                compress_json.dump(
                    performance,
                    "feature_sets_performance_radius2.json",
                )

    performance = pd.DataFrame(performance)
    performance = pd.concat([performance, stored_performance], ignore_index=True)
    performance.to_csv("feature_sets_performance_radius2.csv", index=False)


if __name__ == "__main__":
    compare_feature_sets()
