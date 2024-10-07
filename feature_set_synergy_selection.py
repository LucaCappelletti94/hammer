"""Test whether combining different feature sets improves the performance of the model."""

import silence_tensorflow.auto  # pylint: disable=unused-import
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import compress_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score
from np_classifier.training import Dataset


def compare_feature_set_synergies():
    """Compare the quality of different feature sets."""
    dataset = Dataset(
        include_atom_pair_fingerprint=True,
        include_maccs_fingerprint=True,
        include_morgan_fingerprint=True,
        include_rdkit_fingerprint=True,
        include_avalon_fingerprint=True,
        include_descriptors=True,
        include_feature_morgan_fingerprint=True,
        include_map4_fingerprint=True,
        include_topological_torsion_fingerprint=True,
        include_skfp_autocorr_fingerprint=True,
        include_skfp_ecfp_fingerprint=True,
        include_skfp_erg_fingerprint=True,
        include_skfp_estate_fingerprint=True,
        include_skfp_functional_groups_fingerprint=True,
        include_skfp_ghose_crippen_fingerprint=True,
        include_skfp_klekota_roth_fingerprint=True,
        include_skfp_laggner_fingerprint=True,
        include_skfp_layered_fingerprint=True,
        include_skfp_lingo_fingerprint=True,
        include_skfp_maccs_fingerprint=True,
        include_skfp_map_fingerprint=True,
        include_skfp_mhfp_fingerprint=True,
        include_skfp_mqns_fingerprint=True,
        include_skfp_pattern_fingerprint=True,
        include_skfp_pubchem_fingerprint=True,
        include_skfp_rdkit_fingerprint=True,
        include_skfp_secfp_fingerprint=True,
        include_skfp_topological_torsion_fingerprint=True,
        include_skfp_vsa_fingerprint=True,
    )

    performance = []

    for _, (train_x, train_y), (valid_x, valid_y) in dataset.train_split(augment=False):
        concatenated_train_y = np.hstack(list(train_y.values()))
        concatenated_valid_y = np.hstack(list(valid_y.values()))
        
        # We check that at least one value is true in the concatenated labels
        assert np.any(concatenated_train_y, axis=0).all()
        assert np.any(concatenated_valid_y, axis=0).all()

        for first_feature_name in tqdm(
            train_x.keys(),
            desc="Testing feature set synergies",
            total=len(train_x.keys()),
            unit="feature set",
            leave=False,
        ):

            if np.isnan(train_x[first_feature_name]).any():
                print(f"Skipping {first_feature_name} due to NaN values")
                continue
            for second_feature_name in tqdm(
                train_x.keys(),
                desc=f"Evaluating synergy with {first_feature_name}",
                total=len(train_x.keys()),
                unit="feature set",
                leave=False,
            ):
                if first_feature_name <= second_feature_name:
                    continue
            
                if np.isnan(train_x[second_feature_name]).any():
                    print(f"Skipping {second_feature_name} due to NaN values")
                    continue

                # For each feature set, we train a Random Forest to predict
                # all of the multi-class labels.
                classifier = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=30,
                    n_jobs=-1,
                    verbose=0,
                )
                x_train = np.hstack([train_x[first_feature_name], train_x[second_feature_name]])
                x_valid = np.hstack([valid_x[first_feature_name], valid_x[second_feature_name]])
                classifier.fit(x_train, concatenated_train_y)
                valid_predictions = classifier.predict(x_valid)
                train_predictions = classifier.predict(x_train)

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
                            "first_feature_set": first_feature_name,
                            "second_feature_set": second_feature_name,
                            "set": prediction_set_name,
                            "average_precision": average_precision,
                            "accuracy": accuracy,
                        }
                    )
                    compress_json.dump(
                        performance,
                        "feature_set_synergy_performance.json",
                    )

    performance = pd.DataFrame(performance)
    performance.to_csv("feature_set_synergy_performance.csv", index=False)


if __name__ == "__main__":
    compare_feature_set_synergies()
