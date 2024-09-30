"""Script to generate the train features and store them to a CSV file."""

from typing import Dict
import pandas as pd
from tqdm.auto import tqdm
from np_classifier.training import Dataset


def train_features_to_csv():
    """Train the model."""
    dataset = Dataset()
    train_features: Dict[str, pd.DataFrame] = dataset.train_features_dataframes()

    for feature_name, feature in tqdm(
        train_features.items(),
        desc="Saving train features to CSV",
        total=len(train_features),
        unit="feature",
        leave=False,
        dynamic_ncols=True,
    ):

        head = feature.head()
        head.to_csv(f"{feature_name}_head.csv", index=False)

        feature.to_csv(f"{feature_name}.csv", index=False)


if __name__ == "__main__":
    train_features_to_csv()
