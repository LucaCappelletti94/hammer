"""
Script to retrieve the classification from the original NP Classifier when the data THEY have provided is inconsistent,
and therefore our best next solution is to use the predictions from the original NP Classifier. This is EXTREMELY suboptimal,
as it would be immensely better to have a consistent dataset, but this is the best we can do for now until we recreate the
data ourselves or the original authors decide to share the dataset that they actually used to train the model instead of
a faulty one which definitely does not contain the data that they used to train the model (it contains different classes).
"""

from typing import Dict
from time import sleep
import requests
import pandas as pd
from tqdm.auto import tqdm
import compress_json
from cache_decorator import Cache


@Cache(
    use_source_code=False
)
def get_classifications(smiles: str) -> Dict:
    """Get the classifications for a given SMILES."""
    response = requests.get(
        f"https://npclassifier.gnps2.org/classify?smiles={smiles}",
        timeout=10,
    )
    print(smiles)
    print(response.text)
    sleep(1)
    
    return response.json()


def relabelling():

    df = pd.read_csv("np_classifier/training/categorical.csv")

    df = df[df.class_label == "RiPPs"]

    smiles = [smiles for smiles in df["smiles"]]

    # We exclude smiles that have alredy been manually relabelled
    # in the file np_classifier/training/multi_label.json

    manual_relabelled = [
        entry["smiles"]
        for entry in compress_json.load("np_classifier/training/multi_label.json")
    ]

    smiles = [s for s in smiles if s not in manual_relabelled]

    reports = []

    for s in tqdm(smiles):
        prediction = get_classifications(s)
        reports.append(
            {
                "smiles": s,
                "pathway_labels": prediction["pathway_results"],
                "superclass_labels": prediction["superclass_results"],
                "class_labels": prediction["class_results"],
            }
        )

    compress_json.dump(reports, "np_classifier/training/relabelled.json")


if __name__ == "__main__":
    relabelling()
