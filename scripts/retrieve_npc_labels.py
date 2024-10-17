"""Script to retrieve the classification from the original NP Classifier."""

from typing import Dict
import requests
from time import time
from tqdm.auto import tqdm
import compress_json
from cache_decorator import Cache
from fake_useragent import UserAgent
from hammer.datasets import NPCDataset


@Cache(use_source_code=False, cache_path="{cache_dir}/{function_name}/{_hash}.json")
def get_classifications(smiles: str) -> Dict:
    """Get the classifications for a given SMILES."""

    ua = UserAgent()
    header = {"User-Agent": str(ua.chrome)}

    response = requests.get(
        "https://npclassifier.gnps2.org/classify",
        params={"smiles": smiles},
        headers=header,
        timeout=10,
    )

    if response.status_code != 200:
        print(
            f"Failed to retrieve classifications for {smiles}, "
            f" got status code {response.status_code}"
        )
        print(response.text)

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print(
            f"Failed to convert response to JSON for {smiles}, "
            f" got status code {response.status_code}, raw response: {response.text}"
        )

    return {}


def relabelling():

    dataset: NPCDataset = NPCDataset()

    reports = []

    start = time()

    for labelled_smile in tqdm(
        dataset.iter_labeled_smiles(),
        total=dataset.number_of_smiles(),
        unit="smiles",
        desc="Retrieving classifications from NP Classifier",
        dynamic_ncols=True,
        leave=False,
    ):
        prediction = get_classifications(labelled_smile.smiles)

        if not prediction:
            continue

        reports.append(
            {"smiles": labelled_smile.smiles, **labelled_smile.labels, **prediction}
        )

        if time() - start > 30:
            start = time()
            compress_json.dump(reports, "./relabelled.json", json_kwargs={"indent": 4})

    compress_json.dump(reports, "./relabelled.json", json_kwargs={"indent": 4})


if __name__ == "__main__":
    relabelling()
