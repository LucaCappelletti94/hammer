"""Submodule to retrieve the NP Classifier model from Zenodo and pre-process it for use in the classifier function."""
from downloaders import BaseDownloader

UPDATE_URL = "https://zenodo.org/record/5068687/files/model.zip?download=1"

def retrieve_model():
    """Retrieve the model from Zenodo."""
    downloader = BaseDownloader()
    downloader.download("https://zenodo.org/record/5068687/files/model.zip?download=1", "model.zip")
