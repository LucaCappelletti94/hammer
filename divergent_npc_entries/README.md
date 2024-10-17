# Divergence correction

This directory contains SMILES whose classifications (pathways, superclasses and classes) are different between the original dataset and the model currently being hosted by NPClassifier. Several such divergences are errors.

## Notation for corrected options

We proceed using the following notation:

- `?`: the correct classification is unknown, feel free to correct it
- `original`: the original classification in the dataset is the correct one
- `scraped`: the classification given by the model currently being hosted by NPClassifier is the correct one
- `both`: the classification given by both the original dataset and the model should be merged
- `none`: the classification given by both the original dataset and the model is incorrect. In such cases, we will provide the correct classification in a different CSV file named `pathways_corrected.csv`, `superclasses_corrected.csv` and `classes_corrected.csv`.
