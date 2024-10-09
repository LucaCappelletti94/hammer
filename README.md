# Natural Products Classifier

This repository contains the code for the Natural Products Classifier, a multi-modal multi-task feed-forward neural network that predicts the pathways, classes, and superclasses of natural products based on their molecular structure and physicochemical properties. The classifier leverages a diverse set of molecular fingerprints and descriptors to capture the unique features of natural products and enable accurate predictions across multiple tasks.

The model can be beheaded (remove the output layers) and used either as a feature extractor or as a pre-trained model for transfer learning on other tasks. This package provides also tooling to extract and visualize all of the features used in the model, which can be used to train other models or to perform downstream analyses. **If you intend to use this model for transfer learning, pay attention to not include in your test set SMILEs used for training this model to avoid biasing your evaluations!**

A preview of [the features used headers is available here](https://github.com/LucaCappelletti94/npc_classifier/tree/main/data_preview) ([script to generate feature CSVs available here](https://github.com/LucaCappelletti94/npc_classifier/blob/main/train_features_to_csv.py)), while [t-SNE visualizations of the features are available here](https://github.com/LucaCappelletti94/npc_classifier/tree/main/data_visualizations) ([script to generate visualizations available here](https://github.com/LucaCappelletti94/npc_classifier/blob/main/visualize.py)). All features used are directly derived from the SMILES strings of the molecules, therefore we do not share the rather heavy rasterized dataset (it weights about 50GBs), but we both share the SMILES and their labels ([original single-class SMILES](https://github.com/LucaCappelletti94/npc_classifier/blob/main/hammer/training/categorical.csv.gz) and [new multi-class SMILES](https://github.com/LucaCappelletti94/npc_classifier/blob/main/hammer/training/multi_label.json)), plus the code necessary to generate the features.

The classifiers (available from Zenodo, ACTUALLY TODO!) is implemented in Python using the [TensorFlow](https://www.tensorflow.org/?hl=it)/[Keras](https://keras.io/) deep learning frameworks and the [RDKit cheminformatics library](https://www.rdkit.org/).

## Installation

This library will be available to install via pip, but for now you can install it by cloning the repository and running the following command:

```bash
pip install .
```


## Feature visualization

To visualize the features used in the model, you can run the following command:

```bash
hammer visualize --verbose
```