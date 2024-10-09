# Natural Products Classifier

This repository contains the code for the Natural Products Classifier, a multi-modal multi-task feed-forward neural network that predicts the pathways, classes, and superclasses of natural products based on their molecular structure and physicochemical properties. The classifier leverages a diverse set of molecular fingerprints and descriptors to capture the unique features of natural products and enable accurate predictions across multiple tasks.

The model can be beheaded (remove the output layers) and used either as a feature extractor or as a pre-trained model for transfer learning on other tasks. This package provides also tooling to extract and visualize all of the features used in the model, which can be used to train other models or to perform downstream analyses. **If you intend to use this model for transfer learning, pay attention to not include in your test set SMILEs used for training this model to avoid biasing your evaluations!**

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

## Feature sets evaluation

To evaluate the feature sets used in the model, you can run the following command. This will perform a 10-fold cross-validation evaluation of the feature sets, using as augmentation 64 molecules generated using the precomputed Pickaxe rules, 16 tautomers, and 16 stereoisomers. We limit the number of molecules to augment so that no sample is multiplied an excessive number of times.

The dataset is split using first a stratified split by the rarest class, then subsequently `holdouts` number of stratified Monte Carlo splits into sub-training and validation. It is this the sub-training set that is augmented with the generated molecules. **The test set is not touched during this evaluation process, as we will use it to evaluate the model over the selected feature set.**

```bash
hammer feature-sets-evaluation --verbose --holdouts 10 --include-pickaxe 64 --include-tautomers 16 --include-stereoisomers 16
```
