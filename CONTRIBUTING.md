# Contributing to Hammer

Thank you for your interest in contributing to **Hammer**! Hammer is a hierarchical augmenting multi-modal multitask classifier designed to work with SMILES fingerprints, and we aim to support a wide range of predictive tasks, starting with chemical taxonomy and extending to other areas like species taxonomy. The model is highly flexible, with built-in tools for SMILES augmentation, such as tautomer and stereoisomer generation, but these augmentations are task-dependent - in some tasks the label of the augmented SMILE might be different from the original SMILE. Our goal is to create an open-source tool that offers both open models and traceable datasets.

Below are the guidelines for contributing to Hammer, whether you are interested in adding datasets, improving the model, or extending the augmentation techniques.

## General Guidelines

### 1. Opening an Issue

Before contributing, we ask that you open an issue to discuss your proposal. This allows the community and maintainers to provide feedback and ensure alignment with the project’s direction. Consider opening an issue if you are:

- Proposing a new dataset (e.g., SMILES and their taxonomies, whether chemical, species, or others).
- Suggesting model or dataset extensions (such as new predictive tasks or classification layers).
- Proposing improvements to the existing dataset or hierarchical DAG structure.
- Introducing new SMILES augmentation techniques or improving existing ones.
- Introducing new SMILES features or improving existing ones.

Clearly outline your proposal, its purpose, timeline, and any relevant sources or evidence when creating an issue.

### 2. Adding a New Dataset

We are committed to building high-quality, community-driven datasets that are fully open source. If you would like to contribute a dataset:

- **Provenance is essential.** We only accept datasets where the origin is fully traceable, ensuring accuracy and preventing confusion. If a dataset is produced by a software, there must exist an open-source **tested** script that can reproduce the dataset.
- Begin by opening an issue to discuss the dataset and gather feedback.
- After approval, submit a pull request with the dataset, ensuring:
  - Proper formatting and integration with the dataset pipeline.
  - Clear documentation about the dataset’s origin, purpose, and intended task.

Individual sample contributions are welcome as well, provided there is evidence supporting their accuracy and relevance.

### 3. Extending the Model or DAG

Hammer’s hierarchical model and layered DAG structure are flexible and support a variety of predictive tasks, including multi-label and categorical classification. We encourage contributions that:

- Extend the model for new types of predictions or augment the hierarchical DAG to accommodate new data relationships.
- Provide entirely new DAGs for different tasks or domains.
- Propose changes that improve performance, scalability, or usability for existing tasks.
- Add new features or interfaces to the model that benefit the broader community.

When suggesting extensions, we ask that you first open an issue for discussion before submitting a pull request.

### 4. Contributing to SMILES Augmentation

Hammer comes with built-in SMILES augmentation techniques, such as tautomer and stereoisomer generation. While these augmentations are helpful for some tasks, they may not be applicable to all. We welcome contributions that:

- Introduce new augmentation techniques to diversify the transformations available.
- Improve the effectiveness or efficiency of the existing augmentation methods.

As with any contribution, please open an issue first for discussion.

### 5. Contributing to SMILES Features

SMILES fingerprints are a critical component of Hammer’s predictive capabilities. Hammer uses extensively fingerprints provided by [scikit-fingerprints](https://github.com/scikit-fingerprints/scikit-fingerprints). We welcome contributions that:

- Introduce new SMILES features that enhance the model’s predictive capabilities.
- Improve the efficiency or accuracy of existing SMILES features.

As with any contribution, please open an issue first for discussion. If appropriate, consider working directly with the scikit-fingerprints repository to ensure compatibility and maintainability.

### 6. Documentation and Tests

To maintain a high-quality codebase, we ask that all contributions include:

- **Documentation:** Clear explanations of the purpose of your changes and instructions on how to use any new features or datasets.
- **Tests:** To ensure your contributions function as intended and maintain compatibility with the existing codebase. If your changes affect performance, benchmarks or comparative tests are appreciated.

## Provenance and Data Integrity

We place a strong emphasis on maintaining **data provenance** for any datasets used or contributed. We expect contributors to provide clear and well-defined data origins. This is to avoid issues similar to problematic datasets we’ve encountered in the past, which lacked clarity regarding their sources.

## Acknowledgment for Contributions

We value every contribution, whether it’s a small improvement or a major addition:

- **Minor contributions** will be acknowledged in the project’s “Thanks” section.
- **Significant contributions**—such as new datasets or DAGS, new model capabilities (including augmentations and SMILES features), or major improvements—will be recognized in any future academic papers related to the project.

## Getting Started

To begin contributing, explore the following resources:

- [Dataset Interface](https://github.com/LucaCappelletti94/hammer/blob/main/hammer/datasets/smiles_dataset.py)
- [DAG Interface](https://github.com/LucaCappelletti94/hammer/blob/main/hammer/layered_dags/layered_dag.py)
- [SMILES Feature Interface](https://github.com/LucaCappelletti94/hammer/blob/main/hammer/features/feature_interface.py)
- [SMILES Augmentation Interface](https://github.com/LucaCappelletti94/hammer/blob/main/hammer/augmentation_strategies/augmentation_strategy.py)

We encourage you to open an issue if you have a proposal or check out existing issues for potential contribution ideas. We are excited to collaborate with the community to build a high-quality, open-source resource for predictive tasks!
