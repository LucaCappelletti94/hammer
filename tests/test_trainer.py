"""Submodule to test complete execution of the training pipeline."""

import silence_tensorflow.auto  # pylint: disable=unused-import
from hammer.training import (
    Trainer,
    Dataset,
    FeatureSettings,
    AugmentationSettings,
)


def test_train():
    """Train the model."""
    dataset = Dataset(
        number_of_splits=2,
        maximal_number_of_molecules=1000,
    )
    trainer = Trainer(
        dataset,
        maximal_number_of_epochs=2,
        feature_settings=FeatureSettings.standard(),
        augmentation_settings=AugmentationSettings(),
    )
    _holdout_performance = trainer.holdouts()
