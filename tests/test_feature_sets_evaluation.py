"""Test to ensure the feature_sets_evaluation sub-command works as expected."""

import os
import shutil
import subprocess


def test_feature_sets_evaluation_sub_command():
    """Test the feature_sets_evaluation sub-command."""
    subprocess.run(
        [
            "hammer",
            "feature-sets-evaluation",
            "--dataset",
            "NPC",
            "--smoke-test",
            "--holdouts",
            "1",
            "--performance-path",
            "test_feature_sets_evaluation.csv",
            "--barplot-directory",
            "test_feature_sets_evaluation_barplots",
        ],
        check=True,
    )
    assert os.path.exists("test_feature_sets_evaluation.csv")
    os.remove("test_feature_sets_evaluation.csv")
    assert os.path.exists("test_feature_sets_evaluation_barplots")
    assert os.path.isdir("test_feature_sets_evaluation_barplots")
    shutil.rmtree("test_feature_sets_evaluation_barplots")