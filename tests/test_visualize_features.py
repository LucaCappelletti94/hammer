"""Test to ensure the visualize-features sub-command works as expected."""

import os
import shutil
import subprocess


def test_visualize_features_sub_command():
    """Test the visualize_features sub-command."""
    subprocess.run(
        [
            "hammer",
            "visualize",
            "--verbose",
            "--smoke-test",
            "--output-directory",
            "test_visualize_features",
            "--image-format",
            "png",
        ],
        check=True,
    )
    assert os.path.exists("test_visualize_features")
    shutil.rmtree("test_visualize_features")
