"""Submodule adding arguments used across the different executables."""

from multiprocessing import cpu_count
from argparse import ArgumentParser


def add_shared_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add shared arguments to the parser."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=cpu_count(),
        help="The number of jobs to use for parallel processing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7345767,
        help="The random state to use for reproducibility.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Whether to run a smoke test.",
    )
    return parser
