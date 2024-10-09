"""Script to compute the dataset coverage of the current chemical DAG."""

from argparse import Namespace
from typing import Dict, List, Set, Tuple, Iterator
import compress_json
import pandas as pd
from hammer.training import Dataset
from hammer.executables.argument_parser_utilities import add_shared_arguments


class DAG:
    """Class representing the chemical DAG."""

    def __init__(self):
        dag = compress_json.local_load("../training/dag.json")
        self._classes_to_superclasses: Dict[str, List[str]] = dag["classes"]
        self._superclasses_to_pathways: Dict[str, List[str]] = dag["superclasses"]

    @property
    def unique_classes(self) -> Set[str]:
        """Return the unique classes in the DAG."""
        return set(self._classes_to_superclasses)

    @property
    def unique_superclasses(self) -> Set[str]:
        """Return the unique superclasses in the DAG."""
        return set(self._superclasses_to_pathways)

    @property
    def unique_pathways(self) -> Set[str]:
        """Return the unique pathways in the DAG."""
        return {
            pathway
            for pathways in self._superclasses_to_pathways.values()
            for pathway in pathways
        }

    def iter_dag_triples(self) -> Iterator[Tuple[str, str, str]]:
        """Iterate over the triples (class, superclass, pathway) in the DAG."""
        for class_name, superclasses in self._classes_to_superclasses.items():
            for superclass in superclasses:
                for pathway in self._superclasses_to_pathways[superclass]:
                    yield (class_name, superclass, pathway)


class DAGCoverageReport:
    """Data class with the coverage report of the chemical DAG."""

    def __init__(self):
        dag = DAG()
        dataset = Dataset()

        number_of_classes_only_in_dataset = (
            set(dataset.class_names) - dag.unique_classes
        )
        assert (
            len(number_of_classes_only_in_dataset) == 0
        ), f"Classes only in dataset: {number_of_classes_only_in_dataset}"
        number_of_superclasses_only_in_dataset = (
            set(dataset.superclass_names) - dag.unique_superclasses
        )
        assert (
            len(number_of_superclasses_only_in_dataset) == 0
        ), f"Superclasses only in dataset: {number_of_superclasses_only_in_dataset}"
        number_of_pathways_only_in_dataset = (
            set(dataset.pathway_names) - dag.unique_pathways
        )
        assert (
            len(number_of_pathways_only_in_dataset) == 0
        ), f"Pathways only in dataset: {number_of_pathways_only_in_dataset}"

        self._number_of_classes_only_in_dag = len(
            dag.unique_classes - set(dataset.class_names)
        )
        self._number_of_classes = len(dag.unique_classes)
        self._number_of_superclasses_only_in_dag = len(
            dag.unique_superclasses - set(dataset.superclass_names)
        )
        self._number_of_superclasses = len(dag.unique_superclasses)
        self._number_of_pathways_only_in_dag = len(
            dag.unique_pathways - set(dataset.pathway_names)
        )
        self._number_of_pathways = len(dag.unique_pathways)

        dataset_triples = set(dataset.iter_label_triples())
        dag_triples = set(dag.iter_dag_triples())

        covered_triples = dataset_triples & dag_triples
        self._number_of_triples_only_in_dag = len(dag_triples) - len(covered_triples)
        self._number_of_dag_triples = len(dag_triples)

    @property
    def percentage_of_classes_only_in_dag(self) -> float:
        """Return the percentage of classes only in the DAG."""
        return self._number_of_classes_only_in_dag / self._number_of_classes * 100.0

    @property
    def percentage_of_superclasses_only_in_dag(self) -> float:
        """Return the percentage of superclasses only in the DAG."""
        return (
            self._number_of_superclasses_only_in_dag / self._number_of_superclasses
        ) * 100.0

    @property
    def percentage_of_pathways_only_in_dag(self) -> float:
        """Return the percentage of pathways only in the DAG."""
        return self._number_of_pathways_only_in_dag / self._number_of_pathways * 100.0

    @property
    def percentage_of_triples_only_in_dag(self) -> float:
        """Return the percentage of triples only in the DAG."""
        return self._number_of_triples_only_in_dag / self._number_of_dag_triples * 100.0

    def as_dataframe(self) -> pd.DataFrame:
        """Return the coverage report as a DataFrame."""
        return pd.DataFrame(
            [
                {
                    "label": "Classes only in DAG",
                    "number": self._number_of_classes_only_in_dag,
                    "percentage": self.percentage_of_classes_only_in_dag,
                },
                {
                    "label": "Superclasses only in DAG",
                    "number": self._number_of_superclasses_only_in_dag,
                    "percentage": self.percentage_of_superclasses_only_in_dag,
                },
                {
                    "label": "Pathways only in DAG",
                    "number": self._number_of_pathways_only_in_dag,
                    "percentage": self.percentage_of_pathways_only_in_dag,
                },
                {
                    "label": "Triples only in DAG",
                    "number": self._number_of_triples_only_in_dag,
                    "percentage": self.percentage_of_triples_only_in_dag,
                },
            ],
        )

    def __str__(self) -> str:
        return self.as_dataframe().to_markdown(index=False)


def add_dag_coverage_subcommand(sub_parser_action: "SubParsersAction"):
    """Add the DAG coverage sub-command to the parser."""
    subparser = sub_parser_action.add_parser(
        "dag-coverage",
        help="Compute the dataset coverage of the current chemical DAG.",
    )
    subparser = add_shared_arguments(subparser)

    subparser.set_defaults(func=compute_dag_coverage)


def compute_dag_coverage(_args: Namespace) -> None:
    """Compute the dataset coverage of the current chemical DAG."""
    report = DAGCoverageReport()
    print(report)
