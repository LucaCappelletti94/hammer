"""Test submodule to ensure that the DAG is compatible with the available data."""

import compress_json
import pandas as pd


def test_dag_coherence():
    """Test to ensure that the DAG is compatible with the available data.

    Implementative details
    ----------------------
    The DAG contains the mapping of the classes to their parent superclasses
    and the mapping of the superclasses to their parent pathways. This DAG
    is used in the model as part of a fixed attention mechanism that ensures
    that the model is aware of the hierarchical relationships between the classes.
    If the DAG turns out to have errors, it will have catastrophic effects on the
    model's performance, so it is of the utmost importance to ensure that the DAG
    is coherent with the available data.
    """
    # smiles,pathway_label,superclass_label,class_label
    categorical = pd.read_csv("np_classifier/training/categorical.csv.gz")

    # Load the DAG
    dag = compress_json.load("np_classifier/training/dag.json")

    for row in categorical.itertuples():
        assert row.superclass_label in dag["classes"][row.class_label]
        assert row.pathway_label in dag["superclasses"][row.superclass_label]

    multilabel = compress_json.load(
        "np_classifier/training/multi_label.json"
    ) + compress_json.load("np_classifier/training/relabelled.json")

    for row in multilabel:
        possible_superclasses = set()
        for class_label in row["class_labels"]:
            possible_superclasses.update(dag["classes"][class_label])
        for superclass_label in row["superclass_labels"]:
            assert superclass_label in possible_superclasses

        possible_pathways = set()
        for superclass_label in row["superclass_labels"]:
            possible_pathways.update(dag["superclasses"][superclass_label])
        for pathway_label in row["pathway_labels"]:
            assert pathway_label in possible_pathways, f"Sample not compatible with DAG! {row}"
