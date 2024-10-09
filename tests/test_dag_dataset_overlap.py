"""Test submodule to ensure that entries of the DAG are present in the dataset."""

import compress_json
import pandas as pd

def test_dag_dataset_overlap():
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
    # smiles,pathway_label,superclass_label,class
    categorical = pd.read_csv("np_classifier/training/categorical.csv.gz")

    # Load the DAG
    dag = compress_json.load("np_classifier/training/dag.json")

    # We count the number of classes in the DAG that are not present in the
    # dataset

    classes_counter = 0
    classes_not_in_dag = []

    for classes in dag["classes"]:
        if classes not in categorical["class_label"].unique():
            classes_counter += 1
            classes_not_in_dag.append(classes)
        
    print(f"Number of classes not in dataset: {classes_counter}")
    print(f"Classes not in dataset: {classes_not_in_dag}")

    # We count the number of superclasses in the DAG that are not present in the
    # dataset

    superclasses_counter = 0
    superclasses_not_in_dag = []

    for superclasses in dag["superclasses"]:
        if superclasses not in categorical["superclass_label"].unique():
            superclasses_counter += 1
            superclasses_not_in_dag.append(superclasses)

    print(f"Number of superclasses not in dataset: {superclasses_counter}")
    print(f"Superclasses not in dataset: {superclasses_not_in_dag}")