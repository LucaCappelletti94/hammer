"""Submodule to test the execution of the NP Classifier."""

from np_classifier import NPClassifier


def test_classifier():
    """Test the NP Classifier."""

    # Initialize the NP Classifier
    classifier = NPClassifier()

    # "O=C1c3c(O/C(=C1/O)c2ccc(O)c(O)c2)cc(O)cc3O" # Quercetin
    # "CN1[C@H]2CC[C@@H]1[C@@H](C(OC)=O)[C@@H](OC(C3=CC=CC=C3)=O)C2" # Cocaine

    smiles = "O=C1c3c(O/C(=C1/O)c2ccc(O)c(O)c2)cc(O)cc3O"  # Quercetin
    # Classify the molecule
    classification = classifier.classify(smiles)

    print("Pathways", classification.top_k_pathways(3))
    print("SuperClasses", classification.top_k_superclasses(3))
    print("Classes", classification.top_k_classes(3))
