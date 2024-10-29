"""Script to run model predictions."""

from argparse import Namespace, ArgumentParser
from typing import Dict, Optional, List
import os
import pandas as pd
from tqdm.auto import tqdm
from anytree import Node, RenderTree
from matchms.importing import load_from_mgf
from hammer import Hammer
from hammer.dags import LayeredDAG
from hammer.utils import is_valid_smiles
from hammer.executables.argument_parser_utilities import add_model_predictions_arguments


def add_predict_subcommand(subparser: ArgumentParser):
    """Add the predict sub-command to the parser."""
    subparser = add_model_predictions_arguments(subparser)

    subparser.set_defaults(func=predict)


# ANSI escape codes for colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
COLORS = [GREEN, BLUE, MAGENTA, CYAN, RED, YELLOW]


def print_predictions(smiles: str, dag: LayeredDAG, predictions: Dict[str, pd.Series]):
    """Print the multi-label multi-class predictions to bash as a tree.

    Implementation details
    ----------------------
    Since the predictions are meant to follow the provided DAG, we illustrate
    the predictions as a tree. We use Node and RenderTree from the anytree library
    to generate the tree.
    """
    print(f"{BOLD}{CYAN}SMILES:{RESET} {smiles}")
    nodes: Dict[str, Node] = {}
    last_layer_name: Optional[str] = None
    for i, layer_name in enumerate(dag.layer_names()):
        layer_color: str = COLORS[i % len(COLORS)]
        # We get the predictions associated with the layer.
        layer_predictions: pd.Series = predictions[layer_name]
        # We sort the predictions in descending order.
        layer_predictions = layer_predictions.sort_values(ascending=False)
        # We limit the predictions so that when we find a drop of 10x in the
        # score, we stop considering to display the predictions.
        filtered_predictions = {}
        last_score = None
        for label_name, score in layer_predictions.items():
            if last_score is None or score >= last_score / 10:
                filtered_predictions[label_name] = score
                last_score = score
                continue
            break
        layer_predictions = pd.Series(filtered_predictions)

        # We keep only predictions either with score higher than 0.5, or
        # up until the total prediction score for this layer is higher than
        # 1.25 (so to have so extra margin for the predictions).
        layer_predictions = layer_predictions[
            (layer_predictions > 0.5) | (layer_predictions.cumsum() < 1.0)
        ]

        for label_name, score in layer_predictions.items():
            if last_layer_name is None:
                nodes[label_name] = Node(
                    f"{layer_color}{label_name} ({score:.4f}){RESET}"
                )
                continue
            for parent_node in dag.outbounds(label_name):
                if parent_node in nodes:
                    nodes[label_name] = Node(
                        f"{layer_color}{label_name} ({score:.4f}){RESET}",
                        parent=nodes.get(parent_node),
                    )
        last_layer_name = layer_name

    for root_node in dag.iter_sink_nodes():
        maybe_root_node: Optional[Node] = nodes.get(root_node)
        if maybe_root_node is None:
            continue
        for pre, _fill, node in RenderTree(maybe_root_node):
            print(f"{pre}{node.name}")


def predict(args: Namespace):
    """Run model predictions."""

    if args.version is None and args.model_path is None:
        raise ValueError("Either a version or a model path must be provided.")

    # First, we normalize the input.
    if is_valid_smiles(args.input):
        # If the input is a Single SMILES string
        samples = [args.input]
    elif os.path.isfile(args.input):
        # Otherwise, we check whether the input is a file,
        # and whether it is a CSV, TSV or SSV file.
        valid_extensions = {
            ".csv": ",",
            ".tsv": "\t",
            ".ssv": " ",
            ".mgf": " ",
        }
        valid_compressions = [".gz", ".xz"]
        complete_valid_extensions_separators = {
            extension + compression: separator
            for extension, separator in valid_extensions.items()
            for compression in valid_compressions + [""]
        }
        extension: Optional[str] = None
        separator: Optional[str] = None
        for ext, sep in complete_valid_extensions_separators.items():
            if args.input.endswith(ext):
                separator = sep
                extension = ext
                break
        if separator is None:
            raise ValueError(
                f"Invalid file extension '{args.input}'. Valid extensions are: {valid_extensions}"
            )

        if extension == ".mgf":
            # We load the MGF file.
            samples = []

            for spectrum in tqdm(
                load_from_mgf(args.input),
                desc="Loading Spectra",
                leave=False,
                dynamic_ncols=True,
                disable=not args.verbose,
            ):
                if args.only_smiles:
                    if "smiles" in spectrum.metadata:
                        samples.append(spectrum.metadata["smiles"])
                else:
                    samples.append(spectrum)
        else:
            df = pd.read_csv(args.input, sep=separator, engine="python")

            # We scrape the SMILES strings from the file.
            samples = list(
                {
                    value
                    for row in tqdm(
                        df.values,
                        desc="Scraping SMILES",
                        leave=False,
                        dynamic_ncols=True,
                        disable=not args.verbose,
                    )
                    for value in row
                    if isinstance(value, str) and is_valid_smiles(value)
                }
            )
    else:
        raise ValueError(
            f"Invalid input '{args.input}'. The input must be a SMILES string or a path to a file."
        )

    if args.version:
        model = Hammer.load(args.version)
    elif args.model_path:
        model = Hammer.load_from_path(args.model_path)
    else:
        raise NotImplementedError("This should not happen.")

    model._verbose = args.verbose
    predictions: Dict[str, pd.DataFrame] = model.predict_proba(
        samples,
    )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        for key, value in predictions.items():
            value.to_csv(
                os.path.join(args.output_dir, f"{key}.{args.output_format}"),
                index=True,
            )
    else:
        for sample_identifier in predictions[list(predictions.keys())[0]].index:
            print_predictions(
                sample_identifier,
                model.layered_dag,
                {
                    layer_name: prediction.loc[sample_identifier]
                    for layer_name, prediction in predictions.items()
                },
            )
