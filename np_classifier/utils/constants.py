"""Submodule providing constants."""

import compress_json

PATHWAY_NAMES = compress_json.local_load("pathway_names.json.gz")
SUPERCLASS_NAMES = compress_json.local_load("superclass_names.json.gz")
CLASS_NAMES = compress_json.local_load("class_names.json.gz")
