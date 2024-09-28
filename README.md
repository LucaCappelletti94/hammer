# NP Classifier

Python package to use the [NP classifier](https://github.com/mwang87/NP-Classifier) without server requests.

## What is NP Classifier?

NPClassifier, a deep-learning tool for the automated structural classification of NPs from their counted Morgan fingerprints. The tool is freely available at https://npclassifier.ucsd.edu together with a web-API. NPClassifier was developed using supervised feed-forward networks with 73607 NPs collected from public databases including Pubchem, ChEBI, Chemspider, and the Universal Natural Products Database (UNPD). The distribution of molecular weights and chemical space of the data set are similar to those in the UNPD, a representative natural product database. NPClassifier classifies the structure of an NP at three levels into seven Pathways, 70 Superclasses, and 672 Classes, all of which are generally recognized by the NP research community.

## References

If you use this model, you may want to cite the following paper:

```bibtex
@article{kim2021npclassifier,
  title={NPClassifier: a deep neural network-based structural classification tool for natural products},
  author={Kim, Hyun Woo and Wang, Mingxun and Leber, Christopher A and Nothias, Louis-F{\'e}lix and Reher, Raphael and Kang, Kyo Bin and Van Der Hooft, Justin JJ and Dorrestein, Pieter C and Gerwick, William H and Cottrell, Garrison W},
  journal={Journal of Natural Products},
  volume={84},
  number={11},
  pages={2795--2807},
  year={2021},
  publisher={ACS Publications}
}
```