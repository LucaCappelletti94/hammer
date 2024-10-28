# 🔨 Hammer

Hammer is a Hierarchical Augmented Multi-modal Multi-task classifiER that, given a SMILE as input, computes selected fingerprints and predicts its associated taxonomical ranking.

The classifier can employ a diverse set of molecular fingerprints and descriptors to capture the unique features of the SMILES and enable accurate predictions across multiple tasks.

Furthermore, the model can be beheaded (remove the output layers) and used either as a feature extractor or as a pre-trained model for transfer learning on other tasks. This package provides also tooling to extract and visualize all of the features used in the model, which can be used to train other models or to perform downstream analyses. **If you intend to use this model for transfer learning, pay attention to not include in your test set SMILEs used for training this model to avoid biasing your evaluations!**

## Installation

This library will be available to install via pip, but for now you can install it by cloning the repository and running the following command:

```bash
pip install .
```

## Command line interface and usage

While the package can be entirely used as a library, it also provides a command line interface that can be used to perform a variety of tasks and reproduce the experiments that we have conducted or design new ones.

In the following sections, we will describe the usage of the command line interface of the Hammer package. These commands are readily available after installing the package, no additional setup is required.

### Feature visualization

To visualize the features used in the model using PCA and t-SNE, you can run the following command:

```bash
hammer visualize --verbose --dataset NPC --output-directory "data_visualizations" --image-format "png"
```

This will generate a set of plots that show the distribution of the features used in the model. The plots will be saved in the `data_visualizations` directory in the `png` format. You can change the output directory and image format by changing the `--output-directory` and `--image-format` arguments, respectively. The resulting plots will look like the following (this one illustrates the t-SNE and PCA decomposition of the Topological Torsion 1024 bits):

[![Topological Torsion (1024 bits)](https://github.com/LucaCappelletti94/hammer/blob/main/data_visualizations/Topological%20Torsion%20(1024b).png?raw=true)](https://github.com/LucaCappelletti94/hammer/tree/main/data_visualizations)

It is also possible to visualize specific feature sets, for example the MAP4 features, by using the `--include-map4` argument:

```bash
hammer visualize --verbose\
    --dataset NPC\
    --include-map4\
    --output-directory "data_visualizations"\
    --image-format "png"
```

### DAG Coverage

One of the goals of this project is to, over time and with the help of the community, increase the overall number of pathways, superclasses, and classes that the model can predict. The model employs as a form of static attention a DAG that harmonizes the predictions of the different tasks. At this time, the dataset we are using **DOES NOT** cover all of the combinations of pathways, superclasses and classes that the DAG allows for. We aim to increase the coverage of the DAG over time, and we welcome contributions to the dataset that can help us achieve this goal. *We are starting out from the dataset made available by [NP Classifier](https://github.com/mwang87/NP-Classifier).*

You can compute a summary of the coverage of the DAG using the following command:

```bash
hammer dag-coverage --dataset NPC --verbose
hammer dag-coverage --dataset NPCHarmonized --verbose
```

At the time of writing, the coverage of the DAG is as follows:

| Dataset       | Layer        |   Coverage |
|:--------------|:-------------|-----------:|
| NPC           | pathways     |   1        |
| NPC           | superclasses |   0.922078 |
| NPC           | classes      |   0.936782 |
| NPC           | DAG          |   0.730228 |
| NPCHarmonized | pathways     |   1        |
| NPCHarmonized | superclasses |   0.948052 |
| NPCHarmonized | classes      |   0.95546  |
| NPCHarmonized | DAG          |   0.813651 |

### Feature sets evaluation

To evaluate the feature sets used in the model, you can run the following command. This will perform a 10-fold cross-validation evaluation of the feature sets. The performance for all holdouts and all considered features will be saved in the `feature_sets_evaluation.csv` file, while the barplots will be saved in the `feature_sets_evaluation_barplots` directory.

The dataset is split using first a stratified split by the rarest class, then subsequently `holdouts` number of stratified Monte Carlo splits into sub-training and validation. **The test set is not touched during this evaluation process, as we will use it to evaluate the model over the selected feature set.**

The model used for these evaluations is the same Hammer model that is used for the predictions, changing only the number of input feature sets.

```bash
hammer feature-sets-evaluation \
    --verbose \
    --holdouts 10 \
    --dataset NPC \
    --test-size 0.2 \
    --validation-size 0.2 \
    --performance-path "performance/feature_sets_evaluation.csv" \
    --training-directory "training/feature_selection" \
    --barplot-directory "barplots/feature_sets_evaluation"
```

```bash
hammer feature-sets-evaluation \
    --verbose \
    --holdouts 10 \
    --dataset NPCHarmonized \
    --test-size 0.2 \
    --validation-size 0.2 \
    --performance-path "performance/feature_sets_evaluation_harmonized.csv" \
    --training-directory "training/feature_selection_harmonized" \
    --barplot-directory "barplots/feature_sets_evaluation_harmonized"
```

Executing this command will generate the barplots [you can find in this directory](https://github.com/LucaCappelletti94/hammer/tree/main/feature_sets_evaluation_barplots). In the following barplot, you will find the AUPRC for each class, for validation, test a, for each feature set, averaged over all holdouts:

In the following table, we illustrate the mean and standard deviation of the validation AUPRC for the different feature sets.

| **Feature Set**                               | V2 Mean  | V2 STD   | V1 Mean  | V1 Std   | V0 Mean        | V0 Std         |
|-----------------------------------------------|----------|----------|----------|----------|----------------|----------------|
| Atom Pair (2048b)                             | 0.937478 | 0.031032 | 0.927321 | 0.009292 | 0.857813       | 0.002735       |
| Auto-Correlation                              | 0.842348 | 0.039879 | 0.869845 | 0.021085 | 0.811050       | 0.004208       |
| Avalon (2048b)                                | 0.944524 | 0.023624 | 0.919810 | 0.044258 | 0.900124       | 0.003449       |
| Extended Connectivity (2r, 2048b)             | 0.952927 | 0.001753 | 0.935591 | 0.002339 | 0.884863 (r=1) | 0.003651 (r=1) |
| Functional Groups                             | 0.577002 | 0.015732 | 0.572862 | 0.019137 | 0.589513       | 0.015186       |
| Ghose-Crippen                                 | 0.642189 | 0.020406 | 0.629073 | 0.021072 | 0.659410       | 0.001983       |
| Laggner                                       | 0.812406 | 0.010464 | 0.805593 | 0.014127 | 0.759160       | 0.012833       |
| Layered (2048b)                               | 0.946703 | 0.002667 | 0.929109 | 0.013407 | 0.898108       | 0.003496       |
| Lingo (1024b)                                 | 0.927148 | 0.001787 | 0.915304 | 0.002246 | 0.837630       | 0.002075       |
| MACCS                                         | 0.849523 | 0.016972 | 0.848236 | 0.015830 | 0.810575       | 0.005415       |
| MAP4                                          | 0.945194 | 0.002993 | 0.934615 | 0.002147 | 0.855033       | 0.005111       |
| MinHashed (2r, 2048b)                         | 0.939163 | 0.001409 | 0.928356 | 0.002958 | 0.839570       | 0.009128       |
| Molecular Quantum Numbers                     | 0.617027 | 0.024393 | 0.614444 | 0.029922 | 0.672846       | 0.008695       |
| Pattern (2048b)                               | 0.934383 | 0.036935 | 0.892704 | 0.043349 | 0.893990       | 0.005862       |
| PubChem                                       | 0.936316 | 0.005425 | 0.911609 | 0.025582 | 0.885968       | 0.003264       |
| RDKit (2048b)                                 | 0.942030 | 0.002838 | 0.927816 | 0.003440 | 0.871136       | 0.008087       |
| SMILES Extended Connectivity (1r, 2048b)      | 0.881445 | 0.004091 | 0.870877 | 0.007103 | 0.827976       | 0.004601       |
| Topological Torsion (1024b)                   | 0.946040 | 0.002092 | 0.931260 | 0.002015 | 0.863581       | 0.002356       |
| Van Der Waals Surface Area                    | 0.769931 | 0.084615 | 0.817110 | 0.063946 | 0.795765       | 0.006069       |

[![AUPRC barplot](https://github.com/LucaCappelletti94/hammer/blob/main/feature_sets_evaluation_barplots/classes_auprc_feature_sets.png?raw=true)](https://github.com/LucaCappelletti94/hammer/tree/main/feature_sets_evaluation_barplots)

It is also possible to run the `feature-sets-evaluation` on a subset of features:

```bash
hammer feature-sets-evaluation \
    --verbose \
    --holdouts 5 \
    --dataset NPC \
    --include-map4 \
    --test-size 0.2 \
    --validation-size 0.2 \
    --performance-path "performance/map4_feature_evaluation.csv" \
    --training-directory "training/map4_feature" \
    --barplot-directory "barplots/map4_feature_evaluation"
```

### Features sets synergy

After having evaluated the feature sets for a given dataset, it remains open the question of how the feature sets interact with each other. It may very well be that the performance of the model is not simply the sum of the performance of the individual feature sets, but that there is a synergy between them, or that by extending the input space with redoundant features we may actually decrease the performance of the model by excessively increasing the dimensionality of the input space, thus making the model more prone to overfitting.

This approach fixes a subset of the feature sets as the base feature sets, and then iterates on all of the low-dimensionality (less than 1024) feature sets, adding them one by one to the base feature sets. The performance of the model is then evaluated on the validation set, and the performance of the model is saved in the `feature_sets_synergy_training.csv` file, while the barplots will be saved in the `feature_sets_synergy_barplots` directory.

We pick the base feature sets as the `layered` feature set, as it is nearly the best performing feature set, and differently from Avalon, we know fully how it is computed while there is no paper for the Avalon fingerprints.

```bash
hammer feature-sets-synergy \
    --verbose \
    --holdouts 10 \
    --dataset NPC \
    --base-feature-sets "extended_connectivity" \
    --test-size 0.2 \
    --validation-size 0.2 \
    --performance-path "performance/synergy/extended_connectivity.csv" \
    --training-directory "trainings/synergy/extended_connectivity" \
    --barplot-directory "barplots/synergy/extended_connectivity"
```

For the NPC dataset, we have identified that the secondary feature most synergistic (has the best validation AUPRC) with the base feature sets and also the smallest feature size is the `Van Der Waals Surface Area`, as illustrated in the following barplot:

| **Feature Set + Layered**       | **Mean**   | **Std**   | **Feature size** |
|---------------------------------|------------|-----------|------------------|
| Auto-Correlation                | 0.914921   | 0.002188  | 192              |
| Functional Groups               | 0.905355   | 0.004740  | 85               |
| Ghose-Crippen                   | 0.905958   | 0.004123  | 110              |
| Laggner                         | 0.907647   | 0.004016  | 307              |
| MACCS                           | 0.906761   | 0.010752  | 166              |
| Molecular Quantum Numbers       | 0.909889   | 0.003788  | 42               |
| PubChem                         | 0.913653   | 0.003556  | 881              |
| Van Der Waals Surface Area      | 0.914309   | 0.004208  | 47               |

[![Synergy barplot](https://github.com/LucaCappelletti94/hammer/blob/main/feature_sets_synergy_with_layered_training/classes_auprc_feature_sets.png?raw=true)](https://github.com/LucaCappelletti94/hammer/tree/main/feature_sets_synergy_with_layered_training)

We can now proceed to identify the tertiary feature set that is most synergistic with the base feature sets and the secondary feature set. We cannot simply pick the next secondary feature set that is most synergistic with the base feature sets, as this would not take into account the interaction between the secondary and tertiary feature sets, and the increased dimensionality of the input space. We need to evaluate the performance of the model on the validation set for all possible combinations of the base, secondary and tertiary feature sets, and select the one that has the best performance, if there is still an improvement in the performance of the model.

```bash
hammer feature-sets-synergy \
    --verbose \
    --holdouts 10 \
    --dataset NPC \
    --base-feature-sets "layered" "van_der_waals_surface_area" \
    --test-size 0.2 \
    --validation-size 0.2 \
    --performance-path "tertiary_feature_sets_synergy_training.csv" \
    --training-directory "tertiary_feature_sets_synergy_training" \
    --barplot-directory "tertiary_feature_sets_synergy_barplots"
```

### Train a model variant

```bash
hammer train \
    --verbose \
    --dataset NPCHarmonized \
    --include-extended-connectivity \
    --test-size 0.2 \
    --training-directory "npc.harmonized.v1.tar.gz"
```

### Predict

You can run predictions for a single SMILES using the following command:

```bash
hammer predict \
    --input "CN1[C@H]2CC[C@@H]1[C@@H](C(OC)=O)[C@@H](OC(C3=CC=CC=C3)=O)C2" \
    --version npc.harmonized.v1
```

which will output:

```bash
SMILES: CN1[C@H]2CC[C@@H]1[C@@H](C(OC)=O)[C@@H](OC(C3=CC=CC=C3)=O)C2
Alkaloids (0.9942)
└── Ornithine alkaloids (0.9988)
    └── Tropane alkaloids (0.9999)
```

Analogously, by running the following command for a multi-class compound:

```bash
hammer predict \
    --input "CCC(C)C1NC(=O)C(Cc2ccccc2)N(C)C(=O)C(C(C)CC)N2C(=O)C(CCC2OC)NC(=O)C(CCCN=C(N)N)NC(=O)C(NC(=O)C(CO)OS(=O)(=O)O)C(C)OC1=O" \
    --version npc.harmonized.v1
```

you will get the following output:

```bash
SMILES: CCC(C)C1NC(=O)C(Cc2ccccc2)N(C)C(=O)C(C(C)CC)N2C(=O)C(CCC2OC)NC(=O)C(CCCN=C(N)N)NC(=O)C(NC(=O)C(CO)OS(=O)(=O)O)C(C)OC1=O
Amino acids and Peptides (0.9807)
└── Oligopeptides (0.9994)
Polyketides (0.9821)
└── Oligopeptides (0.9994)
    ├── Cyclic peptides (0.9999)
    ├── Depsipeptides (0.9998)
    └── Ahp-containing cyclodepsipeptides (0.9385)
```

You can also run predictions for SMILES from a CSV, TSV or SSV file:

```bash
hammer predict \
    --input "divergent_npc_entries/divergent_pathways.csv" \
    --version npc.harmonized.v1 \
    --verbose \
    --output-dir "divergent_npc_entries/npc.harmonized.v1/"
```

## Citation

If you use this model in your research, please cite us:

[TODO: we still need to properly publish the model, so this is a placeholder and will be updated in the future]

```bibtex
@software{hammer,
  author = {Cappelletti, Luca, et al.},
  title = {Hammer: Hierarchical Augmented Multi-modal Multi-task classifiER},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LucaCappelletti94/hammer}},
}
```

## Contributing

If you want to contribute to this project, please read the [CONTRIBUTING](CONTRIBUTING.md) file for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
