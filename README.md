# Single_cell_sequencing
Machine Learning for Single-Cell RNA-seq Analysis
This three-person group project presents an end-to-end analysis of single-cell RNA-sequencing data, combining exploratory data analysis, biologically informed preprocessing, unsupervised learning, and supervised classification. The project systematically evaluates a large number of preprocessing and modelling combinations to investigate cellular structure and classify hematopoietic cell populations from their gene-expression profiles.
Project Objective
The objective of the project is to analyse the statistical and biological structure of single-cell RNA-seq data and develop machine learning pipelines capable of:
- identifying meaningful cellular groupings without using class labels;
- classifying cells into ten broad hematopoietic populations;
- comparing alternative preprocessing, representation-learning, clustering, and classification strategies;
- evaluating model-selection procedures while preventing information leakage;
- generating predictions for previously unlabeled cells.
Dataset
The dataset contains single-cell RNA-sequencing profiles for 1,499 cells and 4,290 genes. Gene-expression values and cell-level metadata are stored in an AnnData (.h5ad) file, while a separate training file provides broad hematopoietic lineage labels for approximately two-thirds of the cells. The remaining cells are unlabeled and are used as a hidden prediction set.
The ten target populations include long- and short-term hematopoietic stem cells, multipotent and lymphoid-primed progenitors, and increasingly committed myeloid lineages. The labeled data are strongly imbalanced, making macro-averaged F1 a more appropriate evaluation metric than accuracy alone.
Project Workflow
1. Exploratory Data Analysis
The first stage characterises the quality, sparsity, and biological composition of the dataset before modelling. It includes:
- per-cell library size, number of detected genes, and fraction of zero values;
- per-gene detection rate, sparsity, mean expression, and variability;
- analysis of mitochondrial, ribosomal, hemoglobin, immunoglobulin, and T-cell receptor gene families;
- identification of highly expressed and highly variable genes;
- analysis of class distribution and label coverage;
- investigation of biological signals that may influence downstream modelling.
2. Biologically Informed Preprocessing
Several preprocessing strategies are explored to determine how biological and technical sources of variation affect the learned representations. These include:
- selection of different numbers of highly variable genes (HVGs);
- four levels of gene filtering: none, mild, medium, and aggressive;
- filtering before or after HVG selection;
- optional regression of cell-cycle effects;
- direct use of scaled HVG expression or dimensionality-reduced representations;
- optional inclusion of curated marker-gene scores for hematopoietic populations.
All feature-selection and representation-learning steps are fitted exclusively on the relevant training folds to prevent data leakage.
3. Unsupervised Learning
The unsupervised section investigates the structure of the expression space through multiple dimensionality-reduction methods:
- Principal Component Analysis (PCA);
- t-distributed Stochastic Neighbor Embedding (t-SNE);
- Uniform Manifold Approximation and Projection (UMAP);
- autoencoder-based latent representations.
Each representation is combined with several clustering approaches, including K-means, bisecting K-means, and agglomerative hierarchical clustering with different linkage criteria and distance metrics. The resulting pipelines are compared through silhouette scores and visual inspection, with known biological labels used only as an external reference for interpretation.
4. Supervised Learning
The supervised section addresses the classification of cells into ten hematopoietic populations. The tested model families include:
- logistic regression;
- linear and RBF support vector machines;
- multilayer perceptrons;
- Random Forests;
- Extremely Randomized Trees (ExtraTrees);
- XGBoost;
- Gaussian Naive Bayes.
Each classifier is combined with alternative preprocessing configurations and model-specific hyperparameters. The complete Cartesian search space contains approximately 4.3 million possible pipelines; under the available computational budget, approximately 468 fixed configurations are selected through a balanced sampling strategy and evaluated across a maximum of 15,000 model fits.
Nested Cross-Validation
Model selection and evaluation are performed through a leak-free nested cross-validation procedure:
- an 8-fold outer cross-validation estimates how well the complete model-selection procedure generalises to unseen data;
- a 4-fold inner cross-validation evaluates each fixed pipeline within every outer-training fold;
- preprocessing, HVG selection, filtering, PCA or autoencoder fitting, marker-score construction, and classifier training are repeated from scratch within each fold;
- macro-F1 is used as the primary metric to give equal importance to common and rare cell populations;
- the selected pipeline is refitted on the complete labeled dataset and used to predict the unlabeled cells.
The analysis also includes soft-voting ensemble experiments, a meta-analysis of preprocessing and hyperparameter importance, and a biological interpretation of classification errors among closely related cell populations.

-> Team and contributions

This project was developed collaboratively by a team of three students.  The repository presents the complete group project, including jointly developed analyses, experiments and documentation.

