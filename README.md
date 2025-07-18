# Unsupervised Learning Project: Heart Attack Prediction Analysis

## Overview

This project implements a comprehensive unsupervised learning pipeline for analyzing heart attack prediction data from Indonesia. The project explores various dimensionality reduction techniques (PCA, ICA, Random Projection) and clustering algorithms (K-Means, Gaussian Mixture Models) to understand patterns in cardiovascular health data.

## Project Structure

```
unsupervised/
├── main.py                 # Main pipeline orchestrator
├── download_data.py        # Data download and preprocessing
├── feature_selector.py     # Backward feature elimination
├── sil_scores.py          # Silhouette score computation
├── clusters_as_input.py   # Cluster feature augmentation
├── eval.py                # Model evaluation
├── mlp.py                 # Neural network implementation
├── plotting.py            # Visualization generation
├── datasets/              # Processed datasets
│   ├── og/               # Original data
│   ├── scaled/           # Standardized data
│   ├── pca/              # PCA-transformed data
│   ├── ica/              # ICA-transformed data
│   └── rp/               # Random projection data
├── selected_datasets/     # Feature selection results
├── results/              # Evaluation results
└── plots/                # Generated visualizations
```

## Features

### Data Processing
- **Dataset**: Heart Attack Prediction Indonesia dataset from Kaggle
- **Preprocessing**: Categorical encoding, standardization, train/test splitting
- **Dimensionality Reduction**: PCA, ICA, and Random Projection with 95% variance retention

### Clustering Algorithms
- **K-Means**: Traditional k-means clustering with k=2
- **Gaussian Mixture Models**: GMM with full covariance matrices
- **Evaluation Metrics**: Adjusted Rand Index (ARI) and accuracy

### Feature Selection
- **Backward Elimination**: Iterative feature removal based on clustering performance
- **Multiple Models**: Feature selection using K-Means, GMM, and Neural Network baselines
- **Cross-Validation**: Robust feature selection across different data representations

### Neural Network Baseline
- **Architecture**: 2-hidden layer MLP with ReLU activation
- **Training**: Adam optimizer with early stopping
- **Integration**: scikit-learn compatible wrapper for seamless pipeline integration

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster neural network training)

### Dependencies
```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn tqdm kagglehub joblib
```

## Usage

### Quick Start
Run the complete pipeline:
```bash
python main.py
```

This executes the following steps in sequence:
1. **Data Download & Preprocessing** (`download_data.py`)
2. **Feature Selection** (`feature_selector.py`)
3. **Silhouette Analysis** (`sil_scores.py`)
4. **Cluster Feature Augmentation** (`clusters_as_input.py`)
5. **Model Evaluation** (`eval.py`)
6. **Visualization Generation** (`plotting.py`)

### Individual Components

#### Data Preparation
```bash
python download_data.py
```
Downloads the dataset, performs preprocessing, and creates multiple data representations.

#### Feature Selection
```bash
python feature_selector.py
```
Performs backward feature elimination using clustering and neural network models.

#### Silhouette Analysis
```bash
python sil_scores.py
```
Computes silhouette scores for different numbers of clusters (k=2 to 25).

#### Model Evaluation
```bash
python eval.py
```
Evaluates clustering and neural network models on test data.

#### Visualization
```bash
python plotting.py
```
Generates comprehensive visualizations including:
- Accuracy comparison bar charts
- PCA variance explained curves
- Silhouette score curves
- ARI vs features plots
- Covariance matrix heatmaps

## Data Representations

The project creates four different data representations:

1. **Original (OG)**: Standardized original features
2. **PCA**: Principal Component Analysis with 95% variance retention
3. **ICA**: Independent Component Analysis with same component count as PCA
4. **RP**: Random Projection with same component count as PCA

## Models Evaluated

### Clustering Models
- **K-Means**: Traditional clustering with k=2
- **GMM**: Gaussian Mixture Model with 2 components

### Neural Network Models
- **NN**: Baseline neural network on original features
- **NN+Clust**: Neural network with additional cluster features

## Results

The pipeline generates several output files:

### Performance Metrics
- `results/unsup_test_scores.csv`: Test accuracy and ARI scores
- `results/silhouette_scores.csv`: Silhouette scores for different k values
- `results/nn_cluster_augmented.csv`: Neural network performance with cluster features

### Feature Selection
- `selected_datasets/selection_curves.csv`: Feature selection performance curves
- `selected_datasets/{model}/{dataset}_mask.json`: Selected feature masks

### Visualizations
- `plots/main_results.png`: Main accuracy comparison
- `plots/silhouette_curves.png`: Silhouette analysis
- `plots/pca_variance.png`: PCA variance explained
- `plots/ari_vs_features.png`: Feature selection curves
- `plots/covariance_matrices.png`: Data correlation analysis

## Key Findings

The project demonstrates:
- Effectiveness of different dimensionality reduction techniques for clustering
- Impact of feature selection on clustering performance
- Benefits of combining clustering with supervised learning
- Trade-offs between model complexity and performance

## Configuration

Key configuration parameters can be modified in individual scripts:

- **Sample Size**: Control dataset size for faster experimentation
- **Device**: CPU/GPU selection for neural network training
- **Random State**: Ensure reproducibility across runs
- **Model Parameters**: Adjust clustering and neural network hyperparameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with the original dataset's license terms.

## Acknowledgments

- Dataset: Heart Attack Prediction Indonesia from Kaggle
- Libraries: scikit-learn, PyTorch, pandas, matplotlib, seaborn
- Academic guidance for unsupervised learning methodology 