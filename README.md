# Shapely Regression: Game-theoretic Extensions of Logistic Regression

This repository provides an implementation of Shapley regression, a game-theoretic extension of logistic regression based on the Choquet integral. The framework is designed to model non-linear feature interactions while preserving the interpretability, efficiency, and statistical grounding of classical logistic regression.

## Project Overview

Traditional logistic regression assumes additive feature effects, which limits its ability to model interactions between variables. This project introduces **Shapley regression**, a principled extension of logistic regression based on the Choquet integral, designed to capture pairwise and higher-order feature interactions while preserving interpretability.

The framework is motivated by application settings in which datasets are typically small, heterogeneous, and noisy. In such regimes, standard linear models fail to represent meaningful interaction patterns, while deep learning approaches often suffer from instability and limited interpretability. Shapley regression bridges this gap by explicitly modeling interactions in a structured and transparent manner, while maintaining the convexity and statistical grounding of classical regression models.

At its core, Shapley regression replaces the linear predictor with a *k*-additive cooperative game. This formulation enables controlled interaction modeling with theoretical guarantees on model complexity and performance, allowing predictive power, robustness, and interpretability to be balanced through the choice of *k* and regularization.

The repository provides both theoretical and empirical tools for analyzing these trade-offs, along with experimental pipelines demonstrating the applicability of Shapley regression to biomedical classification tasks and structured clinical data.

### Key Features

- **Multiple Representation Bases**: Supports Game, Möbius, and Shapley representations of the Choquet integral. These bases are mathematically equivalent but offer distinct interpretability and sparsity properties.
- **K-additivity Analysis**: Tools to study the trade-off between model complexity, interpretability, and predictive performance as a function of k-additivity.
- **Robustness Testing**: Comprehensive framework for testing model robustness under various perturbations
- **Visualization Tools**: Specialized visualization functions for each representation basis

## Repository Structure

```
project/
├── core/                    # Core Choquet and regression implementations
│   ├── models/
│   │   ├── choquet.py        # Choquet integral implementations
│   │   └── regression.py     # ChoquisticRegression model
│   └── __init__.py
│
├── paper_code/               # Code used for experiments in the paper
│   ├── APDS/                 # APDS case study
│   │   ├── models.py         # Model definitions
│   │   ├── preprocess.py     # APDS data preprocessing
│   │   ├── run_apds_experiment.py  # Main experiment script
│   │   └── visualisation.py  # Pairwise interaction visualization
│   │
│   └── benchmark/            # Benchmark experiments
│       └── bootstrap_and_noise_robustness/
│           └── robustness_k_add.py       # K-additivity analysis with noise/bootstrap
│
├── utils/                    # Shared utilities
│   ├── plotting.py           # Visualization helpers
│   ├── data_loader.py        # Benchmark data loaders
│   └── metrics.py            # Evaluation metrics
│
├── examples/                 # Example usage scripts
│   ├── comparison_example.py
│   ├── plot_coefficients_example.py
│   └── plot_interaction_matrix_example.py
│
├── results/                  # Experimental results (auto-generated)
│   └── benchmark/
│       ├── noise_robustness/ # Noise robustness results by dataset
│       └── bootstrap/        # Bootstrap stability results by dataset
│
├── data/                     # Data directory (git-ignored)
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
└── README.md                 # Project documentation
```

## Mathematical Background

The project is based on three different mathematical bases that are linearly related:

1. **Game Representation**: The traditional representation of fuzzy measures with the caveat of having fewer restrictions such as monotonicity.
2. **Möbius Representation**: An alternative representation that directly captures the interaction between features.
3. **Shapley Representation**: A representation that uses the Shapley value and the pairwise interaction indices between features.

Each representation has its own interpretability properties and is suitable for different types of analysis.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[anonymous]/shapely_regression.git
cd shapely_regression
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Install additional dependencies if needed:
```bash
pip install -r requirements.txt
```

### Option 2: Use Without Installation

If you prefer not to install the package, you can still run the scripts by adding the project root to your Python path:

1. Clone the repository:
```bash
git clone https://github.com/[anonymous]/shapely_regression.git
cd shapely_regression
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts from the project root:
```bash
python -m examples.comparison_example
```

## Dataset Preparation

Before running the examples, you need to place your datasets in the `data/` directory. The data loader expects the following files:

- `data_apds.csv`: APDS dataset
- `data_banknotes.csv`: Banknote authentication dataset (with header: authentic column)
- `transfusion.csv`: Blood Transfusion Service Center Data Set
- `data_mammographic.data`: Mammographic mass dataset
- `data_raisin.xlsx`: Raisin dataset
- `data_rice.xlsx`: Rice (Commeo and Osmancik) dataset
- `diabetes.csv`: Diabetes (PIMA) dataset
- `data_skin.csv`: Skin segmentation dataset
- `dados_covid_sbpo_atual.csv`: COVID SBPO dataset
- `pure_pairwise_interaction_dataset.csv`: Pure pairwise interaction dataset

**Note**: The data directory is git-ignored to prevent pushing large datasets to the repository.

### Data Preprocessing

The `data_loader.py` automatically applies:
- **Class balancing** via `RandomOverSampler` for imbalanced datasets (not applied to synthetic datasets)
- **Missing value handling** for specific datasets (e.g., COVID, mammographic)
- **Data type conversions** as needed

## Usage

### Basic Example

```python
from core.models.regression import ChoquisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_loader import func_read_data

# Load data
X, y = func_read_data("banknotes")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model with game representation
model_game = ChoquisticRegression(
    representation="game",
    k_add=2,
    scale_data=True
)
model_game.fit(X_train, y_train)

# Create and train model with Mobius representation
model_mobius = ChoquisticRegression(
    representation="mobius",
    k_add=2,
    scale_data=True
)
model_mobius.fit(X_train, y_train)

# Create and train model with Shapley representation (k=2)
model_shapley = ChoquisticRegression(
    representation="shapley",
    k_add=2,
    scale_data=True
)
model_shapley.fit(X_train, y_train)

# Evaluate models
y_pred_game = model_game.predict(X_test)
y_pred_mobius = model_mobius.predict(X_test)
y_pred_shapley = model_shapley.predict(X_test)

print(f"Game representation accuracy: {accuracy_score(y_test, y_pred_game):.4f}")
print(f"Mobius representation accuracy: {accuracy_score(y_test, y_pred_mobius):.4f}")
print(f"Shapley representation accuracy: {accuracy_score(y_test, y_pred_shapley):.4f}")
```

### Running the Example Script

The repository includes an example script that demonstrates the use of different representations:

```bash
# If installed as a package (Option 1):
python examples/comparison_example.py

# If using without installation (Option 2):
python -m examples.comparison_example
```

### K-additivity Analysis with Robustness Testing

Run comprehensive k-additivity analysis with noise robustness and bootstrap stability:

```python
import sys
import os
sys.path.append(os.path.abspath('.'))

from paper_code.benchmark.bootstrap_and_noise_robustness.robustness_k_add import run_analysis_for_dataset

# Run analysis for a single dataset
results = run_analysis_for_dataset(
    dataset="banknotes",
    representation="shapley",
    regularization="l2",
    random_state=42
)

# Results are saved to:
# - results/benchmark/noise_robustness/banknotes/
# - results/benchmark/bootstrap/banknotes/
```

For multiple datasets:

```python
datasets = ['banknotes', 'mammographic', 'diabetes']

for dataset in datasets:
    run_analysis_for_dataset(
        dataset=dataset,
        representation="shapley",
        regularization="l2"
    )
```

### Visualizing Results

```python
from paper_code.benchmark.bootstrap_and_noise_robustness.paper_plots_same_scale import main

# Generate plots for a specific dataset
main(dataset_name='banknotes')

# Plots are saved to:
# - results/benchmark/noise_robustness/banknotes/noise_robustness_scaled.png
# - results/benchmark/bootstrap/banknotes/bootstrap_stability_scaled.png
```

### Understanding the Results

The analysis generates:

1. **CSV files** (`results.csv`) with metrics for each k value:
   - Number of parameters
   - Baseline accuracy
   - Noise robustness (at 0.1, 0.2, 0.3 noise levels)
   - Bootstrap stability (mean ± std)

2. **Summary files** (`summary.txt`) with full results table

3. **Plots**:
   - Noise robustness vs k-additivity
   - Bootstrap stability vs k-additivity
   - Scaled versions for cross-dataset comparison

### Key Features

- **Automatic class balancing**: Uses `RandomOverSampler` to handle imbalanced datasets
- **Noise robustness**: Tests model performance under Gaussian noise (scaled by feature std)
- **Bootstrap stability**: Evaluates prediction consistency across bootstrap samples
- **K-additivity sweep**: Analyzes all k values from 1 to n_features

## Visualization for Interpretability

```python
from utils.plotting import plot_coefficients, plot_interaction_matrix_2add

# Plot model coefficients
plot_coefficients(
    feature_names=X.columns.tolist(),
    all_coefficients=[model_shapley.coef_[0]],
    plot_folder="results/",
    k_add=2
)

# Plot interaction matrix for Shapley representation (2-additive model)
plot_interaction_matrix_2add(
    feature_names=X.columns.tolist(),
    coefs=model_shapley.coef_[0],  # Coefficients from the fitted model
    plot_folder="results/"
)
```

## Troubleshooting

### Import Errors

If you encounter import errors like `ModuleNotFoundError: No module named 'core'`, try one of these solutions:

1. **Install the package** (recommended):
   ```bash
   pip install -e .
   ```

2. **Run from the project root**:
   ```bash
   python -m examples.comparison_example
   ```

3. **Add the project root to your Python path** in your script:
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
