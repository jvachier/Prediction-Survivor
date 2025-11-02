# Prediction-Survival

[![Checks](https://github.com/jvachier/Prediction-Survivor/workflows/Checks/badge.svg)](https://github.com/jvachier/Prediction-Survivor/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Predict survival outcomes for Titanic passengers using both classical machine learning ensemble methods and deep neural networks. This project demonstrates modern Python packaging with `uv`, comprehensive feature engineering, and multiple modeling approaches.

## Project Overview

This repository implements two distinct approaches to the Titanic survival prediction problem:

1. **Ensemble Machine Learning**: A soft-voting classifier combining multiple algorithms (Random Forest, AdaBoost, Logistic Regression, Decision Trees, SGD, and K-Nearest Neighbors) with cross-validation and stratified K-folds
2. **Deep Neural Network**: A dense feedforward network with early stopping and cross-validation

### Key Features

- Feature engineering including title extraction, deck mapping, and age binning
- Support for StandardScaler preprocessing
- Stratified K-fold cross-validation for robust model evaluation
- Automatic caching of preprocessed data for faster iteration
- Comprehensive logging for pipeline transparency
- Modern Python tooling with `uv` package manager

## Prerequisites

- **Python 3.11** (managed automatically by `uv`)
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package installer and resolver

### Installing uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/jvachier/Prediction-Survivor.git
cd Prediction-Survivor
```

### 2. Install dependencies

```bash
uv sync --group dev
```

Or using the Makefile:

```bash
make install
```

### 3. Run predictions

**Ensemble model:**
```bash
uv run python -m src.main --model_ensemble
```

**Neural network:**
```bash
uv run python -m src.main
```

**With StandardScaler preprocessing:**
```bash
uv run python -m src.main --model_ensemble --standardscaler
```

## Project Structure

```
Prediction-Survival/
├── src/
│   ├── main.py                    # Entry point and pipeline orchestration
│   ├── data/                      # Training and test CSV files
│   ├── modules/
│   │   ├── loading.py             # Data loading and caching utilities
│   │   ├── data_preparation.py   # Feature engineering and preprocessing
│   │   └── models.py              # ML ensemble and neural network models
│   ├── pickle_files/              # Cached preprocessed data (gitignored)
│   └── predictions/               # Output CSV files (gitignored)
├── pyproject.toml                 # Project metadata and dependencies
├── Makefile                       # Development automation commands
└── .github/workflows/             # CI/CD configuration
```

## Development

### Available Make Commands

```bash
make install    # Install all dependencies including dev tools
make lint       # Run pylint static analysis
make ruff       # Run Ruff linter with auto-fix and formatting
make black      # Format code with Black
```

### Running Tests Locally

The project uses GitHub Actions for continuous integration. The workflow automatically:
- Installs dependencies with `uv`
- Runs Ruff linting and format checks
- Runs Pylint static analysis

Workflow only triggers on changes to:
- `src/**` - Source code files
- `pyproject.toml` - Dependency changes
- `.github/workflows/checks.yml` - CI configuration

### Code Quality Tools

- **Ruff**: Fast Python linter and formatter
- **Black**: Opinionated code formatter
- **Pylint**: Comprehensive static analysis

## Model Details

### Ensemble Classifier

Uses a `VotingClassifier` with soft voting combining:
- **Random Forest** (50 estimators, max depth 10)
- **AdaBoost** (100 estimators, SAMME algorithm)
- **Logistic Regression** (LBFGS solver)
- **Decision Tree** (max depth 10, sqrt features)
- **SGD Classifier** (log loss, adaptive learning rate)
- **K-Nearest Neighbors** (50 neighbors)

Evaluated using 10-fold stratified cross-validation with ROC-AUC scoring.

### Neural Network

Architecture:
- Input layer: Variable size based on features
- Hidden layers: 512 → 256 → 256 → 128 → 128 (Dropout 0.1) → 64 → 64 → 32 → 32
- Output layer: 2 units (sigmoid activation)
- Optimizer: Adam (learning rate 1e-5)
- Loss: Binary crossentropy
- Early stopping: Patience 50 on validation loss
- Training: 10-fold stratified cross-validation, max 1500 epochs

## Feature Engineering

The pipeline includes extensive feature engineering:

1. **Title Extraction**: Extract titles from passenger names (Mr, Mrs, Miss, Master, Rare)
2. **Deck Mapping**: Convert cabin information to deck levels
3. **Age Binning**: Group ages into meaningful categories
4. **Family Features**: Calculate relatives count and isolation flag
5. **Fare Binning**: Categorize fares into quintiles
6. **Interaction Features**: Age × Class, Fare per person
7. **One-Hot Encoding**: Convert categorical features using `OneHotEncoder`
8. **Optional Scaling**: StandardScaler for numerical features

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Ensure your code passes all checks:
```bash
make ruff
make lint
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Author

**Jeremy Vachier**

## Acknowledgments

- Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic)
- Built with modern Python tooling: `uv`, `ruff`, `scikit-learn`, and `tensorflow`