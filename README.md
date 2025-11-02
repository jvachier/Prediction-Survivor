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

- **Feature Engineering**: Title extraction, deck mapping, age binning, and interaction features
- **Two Model Approaches**: Classical ensemble (StackingClassifier) and deep neural network
- **StandardScaler Preprocessing**: Automatic for neural networks, optional for ensemble
- **Model Persistence**: Automatic saving/loading with timestamps (`.joblib` for ensemble, `.keras` for NN)
- **Prediction Validation**: Automatic format validation before CSV export
- **Robust Error Handling**: Graceful fallback to training if model loading fails
- **Stratified K-fold Cross-validation**: For robust model evaluation
- **Automatic Data Caching**: Preprocessed data cached with `joblib` for faster iteration
- **Comprehensive Testing**: 35+ tests covering models, data preparation, and validation
- **Modern Python Tooling**: Built with `uv`, `ruff`, `scikit-learn`, and `tensorflow`

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

#### Basic Usage

**Neural network (default, with automatic StandardScaler):**
```bash
uv run python -m src.main
```

**Ensemble model:**
```bash
uv run python -m src.main --model_ensemble
```

#### Advanced Options

**Force retrain (ignore saved models):**
```bash
uv run python -m src.main --retrain
```

**Explicitly load existing model:**
```bash
uv run python -m src.main --load_model
```

**Ensemble with StandardScaler preprocessing:**
```bash
uv run python -m src.main --model_ensemble --standardscaler
```

**Use custom config file:**
```bash
uv run python -m src.main --config path/to/config.yaml
```

#### Model Loading Behavior

- **Default**: Automatically loads the most recent saved model if available
- **Model Files**: 
  - Ensemble models: `pickle_files/model/model_ensemble_YYYYMMDD_HHMMSS.joblib`
  - Neural networks: `pickle_files/model/neural_network_YYYYMMDD_HHMMSS.keras`
- **Fallback**: If loading fails, automatically retrains from scratch
- **Force Retrain**: Use `--retrain` to ignore existing models and train fresh

#### Prediction Output

Predictions are saved to `src/predictions/` with automatic validation:
- **Format**: CSV with columns `PassengerId` (int) and `Survived` (int: 0 or 1)
- **Validation**: Checks for correct data types, binary values, and no missing values
- **Filenames**: 
  - `prediction_titanic_RFC_new.csv` (ensemble)
  - `prediction_titanic_NN.csv` (neural network)

## Project Structure

```
Prediction-Survival/
├── src/
│   ├── main.py                    # Entry point and pipeline orchestration
│   ├── data/                      # Training and test CSV files
│   │   ├── train.csv
│   │   └── test.csv
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── loading.py             # Data loading and joblib caching
│   │   ├── data_preparation.py   # Feature engineering and preprocessing
│   │   └── models.py              # ML ensemble and NN implementations
│   ├── pickle_files/              # Cached data and models (gitignored)
│   │   ├── loading/               # Cached raw data
│   │   ├── data_preparation/      # Cached preprocessed data
│   │   └── model/                 # Saved trained models (.joblib, .keras)
│   └── predictions/               # Output CSV files (gitignored)
│       ├── prediction_titanic_RFC_new.csv  # Ensemble predictions
│       └── prediction_titanic_NN.csv       # Neural network predictions
├── tests/
│   ├── test_config.py             # Configuration management tests
│   ├── test_loading.py            # Data loading tests
│   ├── test_data_preparation.py  # Feature engineering tests
│   ├── test_models.py             # Model training and persistence tests
│   └── test_main.py               # Prediction validation tests
├── config.yaml                    # Centralized hyperparameter configuration
├── pyproject.toml                 # Project metadata and dependencies
├── Makefile                       # Development automation commands
├── requirements.txt               # Python dependencies (generated by uv)
└── .github/workflows/             # CI/CD configuration
    └── checks.yml
```

## Development

### Available Make Commands

```bash
make install    # Install all dependencies including dev tools
make lint       # Run pylint static analysis
make ruff       # Run Ruff linter with auto-fix and formatting
make black      # Format code with Black
make test       # Run pytest test suite (35+ tests)
```

### Running Tests

The project includes comprehensive test coverage (35+ tests):

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run specific test modules
uv run pytest tests/test_main.py -v          # Prediction validation tests
uv run pytest tests/test_models.py -v        # Model persistence tests
uv run pytest tests/test_data_preparation.py -v  # Feature engineering tests
```

Test coverage includes:
- **Configuration**: Singleton pattern, YAML loading, dot-notation access
- **Data Loading**: CSV loading with joblib caching
- **Data Preparation**: Feature engineering, binning, scaling, column validation
- **Models**: Ensemble and NN initialization, architecture, save/load persistence
- **Validation**: Prediction format, data types, binary values, edge cases

### Code Quality Tools

- **Ruff**: Fast Python linter and formatter
- **Black**: Opinionated code formatter
- **Pylint**: Comprehensive static analysis
- **pytest**: Testing framework with 35+ comprehensive tests

### Continuous Integration

The project uses GitHub Actions for CI/CD. The workflow automatically:
- Installs dependencies with `uv`
- Runs Ruff linting and format checks
- Runs Pylint static analysis
- Executes full test suite

Workflow triggers on changes to:
- `src/**` - Source code files
- `tests/**` - Test files
- `pyproject.toml` - Dependency changes
- `.github/workflows/checks.yml` - CI configuration

## Model Details

### Ensemble Classifier (StackingClassifier)

Uses a `StackingClassifier` with 9 base models and logistic regression meta-learner:

**Base Models:**
- **Random Forest** (100 estimators, entropy criterion)
- **AdaBoost** (50 estimators, SAMME algorithm)
- **Logistic Regression** (saga solver, L2 penalty)
- **Decision Tree** (entropy criterion, sqrt features)
- **SGD Classifier** (log loss, L2 penalty, adaptive learning)
- **K-Nearest Neighbors** (50 neighbors, uniform weights)
- **XGBoost** (100 estimators, 0.1 learning rate)
- **LightGBM** (100 estimators, 0.1 learning rate)
- **CatBoost** (100 iterations, 0.1 learning rate)

**Meta-Learner:** Logistic Regression (lbfgs solver, L2 penalty)

**Training:** 10-fold stratified cross-validation

### Neural Network

Built with Keras Functional API:

**Architecture:**
- **Input Layer**: Variable size based on features (typically ~60 after one-hot encoding)
- **Hidden Layers**:
  - Dense(256) → BatchNorm → Dropout(0.3) → L2(0.01)
  - Dense(128) → BatchNorm → Dropout(0.3) → L2(0.01)
  - Dense(64) → BatchNorm → Dropout(0.3) → L2(0.01)
  - Dense(32) → L2(0.01)
- **Output Layer**: Dense(1, activation='sigmoid')

**Training Configuration:**
- **Optimizer**: Adam (learning rate 0.001)
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, AUC, Precision, Recall
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16
- **Validation Split**: 20%
- **Early Stopping**: Patience 10 on validation loss, restore best weights
- **Cross-Validation**: 10-fold stratified

**Preprocessing**: Automatically applies StandardScaler to numerical features (Age, Fare)

## Feature Engineering

The pipeline includes extensive feature engineering in `data_preparation.py`:

### Stage 1: Basic Feature Extraction
1. **Title Extraction**: Extract titles from passenger names (Mr, Mrs, Miss, Master, Rare titles)
2. **Deck Mapping**: Convert cabin information to deck levels (A-G, Unknown)
3. **Initial Selection**: Keep essential columns for further processing

### Stage 2: Advanced Feature Creation
4. **Age Binning**: Group ages into 5 meaningful categories using `pd.cut()`
5. **Fare Binning**: Categorize fares into 4 quintiles
6. **Family Features**: 
   - `relatives`: Total family size (SibSp + Parch)
   - `not_alone`: Binary flag for passengers with family
   - `Fare_Per_Person`: Fare divided by family size
7. **Interaction Features**: `Age_Class` (Age × Pclass interaction)

### Stage 3: Encoding and Scaling
8. **One-Hot Encoding**: Convert categorical features using `OneHotEncoder` (drop first category)
9. **StandardScaler** (Optional): Applied to Age and Fare for neural network training

### Validation
- **Column Validation**: Checks for required columns before processing
- **Missing Value Handling**: Automatic imputation during binning
- **Type Safety**: Ensures correct dtypes for all features

All preprocessing steps are cached using `joblib` for efficient reloading.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Quality Standards

Ensure your code passes all checks before submitting:
```bash
make ruff       # Linting and formatting
make lint       # Static analysis with pylint
make test       # Run full test suite (requires 35+ tests passing)
```

### Testing Guidelines

When adding new features, include appropriate tests:
- **Model changes**: Add tests to `tests/test_models.py`
- **Data processing**: Add tests to `tests/test_data_preparation.py`
- **Validation logic**: Add tests to `tests/test_main.py`
- **Configuration**: Add tests to `tests/test_config.py`

Run specific test files during development:
```bash
uv run pytest tests/test_your_module.py -v
```

### Prediction Validation

All predictions must pass automatic validation:
- **PassengerId**: Integer type, positive values
- **Survived**: Integer type, binary values (0 or 1 only)
- **No missing values**: All rows must be complete
- **Correct format**: Exactly 2 columns in specified order

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Author

**Jeremy Vachier**

## Acknowledgments

- Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic)
- Built with modern Python tooling: `uv`, `ruff`, `scikit-learn`, and `tensorflow`