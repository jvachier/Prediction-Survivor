# Contributing to Prediction-Survival

Thank you for your interest in contributing to this Titanic survival prediction project!

## Development Setup

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

```bash
git clone https://github.com/jvachier/Prediction-Survivor.git
cd Prediction-Survivor

# Install dependencies (including dev dependencies)
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Pre-commit Hooks (Optional but Recommended)

```bash
uv run pre-commit install
```

This will automatically run code formatters and linters before each commit.

## Running Tests

```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov

# Run only failed tests (faster)
make test-fast

# Run specific test file
uv run pytest tests/test_models.py -v

# Run specific test
uv run pytest tests/test_models.py::test_neural_network_model_architecture -v
```

## Code Quality Checks

Before submitting a PR, ensure your code passes all quality checks:

```bash
# Format code with Ruff
make format

# Run linter
make lint

# Run all checks
make format && make lint && make test
```

## Code Standards

### General Guidelines

- **Follow PEP 8** style guidelines (enforced by Ruff)
- **Add type hints** to all function signatures
- **Write docstrings** for public APIs (Google style preferred)
- **Keep functions focused** and small (< 50 lines)
- **Use logging** instead of print statements
- **Use configuration system** for all parameters

### Example Code Style

```python
"""Module docstring describing the file."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MyClass:
    """Class docstring describing the purpose.
    
    Attributes:
        data: The input data for processing
        config: Configuration object
    """
    
    data: pd.DataFrame
    config: dict = None
    
    def __post_init__(self) -> None:
        """Load configuration after initialization."""
        if self.config is None:
            self.config = get_config()
    
    def process_data(self, threshold: float = 0.5) -> pd.DataFrame:
        """Process the data with given threshold.
        
        Args:
            threshold: Threshold value for filtering (default: 0.5)
            
        Returns:
            Processed DataFrame
            
        Raises:
            ValueError: If threshold is not between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        logger.info(f"Processing data with threshold {threshold}")
        # Implementation here
        return self.data
```

## Adding New Features

### Adding a New Model

1. **Add model configuration** to `config.yaml`:

```yaml
model_ensemble:
  my_new_model:
    param1: value1
    param2: value2
    random_state: 1
```

2. **Add model to `ModelEnsemble`** in `src/modules/models.py`:

```python
def model_cross(self) -> StackingClassifier:
    """Perform cross validation on individual estimators."""
    # Load config
    my_model_config = self.config.get("model_ensemble.my_new_model")
    
    # Create model
    clf_my_model = MyModelClassifier(**my_model_config, n_jobs=self.n_jobs)
    
    # Add to pipeline
    pipe_my_model = Pipeline([["my_model", clf_my_model]])
    
    # Add to stacking estimators list
    # Add to clf_labels and all_clf lists
```

3. **Add tests** in `tests/test_models.py`:

```python
def test_my_new_model():
    """Test my new model initialization."""
    # Test implementation
```

4. **Update documentation** in README.md

### Modifying Data Preparation

1. **Update** `src/modules/data_preparation.py`
2. **Add configuration** if needed in `config.yaml`
3. **Add tests** in `tests/test_data_preparation.py`
4. **Run full test suite** to ensure no regressions

### Modifying Neural Network Architecture

1. **Update configuration** in `config.yaml` under `neural_network.architecture`
2. **Test changes**: `uv run python src/main.py`
3. **Add tests** if adding new functionality
4. **Document** significant architecture changes

## Configuration System

All parameters should be managed through the configuration system:

```python
from src.config import get_config

# Load configuration
config = get_config()

# Access parameters
random_state = config.get("global.random_state")
learning_rate = config.get("neural_network.training.learning_rate")

# With defaults
batch_size = config.get("neural_network.training.batch_size", default=32)
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clear, focused commits
- Follow commit message guidelines (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Ensure All Checks Pass

```bash
# Format code
make format

# Check linting
make lint

# Run tests
make test

# Check coverage
make test-cov
```

### 4. Update CHANGELOG (if applicable)

Add your changes to the unreleased section of CHANGELOG.md (if it exists).

### 5. Submit Pull Request

- Push your branch: `git push origin feature/your-feature-name`
- Open a PR on GitHub
- Provide clear description of changes
- Link any related issues
- Wait for review

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```bash
feat(models): add CatBoost to ensemble classifiers

Added CatBoost as a new ensemble model with optimized hyperparameters.
Includes configuration in config.yaml and comprehensive tests.

Closes #42
```

```bash
fix(data): correct age binning edge case

Fixed issue where age exactly equal to bin edge was assigned to wrong bin.
Now uses proper inclusive/exclusive boundaries.
```

```bash
docs(readme): update installation instructions

Added section for macOS Apple Silicon users with specific tensorflow-macos
installation steps.
```

```bash
test(neural_network): add tests for Functional API

Added comprehensive tests for new Functional API implementation including
layer naming and architecture validation.
```

## Testing Guidelines

### What to Test

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Edge cases**: Test boundary conditions and error handling
- **Regression tests**: Ensure bugs don't reappear

### Test Structure

```python
def test_function_name():
    """Test that function does X when Y happens."""
    # Arrange
    input_data = create_test_data()
    expected_output = calculate_expected()
    
    # Act
    actual_output = function_under_test(input_data)
    
    # Assert
    assert actual_output == expected_output
```

### Running Specific Tests

```bash
# Run one test file
uv run pytest tests/test_models.py -v

# Run one test function
uv run pytest tests/test_models.py::test_neural_network_initialization -v

# Run tests matching pattern
uv run pytest -k "neural" -v

# Run with coverage for specific file
uv run pytest tests/test_models.py --cov=src.modules.models --cov-report=term-missing
```

## Documentation

### Docstring Style (Google Format)

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """Short description of what the function does.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 0)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string
        
    Example:
        >>> function_name("test", 42)
        True
    """
```

### Updating Documentation

- Update README.md for user-facing changes
- Update CONFIG.md for new configuration options
- Update docstrings for API changes
- Add examples for new features

## Questions or Issues?

- **Bug reports**: Open an issue with reproduction steps
- **Feature requests**: Open an issue describing the feature
- **Questions**: Open a discussion or issue
- **Security issues**: Email maintainers directly (don't open public issue)

## Code of Conduct

### Our Standards

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be collaborative** and helpful
- **Be patient** with others
- **Focus on** what's best for the project

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information
- Other unprofessional conduct

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Getting Help

- Read the [README.md](README.md)
- Check [CONFIG.md](CONFIG.md) for configuration help
- Review existing code for examples
- Open a discussion for questions
- Open an issue for bugs

## Recognition

All contributors will be recognized in the project. Thank you for making this project better! 

---

Happy coding! 
