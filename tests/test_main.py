"""Tests for main module and prediction validation."""

import numpy as np
import pandas as pd
import pytest

from src.main import validate_predictions


def test_validate_predictions_correct() -> None:
    """Test validation passes for correct predictions."""
    predictions = pd.DataFrame(
        {"PassengerId": [1, 2, 3, 4, 5], "Survived": [0, 1, 0, 1, 1]}
    )

    # Should not raise
    validate_predictions(predictions)


def test_validate_predictions_wrong_columns() -> None:
    """Test validation fails with wrong columns."""
    predictions = pd.DataFrame({"ID": [1, 2, 3], "Prediction": [0, 1, 0]})

    with pytest.raises(ValueError, match="must have columns"):
        validate_predictions(predictions)


def test_validate_predictions_missing_column() -> None:
    """Test validation fails with missing column."""
    predictions = pd.DataFrame({"PassengerId": [1, 2, 3]})

    with pytest.raises(ValueError, match="must have columns"):
        validate_predictions(predictions)


def test_validate_predictions_wrong_survived_values() -> None:
    """Test validation fails with non-binary Survived values."""
    predictions = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 2],  # 2 is invalid
        }
    )

    with pytest.raises(ValueError, match="must be 0 or 1"):
        validate_predictions(predictions)


def test_validate_predictions_negative_values() -> None:
    """Test validation fails with negative Survived values."""
    predictions = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, -1],  # -1 is invalid
        }
    )

    with pytest.raises(ValueError, match="must be 0 or 1"):
        validate_predictions(predictions)


def test_validate_predictions_missing_values() -> None:
    """Test validation fails with missing values.

    Note: NaN values cause float conversion, so type check catches it first.
    """
    predictions = pd.DataFrame({"PassengerId": [1, 2, 3], "Survived": [0, np.nan, 1]})

    with pytest.raises(ValueError, match="must be integer type"):
        validate_predictions(predictions)


def test_validate_predictions_wrong_dtype_passengerid() -> None:
    """Test validation fails with wrong PassengerId data type."""
    predictions = pd.DataFrame(
        {
            "PassengerId": ["1", "2", "3"],  # String instead of int
            "Survived": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="PassengerId must be integer type"):
        validate_predictions(predictions)


def test_validate_predictions_wrong_dtype_survived() -> None:
    """Test validation fails with wrong Survived data type."""
    predictions = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0.0, 1.0, 0.0],  # Float instead of int
        }
    )

    with pytest.raises(ValueError, match="Survived must be integer type"):
        validate_predictions(predictions)


def test_validate_predictions_empty_dataframe() -> None:
    """Test validation fails with empty DataFrame."""
    predictions = pd.DataFrame({"PassengerId": [], "Survived": []})

    # Empty is technically valid, but let's ensure types are correct
    predictions["PassengerId"] = predictions["PassengerId"].astype(int)
    predictions["Survived"] = predictions["Survived"].astype(int)

    # Should not raise
    validate_predictions(predictions)


def test_validate_predictions_large_dataset() -> None:
    """Test validation works with larger dataset."""
    np.random.seed(42)
    predictions = pd.DataFrame(
        {
            "PassengerId": np.arange(1, 419),  # Titanic test set size
            "Survived": np.random.randint(0, 2, 418),
        }
    )

    # Should not raise
    validate_predictions(predictions)


def test_validate_predictions_all_zeros() -> None:
    """Test validation passes when all predictions are 0."""
    predictions = pd.DataFrame(
        {"PassengerId": [1, 2, 3, 4, 5], "Survived": [0, 0, 0, 0, 0]}
    )

    # Should not raise
    validate_predictions(predictions)


def test_validate_predictions_all_ones() -> None:
    """Test validation passes when all predictions are 1."""
    predictions = pd.DataFrame(
        {"PassengerId": [1, 2, 3, 4, 5], "Survived": [1, 1, 1, 1, 1]}
    )

    # Should not raise
    validate_predictions(predictions)
