"""Tests for data preparation module."""

import numpy as np
import pandas as pd
import pytest

from src.modules.data_preparation import DataPreparation, LoadSave


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample Titanic data for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Smith, Miss. Emily",
            ],
            "Sex": ["male", "female", "female"],
            "Age": [22.0, 38.0, 15.0],
            "SibSp": [1, 1, 0],
            "Parch": [0, 0, 2],
            "Ticket": ["A/5 21171", "PC 17599", "113803"],
            "Fare": [7.25, 71.2833, 8.05],
            "Cabin": [np.nan, "C85", np.nan],
            "Embarked": ["S", "C", "S"],
            "Pclass": [3, 1, 3],
        }
    )


def test_preparation_first(sample_df: pd.DataFrame) -> None:
    """Test that titles and deck are extracted correctly."""
    prep = DataPreparation(sample_df)
    result = prep.preparation_first()

    # Check that Title column is created
    assert "Title" in result.columns
    assert result["Title"].iloc[0] == 1  # Mr
    assert result["Title"].iloc[1] == 3  # Mrs
    assert result["Title"].iloc[2] == 2  # Miss

    # Check that Deck column is created
    assert "Deck" in result.columns
    assert result["Deck"].iloc[1] == 3  # C deck

    # Check that Name is dropped
    assert "Name" not in result.columns


def test_selection(sample_df: pd.DataFrame) -> None:
    """Test that selection drops correct columns and fills missing values."""
    prep = DataPreparation(sample_df)
    result_first = prep.preparation_first()
    result = prep.selection(result_first)

    # Check that Cabin and Ticket are dropped
    assert "Cabin" not in result.columns
    assert "Ticket" not in result.columns

    # Check that Age is filled (no NaN)
    assert not result["Age"].isnull().any()

    # Check that other missing values are filled with 0
    assert not result.isnull().any().any()


def test_preparation_second(sample_df: pd.DataFrame) -> None:
    """Test feature engineering in preparation_second."""
    prep = DataPreparation(sample_df)
    result_first = prep.preparation_first()
    result_selected = prep.selection(result_first)
    result = prep.preparation_second(result_selected)

    # Check that Sex is encoded
    assert result["Sex"].dtype in [np.int64, np.int32]

    # Check that Embarked is encoded
    assert result["Embarked"].dtype in [np.int64, np.int32]

    # Check that Age is binned
    assert result["Age"].max() <= 6

    # Check that relatives column is created
    assert "relatives" in result.columns
    assert result["relatives"].iloc[0] == 1  # SibSp=1, Parch=0

    # Check that not_alone column is created
    assert "not_alone" in result.columns

    # Check that Fare_Per_Person is created
    assert "Fare_Per_Person" in result.columns

    # Check that Age_Class is created
    assert "Age_Class" in result.columns


def test_preparation_dummies(sample_df: pd.DataFrame) -> None:
    """Test that OneHotEncoder creates proper dummy columns."""
    prep = DataPreparation(sample_df)
    result_first = prep.preparation_first()
    result_selected = prep.selection(result_first)
    result_second = prep.preparation_second(result_selected)
    result = prep.preparation_dummies(result_second)

    # Check that one-hot encoded columns are created
    title_cols = [col for col in result.columns if col.startswith("Title_")]
    assert len(title_cols) > 0

    pclass_cols = [col for col in result.columns if col.startswith("Pclass_")]
    assert len(pclass_cols) > 0

    age_cols = [col for col in result.columns if col.startswith("Age_")]
    assert len(age_cols) > 0

    # Check that original categorical columns are dropped
    assert "Title" not in result.columns
    assert "Pclass" not in result.columns


def test_preparation_second_standardscaler(sample_df: pd.DataFrame) -> None:
    """Test StandardScaler preprocessing."""
    prep = DataPreparation(sample_df)
    result_first = prep.preparation_first()
    result_selected = prep.selection(result_first)
    result = prep.preparation_second_standardscaler(result_selected)

    # Check that Age and Fare are scaled (should have mean ~0, std ~1)
    assert abs(result["Age"].mean()) < 1.0
    assert abs(result["Fare"].mean()) < 10.0

    # Check that relatives column is created
    assert "relatives" in result.columns


def test_load_save_name() -> None:
    """Test LoadSave initialization."""
    loader = LoadSave("test")
    assert loader.name == "test"
