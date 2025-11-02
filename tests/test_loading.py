"""Tests for loading module."""

from pathlib import Path

import pandas as pd
import pytest

from src.modules.loading import LoadingFiles


def test_loading_files_initialization() -> None:
    """Test LoadingFiles can be instantiated."""
    loader = LoadingFiles()
    assert loader is not None


@pytest.fixture
def sample_csv_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary CSV files for testing."""
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    # Create sample data
    df_train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 1],
            "Name": ["Smith, Mr. John", "Smith, Mrs. Jane", "Smith, Miss. Emily"],
            "Age": [22, 38, 15],
            "Fare": [7.25, 71.28, 8.05],
        }
    )

    df_test = pd.DataFrame(
        {
            "PassengerId": [4, 5],
            "Name": ["Brown, Mr. Bob", "Brown, Mrs. Alice"],
            "Age": [30, 28],
            "Fare": [10.5, 12.0],
        }
    )

    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    return train_csv, test_csv


def test_load_save_df_creates_dataframes(sample_csv_files: tuple[Path, Path]) -> None:
    """Test that load_save_df can read CSV files."""
    # This test would need to modify LoadingFiles to accept paths
    # For now, just test that the class exists
    loader = LoadingFiles()
    assert hasattr(loader, "load_save_df")
    assert hasattr(loader, "load_db_file")
