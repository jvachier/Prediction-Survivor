"""Utility helpers for loading Titanic CSV inputs and caching them."""

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import joblib
import pandas as pd
from src.config import get_config

# Define project root for reliable path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
config = get_config()


@dataclass(slots=True)
class LoadingFiles:
    """Load training and test data sets and persist cached copies.

    This class handles loading Titanic CSV files and caching them as joblib
    files for faster subsequent loads.
    """

    def load_save_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files and save them to disk as joblib backups.

        Reads the train.csv and test.csv files from the data directory,
        then saves compressed copies using joblib for faster future loading.

        Returns:
            Tuple containing (train_dataframe, test_dataframe)

        Raises:
            FileNotFoundError: If CSV files don't exist in the data directory
        """
        train_csv = PROJECT_ROOT / config.get("paths.data_dir") / "train.csv"
        test_csv = PROJECT_ROOT / config.get("paths.data_dir") / "test.csv"

        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        pickle_dir = PROJECT_ROOT / config.get("paths.pickle_dir") / "loading"
        pickle_dir.mkdir(parents=True, exist_ok=True)

        # Use joblib for efficient serialization with compression
        joblib.dump(df_train, pickle_dir / "train.joblib", compress=3)
        joblib.dump(df_test, pickle_dir / "test.joblib", compress=3)

        return (
            df_train,
            df_test,
        )

    def load_db_file(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load previously cached training and test data sets.

        Loads dataframes from joblib-compressed files created by load_save_df().
        This is significantly faster than reading from CSV.

        Returns:
            Tuple containing (train_dataframe, test_dataframe)

        Raises:
            FileNotFoundError: If joblib cache files don't exist
        """
        pickle_dir = PROJECT_ROOT / config.get("paths.pickle_dir") / "loading"

        # Load from joblib files
        df_train = joblib.load(pickle_dir / "train.joblib")
        df_test = joblib.load(pickle_dir / "test.joblib")

        return (
            df_train,
            df_test,
        )
