"""Utility helpers for loading Titanic CSV inputs and caching them."""

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import joblib
import pandas as pd

# Define project root for reliable path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass(slots=True)
class LoadingFiles:
    """Load training and test data sets and persist cached copies."""

    def load_save_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files and save them to disk as joblib backups."""
        train_csv = PROJECT_ROOT / "src/data/train.csv"
        test_csv = PROJECT_ROOT / "src/data/test.csv"

        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        pickle_dir = PROJECT_ROOT / "pickle_files/loading"
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
        """Load previously cached training and test data sets."""
        pickle_dir = PROJECT_ROOT / "pickle_files/loading"

        # Load from joblib files
        df_train = joblib.load(pickle_dir / "train.joblib")
        df_test = joblib.load(pickle_dir / "test.joblib")
        
        return (
            df_train,
            df_test,
        )
