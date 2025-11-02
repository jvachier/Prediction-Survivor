"""Utility helpers for loading Titanic CSV inputs and caching them."""

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import pickle
import pandas as pd

# Define project root for reliable path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass(slots=True)
class LoadingFiles:
    """Load training and test data sets and persist cached copies."""

    def load_save_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files and save them to disk as pickle backups."""
        train_csv = PROJECT_ROOT / "src/data/train.csv"
        test_csv = PROJECT_ROOT / "src/data/test.csv"

        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        pickle_dir = PROJECT_ROOT / "pickle_files/loading"
        pickle_dir.mkdir(parents=True, exist_ok=True)

        with open(pickle_dir / "train", "ab") as dbfile_train:
            pickle.dump(df_train, dbfile_train)

        with open(pickle_dir / "test", "ab") as dbfile_test:
            pickle.dump(df_test, dbfile_test)

        return (
            df_train,
            df_test,
        )

    def load_db_file(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load previously cached training and test data sets."""
        pickle_dir = PROJECT_ROOT / "pickle_files/loading"

        with open(pickle_dir / "train", "rb") as dbfile_train:
            df_train = pickle.load(dbfile_train)

        with open(pickle_dir / "test", "rb") as dbfile_test:
            df_test = pickle.load(dbfile_test)
        return (
            df_train,
            df_test,
        )
