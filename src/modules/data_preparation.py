"""Utilities for preparing Titanic data sets for modelling."""

from dataclasses import dataclass
import joblib
from pathlib import Path

import re
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import get_config

# Define project root for reliable path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
config = get_config()


@dataclass(slots=True)
class DataPreparation:
    """Encapsulate feature engineering steps for Titanic data frames.

    This class handles the complete data preparation pipeline including:
    - Title extraction from names
    - Deck extraction from cabin information
    - Feature engineering (relatives, fare per person, age classes)
    - Categorical encoding and standardization

    Attributes:
        data: Input DataFrame with raw Titanic passenger data
    """

    data: pd.DataFrame

    def preparation_first(self) -> pd.DataFrame:
        """Extract title and deck features and clean categorical columns.

        This is the first stage of data preparation that:
        - Validates required columns exist
        - Extracts passenger titles from names (Mr, Mrs, Miss, Master, Rare)
        - Maps cabin letters to deck numbers (A-G, U for unknown)
        - Fills missing age values with median
        - Fills missing fare values with median
        - Maps sex to Title for consistency

        Returns:
            DataFrame with extracted features (Title, Deck) and cleaned data

        Raises:
            ValueError: If required columns are missing from input data
        """
        # Validate required columns exist
        required_cols = [
            "Name",
            "Age",
            "Fare",
            "Pclass",
            "SibSp",
            "Parch",
            "Embarked",
            "Sex",
            "Cabin",
        ]
        missing_cols = set(required_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        df_data = self.data.copy()
        deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
        df_data["Cabin"] = df_data["Cabin"].fillna("U0")
        df_data["Deck"] = df_data["Cabin"].map(
            lambda x: re.compile("([a-zA-Z]+)").search(x).group()
        )
        df_data["Deck"] = df_data["Deck"].map(deck)
        df_data["Deck"] = df_data["Deck"].fillna(0)
        df_data["Deck"] = df_data["Deck"].astype(int)

        titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df_data["Title"] = df_data.Name.str.extract(" ([A-Za-z]+)", expand=False)

        df_data["Title"] = df_data["Title"].replace(
            [
                "Lady",
                "Countess",
                "Capt",
                "Col",
                "Don",
                "Dr",
                "Major",
                "Rev",
                "Sir",
                "Jonkheer",
                "Dona",
            ],
            "Rare",
        )
        df_data["Title"] = df_data["Title"].replace("Mlle", "Miss")
        df_data["Title"] = df_data["Title"].replace("Ms", "Miss")
        df_data["Title"] = df_data["Title"].replace("Mme", "Mrs")

        df_data["Title"] = df_data["Title"].map(titles)

        df_data["Title"] = df_data["Title"].fillna(0)

        df_data = df_data.drop("Name", axis=1)
        return df_data

    def selection(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Select relevant columns for model training.

        Drops unnecessary columns and keeps only features useful for prediction.
        Removes: Ticket, PassengerId, Cabin, Name

        Args:
            df_data: DataFrame after first preparation stage

        Returns:
            DataFrame with only selected feature columns
        """
        """Select relevant attributes and impute missing values."""
        df_selected = df_data.drop(columns=["Cabin", "Ticket"])

        rng = np.random.default_rng(0)
        mu = df_selected["Age"].mean()
        sd = df_selected["Age"].std()

        filler = pd.Series(rng.normal(loc=mu, scale=sd, size=len(df_selected)))
        df_selected["Age"] = df_selected["Age"].fillna(filler)

        df_selected = df_selected.fillna(0)
        return df_selected

    def preparation_second(self, df_selected: pd.DataFrame) -> pd.DataFrame:
        """Bin numerical features and derive helper columns for models.

        This is the second stage of preparation (for non-StandardScaler models) that:
        - Maps categorical values (Embarked, Sex)
        - Calculates family-related features (relatives, not_alone)
        - Calculates Fare_Per_Person BEFORE binning (fixes division bug)
        - Bins Age into 7 categories
        - Bins Fare into 6 categories
        - Bins Fare_Per_Person into 5 categories
        - Creates Age_Class interaction feature

        Args:
            df_selected: DataFrame after column selection

        Returns:
            DataFrame with binned features and derived columns
        """
        df_pre2 = df_selected.copy()
        # Map Embarked, filling unmapped values (0) with a default
        df_pre2["Embarked"] = df_pre2["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df_pre2["Embarked"] = df_pre2["Embarked"].fillna(0).astype(int)

        df_pre2["Sex"] = df_pre2["Sex"].map({"female": 0, "male": 1}).astype(int)

        # Calculate relatives before any transformations
        df_pre2["relatives"] = df_pre2["SibSp"] + df_pre2["Parch"]
        df_pre2["not_alone"] = (df_pre2["relatives"] == 0).astype(int)

        # Calculate Fare_Per_Person using ORIGINAL continuous Fare values before binning
        df_pre2["Fare_Per_Person"] = df_pre2["Fare"] / (df_pre2["relatives"] + 1)

        # Now bin Age and Fare
        df_pre2["Age"] = pd.cut(
            df_pre2["Age"],
            bins=[-np.inf, 11, 18, 22, 27, 33, 40, np.inf],
            labels=[0, 1, 2, 3, 4, 5, 6],
        ).astype(int)

        df_pre2["Fare"] = pd.cut(
            df_pre2["Fare"],
            bins=[-np.inf, 7.91, 14.454, 31, 99, 250, np.inf],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(int)

        # Bin Fare_Per_Person as well for consistency
        df_pre2["Fare_Per_Person"] = pd.cut(
            df_pre2["Fare_Per_Person"],
            bins=[-np.inf, 7, 14, 30, 100, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(int)

        df_pre2["Age_Class"] = df_pre2["Age"] * df_pre2["Pclass"]
        df_pre2["Age_Class"] = df_pre2["Age_Class"].astype(int)
        return df_pre2

    def preparation_dummies(self, df_pre2: pd.DataFrame) -> pd.DataFrame:
        """Expand categorical features into indicator columns.

        Applies one-hot encoding to categorical columns using scikit-learn's
        OneHotEncoder. Drops the first category to avoid multicollinearity.

        Encoded columns: Title, Pclass, Age, Embarked, Fare

        Args:
            df_pre2: DataFrame after second preparation stage (with binned features)

        Returns:
            DataFrame with one-hot encoded categorical features
        """
        encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=int)
        categorical_cols = ["Title", "Pclass", "Age", "Embarked", "Fare"]

        # Fit and transform the categorical columns
        encoded_array = encoder.fit_transform(df_pre2[categorical_cols])

        # Get feature names for the encoded columns
        feature_names = encoder.get_feature_names_out(categorical_cols)

        # Create a DataFrame with encoded columns
        encoded_df = pd.DataFrame(
            encoded_array, columns=feature_names, index=df_pre2.index
        )

        # Drop original categorical columns and concatenate encoded ones
        df_dummies = df_pre2.drop(columns=categorical_cols).join(encoded_df)
        return df_dummies

    def preparation_second_standardscaler(
        self, df_selected: pd.DataFrame
    ) -> pd.DataFrame:
        """Scale numerical features while keeping core transformations consistent.

        This is the StandardScaler variant of preparation_second, used for neural networks.
        Key differences from preparation_second:
        - Calculates Fare_Per_Person BEFORE scaling (fixes division bug)
        - Applies StandardScaler to Age, Fare, and Fare_Per_Person (z-score normalization)
        - Does NOT bin numerical features
        - Creates Age_Class interaction from scaled values

        Args:
            df_selected: DataFrame after column selection

        Returns:
            DataFrame with standardized numerical features (mean=0, std=1)
        """
        df_pre2 = df_selected.copy()
        # Map Embarked, filling unmapped values (0) with a default
        df_pre2["Embarked"] = df_pre2["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df_pre2["Embarked"] = df_pre2["Embarked"].fillna(0).astype(int)

        df_pre2["Sex"] = df_pre2["Sex"].map({"female": 0, "male": 1}).astype(int)

        # Calculate relatives before any transformations
        df_pre2["relatives"] = df_pre2["SibSp"] + df_pre2["Parch"]
        df_pre2["not_alone"] = (df_pre2["relatives"] == 0).astype(int)

        # Calculate Fare_Per_Person using ORIGINAL continuous Fare values before scaling
        df_pre2["Fare_Per_Person"] = df_pre2["Fare"] / (df_pre2["relatives"] + 1)

        # Now apply StandardScaler to Age, Fare, and Fare_Per_Person
        scaler = StandardScaler()
        cols = ["Age", "Fare", "Fare_Per_Person"]
        df_pre2[cols] = scaler.fit_transform(df_pre2[cols])

        df_pre2["Age_Class"] = df_pre2["Age"] * df_pre2["Pclass"]
        return df_pre2

    def preparation_dummies_standardscaler(self, df_pre2: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode scaled data set variant.

        Applies one-hot encoding for use with StandardScaler-prepared data.
        Encodes: Title, Pclass, Embarked (no Age/Fare as they're continuous)

        Args:
            df_pre2: DataFrame after StandardScaler preparation

        Returns:
            DataFrame with one-hot encoded categorical features (for NN models)
        """
        encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=int)
        categorical_cols = ["Title", "Pclass", "Embarked"]

        # Fit and transform the categorical columns
        encoded_array = encoder.fit_transform(df_pre2[categorical_cols])

        # Get feature names for the encoded columns
        feature_names = encoder.get_feature_names_out(categorical_cols)

        # Create a DataFrame with encoded columns
        encoded_df = pd.DataFrame(
            encoded_array, columns=feature_names, index=df_pre2.index
        )

        # Drop original categorical columns and concatenate encoded ones
        df_dummies = df_pre2.drop(columns=categorical_cols).join(encoded_df)
        return df_dummies


@dataclass(slots=True)
class LoadSave:
    """Handle persistence of pre-processed data frames.

    Provides methods to save and load preprocessed DataFrames using joblib
    compression for efficient storage and fast loading.

    Attributes:
        name: Identifier for the dataset (e.g., "train", "test", "train_standardscaler")
    """

    name: str

    def load_dataframe(self) -> pd.DataFrame:
        """Load a cached data frame from joblib storage.

        Returns:
            Previously saved DataFrame

        Raises:
            FileNotFoundError: If the cached file doesn't exist
        """
        pickle_dir = PROJECT_ROOT / config.get("paths.pickle_dir") / "data_preparation"
        pickle_path = pickle_dir / f"data_set_{self.name}.joblib"

        data_set = joblib.load(pickle_path)
        return data_set

    def save_dataframe(self, data_set: pd.DataFrame) -> None:
        """Persist a data frame to joblib storage for reuse.

        Saves with compression level 3 (good balance of speed and size).

        Args:
            data_set: DataFrame to save
        """
        pickle_dir = PROJECT_ROOT / config.get("paths.pickle_dir") / "data_preparation"
        pickle_dir.mkdir(parents=True, exist_ok=True)
        pickle_path = pickle_dir / f"data_set_{self.name}.joblib"

        # Use joblib with compression for efficient storage
        joblib.dump(data_set, pickle_path, compress=3)
