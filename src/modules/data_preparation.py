"""Utilities for preparing Titanic data sets for modelling."""

from dataclasses import dataclass
import joblib
from pathlib import Path

import re
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define project root for reliable path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass(slots=True)
class DataPreparation:
    """Encapsulate feature engineering steps for Titanic data frames."""

    data: pd.DataFrame

    def preparation_first(self) -> pd.DataFrame:
        """Extract title and deck features and clean categorical columns."""
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
        """Bin numerical features and derive helper columns for models."""
        df_pre2 = df_selected.copy()
        # Map Embarked, filling unmapped values (0) with a default
        df_pre2["Embarked"] = df_pre2["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df_pre2["Embarked"] = df_pre2["Embarked"].fillna(0).astype(int)

        df_pre2["Sex"] = df_pre2["Sex"].map({"female": 0, "male": 1}).astype(int)

        # Use pd.cut for cleaner binning
        df_pre2["Age"] = pd.cut(
            df_pre2["Age"],
            bins=[-np.inf, 11, 18, 22, 27, 33, 40, np.inf],
            labels=[0, 1, 2, 3, 4, 5, 6],
        ).astype(int)

        df_pre2["relatives"] = df_pre2["SibSp"] + df_pre2["Parch"]
        df_pre2["not_alone"] = (df_pre2["relatives"] == 0).astype(int)

        df_pre2["Fare"] = pd.cut(
            df_pre2["Fare"],
            bins=[-np.inf, 7.91, 14.454, 31, 99, 250, np.inf],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(int)

        df_pre2["Fare_Per_Person"] = df_pre2["Fare"] / (df_pre2["relatives"] + 1)
        df_pre2["Fare_Per_Person"] = df_pre2["Fare_Per_Person"].astype(int)

        df_pre2["Age_Class"] = df_pre2["Age"] * df_pre2["Pclass"]
        df_pre2["Age_Class"] = df_pre2["Age_Class"].astype(int)
        return df_pre2

    def preparation_dummies(self, df_pre2: pd.DataFrame) -> pd.DataFrame:
        """Expand categorical features into indicator columns."""
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
        """Scale numerical features while keeping core transformations consistent."""
        df_pre2 = df_selected.copy()
        # Map Embarked, filling unmapped values (0) with a default
        df_pre2["Embarked"] = df_pre2["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df_pre2["Embarked"] = df_pre2["Embarked"].fillna(0).astype(int)

        df_pre2["Sex"] = df_pre2["Sex"].map({"female": 0, "male": 1}).astype(int)

        scaler = StandardScaler()
        cols = ["Age", "Fare"]
        df_pre2[cols] = scaler.fit_transform(df_pre2[cols])

        df_pre2["relatives"] = df_pre2["SibSp"] + df_pre2["Parch"]
        df_pre2["not_alone"] = (df_pre2["relatives"] == 0).astype(int)

        df_pre2["Fare_Per_Person"] = df_pre2["Fare"] / (df_pre2["relatives"] + 1)
        df_pre2["Fare_Per_Person"] = df_pre2["Fare_Per_Person"].astype(int)

        df_pre2["Age_Class"] = df_pre2["Age"] * df_pre2["Pclass"]
        df_pre2["Age_Class"] = df_pre2["Age_Class"].astype(int)
        return df_pre2

    def preparation_dummies_standardscaler(self, df_pre2: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode scaled data set variant."""
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
    """Handle persistence of pre-processed data frames."""

    name: str

    def load_dataframe(self) -> pd.DataFrame:
        """Load a cached data frame from joblib storage."""
        pickle_dir = PROJECT_ROOT / "pickle_files/data_preparation"
        pickle_path = pickle_dir / f"data_set_{self.name}.joblib"

        data_set = joblib.load(pickle_path)
        return data_set

    def save_dataframe(self, data_set: pd.DataFrame) -> None:
        """Persist a data frame to joblib storage for reuse."""
        pickle_dir = PROJECT_ROOT / "pickle_files/data_preparation"
        pickle_dir.mkdir(parents=True, exist_ok=True)
        pickle_path = pickle_dir / f"data_set_{self.name}.joblib"

        # Use joblib with compression for efficient storage
        joblib.dump(data_set, pickle_path, compress=3)
