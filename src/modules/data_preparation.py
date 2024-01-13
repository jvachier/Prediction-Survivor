import pandas as pd
import numpy as np
import random
import re
from dataclasses import dataclass


@dataclass(slots=True)
class Data_preparation:
    data: pd.DataFrame
    # def __init__(self, df) -> None:
    #     self.data: pd.DataFrame = df

    def preparation_first(self) -> pd.DataFrame:
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
        df_data["Title"] = df_data.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

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
        df_selected = df_data.drop(columns=["Cabin", "Ticket"])
        # f = lambda row: random.gauss(
        #     df_selected["Age"].mean(), np.sqrt(df_selected["Age"].std())
        # )

        # m = df_selected["Age"].isna()
        # df_selected["Age"].loc[m] = df_selected["Age"].loc[m].apply(f)

        rng = np.random.default_rng(0)
        mu = df_selected["Age"].mean()
        sd = df_selected["Age"].std()

        filler = pd.Series(rng.normal(loc=mu, scale=sd, size=len(df_selected)))
        df_selected["Age"] = df_selected["Age"].fillna(filler)

        # mean_value = df_selected["Age"].mean()
        # df_selected["Age"].fillna(value=mean_value)

        df_selected = df_selected.fillna(0)
        # df_selected = df_selected.dropna()
        return df_selected

    def preparation_second(self, df_selected: pd.DataFrame):
        df_pre2 = df_selected.copy()
        df_pre2["Embarked"] = df_pre2["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])
        df_pre2["Sex"] = df_pre2["Sex"].replace(["female", "male"], [0, 1])

        df_pre2.loc[df_pre2.Age <= 11, "Age"] = 0
        df_pre2.loc[(df_pre2.Age <= 18) & (11 < df_pre2.Age), "Age"] = 1
        df_pre2.loc[(df_pre2.Age <= 22) & (18 < df_pre2.Age), "Age"] = 2
        df_pre2.loc[(df_pre2.Age <= 27) & (22 < df_pre2.Age), "Age"] = 3
        df_pre2.loc[(df_pre2.Age <= 33) & (27 < df_pre2.Age), "Age"] = 4
        df_pre2.loc[(df_pre2.Age <= 40) & (33 < df_pre2.Age), "Age"] = 5
        df_pre2.loc[df_pre2.Age > 40, "Age"] = 6
        df_pre2["Age"] = df_pre2["Age"].astype(int)

        df_pre2["relatives"] = df_pre2["SibSp"] + df_pre2["Parch"]
        df_pre2.loc[df_pre2["relatives"] > 0, "not_alone"] = 0
        df_pre2.loc[df_pre2["relatives"] == 0, "not_alone"] = 1
        df_pre2["not_alone"] = df_pre2["not_alone"].astype(int)

        df_pre2.loc[df_pre2["Fare"] <= 7.91, "Fare"] = 0
        df_pre2.loc[(df_pre2["Fare"] > 7.91) & (df_pre2["Fare"] <= 14.454), "Fare"] = 1
        df_pre2.loc[(df_pre2["Fare"] > 14.454) & (df_pre2["Fare"] <= 31), "Fare"] = 2
        df_pre2.loc[(df_pre2["Fare"] > 31) & (df_pre2["Fare"] <= 99), "Fare"] = 3
        df_pre2.loc[(df_pre2["Fare"] > 99) & (df_pre2["Fare"] <= 250), "Fare"] = 4
        df_pre2.loc[df_pre2["Fare"] > 250, "Fare"] = 5
        df_pre2["Fare"] = df_pre2["Fare"].astype(int)

        df_pre2["Fare_Per_Person"] = df_pre2["Fare"] / (df_pre2["relatives"] + 1)
        df_pre2["Fare_Per_Person"] = df_pre2["Fare_Per_Person"].astype(int)

        df_pre2["Age_Class"] = df_pre2["Age"] * df_pre2["Pclass"]
        df_pre2["Age_Class"] = df_pre2["Age_Class"].astype(int)
        return df_pre2

    def preparation_dummies(self, df_pre2: pd.DataFrame) -> pd.DataFrame:
        # dummies for Embarked, Pclass, Title, Age, Fare
        df_dummies = pd.get_dummies(
            df_pre2,
            columns=["Title", "Pclass", "Age", "Embarked"],
            drop_first=True,
            dtype=int,
        )
        return df_dummies
