from dataclasses import dataclass
from typing import Tuple

import pickle
import pandas as pd


@dataclass(slots=True)
class LoadingFiles:
    def load_save_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = pd.read_csv("../src/data/train.csv")
        df_test = pd.read_csv("../src/data/test.csv")

        with open("./pickle_files/loading/train", "ab") as dbfile_train:
            pickle.dump(df_train, dbfile_train)

        with open("./pickle_files/loading/test", "ab") as dbfile_test:
            pickle.dump(df_test, dbfile_test)

        return (
            df_train,
            df_test,
        )

    def load_db_file(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with open("./pickle_files/loading/train", "rb") as dbfile_train:
            df_train = pickle.load(dbfile_train)

        with open("./pickle_files/loading/test", "rb") as dbfile_test:
            df_test = pickle.load(dbfile_test)
        return (
            df_train,
            df_test,
        )
