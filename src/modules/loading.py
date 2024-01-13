import pandas as pd
import pickle

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class Loading_files:
    def load_save_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train = pd.read_csv("../src/data/train.csv")
        df_test = pd.read_csv("../src/data/test.csv")

        dbfile_train = open("./pickle_files/loading/train", "ab")
        dbfile_test = open("./pickle_files/loading/test", "ab")

        pickle.dump(df_train, dbfile_train)
        pickle.dump(df_test, dbfile_test)

        dbfile_train.close()
        dbfile_test.close()
        return (
            df_train,
            df_test,
        )

    def load_db_file(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dbfile_train = open("./pickle_files/loading/train", "rb")
        dbfile_test = open("./pickle_files/loading/test", "rb")

        df_train = pickle.load(dbfile_train)
        df_test = pickle.load(dbfile_test)

        dbfile_train.close()
        dbfile_test.close()
        return (
            df_train,
            df_test,
        )
