import os.path
import numpy as np
import pandas as pd

from argparse import ArgumentParser

from tensorflow.keras.utils import to_categorical

import modules.data_preparation as data_preparation
import modules.models as models
import modules.loading as loading


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model_ensemble", action="store_true")
    parser.add_argument("--standardscaler", action="store_true")

    args = parser.parse_args()

    print("Loading data\n")

    LOAD = loading.Loading_files()

    if os.path.isfile("./pickle_files/loading/train") is False:
        (
            df_train,
            df_test,
        ) = LOAD.load_save_df()
    else:
        (
            df_train,
            df_test,
        ) = LOAD.load_db_file()

    print("Data Preparation\n")
    LOAD_DATA_train = data_preparation.Load_Save("train")
    LOAD_DATA_test = data_preparation.Load_Save("test")

    LOAD_DATA_train_standardscaler = data_preparation.Load_Save("train_standardscaler")
    LOAD_DATA_test_standardscaler = data_preparation.Load_Save("test_standardscaler")

    if (
        os.path.isfile("./pickle_files/data_preparation/data_set_train")
        & os.path.isfile("./pickle_files/data_preparation/data_set_train_standarscaler")
        is False
    ):
        Train = data_preparation.Data_preparation(df_train)
        train_prep1 = Train.preparation_first()
        train_selec = Train.selection(train_prep1)

        Test = data_preparation.Data_preparation(df_test)
        test_prep1 = Test.preparation_first()
        test_selec = Test.selection(test_prep1)

        if args.standardscaler:
            train_prep2 = Train.preparation_second_standardscaler(train_selec)
            train = Train.preparation_dummies_standardscaler(train_prep2)

            test_prep2 = Test.preparation_second_standardscaler(test_selec)
            test = Test.preparation_dummies_standardscaler(test_prep2)

            LOAD_DATA_train_standardscaler.save_dataframe(train)
            LOAD_DATA_test_standardscaler.save_dataframe(test)
        else:
            train_prep2 = Train.preparation_second(train_selec)
            train = Train.preparation_dummies(train_prep2)

            test_prep2 = Test.preparation_second(test_selec)
            test = Test.preparation_dummies(test_prep2)

            LOAD_DATA_train.save_dataframe(train)
            LOAD_DATA_test.save_dataframe(test)
    else:
        if args.standardscaler:
            train = LOAD_DATA_train_standardscaler.load_dataframe()
            test = LOAD_DATA_test_standardscaler.load_dataframe()
        else:
            train = LOAD_DATA_train.load_dataframe()
            test = LOAD_DATA_test.load_dataframe()

    print("Models\n")
    Split = models.split(train)
    X_train, X_test, y_train, y_test = Split.train_split()

    if args.model_ensemble:
        print("Model Ensemble\n")
        RF = models.Model_Ensemble(X_train, X_test, y_train, y_test)

        mv_clf = RF.model_cross()

        prediction_rfc = mv_clf.predict(test.values)
        test_result_rfc = test.copy()
        test_result_rfc["Survived"] = prediction_rfc
        results_rfc = test_result_rfc[["PassengerId", "Survived"]]

        if os.path.isfile("./predictions/prediction_titanic_RFC_new.csv") is False:
            results_rfc.to_csv(
                "./predictions/prediction_titanic_RFC_new.csv", index=False
            )
    else:
        print("Deep Neural Network\n")
        y_train = train.loc[:, "Survived"].to_numpy()
        features_train = train.drop(columns=["Survived"]).to_numpy()
        y_train = to_categorical(
            y_train, num_classes=2
        )  # not to be used with StratifiedKFold
        n_xtrain, m_xtrain = features_train.T.shape
        # n_ytrain, m_xtrain = y_train.shape

        test_result_NN = test.copy()
        NN = models.NN(X_train, y_train, X_test, y_test)
        # NN.model_parameters()
        modell_NN = NN.model_NN(n_xtrain)
        NN.fit_NN(features_train, y_train)

        print(modell_NN.summary())

        predictions = modell_NN.predict(x=test.to_numpy(), verbose=2)
        n_test, m_test = predictions.shape
        Label = []
        for i in range(n_test):
            Label.append(np.argmax(predictions[i], 0))
        Label_array = np.asarray(Label).reshape(-1)

        test_result_NN["Survived"] = Label_array
        results_NN = test_result_NN[["PassengerId", "Survived"]]

        if os.path.isfile("./predictions/prediction_titanic_NN.csv") is False:
            results_NN.to_csv("./predictions/prediction_titanic_NN.csv", index=False)


if __name__ == "__main__":
    main()
