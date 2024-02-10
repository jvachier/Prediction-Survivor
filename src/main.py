import os.path
from argparse import ArgumentParser

import numpy as np

from src.modules import data_preparation
from src.modules import models
from src.modules import loading


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model_ensemble", action="store_true")
    parser.add_argument("--standardscaler", action="store_true")

    args = parser.parse_args()

    print("Loading data\n")

    load = loading.Loading_files()

    if os.path.isfile("./pickle_files/loading/train") is False:
        (
            df_train,
            df_test,
        ) = load.load_save_df()
    else:
        (
            df_train,
            df_test,
        ) = load.load_db_file()

    print("Data Preparation\n")
    load_data_train = data_preparation.Load_Save("train")
    load_data_test = data_preparation.Load_Save("test")

    load_data_train_standardscaler = data_preparation.Load_Save("train_standardscaler")
    load_data_test_standardscaler = data_preparation.Load_Save("test_standardscaler")

    if (
        os.path.isfile("./pickle_files/data_preparation/data_set_train")
        & os.path.isfile("./pickle_files/data_preparation/data_set_train_standarscaler")
        is False
    ):
        train = data_preparation.Data_preparation(df_train)
        train_prep1 = train.preparation_first()
        train_selec = train.selection(train_prep1)

        test = data_preparation.Data_preparation(df_test)
        test_prep1 = test.preparation_first()
        test_selec = test.selection(test_prep1)

        if args.standardscaler:
            print("Standarscaler\n")
            train_prep2 = train.preparation_second_standardscaler(train_selec)
            train_final = train.preparation_dummies_standardscaler(train_prep2)

            test_prep2 = test.preparation_second_standardscaler(test_selec)
            test_final = test.preparation_dummies_standardscaler(test_prep2)

            load_data_train_standardscaler.save_dataframe(train_final)
            load_data_test_standardscaler.save_dataframe(test_final)
        else:
            train_prep2 = train.preparation_second(train_selec)
            train_final = train.preparation_dummies(train_prep2)

            test_prep2 = test.preparation_second(test_selec)
            test_final = test.preparation_dummies(test_prep2)

            load_data_train.save_dataframe(train_final)
            load_data_test.save_dataframe(test_final)
    else:
        if args.standardscaler:
            print("Standarscaler\n")
            train_final = load_data_train_standardscaler.load_dataframe()
            test_final = load_data_test_standardscaler.load_dataframe()
        else:
            train_final = load_data_train.load_dataframe()
            test_final = load_data_test.load_dataframe()

    print("Models\n")
    Split = models.split(train_final)
    x_train, x_test, y_train, y_test = Split.train_split()

    if args.model_ensemble:
        print("Model Ensemble\n")
        voting = models.Model_Ensemble(x_train, x_test, y_train, y_test)

        mv_clf = voting.model_cross()

        prediction_rfc = mv_clf.predict(test_final.values)
        test_result_rfc = test_final.copy()
        test_result_rfc["Survived"] = prediction_rfc
        results_rfc = test_result_rfc[["PassengerId", "Survived"]]

        if os.path.isfile("./predictions/prediction_titanic_RFC_new.csv") is False:
            results_rfc.to_csv(
                "./predictions/prediction_titanic_RFC_new.csv", index=False
            )
    else:
        print("Deep Neural Network\n")
        y_train = train_final.loc[:, "Survived"].to_numpy()
        features_train = train_final.drop(columns=["Survived"]).to_numpy()

        test_result_nn = test_final.copy()
        NN = models.NN(features_train, y_train)
        modell_nn = NN.model_NN()
        print(modell_nn.summary())
        NN.fit_NN()

        predictions = modell_nn.predict(x=test.to_numpy(), verbose=2)
        n_test, m_test = predictions.shape
        label = []
        for i in range(n_test):
            label.append(np.argmax(predictions[i], 0))
        label_array = np.asarray(label).reshape(-1)

        test_result_nn["Survived"] = label_array
        results_nn = test_result_nn[["PassengerId", "Survived"]]

        if os.path.isfile("./predictions/prediction_titanic_NN.csv") is False:
            results_nn.to_csv("./predictions/prediction_titanic_NN.csv", index=False)


if __name__ == "__main__":
    main()
