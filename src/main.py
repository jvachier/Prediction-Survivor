import os.path
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical

import modules.data_preparation as data_preparation
import modules.models as models


def main() -> None:
    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")

    Train = data_preparation.Data_preparation(df_train)
    train_prep1 = Train.preparation_first()
    train_selec = Train.selection(train_prep1)
    train_prep2 = Train.preparation_second(train_selec)
    train = Train.preparation_dummies(train_prep2)

    Test = data_preparation.Data_preparation(df_test)
    test_prep1 = Test.preparation_first()
    test_selec = Test.selection(test_prep1)
    test_prep2 = Test.preparation_second(test_selec)
    test = Test.preparation_dummies(test_prep2)

    Split = models.split(train)
    X_train, X_test, y_train, y_test = Split.train_split()

    # Random Forest
    RF = models.random_forest(X_train, X_test, y_train, y_test)

    mv_clf = RF.model_cross()

    prediction_rfc = mv_clf.predict(test.values)
    test_result_rfc = test.copy()
    test_result_rfc["Survived"] = prediction_rfc
    results_rfc = test_result_rfc[["PassengerId", "Survived"]]

    if os.path.isfile("./predictions/prediction_titanic_RFC_new.csv") is False:
        results_rfc.to_csv("./predictions/prediction_titanic_RFC_new.csv", index=False)

    # Deep Neural Netowork

    y_train = train.loc[:, "Survived"].to_numpy()
    features_train = train.drop(columns=["Survived"]).to_numpy()
    y_train = to_categorical(
        y_train, num_classes=2
    )  # not to be used with StratifiedKFold
    n_xtrain, m_xtrain = features_train.T.shape
    # n_ytrain, m_xtrain = y_train.shape

    test_result_NN = test.copy()
    NN = models.NN(X_train, X_test, y_train, y_test)
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
