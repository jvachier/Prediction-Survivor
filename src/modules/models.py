from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from dataclasses import dataclass

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import (
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    Nadam,
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping


@dataclass(slots=True)
class split:
    train: pd.DataFrame

    def train_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.train.drop(columns=["Survived"]).values,
            self.train["Survived"].values,
            test_size=0.20,
            stratify=self.train["Survived"].values,
            random_state=1,
        )
        return X_train, X_test, y_train, y_test


@dataclass(slots=True)
class random_forest:
    X_train: np.array
    X_test: np.array
    y_train: list
    y_test: list

    def model_cross(self):
        clf_RFC = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=1,
            n_jobs=4,
        )
        clf_adaboost = AdaBoostClassifier(
            n_estimators=100,
            random_state=2,
        )
        clf_lr = LogisticRegression(solver="lbfgs", max_iter=10000, random_state=3)
        clf_dt = DecisionTreeClassifier(
            max_depth=10, max_features="sqrt", random_state=5
        )

        clf_sgdc = SGDClassifier(loss="log_loss", max_iter=10000, random_state=4)

        clf_knnc = KNeighborsClassifier(n_neighbors=3)

        pipe_RFC = Pipeline(
            [
                # ["pca", PCA(n_components=2)],
                ["rfc", clf_RFC]
            ]
        )
        pipe_adaboost = Pipeline(
            [
                #    ["pca", PCA(n_components=2)],
                ["adaboost", clf_adaboost]
            ]
        )
        pipe_lr = Pipeline([["lr", clf_lr]])
        pipe_dt = Pipeline([["dt", clf_dt]])
        pipe_sgdv = Pipeline([["sgdc", clf_sgdc]])
        pipe_knnc = Pipeline([["knnc", clf_knnc]])

        mv_clf = VotingClassifier(
            estimators=[
                ["rfc", clf_RFC],
                ["adaboost", clf_adaboost],
                ["lr", clf_lr],
                ["dt", clf_dt],
                ["sgdc", clf_sgdc],
                ["knnc", clf_knnc],
            ],
            voting="soft",  # proba for roc auc otherwise hard
            # votin="hard",
        )
        clf_labels = [
            "Random Forest",
            "Adaboost",
            "Logistic Regression",
            "Decision Tree",
            "Stochastic Gradient Descent",
            "Nearest Neighbors",
            "Voting Classifier",
        ]
        all_clf = [
            pipe_RFC,
            pipe_adaboost,
            pipe_lr,
            pipe_dt,
            pipe_sgdv,
            pipe_knnc,
            mv_clf,
        ]
        for clf, label in zip(all_clf, clf_labels):
            scores = cross_val_score(
                estimator=clf,
                X=self.X_train,
                y=self.y_train,
                cv=20,
                n_jobs=2,
                scoring="roc_auc",
            )
            print(
                "ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label)
            )
        mv_clf.fit(self.X_train, self.y_train)
        return mv_clf


class NN:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.n_xtrain = None
        self.m_xtrain = None
        self.n_ytrain = None
        self.m_xtrain = None

        self.modell_NN = None
        self.modell = None

    def model_parameters(self) -> None:
        self.y_train = to_categorical(self.y_train, num_classes=2)
        self.n_xtrain, self.m_xtrain = self.X_train.T.shape
        self.n_ytrain, self.m_xtrain = self.y_train.shape

    def model_NN(self, n_xtrain) -> tf:
        self.modell_NN = Sequential()
        self.modell_NN.add(Dense(units=128, activation="relu", input_shape=(n_xtrain,)))
        self.modell_NN.add(Dense(units=256, activation="relu"))
        self.modell_NN.add(Dense(units=256, activation="relu"))
        # self.modell_NN.add(Dropout(0.20))
        self.modell_NN.add(Dense(units=64, activation="relu"))
        self.modell_NN.add(Dense(units=128, activation="relu"))
        # self.modell_NN.add(Dropout(0.10))
        self.modell_NN.add(Dense(units=64, activation="relu"))
        self.modell_NN.add(Dense(units=64, activation="relu"))
        # self.modell_NN.add(Dropout(0.10))
        self.modell_NN.add(Dense(units=32, activation="relu"))
        self.modell_NN.add(Dense(units=32, activation="relu"))
        self.modell_NN.add(Dense(2, activation="sigmoid"))
        self.modell_NN.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            # metrics=["binary_accuracy"],
            metrics=["accuracy"],
        )
        return self.modell_NN

    def fit_NN(self, X_train, y_train) -> None:
        scores_NN = []
        callback = EarlyStopping(monitor="val_loss", patience=50)

        fold = KFold(n_splits=5).split(X_train, y_train)
        for k, (train, test) in enumerate(fold):
            self.modell_NN.fit(
                X_train[train],
                y_train[train],
                # epochs=self.epochs,
                epochs=1500,
                callbacks=[callback],
                verbose=0,
                validation_split=0.3,
            )
            score_NN = self.modell_NN.evaluate(
                X_train[test],
                y_train[test],
                verbose=0,
            )
            scores_NN.append(score_NN)
            print(
                "NN - Fold: %2d, Acc.: %.3f, Loss: %.3f"
                % (k + 1, score_NN[1], score_NN[0])
            )
