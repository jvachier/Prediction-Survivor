"""Model utilities for ensemble classifiers and neural networks."""

from dataclasses import dataclass

from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.optimizers.legacy import (
    Adam,
)
from keras.layers import (
    Dense,
    Dropout,
)
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


@dataclass(slots=True)
class Split:
    """Utility class to create train/test splits for the Titanic dataset."""

    train: pd.DataFrame

    def train_split(self) -> Tuple[np.array, np.array, list, list]:
        """Split the training data into hold-out train and validation sets."""
        x_train, x_test, y_train, y_test = train_test_split(
            self.train.drop(columns=["Survived"]).values,
            self.train["Survived"].values,
            test_size=0.20,
            stratify=self.train["Survived"].values,
            random_state=1,
        )
        return x_train, x_test, y_train, y_test


@dataclass(slots=True)
class ModelEnsemble:
    """Train a soft-voting ensemble comprised of classic ML estimators."""

    x_train: np.array
    x_test: np.array
    y_train: list
    y_test: list

    def model_cross(self) -> object:
        """Perform cross validation on individual estimators and fit the ensemble."""
        clf_rfc = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=1,
            n_jobs=4,
        )

        clf_adaboost = AdaBoostClassifier(
            n_estimators=100,
            algorithm="SAMME",
            random_state=2,
        )
        clf_lr = LogisticRegression(solver="lbfgs", max_iter=10000, random_state=3)
        clf_dt = DecisionTreeClassifier(
            max_depth=10, max_features="sqrt", random_state=5
        )

        clf_sgdc = SGDClassifier(
            loss="log_loss",
            max_iter=10000,
            random_state=4,
            learning_rate="adaptive",
            eta0=1e-4,
        )

        clf_knnc = KNeighborsClassifier(n_neighbors=50)

        pipe_rfc = Pipeline([["rfc", clf_rfc]])
        pipe_adaboost = Pipeline([["adaboost", clf_adaboost]])
        pipe_lr = Pipeline([["lr", clf_lr]])
        pipe_dt = Pipeline([["dt", clf_dt]])
        pipe_sgdv = Pipeline([["sgdc", clf_sgdc]])
        pipe_knnc = Pipeline([["knnc", clf_knnc]])

        mv_clf = VotingClassifier(
            estimators=[
                ["rfc", clf_rfc],
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
            pipe_rfc,
            pipe_adaboost,
            pipe_lr,
            pipe_dt,
            pipe_sgdv,
            pipe_knnc,
            mv_clf,
        ]

        stratiKfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

        for clf, label in zip(all_clf, clf_labels):
            scores = cross_val_score(
                estimator=clf,
                X=self.x_train,
                y=self.y_train,
                cv=stratiKfold,
                n_jobs=4,
                scoring="roc_auc",
            )
            print(
                "ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label)
            )
        mv_clf.fit(self.x_train, self.y_train)
        return mv_clf


@dataclass(slots=True)
class NeuralNetwork:
    """Define and train a dense neural network for Titanic survival prediction."""

    x_train: np.array
    y_train: list
    n_xtrain: int = None
    m_xtrain: list = None
    modell_nn: Sequential = None

    def __post_init__(self):
        """Capture input dimensionality after initialisation."""
        self.n_xtrain, self.m_xtrain = self.x_train.T.shape

    def model_nn(self) -> Sequential:
        """Build and compile the sequential neural network architecture."""
        self.modell_nn = Sequential()
        self.modell_nn.add(
            Dense(units=512, activation="relu", input_shape=(self.n_xtrain,))
        )
        self.modell_nn.add(Dense(units=256, activation="relu"))
        self.modell_nn.add(Dense(units=256, activation="relu"))
        # self.modell_nn.add(Dropout(0.20))
        self.modell_nn.add(Dense(units=128, activation="relu"))
        self.modell_nn.add(Dense(units=128, activation="relu"))
        self.modell_nn.add(Dropout(0.10))
        self.modell_nn.add(Dense(units=64, activation="relu"))
        self.modell_nn.add(Dense(units=64, activation="relu"))
        # self.modell_nn.add(Dropout(0.10))
        self.modell_nn.add(Dense(units=32, activation="relu"))
        self.modell_nn.add(Dense(units=32, activation="relu"))
        self.modell_nn.add(Dense(2, activation="sigmoid"))
        self.modell_nn.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return self.modell_nn

    def fit_nn(self) -> None:
        """Train the neural network with early stopping across cross-validation folds."""
        scores_nn = []
        callback = EarlyStopping(monitor="val_loss", patience=50)

        fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3).split(
            self.x_train, self.y_train
        )

        y_train_categorical = to_categorical(self.y_train, num_classes=2)

        for k, (train, test) in enumerate(fold):
            self.modell_nn.fit(
                self.x_train[train],
                y_train_categorical[train],
                epochs=1500,
                callbacks=[callback],
                verbose=0,
                validation_split=0.3,
            )
            score_nn = self.modell_nn.evaluate(
                self.x_train[test],
                y_train_categorical[test],
                verbose=0,
            )
            scores_nn.append(score_nn)
            print(
                "NN - Fold: %2d, Acc.: %.3f, Loss: %.3f"
                % (k + 1, score_nn[1], score_nn[0])
            )
