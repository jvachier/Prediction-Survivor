"""Model utilities for ensemble classifiers and neural networks."""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.metrics import AUC, Precision, Recall
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Split:
    """Utility class to create train/test splits for the Titanic dataset."""

    train: pd.DataFrame

    def train_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split the training data into hold-out train and validation sets.

        Returns DataFrames for X to preserve feature names for tree-based models.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.train.drop(columns=["Survived"]),
            self.train["Survived"].values,
            test_size=0.20,
            stratify=self.train["Survived"].values,
            random_state=1,
        )
        return x_train, x_test, y_train, y_test


@dataclass(slots=True)
class ModelEnsemble:
    """Train a soft-voting ensemble comprised of classic ML estimators."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray

    def model_cross(self) -> StackingClassifier:
        """Perform cross validation on individual estimators and fit the stacking ensemble."""
        # Base estimators
        clf_rfc = RandomForestClassifier(
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

        clf_sgdc = SGDClassifier(
            loss="log_loss",
            max_iter=10000,
            random_state=4,
            learning_rate="adaptive",
            eta0=1e-4,
        )

        clf_knnc = KNeighborsClassifier(n_neighbors=50)

        # Gradient Boosting estimators
        clf_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=6,
            n_jobs=4,
            eval_metric="logloss",
        )

        clf_lgbm = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=7,
            n_jobs=4,
            verbose=-1,
        )

        clf_catboost = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=8,
            verbose=0,
        )

        # Pipelines for base estimators
        pipe_rfc = Pipeline([["rfc", clf_rfc]])
        pipe_adaboost = Pipeline([["adaboost", clf_adaboost]])
        pipe_lr = Pipeline([["lr", clf_lr]])
        pipe_dt = Pipeline([["dt", clf_dt]])
        pipe_sgdv = Pipeline([["sgdc", clf_sgdc]])
        pipe_knnc = Pipeline([["knnc", clf_knnc]])
        pipe_xgb = Pipeline([["xgb", clf_xgb]])
        pipe_lgbm = Pipeline([["lgbm", clf_lgbm]])
        pipe_catboost = Pipeline([["catboost", clf_catboost]])

        # Stacking Classifier with all base estimators
        stacking_clf = StackingClassifier(
            estimators=[
                ["rfc", clf_rfc],
                ["adaboost", clf_adaboost],
                ["lr", clf_lr],
                ["dt", clf_dt],
                ["sgdc", clf_sgdc],
                ["knnc", clf_knnc],
                ["xgb", clf_xgb],
                ["lgbm", clf_lgbm],
                ["catboost", clf_catboost],
            ],
            final_estimator=LogisticRegression(solver="lbfgs", max_iter=10000),
            cv=5,
            n_jobs=4,
        )

        clf_labels = [
            "Random Forest",
            "Adaboost",
            "Logistic Regression",
            "Decision Tree",
            "Stochastic Gradient Descent",
            "Nearest Neighbors",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "Stacking Classifier",
        ]
        all_clf = [
            pipe_rfc,
            pipe_adaboost,
            pipe_lr,
            pipe_dt,
            pipe_sgdv,
            pipe_knnc,
            pipe_xgb,
            pipe_lgbm,
            pipe_catboost,
            stacking_clf,
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
            logger.info(
                "ROC AUC: %0.2f (+/- %0.2f) [%s]",
                scores.mean(),
                scores.std(),
                label,
            )
        stacking_clf.fit(self.x_train, self.y_train)
        return stacking_clf


@dataclass(slots=True)
class NeuralNetwork:
    """Define and train a dense neural network for Titanic survival prediction."""

    x_train: np.ndarray
    y_train: np.ndarray
    n_xtrain: int = None
    modell_nn: Model = None

    def __post_init__(self) -> None:
        """Capture input dimensionality after initialisation."""
        self.n_xtrain = self.x_train.shape[1]

    def model_nn(self) -> Model:
        """Build and compile the neural network using Functional API with modern improvements."""
        # Input layer
        inputs = Input(shape=(self.n_xtrain,), name="input")

        # First block
        x = Dense(
            units=256,
            activation="gelu",
            kernel_regularizer=l2(0.001),
            name="dense_1",
        )(inputs)
        x = BatchNormalization(name="batch_norm_1")(x)
        x = Dropout(0.3, name="dropout_1")(x)

        # Second block
        x = Dense(
            units=128,
            activation="gelu",
            kernel_regularizer=l2(0.001),
            name="dense_2",
        )(x)
        x = BatchNormalization(name="batch_norm_2")(x)
        x = Dropout(0.3, name="dropout_2")(x)

        # Third block
        x = Dense(
            units=64,
            activation="gelu",
            kernel_regularizer=l2(0.001),
            name="dense_3",
        )(x)
        x = BatchNormalization(name="batch_norm_3")(x)
        x = Dropout(0.2, name="dropout_3")(x)

        # Fourth block
        x = Dense(
            units=32,
            activation="gelu",
            kernel_regularizer=l2(0.001),
            name="dense_4",
        )(x)
        x = BatchNormalization(name="batch_norm_4")(x)
        x = Dropout(0.2, name="dropout_4")(x)

        # Output layer - Binary classification
        outputs = Dense(1, activation="sigmoid", name="output")(x)

        # Create model
        self.modell_nn = Model(
            inputs=inputs, outputs=outputs, name="titanic_survival_nn"
        )

        self.modell_nn.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(name="auc"),
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )
        return self.modell_nn

    def fit_nn(self) -> None:
        """Train the neural network with early stopping and learning rate reduction."""
        scores_nn = []

        # Callbacks for better training
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
            verbose=0,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            min_lr=1e-7,
            verbose=0,
        )

        fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3).split(
            self.x_train, self.y_train
        )

        for k, (train, test) in enumerate(fold):
            self.modell_nn.fit(
                self.x_train[train],
                self.y_train[train],
                batch_size=32,
                epochs=1000,  # Reduced from 1500, early stopping will handle it
                callbacks=[early_stop, reduce_lr],
                verbose=0,
                validation_split=0.2,  # 20% for validation (was 30%)
            )
            score_nn = self.modell_nn.evaluate(
                self.x_train[test],
                self.y_train[test],
                verbose=0,
            )
            scores_nn.append(score_nn)
            logger.info(
                "NN - Fold: %2d, Loss: %.3f, Acc.: %.3f, AUC: %.3f, Prec.: %.3f, Rec.: %.3f",
                k + 1,
                score_nn[0],  # loss
                score_nn[1],  # accuracy
                score_nn[2],  # auc
                score_nn[3],  # precision
                score_nn[4],  # recall
            )
