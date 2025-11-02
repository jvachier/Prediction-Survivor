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

from src.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


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
        test_size = config.get("data_preparation.test_size", 0.20)
        random_state = config.get("global.random_state", 42)

        x_train, x_test, y_train, y_test = train_test_split(
            self.train.drop(columns=["Survived"]),
            self.train["Survived"].values,
            test_size=test_size,
            stratify=self.train["Survived"].values,
            random_state=random_state,
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
        # Get configuration
        n_jobs = config.get("global.n_jobs", 4)
        cv_folds = config.get("global.cv_folds", 10)

        # Base estimators with config
        rfc_config = config.model_ensemble["random_forest"]
        clf_rfc = RandomForestClassifier(
            n_estimators=rfc_config["n_estimators"],
            max_depth=rfc_config["max_depth"],
            random_state=rfc_config["random_state"],
            n_jobs=n_jobs,
        )

        adaboost_config = config.model_ensemble["adaboost"]
        clf_adaboost = AdaBoostClassifier(
            n_estimators=adaboost_config["n_estimators"],
            random_state=adaboost_config["random_state"],
        )

        lr_config = config.model_ensemble["logistic_regression"]
        clf_lr = LogisticRegression(
            solver=lr_config["solver"],
            max_iter=lr_config["max_iter"],
            random_state=lr_config["random_state"],
        )

        dt_config = config.model_ensemble["decision_tree"]
        clf_dt = DecisionTreeClassifier(
            max_depth=dt_config["max_depth"],
            max_features=dt_config["max_features"],
            random_state=dt_config["random_state"],
        )

        sgdc_config = config.model_ensemble["sgd_classifier"]
        clf_sgdc = SGDClassifier(
            loss=sgdc_config["loss"],
            max_iter=sgdc_config["max_iter"],
            random_state=sgdc_config["random_state"],
            learning_rate=sgdc_config["learning_rate"],
            eta0=sgdc_config["eta0"],
        )

        knn_config = config.model_ensemble["knn"]
        clf_knnc = KNeighborsClassifier(n_neighbors=knn_config["n_neighbors"])

        # Gradient Boosting estimators with config
        xgb_config = config.model_ensemble["xgboost"]
        clf_xgb = XGBClassifier(
            n_estimators=xgb_config["n_estimators"],
            max_depth=xgb_config["max_depth"],
            learning_rate=xgb_config["learning_rate"],
            random_state=xgb_config["random_state"],
            n_jobs=n_jobs,
            eval_metric=xgb_config["eval_metric"],
        )

        lgbm_config = config.model_ensemble["lightgbm"]
        clf_lgbm = LGBMClassifier(
            n_estimators=lgbm_config["n_estimators"],
            max_depth=lgbm_config["max_depth"],
            learning_rate=lgbm_config["learning_rate"],
            random_state=lgbm_config["random_state"],
            n_jobs=n_jobs,
            verbose=lgbm_config["verbose"],
        )

        catboost_config = config.model_ensemble["catboost"]
        clf_catboost = CatBoostClassifier(
            iterations=catboost_config["iterations"],
            depth=catboost_config["depth"],
            learning_rate=catboost_config["learning_rate"],
            random_state=catboost_config["random_state"],
            verbose=catboost_config["verbose"],
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
            final_estimator=LogisticRegression(
                solver=lr_config["solver"], max_iter=lr_config["max_iter"]
            ),
            cv=cv_folds,
            n_jobs=n_jobs,
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

        stratiKfold = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=config.get("global.random_state", 42),
        )

        for clf, label in zip(all_clf, clf_labels):
            scores = cross_val_score(
                estimator=clf,
                X=self.x_train,
                y=self.y_train,
                cv=stratiKfold,
                n_jobs=n_jobs,
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
        # Get configuration
        nn_config = config.neural_network
        arch = nn_config["architecture"]
        training = nn_config["training"]

        # Input layer
        inputs = Input(shape=(self.n_xtrain,), name="input")

        # First block
        x = Dense(
            units=arch["layer_1"]["units"],
            activation=arch["layer_1"]["activation"],
            kernel_regularizer=l2(arch["layer_1"]["l2_regularization"]),
            name="dense_1",
        )(inputs)
        x = BatchNormalization(name="batch_norm_1")(x)
        x = Dropout(arch["layer_1"]["dropout"], name="dropout_1")(x)

        # Second block
        x = Dense(
            units=arch["layer_2"]["units"],
            activation=arch["layer_2"]["activation"],
            kernel_regularizer=l2(arch["layer_2"]["l2_regularization"]),
            name="dense_2",
        )(x)
        x = BatchNormalization(name="batch_norm_2")(x)
        x = Dropout(arch["layer_2"]["dropout"], name="dropout_2")(x)

        # Third block
        x = Dense(
            units=arch["layer_3"]["units"],
            activation=arch["layer_3"]["activation"],
            kernel_regularizer=l2(arch["layer_3"]["l2_regularization"]),
            name="dense_3",
        )(x)
        x = BatchNormalization(name="batch_norm_3")(x)
        x = Dropout(arch["layer_3"]["dropout"], name="dropout_3")(x)

        # Fourth block
        x = Dense(
            units=arch["layer_4"]["units"],
            activation=arch["layer_4"]["activation"],
            kernel_regularizer=l2(arch["layer_4"]["l2_regularization"]),
            name="dense_4",
        )(x)
        x = BatchNormalization(name="batch_norm_4")(x)
        x = Dropout(arch["layer_4"]["dropout"], name="dropout_4")(x)

        # Output layer - Binary classification
        outputs = Dense(
            arch["output"]["units"], 
            activation=arch["output"]["activation"], 
            name="output"
        )(x)

        # Create model
        self.modell_nn = Model(
            inputs=inputs, outputs=outputs, name="titanic_survival_nn"
        )

        self.modell_nn.compile(
            optimizer=Adam(learning_rate=training["learning_rate"]),
            loss=training["loss"],
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

        # Get configuration
        training = config.neural_network["training"]
        callbacks_config = config.neural_network["callbacks"]

        # Callbacks for better training
        early_stop = EarlyStopping(
            monitor=callbacks_config["early_stopping"]["monitor"],
            patience=callbacks_config["early_stopping"]["patience"],
            restore_best_weights=callbacks_config["early_stopping"][
                "restore_best_weights"
            ],
            verbose=callbacks_config["early_stopping"]["verbose"],
        )
        reduce_lr = ReduceLROnPlateau(
            monitor=callbacks_config["reduce_lr"]["monitor"],
            factor=callbacks_config["reduce_lr"]["factor"],
            patience=callbacks_config["reduce_lr"]["patience"],
            min_lr=callbacks_config["reduce_lr"]["min_lr"],
            verbose=callbacks_config["reduce_lr"]["verbose"],
        )

        fold = StratifiedKFold(
            n_splits=config.get("global.cv_folds", 10),
            shuffle=True,
            random_state=config.get("global.random_state", 42),
        ).split(self.x_train, self.y_train)

        for k, (train, test) in enumerate(fold):
            self.modell_nn.fit(
                self.x_train[train],
                self.y_train[train],
                batch_size=training["batch_size"],
                epochs=training["epochs"],
                callbacks=[early_stop, reduce_lr],
                verbose=training["verbose"],
                validation_split=training["validation_split"],
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
