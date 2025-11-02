"""Command line entry point for preparing data and training models."""

import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import pandas as pd
from keras.models import load_model

from src.config import get_config
from src.modules import data_preparation
from src.modules import models
from src.modules import loading

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Define project root for reliable path resolution
PROJECT_ROOT = Path(__file__).parent.parent


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Validate prediction DataFrame format.

    Args:
        predictions: DataFrame with PassengerId and Survived columns

    Raises:
        ValueError: If predictions are not in correct format
    """
    # Check required columns
    required_cols = ["PassengerId", "Survived"]
    if not all(col in predictions.columns for col in required_cols):
        raise ValueError(f"Predictions must have columns: {required_cols}")

    # Check data types
    if not pd.api.types.is_integer_dtype(predictions["PassengerId"]):
        raise ValueError("PassengerId must be integer type")

    if not pd.api.types.is_integer_dtype(predictions["Survived"]):
        raise ValueError("Survived must be integer type")

    # Check Survived values are binary
    unique_values = predictions["Survived"].unique()
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"Survived must be 0 or 1, got: {unique_values}")

    # Check for missing values
    if predictions.isnull().any().any():
        raise ValueError("Predictions contain missing values")

    logger.info(
        f"Predictions validated: {len(predictions)} rows, "
        f"{predictions['Survived'].sum()} survived, "
        f"{len(predictions) - predictions['Survived'].sum()} died"
    )


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training and test data, using cache if available.

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Loading data")
    load = loading.LoadingFiles()
    pickle_load_dir = PROJECT_ROOT / config.get("paths.pickle_dir")
    pickle_load_train = pickle_load_dir / "loading/train.joblib"

    if pickle_load_train.exists():
        return load.load_db_file()
    return load.load_save_df()


def prepare_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame, use_standardscaler: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare and transform data with optional StandardScaler.

    Args:
        df_train: Raw training dataframe
        df_test: Raw test dataframe
        use_standardscaler: Whether to apply StandardScaler preprocessing

    Returns:
        Tuple of (prepared_train, prepared_test)
    """
    logger.info("Data Preparation")
    pickle_load_dir = PROJECT_ROOT / config.get("paths.pickle_dir")

    # Select appropriate cache files and loaders
    suffix = "_standardscaler" if use_standardscaler else ""
    cache_path = pickle_load_dir / f"data_preparation/data_set_train{suffix}.joblib"

    if cache_path.exists():
        logger.info(
            f"Loading cached data {'with' if use_standardscaler else 'without'} StandardScaler"
        )
        load_train = data_preparation.LoadSave(f"train{suffix}")
        load_test = data_preparation.LoadSave(f"test{suffix}")
        return load_train.load_dataframe(), load_test.load_dataframe()

    # Process data from scratch
    train = data_preparation.DataPreparation(df_train)
    train_prep1 = train.preparation_first()
    train_selec = train.selection(train_prep1)

    test = data_preparation.DataPreparation(df_test)
    test_prep1 = test.preparation_first()
    test_selec = test.selection(test_prep1)

    if use_standardscaler:
        logger.info("Applying StandardScaler")
        train_prep2 = train.preparation_second_standardscaler(train_selec)
        train_final = train.preparation_dummies_standardscaler(train_prep2)
        test_prep2 = test.preparation_second_standardscaler(test_selec)
        test_final = test.preparation_dummies_standardscaler(test_prep2)

        load_train = data_preparation.LoadSave("train_standardscaler")
        load_test = data_preparation.LoadSave("test_standardscaler")
    else:
        train_prep2 = train.preparation_second(train_selec)
        train_final = train.preparation_dummies(train_prep2)
        test_prep2 = test.preparation_second(test_selec)
        test_final = test.preparation_dummies(test_prep2)

        load_train = data_preparation.LoadSave("train")
        load_test = data_preparation.LoadSave("test")

    load_train.save_dataframe(train_final)
    load_test.save_dataframe(test_final)

    return train_final, test_final


def try_load_model(model_path: Path, load_fn) -> Optional[Any]:
    """Attempt to load a model from a path.

    Args:
        model_path: Path to the model file
        load_fn: Function to use for loading (joblib.load or load_model)

    Returns:
        Loaded model or None if loading failed
    """
    try:
        logger.info(f"Loading model from {model_path}")
        return load_fn(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        logger.info("Falling back to training new model")
        return None


def get_model_path(args, model_dir: Path, pattern: str) -> Optional[Path]:
    """Determine which model path to load based on CLI arguments.

    Args:
        args: Parsed command line arguments
        model_dir: Directory containing saved models
        pattern: Glob pattern to find models (e.g., "*.keras")

    Returns:
        Path to model file or None if no model should be loaded
    """
    if args.load_model:
        # Load specific model file
        model_path = Path(args.load_model)
        return model_path if model_path.is_absolute() else PROJECT_ROOT / model_path

    if args.retrain:
        return None

    # Find latest trained model
    existing_models = sorted(model_dir.glob(pattern))
    return existing_models[-1] if existing_models else None


def train_ensemble_model(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train, y_test, model_dir: Path
):
    """Train and save a new ensemble model.

    Args:
        x_train: Training features
        x_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_dir: Directory to save the model

    Returns:
        Trained model
    """
    logger.info("Training new ensemble model")
    stacking = models.ModelEnsemble(x_train, x_test, y_train, y_test)
    mv_clf = stacking.model_cross()

    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"stacking_classifier_{timestamp}.joblib"
    joblib.dump(mv_clf, model_path, compress=3)
    logger.info(f"Model saved to {model_path}")

    return mv_clf


def train_neural_network(features_train: pd.DataFrame, y_train, model_dir: Path):
    """Train and save a new neural network model.

    Args:
        features_train: Training features
        y_train: Training labels
        model_dir: Directory to save the model

    Returns:
        Trained model
    """
    logger.info("Training new neural network")
    neural_network = models.NeuralNetwork(features_train, y_train)
    modell_nn = neural_network.model_nn()
    logger.info(modell_nn.summary())
    neural_network.fit_nn()

    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"neural_network_{timestamp}.keras"
    modell_nn.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return modell_nn


def save_predictions(predictions: pd.DataFrame, filename: str, model_type: str) -> None:
    """Validate and save predictions to CSV.

    Args:
        predictions: DataFrame with PassengerId and Survived columns
        filename: Name of the output CSV file
        model_type: Type of model for logging (e.g., "Ensemble", "Neural network")
    """
    # Create a copy to avoid SettingWithCopyWarning
    predictions = predictions.copy()

    # Ensure correct data types
    predictions["PassengerId"] = predictions["PassengerId"].astype(int)
    predictions["Survived"] = predictions["Survived"].astype(int)

    # Validate predictions
    validate_predictions(predictions)

    # Save to CSV
    predictions_dir = PROJECT_ROOT / config.get("paths.predictions_dir")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = predictions_dir / filename

    predictions.to_csv(output_path, index=False)
    logger.info(f"{model_type} predictions saved to {output_path}")


def run_ensemble_pipeline(
    args, train_final: pd.DataFrame, test_final: pd.DataFrame
) -> None:
    """Execute ensemble model training and prediction pipeline.

    Args:
        args: Parsed command line arguments
        train_final: Prepared training data
        test_final: Prepared test data
    """
    logger.info("Model Ensemble")
    model_dir = PROJECT_ROOT / config.get("paths.model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Prepare train/test split
    split = models.Split(train_final)
    x_train, x_test, y_train, y_test = split.train_split()

    # Try to load existing model
    model_path = get_model_path(args, model_dir, "stacking_classifier_*.joblib")
    mv_clf = try_load_model(model_path, joblib.load) if model_path else None

    # Train new model if needed
    if mv_clf is None:
        mv_clf = train_ensemble_model(x_train, x_test, y_train, y_test, model_dir)

    # Make predictions
    prediction_rfc = mv_clf.predict(test_final)
    test_result_rfc = test_final.copy()
    test_result_rfc["Survived"] = prediction_rfc.astype(int)
    results_rfc = test_result_rfc[["PassengerId", "Survived"]]

    # Save predictions
    save_predictions(results_rfc, "prediction_titanic_RFC_new.csv", "Ensemble")


def run_neural_network_pipeline(
    args, train_final: pd.DataFrame, test_final: pd.DataFrame
) -> None:
    """Execute neural network training and prediction pipeline.

    Args:
        args: Parsed command line arguments
        train_final: Prepared training data
        test_final: Prepared test data
    """
    logger.info("Training Deep Neural Network")
    model_dir = PROJECT_ROOT / config.get("paths.model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training data
    y_train = train_final.loc[:, "Survived"].to_numpy()
    features_train = train_final.drop(columns=["Survived"]).to_numpy()

    # Try to load existing model
    model_path = get_model_path(args, model_dir, "neural_network_*.keras")
    modell_nn = try_load_model(model_path, load_model) if model_path else None

    # Train new model if needed
    if modell_nn is None:
        modell_nn = train_neural_network(features_train, y_train, model_dir)

    # Make predictions
    predictions = modell_nn.predict(x=test_final.to_numpy(), verbose=2)
    label_array = (predictions > 0.5).astype(int).flatten()

    test_result_nn = test_final.copy()
    test_result_nn["Survived"] = label_array
    results_nn = test_result_nn[["PassengerId", "Survived"]]

    # Save predictions
    save_predictions(results_nn, "prediction_titanic_NN.csv", "Neural network")


def main() -> None:
    """Execute the full training or inference pipeline based on CLI flags."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model_ensemble",
        action="store_true",
        help="Train ensemble models (default: train neural network)",
    )
    parser.add_argument(
        "--standardscaler",
        action="store_true",
        help="Force StandardScaler preprocessing (NN uses it by default)",
    )
    parser.add_argument(
        "--retrain", action="store_true", help="Force retraining even if model exists"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Path to existing model file to load (skips training)",
    )

    args = parser.parse_args()

    # Neural networks need standardized features by default
    use_standardscaler = args.standardscaler or not args.model_ensemble

    # Load and prepare data
    df_train, df_test = load_raw_data()
    train_final, test_final = prepare_data(df_train, df_test, use_standardscaler)

    # Run appropriate pipeline
    if args.model_ensemble:
        run_ensemble_pipeline(args, train_final, test_final)
    else:
        run_neural_network_pipeline(args, train_final, test_final)


if __name__ == "__main__":
    main()
