"""Command line entry point for preparing data and training models."""

import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib
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


def main() -> None:
    """Execute the full training or inference pipeline based on CLI flags."""
    parser = ArgumentParser()
    parser.add_argument("--model_ensemble", action="store_true", 
                       help="Train ensemble models (default: train neural network)")
    parser.add_argument("--standardscaler", action="store_true",
                       help="Force StandardScaler preprocessing (NN uses it by default)")
    parser.add_argument("--retrain", action="store_true",
                       help="Force retraining even if model exists")
    parser.add_argument("--load_model", type=str,
                       help="Path to existing model file to load (skips training)")

    args = parser.parse_args()
    
    # Neural networks need standardized features by default
    use_standardscaler = args.standardscaler or not args.model_ensemble

    logger.info("Loading data")

    load = loading.LoadingFiles()

    # Use config paths
    pickle_load_dir = PROJECT_ROOT / config.get("paths.pickle_dir")
    pickle_load_train = pickle_load_dir / "loading/train.joblib"

    if not pickle_load_train.exists():
        (
            df_train,
            df_test,
        ) = load.load_save_df()
    else:
        (
            df_train,
            df_test,
        ) = load.load_db_file()

    logger.info("Data Preparation")
    load_data_train = data_preparation.LoadSave("train")
    load_data_test = data_preparation.LoadSave("test")

    load_data_train_standardscaler = data_preparation.LoadSave("train_standardscaler")
    load_data_test_standardscaler = data_preparation.LoadSave("test_standardscaler")

    pickle_train_path = pickle_load_dir / "data_preparation/data_set_train.joblib"
    pickle_train_std_path = (
        pickle_load_dir / "data_preparation/data_set_train_standardscaler.joblib"
    )

    # Check if the correct cache file exists based on what we need
    cache_exists = (
        pickle_train_std_path.exists() if use_standardscaler else pickle_train_path.exists()
    )

    if not cache_exists:
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
        if use_standardscaler:
            logger.info("Loading cached data with StandardScaler")
            train_final = load_data_train_standardscaler.load_dataframe()
            test_final = load_data_test_standardscaler.load_dataframe()
        else:
            logger.info("Loading cached data without StandardScaler")
            train_final = load_data_train.load_dataframe()
            test_final = load_data_test.load_dataframe()

    logger.info("Preparing models")
    split = models.Split(train_final)
    x_train, x_test, y_train, y_test = split.train_split()

    if args.model_ensemble:
        logger.info("Model Ensemble")
        model_dir = PROJECT_ROOT / config.get("paths.model_dir")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we should load an existing model
        if args.load_model:
            # Load specific model file
            model_path = Path(args.load_model)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            logger.info(f"Loading ensemble model from {model_path}")
            mv_clf = joblib.load(model_path)
        else:
            # Check for latest trained model in model_dir
            existing_models = sorted(model_dir.glob("stacking_classifier_*.joblib"))
            
            if existing_models and not args.retrain:
                # Load the most recent model
                latest_model = existing_models[-1]
                logger.info(f"Loading existing ensemble model from {latest_model}")
                mv_clf = joblib.load(latest_model)
            else:
                # Train new model
                if args.retrain and existing_models:
                    logger.info("Retraining ensemble model (--retrain flag set)")
                else:
                    logger.info("No existing model found, training new ensemble model")
                    
                stacking = models.ModelEnsemble(x_train, x_test, y_train, y_test)
                mv_clf = stacking.model_cross()

                # Save the trained ensemble model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = model_dir / f"stacking_classifier_{timestamp}.joblib"
                joblib.dump(mv_clf, model_path, compress=3)
                logger.info(f"Model saved to {model_path}")

        prediction_rfc = mv_clf.predict(test_final)
        test_result_rfc = test_final.copy()
        test_result_rfc["Survived"] = prediction_rfc
        results_rfc = test_result_rfc[["PassengerId", "Survived"]]

        predictions_dir = PROJECT_ROOT / config.get("paths.predictions_dir")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_path = predictions_dir / "prediction_titanic_RFC_new.csv"

        if not output_path.exists():
            results_rfc.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
    else:
        logger.info("Training Deep Neural Network")
        y_train = train_final.loc[:, "Survived"].to_numpy()
        features_train = train_final.drop(columns=["Survived"]).to_numpy()

        test_result_nn = test_final.copy()
        model_dir = PROJECT_ROOT / config.get("paths.model_dir")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we should load an existing model
        if args.load_model:
            # Load specific model file
            model_path = Path(args.load_model)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            logger.info(f"Loading model from {model_path}")
            from keras.models import load_model
            modell_nn = load_model(model_path)
        else:
            # Check for latest trained model in model_dir
            existing_models = sorted(model_dir.glob("neural_network_*.keras"))
            
            if existing_models and not args.retrain:
                # Load the most recent model
                latest_model = existing_models[-1]
                logger.info(f"Loading existing model from {latest_model}")
                from keras.models import load_model
                modell_nn = load_model(latest_model)
            else:
                # Train new model
                if args.retrain and existing_models:
                    logger.info("Retraining model (--retrain flag set)")
                else:
                    logger.info("No existing model found, training new model")
                    
                neural_network = models.NeuralNetwork(features_train, y_train)
                modell_nn = neural_network.model_nn()
                logger.info(modell_nn.summary())
                neural_network.fit_nn()

                # Save the trained neural network
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = model_dir / f"neural_network_{timestamp}.keras"
                modell_nn.save(model_path)
                logger.info(f"Model saved to {model_path}")

        # Binary classification: predictions are probabilities (0-1)
        # Threshold at 0.5 to get class labels
        predictions = modell_nn.predict(x=test_final.to_numpy(), verbose=2)
        label_array = (predictions > 0.5).astype(int).flatten()

        test_result_nn["Survived"] = label_array
        results_nn = test_result_nn[["PassengerId", "Survived"]]

        predictions_dir = PROJECT_ROOT / config.get("paths.predictions_dir")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_path = predictions_dir / "prediction_titanic_NN.csv"

        if not output_path.exists():
            results_nn.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
