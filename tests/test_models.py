"""Tests for model classes."""

import warnings
import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

from src.modules.models import Split, ModelEnsemble, NeuralNetwork


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Create sample training data with Survived column."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Survived": np.random.randint(0, 2, 100),
            "Feature1": np.random.randn(100),
            "Feature2": np.random.randn(100),
            "Feature3": np.random.randn(100),
            "Feature4": np.random.randn(100),
        }
    )


def test_split_initialization(sample_train_data: pd.DataFrame) -> None:
    """Test that Split can be initialized."""
    splitter = Split(train=sample_train_data)
    assert splitter is not None
    assert splitter.train.shape == sample_train_data.shape


def test_split_train_split(sample_train_data: pd.DataFrame) -> None:
    """Test that train_split creates proper splits."""
    splitter = Split(train=sample_train_data)
    x_train, x_test, y_train, y_test = splitter.train_split()

    # Check shapes
    assert len(x_train) + len(x_test) == len(sample_train_data)
    assert x_train.shape[1] == 4  # 4 features
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)

    # Check that it's a proper split (20% test)
    assert len(x_test) == 20
    assert len(x_train) == 80

    # Check stratification - ratios should be similar
    train_ratio = np.mean(y_train)
    test_ratio = np.mean(y_test)
    assert abs(train_ratio - test_ratio) < 0.2


def test_model_ensemble_initialization() -> None:
    """Test ModelEnsemble can be initialized."""
    x_train = np.random.randn(80, 5)
    x_test = np.random.randn(20, 5)
    y_train = np.random.randint(0, 2, 80).tolist()
    y_test = np.random.randint(0, 2, 20).tolist()

    ensemble = ModelEnsemble(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )

    assert ensemble.x_train.shape == (80, 5)
    assert ensemble.y_train == y_train
    assert len(ensemble.y_test) == 20


def test_neural_network_initialization() -> None:
    """Test NeuralNetwork initialization and post_init."""
    x_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100).tolist()

    nn = NeuralNetwork(x_train=x_train, y_train=y_train)

    # Check post_init worked
    assert nn.n_xtrain == 10  # 10 features


def test_neural_network_model_architecture() -> None:
    """Test that NN model is built correctly."""
    x_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100).tolist()

    nn = NeuralNetwork(x_train=x_train, y_train=y_train)
    model = nn.model_nn()

    # Check model exists
    assert model is not None

    # Check input shape
    assert model.input_shape == (None, 10)

    # Check output shape (binary classification with 1 unit)
    assert model.output_shape == (None, 1)

    # Check that model is compiled
    assert model.optimizer is not None
    assert model.loss is not None


def test_model_ensemble_save_load(tmp_path: Path) -> None:
    """Test saving and loading ensemble model."""
    # Suppress CatBoost sklearn compatibility warning (known issue, will be fixed in sklearn 1.8)
    warnings.filterwarnings(
        "ignore",
        message=".*CatBoostClassifier.*__sklearn_tags__.*",
        category=DeprecationWarning,
    )

    # Create sample data with proper DataFrame format
    x_train = pd.DataFrame(np.random.randn(80, 3), columns=["f1", "f2", "f3"])
    x_test = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])
    y_train = np.random.randint(0, 2, 80)
    y_test = np.random.randint(0, 2, 20)

    ensemble = ModelEnsemble(x_train, x_test, y_train, y_test)

    # Train model (this will take a while due to cross-validation)
    model = ensemble.model_cross()

    # Save model
    model_path = tmp_path / "test_ensemble.joblib"
    joblib.dump(model, model_path, compress=3)

    # Verify file exists
    assert model_path.exists()

    # Load model
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None

    # Make predictions with both models
    pred1 = model.predict(x_test)
    pred2 = loaded_model.predict(x_test)

    # Predictions should be identical
    assert np.array_equal(pred1, pred2)


def test_neural_network_save_load(tmp_path: Path) -> None:
    """Test saving and loading neural network."""
    # Suppress Keras optimizer loading warning (expected for untrained models)
    warnings.filterwarnings(
        "ignore",
        message=".*Skipping variable loading for optimizer.*",
        category=UserWarning,
    )

    x_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)

    nn = NeuralNetwork(x_train, y_train)
    model = nn.model_nn()

    # Save model
    model_path = tmp_path / "test_nn.keras"
    model.save(model_path)

    # Verify file exists
    assert model_path.exists()

    # Load model
    from keras.models import load_model

    loaded_model = load_model(model_path)

    # Verify architecture
    assert loaded_model.input_shape == model.input_shape
    assert loaded_model.output_shape == model.output_shape
    assert len(loaded_model.layers) == len(model.layers)

    # Make predictions with both models
    test_input = np.random.randn(5, 10)
    pred1 = model.predict(test_input, verbose=0)
    pred2 = loaded_model.predict(test_input, verbose=0)

    # Predictions should be very close (some float precision differences possible)
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
