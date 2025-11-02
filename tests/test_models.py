"""Tests for model classes."""

import numpy as np
import pandas as pd
import pytest

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
