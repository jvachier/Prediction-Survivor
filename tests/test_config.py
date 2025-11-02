"""Tests for configuration module."""

from src.config import Config, get_config


def test_config_loads_successfully() -> None:
    """Test that config file loads without errors."""
    config = Config()
    assert config is not None


def test_config_get_global_settings() -> None:
    """Test accessing global settings."""
    config = Config()
    random_state = config.get("global.random_state")
    assert random_state == 42

    n_jobs = config.get("global.n_jobs")
    assert n_jobs == 4


def test_config_get_neural_network_settings() -> None:
    """Test accessing neural network settings."""
    config = Config()
    learning_rate = config.get("neural_network.training.learning_rate")
    assert learning_rate == 0.001

    layer1_units = config.get("neural_network.architecture.layer_1.units")
    assert layer1_units == 256


def test_config_get_model_ensemble_settings() -> None:
    """Test accessing model ensemble settings."""
    config = Config()
    rf_estimators = config.get("model_ensemble.random_forest.n_estimators")
    assert rf_estimators == 50

    xgb_max_depth = config.get("model_ensemble.xgboost.max_depth")
    assert xgb_max_depth == 5


def test_config_get_with_default() -> None:
    """Test getting non-existent key returns default."""
    config = Config()
    value = config.get("nonexistent.key", default="default_value")
    assert value == "default_value"


def test_config_properties() -> None:
    """Test configuration properties."""
    config = Config()

    assert "random_state" in config.global_settings
    assert "age_bins" in config.data_preparation
    assert "random_forest" in config.model_ensemble
    assert "architecture" in config.neural_network
    assert "data_dir" in config.paths


def test_config_dict_access() -> None:
    """Test dictionary-style access."""
    config = Config()
    global_settings = config["global"]
    assert global_settings["random_state"] == 42


def test_get_config_singleton() -> None:
    """Test that get_config returns singleton instance."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2
