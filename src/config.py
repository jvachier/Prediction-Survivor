"""Configuration loader for Titanic Survival Prediction."""

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Load and access configuration from YAML file."""

    def __init__(self, config_path: Path = None):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to config.yaml file. Defaults to project root.
        """
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"

        with open(config_path, "r") as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "global.random_state")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = Config()
            >>> config.get("global.random_state")
            42
            >>> config.get("neural_network.training.learning_rate")
            0.001
        """
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def global_settings(self) -> Dict[str, Any]:
        """Get global settings."""
        return self._config.get("global", {})

    @property
    def data_preparation(self) -> Dict[str, Any]:
        """Get data preparation settings."""
        return self._config.get("data_preparation", {})

    @property
    def model_ensemble(self) -> Dict[str, Any]:
        """Get model ensemble settings."""
        return self._config.get("model_ensemble", {})

    @property
    def neural_network(self) -> Dict[str, Any]:
        """Get neural network settings."""
        return self._config.get("neural_network", {})

    @property
    def paths(self) -> Dict[str, Any]:
        """Get path settings."""
        return self._config.get("paths", {})

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]


# Singleton instance
_config_instance = None


def get_config(config_path: Path = None) -> Config:
    """Get or create configuration singleton.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
