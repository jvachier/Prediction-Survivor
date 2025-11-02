# Configuration System

This project uses a YAML-based configuration system to manage all model parameters and global settings.

## Files

- **`config.yaml`** - Main configuration file with all parameters
- **`src/config.py`** - Configuration loader module
- **`examples/config_usage.py`** - Usage examples

## Quick Start

```python
from src.config import get_config

# Load configuration
config = get_config()

# Access settings using dot notation
random_state = config.get("global.random_state")  # 42
learning_rate = config.get("neural_network.training.learning_rate")  # 0.001

# Access entire sections
nn_config = config.neural_network
rf_config = config.model_ensemble["random_forest"]
```

## Configuration Structure

### Global Settings
```yaml
global:
  random_state: 42      # Seed for reproducibility
  n_jobs: 4             # Parallel processing threads
  test_size: 0.20       # Train/test split ratio
  cv_folds: 10          # Cross-validation folds
  cv_shuffle: true      # Shuffle during CV
```

### Data Preparation
```yaml
data_preparation:
  age_bins: [-inf, 11, 18, 22, 27, 33, 40, inf]
  fare_bins: [-inf, 7.91, 14.454, 31, 99, 250, inf]
  impute_age_distribution: true
  random_seed_age: 0
```

### Model Ensemble
Configure all ensemble models (Random Forest, XGBoost, LightGBM, CatBoost, etc.):

```yaml
model_ensemble:
  scoring_metric: "roc_auc"
  
  random_forest:
    n_estimators: 50
    max_depth: 10
    random_state: 1
  
  xgboost:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
    # ... more parameters
```

### Neural Network
Complete neural network architecture and training configuration:

```yaml
neural_network:
  architecture:
    layer_1:
      units: 256
      activation: "gelu"
      l2_regularization: 0.001
      dropout: 0.3
      batch_normalization: true
    # ... more layers
  
  training:
    optimizer: "adam"
    learning_rate: 0.001
    batch_size: 32
    epochs: 1000
    validation_split: 0.2
  
  callbacks:
    early_stopping:
      patience: 50
      restore_best_weights: true
    reduce_lr:
      factor: 0.5
      patience: 20
```

## Usage Examples

### Example 1: Create Model from Config

```python
from src.config import get_config
from sklearn.ensemble import RandomForestClassifier

config = get_config()
rf_config = config.get("model_ensemble.random_forest")

clf = RandomForestClassifier(
    n_estimators=rf_config["n_estimators"],
    max_depth=rf_config["max_depth"],
    random_state=rf_config["random_state"],
    n_jobs=config.get("global.n_jobs"),
)
```

### Example 2: Neural Network Training

```python
from src.config import get_config

config = get_config()

# Get training parameters
batch_size = config.get("neural_network.training.batch_size")
epochs = config.get("neural_network.training.epochs")

# Get callback settings
early_stop = config.get("neural_network.callbacks.early_stopping")
patience = early_stop["patience"]

# Train model
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[...],
)
```

### Example 3: Access with Default Values

```python
# If key doesn't exist, return default
value = config.get("some.missing.key", default=100)
```

## Benefits

**Centralized Configuration** - All parameters in one place  
**Easy Experimentation** - Change parameters without modifying code  
**Version Control** - Track configuration changes in git  
**Reproducibility** - Share exact configuration with others  
**Type Safety** - YAML structure validation  
**Documentation** - Self-documenting with comments in YAML  

## Modifying Configuration

1. Edit `config.yaml`
2. No code changes needed
3. Run your script - new values are automatically used

Example:
```bash
# Try different neural network architectures
vim config.yaml  # Change layer_1.units from 256 to 512
uv run python src/main.py  # Automatically uses new value
```

## Testing

Configuration has its own test suite:

```bash
uv run pytest tests/test_config.py -v
```

## Advanced Usage

See `examples/config_usage.py` for complete working examples:

```bash
uv run python examples/config_usage.py
```

## Integration Checklist

To integrate configuration into existing code:

- [ ] Import: `from src.config import get_config`
- [ ] Load: `config = get_config()`
- [ ] Replace hardcoded values with: `config.get("path.to.parameter")`
- [ ] Update tests if needed
- [ ] Document any new configuration parameters

## Tips

1. **Use dot notation** for nested access: `config.get("neural_network.training.learning_rate")`
2. **Provide defaults** for optional parameters: `config.get("key", default=value)`
3. **Use properties** for common sections: `config.neural_network`, `config.global_settings`
4. **Keep config.yaml organized** with comments for each section
5. **Version control** your config changes for reproducibility

---

For more examples, see `examples/config_usage.py`
