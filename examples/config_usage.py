"""
Example: How to Use Configuration in Models

This file demonstrates how to refactor your models to use the configuration system.
You can apply these patterns to your actual model files.
"""

from src.config import get_config
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.metrics import AUC, Precision, Recall
from keras.regularizers import l2


def example_random_forest():
    """Example: Creating Random Forest with config."""
    config = get_config()

    # Load configuration
    rf_config = config.get("model_ensemble.random_forest")
    n_jobs = config.get("global.n_jobs")

    # Create model with config parameters
    clf = RandomForestClassifier(
        n_estimators=rf_config["n_estimators"],
        max_depth=rf_config["max_depth"],
        random_state=rf_config["random_state"],
        n_jobs=n_jobs,
    )

    return clf


def example_neural_network(n_features: int):
    """Example: Creating Neural Network with config."""
    config = get_config()

    # Load architecture config
    layer1 = config.get("neural_network.architecture.layer_1")
    layer2 = config.get("neural_network.architecture.layer_2")
    layer3 = config.get("neural_network.architecture.layer_3")
    layer4 = config.get("neural_network.architecture.layer_4")
    output_config = config.get("neural_network.architecture.output")

    # Load training config
    training = config.neural_network["training"]

    # Build model
    inputs = Input(shape=(n_features,), name="input")

    # Layer 1
    x = Dense(
        units=layer1["units"],
        activation=layer1["activation"],
        kernel_regularizer=l2(layer1["l2_regularization"]),
        name="dense_1",
    )(inputs)
    if layer1["batch_normalization"]:
        x = BatchNormalization(name="batch_norm_1")(x)
    x = Dropout(layer1["dropout"], name="dropout_1")(x)

    # Layer 2
    x = Dense(
        units=layer2["units"],
        activation=layer2["activation"],
        kernel_regularizer=l2(layer2["l2_regularization"]),
        name="dense_2",
    )(x)
    if layer2["batch_normalization"]:
        x = BatchNormalization(name="batch_norm_2")(x)
    x = Dropout(layer2["dropout"], name="dropout_2")(x)

    # Layer 3
    x = Dense(
        units=layer3["units"],
        activation=layer3["activation"],
        kernel_regularizer=l2(layer3["l2_regularization"]),
        name="dense_3",
    )(x)
    if layer3["batch_normalization"]:
        x = BatchNormalization(name="batch_norm_3")(x)
    x = Dropout(layer3["dropout"], name="dropout_3")(x)

    # Layer 4
    x = Dense(
        units=layer4["units"],
        activation=layer4["activation"],
        kernel_regularizer=l2(layer4["l2_regularization"]),
        name="dense_4",
    )(x)
    if layer4["batch_normalization"]:
        x = BatchNormalization(name="batch_norm_4")(x)
    x = Dropout(layer4["dropout"], name="dropout_4")(x)

    # Output
    outputs = Dense(
        output_config["units"], activation=output_config["activation"], name="output"
    )(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs, name="titanic_nn")

    model.compile(
        optimizer=Adam(learning_rate=training["learning_rate"]),
        loss=training["loss"],
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )

    return model


def example_training_config():
    """Example: Getting training configuration."""
    config = get_config()

    # Get training parameters
    batch_size = config.get("neural_network.training.batch_size")
    epochs = config.get("neural_network.training.epochs")
    validation_split = config.get("neural_network.training.validation_split")

    # Get early stopping config
    early_stop_config = config.get("neural_network.callbacks.early_stopping")

    print(f"Training with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Validation split: {validation_split}")
    print(f"  Early stopping patience: {early_stop_config['patience']}")


def example_all_ensemble_models():
    """Example: Creating all ensemble models from config."""
    config = get_config()
    ensemble_config = config.model_ensemble

    models = {}

    # Just showing structure - you'd create actual model instances
    for model_name, model_config in ensemble_config.items():
        if model_name == "scoring_metric":
            continue
        if model_name == "stacking":
            continue
        print(f"{model_name}: {model_config}")

    return models


if __name__ == "__main__":
    print("=" * 60)
    print("Configuration System Examples")
    print("=" * 60)

    print("\n1. Random Forest from Config:")
    rf = example_random_forest()
    print(f"   Created: {rf}")

    print("\n2. Neural Network from Config:")
    nn = example_neural_network(n_features=10)
    print(f"   Created model with {len(nn.layers)} layers")

    print("\n3. Training Configuration:")
    example_training_config()

    print("\n4. All Ensemble Model Configs:")
    example_all_ensemble_models()
