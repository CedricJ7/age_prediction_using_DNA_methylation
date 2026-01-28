"""
Neural network models for age prediction.
"""

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.config import ModelConfig
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_mlp_model(config: ModelConfig) -> Pipeline:
    """
    Create Multi-Layer Perceptron (MLP) regressor.

    This is an implementation similar to AltumAge, an epigenetic age prediction model.

    Args:
        config: Model configuration

    Returns:
        Pipeline with scaler and MLP model

    Example:
        >>> model = create_mlp_model(ModelConfig())
        >>> model.named_steps['model'].hidden_layer_sizes
        (128, 64, 32)
    """
    logger.info(f"Creating MLP (AltumAge) with hidden_layers={config.mlp_hidden_layers}, "
                f"alpha={config.mlp_alpha}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=config.mlp_hidden_layers,
            activation="relu",
            solver="adam",
            alpha=config.mlp_alpha,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])
