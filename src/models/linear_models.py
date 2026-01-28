"""
Linear regression models with regularization for age prediction.
"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.config import ModelConfig
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_ridge_model(config: ModelConfig) -> Pipeline:
    """
    Create Ridge regression model with L2 regularization.

    Args:
        config: Model configuration

    Returns:
        Pipeline with scaler and Ridge model

    Example:
        >>> from src.utils.config import ModelConfig
        >>> model = create_ridge_model(ModelConfig())
        >>> model.named_steps['model'].alpha
        100.0
    """
    logger.info(f"Creating Ridge model with alpha={config.ridge_alpha}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=config.ridge_alpha)),
    ])


def create_lasso_model(config: ModelConfig) -> Pipeline:
    """
    Create Lasso regression model with L1 regularization.

    Args:
        config: Model configuration

    Returns:
        Pipeline with scaler and Lasso model

    Example:
        >>> model = create_lasso_model(ModelConfig())
        >>> model.named_steps['model'].alpha
        0.1
    """
    logger.info(f"Creating Lasso model with alpha={config.lasso_alpha}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(
            alpha=config.lasso_alpha,
            max_iter=50000,
            tol=1e-4
        )),
    ])


def create_elasticnet_model(config: ModelConfig) -> Pipeline:
    """
    Create ElasticNet model with L1+L2 regularization.

    Args:
        config: Model configuration

    Returns:
        Pipeline with scaler and ElasticNet model

    Example:
        >>> model = create_elasticnet_model(ModelConfig())
        >>> model.named_steps['model'].alpha
        0.1
        >>> model.named_steps['model'].l1_ratio
        0.5
    """
    logger.info(f"Creating ElasticNet model with alpha={config.elasticnet_alpha}, "
                f"l1_ratio={config.elasticnet_l1_ratio}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
            max_iter=50000,
            tol=1e-4
        )),
    ])
