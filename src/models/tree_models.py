"""
Tree-based ensemble models for age prediction.
"""

from sklearn.ensemble import RandomForestRegressor

from ..utils.config import ModelConfig
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_random_forest_model(config: ModelConfig) -> RandomForestRegressor:
    """
    Create Random Forest regressor.

    Args:
        config: Model configuration

    Returns:
        RandomForestRegressor instance

    Example:
        >>> model = create_random_forest_model(ModelConfig())
        >>> model.n_estimators
        300
    """
    logger.info(f"Creating Random Forest with n_estimators={config.rf_n_estimators}, "
                f"max_depth={config.rf_max_depth}")

    return RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )


def create_xgboost_model(config: ModelConfig):
    """
    Create XGBoost regressor with strong regularization.

    Args:
        config: Model configuration

    Returns:
        XGBRegressor instance

    Raises:
        ImportError: If xgboost is not installed

    Example:
        >>> model = create_xgboost_model(ModelConfig())
        >>> model.n_estimators
        400
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.error("XGBoost not installed. Install with: pip install xgboost")
        raise

    logger.info(f"Creating XGBoost with n_estimators={config.xgboost_n_estimators}, "
                f"learning_rate={config.xgboost_learning_rate}, "
                f"max_depth={config.xgboost_max_depth}, "
                f"reg_alpha={config.xgboost_reg_alpha}, "
                f"reg_lambda={config.xgboost_reg_lambda}, "
                f"early_stopping_rounds={config.xgboost_early_stopping_rounds}")

    return XGBRegressor(
        n_estimators=config.xgboost_n_estimators,
        learning_rate=config.xgboost_learning_rate,
        max_depth=config.xgboost_max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=config.xgboost_reg_alpha,
        reg_lambda=config.xgboost_reg_lambda,
        early_stopping_rounds=config.xgboost_early_stopping_rounds,
        objective="reg:squarederror",
        eval_metric="mae",
        n_jobs=-1,
        random_state=42,
    )
