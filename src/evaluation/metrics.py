"""
Model evaluation metrics for age prediction.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: int = 10,
) -> Dict[str, float]:
    """
    Evaluate a model comprehensively on test and cross-validation sets.

    Computes multiple metrics including:
    - MAE (Mean Absolute Error)
    - MAD (Median Absolute Deviation)
    - RMSE (Root Mean Squared Error)
    - R² (Coefficient of Determination)
    - Pearson correlation
    - Cross-validation scores
    - Overfitting ratio

    Args:
        model: Fitted model with predict() method
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        cv: Number of cross-validation folds

    Returns:
        Dictionary of evaluation metrics

    Example:
        >>> metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        >>> print(f"Test MAE: {metrics['mae']:.2f} years")
        >>> print(f"Overfitting ratio: {metrics['overfitting_ratio']:.2f}x")
    """
    logger.info("Evaluating model performance")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Test set metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    mad = float(np.median(np.abs(y_test - y_pred_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    # Correlation
    try:
        corr, _ = stats.pearsonr(y_test, y_pred_test)
    except Exception as e:
        logger.warning(f"Could not compute correlation: {e}")
        corr = np.nan

    # Training set metrics (for overfitting detection)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Cross-validation
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="neg_mean_absolute_error"
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        cv_mae = np.nan
        cv_std = np.nan

    # Overfitting ratio
    overfitting_ratio = mae / mae_train if mae_train > 0 else np.nan

    # Warn if severe overfitting detected
    if overfitting_ratio > 10:
        logger.warning(f"⚠️  Severe overfitting detected! Ratio: {overfitting_ratio:.1f}x")
        logger.warning("Consider: stronger regularization, fewer features, or ensemble methods")

    metrics = {
        "mae": mae,
        "mad": mad,
        "rmse": rmse,
        "r2": r2,
        "correlation": corr,
        "mae_train": mae_train,
        "r2_train": r2_train,
        "cv_mae": cv_mae,
        "cv_std": cv_std,
        "overfitting_ratio": overfitting_ratio,
    }

    logger.info(f"Test MAE: {mae:.3f}, R²: {r2:.4f}, Overfitting: {overfitting_ratio:.2f}x")

    return metrics


def compare_models(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compare multiple models and rank by performance.

    Args:
        results: DataFrame with model metrics

    Returns:
        DataFrame sorted by test MAE (ascending)

    Example:
        >>> comparison = compare_models(results_df)
        >>> print(comparison[['model', 'mae', 'r2', 'overfitting_ratio']])
    """
    if 'mae' not in results.columns:
        logger.error("Results DataFrame must have 'mae' column")
        return results

    # Sort by MAE (lower is better)
    sorted_results = results.sort_values('mae').reset_index(drop=True)

    # Add rank
    sorted_results.insert(0, 'rank', range(1, len(sorted_results) + 1))

    logger.info("Model comparison:")
    for idx, row in sorted_results.head(3).iterrows():
        logger.info(f"  #{row['rank']}: {row['model']} - MAE: {row['mae']:.3f}")

    return sorted_results


def calculate_prediction_intervals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals based on residual distribution.

    Args:
        y_pred: Predicted values
        residuals: Prediction residuals (y_true - y_pred)
        confidence: Confidence level (default: 95%)

    Returns:
        Tuple of (lower bounds, upper bounds)

    Example:
        >>> residuals = y_test - y_pred
        >>> lower, upper = calculate_prediction_intervals(y_pred, residuals)
        >>> print(f"95% PI width: {(upper - lower).mean():.2f} years")
    """
    # Calculate quantiles of residuals
    alpha = 1 - confidence
    lower_q = alpha / 2
    upper_q = 1 - (alpha / 2)

    lower_residual = np.quantile(residuals, lower_q)
    upper_residual = np.quantile(residuals, upper_q)

    lower_bound = y_pred + lower_residual
    upper_bound = y_pred + upper_residual

    logger.debug(f"{confidence*100}% prediction interval width: "
                 f"{upper_residual - lower_residual:.2f}")

    return lower_bound, upper_bound
