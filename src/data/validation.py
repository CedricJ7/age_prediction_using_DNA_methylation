"""
Data validation utilities for DNA methylation age prediction.

Provides schema validation, data quality checks, and integrity verification.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_annotations(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    age_range: Tuple[float, float] = (0, 120),
) -> pd.DataFrame:
    """
    Validate annotation DataFrame.
    
    Args:
        df: Annotations DataFrame
        required_columns: List of required column names
        age_range: Valid age range (min, max)
        
    Returns:
        Validated DataFrame (may have rows removed)
        
    Raises:
        DataValidationError: If validation fails critically
    """
    if required_columns is None:
        required_columns = ["age", "Sample_description"]
    
    logger.info(f"Validating annotations: {len(df)} rows")
    
    # Check required columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    
    # Check for empty DataFrame
    if len(df) == 0:
        raise DataValidationError("Empty DataFrame")
    
    initial_count = len(df)
    
    # Remove rows with missing age
    if "age" in df.columns:
        df = df.dropna(subset=["age"])
        logger.debug(f"Removed {initial_count - len(df)} rows with missing age")
    
    # Validate age range
    if "age" in df.columns:
        min_age, max_age = age_range
        invalid_age = (df["age"] < min_age) | (df["age"] > max_age)
        if invalid_age.any():
            n_invalid = invalid_age.sum()
            logger.warning(f"Found {n_invalid} samples with age outside [{min_age}, {max_age}]")
            df = df[~invalid_age]
    
    # Check for duplicate sample IDs
    if "Sample_description" in df.columns:
        duplicates = df["Sample_description"].duplicated()
        if duplicates.any():
            n_dup = duplicates.sum()
            logger.warning(f"Found {n_dup} duplicate Sample_description values")
            df = df[~duplicates]
    
    final_count = len(df)
    if final_count < initial_count:
        logger.info(f"Validation removed {initial_count - final_count} rows")
    
    logger.info(f"Validation complete: {final_count} valid rows")
    
    return df


def validate_methylation_matrix(
    df: pd.DataFrame,
    expected_samples: List[str] = None,
    beta_range: Tuple[float, float] = (0.0, 1.0),
    max_missing_rate: float = 0.2,
) -> pd.DataFrame:
    """
    Validate methylation beta value matrix.
    
    Args:
        df: Methylation matrix (CpGs x samples)
        expected_samples: List of expected sample IDs
        beta_range: Valid beta value range
        max_missing_rate: Maximum allowed missing rate per CpG
        
    Returns:
        Validated matrix
        
    Raises:
        DataValidationError: If critical validation fails
    """
    logger.info(f"Validating methylation matrix: {df.shape}")
    
    # Check for expected samples
    if expected_samples is not None:
        missing_samples = set(expected_samples) - set(df.columns)
        if missing_samples:
            logger.warning(f"Missing {len(missing_samples)} expected samples")
    
    # Check beta value range
    min_beta, max_beta = beta_range
    out_of_range = (df < min_beta) | (df > max_beta)
    out_of_range_count = out_of_range.sum().sum()
    
    if out_of_range_count > 0:
        logger.warning(f"Found {out_of_range_count} values outside [{min_beta}, {max_beta}]")
        # Clip values to valid range
        df = df.clip(min_beta, max_beta)
    
    # Check missing rate per CpG
    missing_rates = df.isna().mean(axis=1)
    high_missing = missing_rates > max_missing_rate
    
    if high_missing.any():
        n_remove = high_missing.sum()
        logger.info(f"Removing {n_remove} CpGs with >{max_missing_rate*100}% missing")
        df = df[~high_missing]
    
    logger.info(f"Validation complete: {df.shape}")
    
    return df


def validate_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None,
) -> bool:
    """
    Validate feature matrix and target vector.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: Optional feature names
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    # Check shapes match
    if X.shape[0] != len(y):
        raise DataValidationError(
            f"X rows ({X.shape[0]}) != y length ({len(y)})"
        )
    
    # Check for NaN in features
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        raise DataValidationError(
            f"Feature matrix contains {nan_count} NaN values"
        )
    
    # Check for NaN in target
    if np.isnan(y).any():
        raise DataValidationError("Target vector contains NaN values")
    
    # Check for infinite values
    if np.isinf(X).any():
        raise DataValidationError("Feature matrix contains infinite values")
    
    # Check feature names match
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise DataValidationError(
            f"Feature names ({len(feature_names)}) != features ({X.shape[1]})"
        )
    
    logger.info(f"Feature matrix validated: {X.shape}")
    
    return True


def check_data_quality(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Compute data quality metrics.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "missing_rate": np.isnan(X).mean() if np.isnan(X).any() else 0.0,
        "zero_variance_features": (X.std(axis=0) == 0).sum(),
        "target_mean": y.mean(),
        "target_std": y.std(),
        "target_min": y.min(),
        "target_max": y.max(),
        "samples_per_feature_ratio": X.shape[0] / X.shape[1],
    }
    
    # Check for high collinearity (sample of features)
    if X.shape[1] > 100:
        sample_idx = np.random.choice(X.shape[1], 100, replace=False)
        corr_matrix = np.corrcoef(X[:, sample_idx].T)
        high_corr = np.abs(corr_matrix) > 0.95
        np.fill_diagonal(high_corr, False)
        metrics["high_correlation_pairs"] = high_corr.sum() // 2
    
    logger.info(f"Data quality check complete")
    for key, value in metrics.items():
        logger.debug(f"  {key}: {value}")
    
    return metrics


def validate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_error: float = 50.0,
) -> Tuple[bool, List[str]]:
    """
    Validate model predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        max_error: Maximum allowed absolute error
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check lengths match
    if len(y_true) != len(y_pred):
        return False, ["Length mismatch between true and predicted"]
    
    # Check for NaN predictions
    if np.isnan(y_pred).any():
        warnings.append(f"Found {np.isnan(y_pred).sum()} NaN predictions")
    
    # Check for extreme errors
    errors = np.abs(y_true - y_pred)
    extreme_errors = errors > max_error
    if extreme_errors.any():
        warnings.append(
            f"Found {extreme_errors.sum()} predictions with error > {max_error}"
        )
    
    # Check for negative predictions (age should be positive)
    if (y_pred < 0).any():
        warnings.append(f"Found {(y_pred < 0).sum()} negative predictions")
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings
