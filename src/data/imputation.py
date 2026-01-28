"""
Missing value imputation strategies for DNA methylation data.
"""

from typing import Dict
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNet

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def get_available_imputers() -> Dict[str, object]:
    """
    Get dictionary of available imputation methods.

    Returns:
        Dictionary mapping method names to imputer instances

    Example:
        >>> imputers = get_available_imputers()
        >>> 'KNN (k=5)' in imputers
        True
    """
    return {
        "Mean": SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
        "Most Frequent": SimpleImputer(strategy="most_frequent"),
        "KNN (k=5)": KNNImputer(n_neighbors=5),
        "KNN (k=10)": KNNImputer(n_neighbors=10),
        "KNN (k=20)": KNNImputer(n_neighbors=20),
        "Iterative (BayesianRidge)": IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=42
        ),
        "Iterative (ElasticNet)": IterativeImputer(
            estimator=ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
            max_iter=10,
            random_state=42
        ),
    }


def impute_data(
    X: pd.DataFrame,
    method: str = "KNN (k=5)"
) -> pd.DataFrame:
    """
    Impute missing values using specified method.

    Args:
        X: DataFrame with missing values
        method: Imputation method name (see get_available_imputers())

    Returns:
        DataFrame with imputed values

    Raises:
        ValueError: If method not recognized

    Example:
        >>> X_imputed = impute_data(X_raw, method="KNN (k=5)")
        >>> X_imputed.isna().sum().sum()
        0
    """
    imputers = get_available_imputers()

    if method not in imputers:
        raise ValueError(
            f"Unknown imputation method: {method}. "
            f"Available: {list(imputers.keys())}"
        )

    logger.info(f"Imputing missing values using: {method}")

    missing_before = X.isna().sum().sum()
    missing_rate = missing_before / (X.shape[0] * X.shape[1])

    logger.info(f"Missing values before imputation: {missing_before} "
                f"({missing_rate:.2%})")

    imputer = imputers[method]

    # Fit and transform
    X_imputed_array = imputer.fit_transform(X)

    # Determine which columns were kept after imputation
    # SimpleImputer may drop columns that are all NaN
    n_features_out = X_imputed_array.shape[1]
    n_features_in = X.shape[1]

    if n_features_out < n_features_in:
        # Some columns were dropped - find which ones remain
        # Columns with all NaN are dropped by SimpleImputer
        kept_columns = [col for col in X.columns if not X[col].isna().all()]
        output_columns = kept_columns[:n_features_out]
    else:
        # All columns kept
        output_columns = X.columns

    # Convert back to DataFrame with original index
    X_imputed = pd.DataFrame(
        X_imputed_array,
        index=X.index,
        columns=output_columns
    )

    missing_after = X_imputed.isna().sum().sum()

    if missing_after > 0:
        logger.warning(f"Still have {missing_after} missing values after imputation")
    else:
        logger.info("All missing values successfully imputed")

    return X_imputed


def create_knn_imputer(n_neighbors: int = 5) -> KNNImputer:
    """
    Create a K-Nearest Neighbors imputer.

    This is the recommended default for DNA methylation data as it
    preserves local methylation patterns.

    Args:
        n_neighbors: Number of neighbors to use (default: 5)

    Returns:
        KNNImputer instance

    Example:
        >>> imputer = create_knn_imputer(n_neighbors=5)
        >>> X_imputed = imputer.fit_transform(X)
    """
    logger.info(f"Creating KNN imputer with k={n_neighbors}")
    return KNNImputer(n_neighbors=n_neighbors)
