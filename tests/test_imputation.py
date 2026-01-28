"""
Unit tests for data imputation module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.data.imputation import (
    get_available_imputers,
    impute_data,
    create_knn_imputer
)


def test_get_available_imputers():
    """Test that available imputers are returned."""
    imputers = get_available_imputers()

    assert isinstance(imputers, dict)
    assert len(imputers) > 0
    assert "KNN (k=5)" in imputers
    assert "Mean" in imputers
    assert "Median" in imputers


def test_impute_data_removes_missing_values():
    """Test that imputation removes all missing values."""
    # Create data with missing values
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [np.nan, 2.0, 3.0, 4.0, 5.0],
        'feature3': [1.0, 2.0, 3.0, np.nan, 5.0],
    })

    X_imputed = impute_data(X, method="Mean")

    # Check all missing values are gone
    assert X_imputed.isna().sum().sum() == 0
    assert X_imputed.shape == X.shape


def test_impute_data_preserves_non_missing():
    """Test that non-missing values are preserved."""
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0],
        'feature2': [5.0, 6.0, 7.0, 8.0],  # No missing
    })

    X_imputed = impute_data(X, method="Mean")

    # Feature2 should be unchanged
    pd.testing.assert_series_equal(
        X_imputed['feature2'],
        X['feature2'],
        check_names=False
    )


def test_impute_data_knn():
    """Test KNN imputation specifically."""
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [2.0, 4.0, 6.0, 8.0, 10.0],
    })

    X_imputed = impute_data(X, method="KNN (k=5)")

    assert X_imputed.isna().sum().sum() == 0
    # KNN should impute based on neighbors
    assert X_imputed.loc[2, 'feature1'] > 0


def test_impute_data_median():
    """Test median imputation."""
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 100.0],  # Outlier affects mean
    })

    X_imputed = impute_data(X, method="Median")

    # Median of [1, 2, 4, 100] = 3.0
    assert X_imputed.loc[2, 'feature1'] == pytest.approx(3.0, abs=0.1)


def test_impute_data_unknown_method():
    """Test error handling for unknown method."""
    X = pd.DataFrame({'feature1': [1.0, np.nan, 3.0]})

    with pytest.raises(ValueError, match="Unknown imputation method"):
        impute_data(X, method="NonexistentMethod")


def test_impute_data_preserves_index_columns():
    """Test that DataFrame structure is preserved."""
    index = ['sample1', 'sample2', 'sample3']
    columns = ['cpg1', 'cpg2', 'cpg3']

    X = pd.DataFrame(
        [[1.0, np.nan, 3.0],
         [4.0, 5.0, np.nan],
         [7.0, 8.0, 9.0]],
        index=index,
        columns=columns
    )

    X_imputed = impute_data(X, method="Mean")

    assert list(X_imputed.index) == index
    assert list(X_imputed.columns) == columns


def test_create_knn_imputer():
    """Test KNN imputer creation."""
    imputer = create_knn_imputer(n_neighbors=3)

    assert isinstance(imputer, KNNImputer)
    assert imputer.n_neighbors == 3


def test_create_knn_imputer_default():
    """Test KNN imputer with default parameters."""
    imputer = create_knn_imputer()

    assert isinstance(imputer, KNNImputer)
    assert imputer.n_neighbors == 5


def test_impute_data_all_missing_feature():
    """Test behavior when entire feature is missing."""
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [np.nan, np.nan, np.nan],  # All missing
    })

    X_imputed = impute_data(X, method="Mean")

    # Mean imputer removes columns with all missing values
    # Check that at least feature1 is preserved
    assert 'feature1' in X_imputed.columns
    assert not X_imputed['feature1'].isna().any()
