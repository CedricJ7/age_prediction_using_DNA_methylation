"""
Unit tests for feature selection module.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.selection import filter_cpgs_by_missing_rate


def test_filter_cpgs_by_missing_rate_removes_high_missing():
    """Test that CpG sites with high missing rate are removed."""
    # Create matrix with CpGs as rows (index) and samples as columns
    df = pd.DataFrame({
        'S1': [1.0, np.nan, 3.0, np.nan],
        'S2': [2.0, np.nan, 4.0, np.nan],
        'S3': [3.0, np.nan, 5.0, np.nan],
    }, index=['cpg0', 'cpg1', 'cpg2', 'cpg3'])

    # Row 0: 0% missing (all samples have values)
    # Row 1: 100% missing (all samples are NaN)
    # Row 2: 0% missing
    # Row 3: 100% missing

    filtered = filter_cpgs_by_missing_rate(df, max_missing_rate=0.5)

    # Should keep rows 0 and 2 (0% missing), remove rows 1 and 3 (100% missing)
    assert len(filtered) == 2
    assert 'cpg0' in filtered.index
    assert 'cpg2' in filtered.index


def test_filter_cpgs_by_missing_rate_zero_threshold():
    """Test that zero threshold returns original matrix."""
    df = pd.DataFrame({
        'S1': [1.0, np.nan, 3.0],
        'S2': [2.0, np.nan, 4.0],
    }).T

    filtered = filter_cpgs_by_missing_rate(df, max_missing_rate=0.0)

    assert len(filtered) == len(df)
    pd.testing.assert_frame_equal(filtered, df)


def test_filter_cpgs_by_missing_rate_all_pass():
    """Test when all CpG sites pass threshold."""
    df = pd.DataFrame({
        'S1': [1.0, 2.0, 3.0],
        'S2': [4.0, 5.0, 6.0],
        'S3': [7.0, 8.0, 9.0],
    }).T

    filtered = filter_cpgs_by_missing_rate(df, max_missing_rate=0.1)

    assert len(filtered) == len(df)
    pd.testing.assert_frame_equal(filtered, df)


def test_filter_cpgs_by_missing_rate_partial_missing():
    """Test with partial missing values."""
    df = pd.DataFrame({
        'S1': [1.0, 2.0, 3.0, 4.0],
        'S2': [np.nan, 5.0, 6.0, 7.0],
        'S3': [8.0, 9.0, 10.0, 11.0],
    }, index=['cpg0', 'cpg1', 'cpg2', 'cpg3'])

    # cpg0: 1/3 missing = 33% missing (should be removed with 20% threshold)
    # cpg1, cpg2, cpg3: 0% missing

    # With 20% threshold, should remove cpg0 (33% missing)
    filtered = filter_cpgs_by_missing_rate(df, max_missing_rate=0.2)

    assert len(filtered) == 3  # cpg1, cpg2, cpg3 kept
    assert 'cpg0' not in filtered.index
    assert 'cpg1' in filtered.index


def test_filter_cpgs_by_missing_rate_negative_threshold():
    """Test with negative threshold (should behave like zero)."""
    df = pd.DataFrame({
        'S1': [1.0, 2.0],
        'S2': [3.0, 4.0],
    }).T

    filtered = filter_cpgs_by_missing_rate(df, max_missing_rate=-0.1)

    assert len(filtered) == len(df)
