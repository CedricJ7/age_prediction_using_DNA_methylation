"""
Feature selection functions for DNA methylation data.

This module provides efficient methods for selecting CpG sites based on
correlation with age and dimensionality reduction via PCA.
"""

from pathlib import Path
import heapq
import numpy as np
import pandas as pd

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def select_top_k_cpgs(
    data_path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    cpg_names: list[str],
    top_k: int,
    chunk_size: int = 2000,
) -> tuple[list[int], list[str]]:
    """
    Select top-k CpG sites most correlated with age.

    Uses Pearson correlation to identify CpG sites whose methylation levels
    are most predictive of chronological age. Processes data in chunks to
    handle large files efficiently.

    Args:
        data_path: Path to methylation CSV file
        sample_ids: List of sample IDs to include
        y: Array of target ages
        cpg_names: List of all CpG site names
        top_k: Number of top CpG sites to select
        chunk_size: Number of CpG sites to process at once

    Returns:
        Tuple of (selected indices, selected names)

    Example:
        >>> indices, names = select_top_k_cpgs(
        ...     Path("Data/c_sample.csv"),
        ...     sample_ids=["S1", "S2"],
        ...     y=np.array([25, 30]),
        ...     cpg_names=all_cpg_names,
        ...     top_k=1000
        ... )
        >>> len(indices) == 1000
        True
    """
    logger.info(f"Selecting top {top_k} CpG sites by correlation with age")

    # Center target variable for correlation calculation
    y_centered = y - y.mean()
    y_den = np.sqrt(np.sum(y_centered**2))

    # Use min-heap to track top-k correlations efficiently
    best: list[tuple[float, int]] = []

    start = 0
    for chunk_idx, chunk in enumerate(
        pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size)
    ):
        # Convert to numpy for efficient computation
        x = chunk.to_numpy(dtype=np.float32, copy=False)

        # Simple imputation: replace NaN with row means
        if np.isnan(x).any():
            row_means = np.zeros((x.shape[0], 1), dtype=x.dtype)
            valid_rows = ~np.isnan(x).all(axis=1)
            if np.any(valid_rows):
                row_means[valid_rows] = np.nanmean(x[valid_rows], axis=1, keepdims=True)
            x = np.where(np.isnan(x), row_means, x)

        # Center features
        x_centered = x - x.mean(axis=1, keepdims=True)

        # Compute Pearson correlation
        num = x_centered @ y_centered
        den = np.sqrt(np.sum(x_centered**2, axis=1)) * y_den
        corr = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        abs_corr = np.abs(corr)

        # Update heap with best correlations
        for i, c in enumerate(abs_corr):
            idx = start + i
            if len(best) < top_k:
                heapq.heappush(best, (c, idx))
            elif c > best[0][0]:
                heapq.heapreplace(best, (c, idx))

        start += len(chunk)

        if (chunk_idx + 1) % 100 == 0:
            logger.debug(f"Processed {start} CpG sites")

    # Sort by correlation (descending)
    best_sorted = sorted(best, key=lambda t: t[0], reverse=True)
    indices = [idx for _, idx in best_sorted]
    names = [
        cpg_names[i] if i < len(cpg_names) else f"cpg_{i}"
        for i in indices
    ]

    logger.info(f"Selected {len(indices)} CpG sites with correlation range: "
                f"{best_sorted[-1][0]:.4f} to {best_sorted[0][0]:.4f}")

    return indices, names


def compute_pca_scores_streaming(
    data_path: Path,
    sample_ids: list[str],
    n_components: int = 400,
    chunk_size: int = 2000,
    max_missing_rate: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA scores via streaming Gram matrix calculation.

    Implements memory-efficient PCA for high-dimensional data by computing
    the Gram matrix X^T @ X incrementally, then eigendecomposing.

    Args:
        data_path: Path to methylation CSV file
        sample_ids: List of sample IDs to include
        n_components: Number of principal components to compute
        chunk_size: Number of CpG sites to process at once
        max_missing_rate: Maximum missing rate threshold for CpG filtering

    Returns:
        Tuple of (PC scores, explained variance ratios)

    Example:
        >>> scores, explained = compute_pca_scores_streaming(
        ...     Path("Data/c_sample.csv"),
        ...     sample_ids=["S1", "S2"],
        ...     n_components=50
        ... )
        >>> scores.shape[1] == 50
        True
    """
    logger.info(f"Computing PCA with {n_components} components")

    n_samples = len(sample_ids)
    gram = np.zeros((n_samples, n_samples), dtype=np.float64)

    cpgs_processed = 0
    cpgs_kept = 0

    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        x = chunk.to_numpy(dtype=np.float32, copy=False)
        cpgs_processed += len(x)

        # Filter by missing rate
        if max_missing_rate > 0:
            missing_rate = np.isnan(x).mean(axis=1)
            keep = missing_rate <= max_missing_rate
            x = x[keep]
            cpgs_kept += len(x)

        if x.size == 0:
            continue

        # Simple imputation
        if np.isnan(x).any():
            row_means = np.zeros((x.shape[0], 1), dtype=x.dtype)
            valid_rows = ~np.isnan(x).all(axis=1)
            if np.any(valid_rows):
                row_means[valid_rows] = np.nanmean(x[valid_rows], axis=1, keepdims=True)
            x = np.where(np.isnan(x), row_means, x)

        # Center and accumulate Gram matrix
        x = x - x.mean(axis=1, keepdims=True)
        gram += x.T @ x

    logger.info(f"Kept {cpgs_kept}/{cpgs_processed} CpG sites "
                f"(missing rate <= {max_missing_rate})")

    # Eigendecomposition
    logger.debug("Computing eigendecomposition")
    eigvals, eigvecs = np.linalg.eigh(gram)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top components
    k = min(n_components, n_samples)
    eigvals_k = np.maximum(eigvals[:k], 0.0)

    # Compute scores
    scores = eigvecs[:, :k] * np.sqrt(eigvals_k)

    # Explained variance
    explained = eigvals_k / eigvals.sum() if eigvals.sum() > 0 else np.zeros(k)

    logger.info(f"Cumulative explained variance (first 10 PCs): "
                f"{explained[:10].sum():.4f}")

    return scores, explained


def filter_cpgs_by_missing_rate(
    matrix: pd.DataFrame,
    max_missing_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Filter out CpG sites with too many missing values.

    Args:
        matrix: DataFrame with CpG sites as rows and samples as columns
        max_missing_rate: Maximum proportion of missing values allowed

    Returns:
        Filtered DataFrame

    Example:
        >>> df = pd.DataFrame([[1, 2], [np.nan, np.nan], [3, 4]])
        >>> filtered = filter_cpgs_by_missing_rate(df, max_missing_rate=0.3)
        >>> len(filtered)
        2
    """
    if max_missing_rate <= 0:
        return matrix

    logger.info(f"Filtering CpGs with missing rate > {max_missing_rate}")

    missing_rate = matrix.isna().mean(axis=1)
    kept = missing_rate <= max_missing_rate

    initial_count = len(matrix)
    matrix_filtered = matrix[kept].copy()
    filtered_count = initial_count - len(matrix_filtered)

    logger.info(f"Kept {len(matrix_filtered)}/{initial_count} CpG sites "
                f"(removed {filtered_count})")

    return matrix_filtered
