"""
Feature engineering for DNA methylation age prediction.

Provides polynomial features, interaction terms, and other transformations.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_polynomial_features(
    X: np.ndarray,
    feature_names: List[str],
    top_k: int = 100,
    degree: int = 2,
    interaction_only: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Create polynomial and interaction features for top CpG sites.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        top_k: Number of top features to create polynomials for
        degree: Polynomial degree
        interaction_only: If True, only interaction terms (no x^2)
        
    Returns:
        Tuple of (transformed_matrix, new_feature_names)
        
    Example:
        >>> X_poly, names = create_polynomial_features(X, cpg_names, top_k=50)
    """
    n_samples, n_features = X.shape
    
    if top_k > n_features:
        top_k = n_features
        logger.warning(f"top_k reduced to {top_k} (total features)")
    
    logger.info(f"Creating polynomial features for top {top_k} features")
    
    # Select top k features (assuming they're already sorted by importance)
    X_top = X[:, :top_k]
    top_names = feature_names[:top_k]
    
    # Create polynomial features
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False,
    )
    
    X_poly = poly.fit_transform(X_top)
    poly_names = poly.get_feature_names_out(top_names)
    
    # Combine with remaining features
    X_remaining = X[:, top_k:]
    remaining_names = feature_names[top_k:]
    
    X_combined = np.hstack([X_poly, X_remaining])
    combined_names = list(poly_names) + list(remaining_names)
    
    logger.info(f"Feature matrix: {n_features} -> {X_combined.shape[1]} features")
    logger.info(f"Added {len(poly_names) - top_k} polynomial/interaction terms")
    
    return X_combined, combined_names


def create_demographic_interactions(
    X_cpg: np.ndarray,
    X_demo: np.ndarray,
    cpg_names: List[str],
    demo_names: List[str],
    top_k: int = 50,
) -> Tuple[np.ndarray, List[str]]:
    """
    Create interaction terms between top CpG sites and demographic features.
    
    Args:
        X_cpg: CpG methylation matrix
        X_demo: Demographic features matrix
        cpg_names: CpG feature names
        demo_names: Demographic feature names
        top_k: Number of top CpG sites to interact
        
    Returns:
        Tuple of (interaction_matrix, interaction_names)
    """
    n_samples = X_cpg.shape[0]
    n_demo = X_demo.shape[1]
    
    logger.info(f"Creating CpG x demographic interactions for top {top_k} CpGs")
    
    # Select top CpGs
    X_top_cpg = X_cpg[:, :top_k]
    top_cpg_names = cpg_names[:top_k]
    
    # Create interaction terms
    interactions = []
    interaction_names = []
    
    for i, demo_name in enumerate(demo_names):
        for j, cpg_name in enumerate(top_cpg_names):
            interaction = X_demo[:, i] * X_top_cpg[:, j]
            interactions.append(interaction)
            interaction_names.append(f"{cpg_name}x{demo_name}")
    
    X_interactions = np.column_stack(interactions)
    
    logger.info(f"Created {len(interaction_names)} interaction features")
    
    return X_interactions, interaction_names


def create_age_group_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    age_bins: List[float] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Create features that capture age-specific patterns.
    
    For each age group, identifies top features and creates
    group-specific indicators.
    
    Args:
        X: Feature matrix
        y: Age target vector
        feature_names: Feature names
        age_bins: Age group boundaries (default: [0, 30, 50, 70, 120])
        
    Returns:
        Tuple of (augmented_matrix, augmented_names)
    """
    if age_bins is None:
        age_bins = [0, 30, 50, 70, 120]
    
    logger.info(f"Creating age-group features with bins: {age_bins}")
    
    # Create age group indicators
    n_samples = X.shape[0]
    n_groups = len(age_bins) - 1
    
    group_indicators = np.zeros((n_samples, n_groups))
    group_names = []
    
    for i in range(n_groups):
        mask = (y >= age_bins[i]) & (y < age_bins[i + 1])
        group_indicators[:, i] = mask.astype(float)
        group_names.append(f"age_group_{age_bins[i]}_{age_bins[i+1]}")
    
    # Combine with original features
    X_augmented = np.hstack([X, group_indicators])
    augmented_names = list(feature_names) + group_names
    
    logger.info(f"Added {n_groups} age group indicator features")
    
    return X_augmented, augmented_names


def compute_cpg_region_scores(
    X: np.ndarray,
    cpg_names: List[str],
    region_mapping: dict = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate CpG methylation by genomic regions.
    
    Args:
        X: CpG methylation matrix
        cpg_names: CpG site names
        region_mapping: Dict mapping region names to CpG indices
                       (if None, groups by prefix)
        
    Returns:
        Tuple of (region_scores, region_names)
    """
    if region_mapping is None:
        # Simple grouping by CpG prefix (first 5 chars)
        # This is a placeholder - real implementation would use
        # genomic annotations
        logger.warning("Using simple prefix-based grouping (placeholder)")
        
        prefixes = {}
        for i, name in enumerate(cpg_names):
            prefix = name[:5] if len(name) >= 5 else name
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(i)
        
        region_mapping = prefixes
    
    logger.info(f"Computing scores for {len(region_mapping)} regions")
    
    region_scores = []
    region_names = []
    
    for region, indices in region_mapping.items():
        if len(indices) > 0:
            # Mean methylation across CpGs in region
            score = X[:, indices].mean(axis=1)
            region_scores.append(score)
            region_names.append(f"region_{region}_mean")
    
    X_regions = np.column_stack(region_scores)
    
    logger.info(f"Created {len(region_names)} region features")
    
    return X_regions, region_names


def select_features_by_variance(
    X: np.ndarray,
    feature_names: List[str],
    min_variance: float = 0.01,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Remove low-variance features.
    
    Args:
        X: Feature matrix
        feature_names: Feature names
        min_variance: Minimum variance threshold
        
    Returns:
        Tuple of (filtered_X, filtered_names, mask)
    """
    variances = X.var(axis=0)
    mask = variances >= min_variance
    
    X_filtered = X[:, mask]
    filtered_names = [name for name, m in zip(feature_names, mask) if m]
    
    n_removed = (~mask).sum()
    logger.info(f"Removed {n_removed} features with variance < {min_variance}")
    
    return X_filtered, filtered_names, mask


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    method: str = "standard",
) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Normalize features using specified method.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (normalized_train, normalized_test, scaler)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_train_norm = scaler.fit_transform(X_train)
    
    X_test_norm = None
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
    
    logger.info(f"Applied {method} normalization")
    
    return X_train_norm, X_test_norm, scaler
