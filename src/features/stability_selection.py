"""
Stability selection for robust feature selection.

Implements bootstrap-based stability selection to identify features
that consistently appear across multiple resamples.
"""

from typing import List, Tuple
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    threshold: float = 0.8,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stability selection to identify robust features.
    
    Uses bootstrap resampling with ElasticNet to select features that
    appear in a high proportion of models, reducing the curse of dimensionality.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        n_bootstrap: Number of bootstrap iterations
        sample_fraction: Fraction of samples per bootstrap
        threshold: Minimum selection frequency to keep feature (0-1)
        alpha: ElasticNet regularization strength
        l1_ratio: ElasticNet L1/L2 balance
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (selected_indices, selection_frequencies)
        
    Example:
        >>> selected_idx, frequencies = stability_selection(X, y, threshold=0.8)
        >>> X_selected = X[:, selected_idx]
    """
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state)
    
    logger.info(f"Starting stability selection with {n_bootstrap} iterations")
    logger.info(f"Features: {n_features}, Threshold: {threshold}")
    
    # Track how often each feature is selected
    selection_counts = np.zeros(n_features)
    
    scaler = StandardScaler()
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = rng.choice(
            n_samples,
            size=int(n_samples * sample_fraction),
            replace=True
        )
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Scale features
        X_scaled = scaler.fit_transform(X_boot)
        
        # Fit ElasticNet
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
            tol=1e-4,
            random_state=random_state + i,
        )
        model.fit(X_scaled, y_boot)
        
        # Track non-zero coefficients
        selected = np.abs(model.coef_) > 1e-10
        selection_counts += selected
        
        if (i + 1) % 20 == 0:
            logger.debug(f"Bootstrap iteration {i + 1}/{n_bootstrap}")
    
    # Compute selection frequencies
    selection_frequencies = selection_counts / n_bootstrap
    
    # Select features above threshold
    selected_mask = selection_frequencies >= threshold
    selected_indices = np.where(selected_mask)[0]
    
    logger.info(f"Selected {len(selected_indices)} features (threshold={threshold})")
    logger.info(f"Selection rate: {len(selected_indices)/n_features*100:.1f}%")
    
    return selected_indices, selection_frequencies


def stability_selection_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    threshold: float = 0.8,
    alpha: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-validated stability selection.
    
    Performs stability selection across CV folds to identify features
    that are consistently selected regardless of training data split.
    
    Args:
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        threshold: Minimum selection frequency
        alpha: ElasticNet alpha
        random_state: Random seed
        
    Returns:
        Tuple of (selected_indices, fold_selection_counts)
    """
    from sklearn.model_selection import KFold
    
    n_features = X.shape[1]
    fold_selections = np.zeros((cv_folds, n_features))
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    logger.info(f"Running {cv_folds}-fold stability selection")
    
    for fold, (train_idx, _) in enumerate(kf.split(X)):
        X_fold = X[train_idx]
        y_fold = y[train_idx]
        
        # Run stability selection on this fold
        selected_idx, _ = stability_selection(
            X_fold, y_fold,
            n_bootstrap=50,  # Fewer iterations per fold
            threshold=0.6,   # Lower threshold per fold
            alpha=alpha,
            random_state=random_state + fold,
        )
        
        # Mark selected features
        fold_selections[fold, selected_idx] = 1
        
        logger.debug(f"Fold {fold + 1}: selected {len(selected_idx)} features")
    
    # Count selections across folds
    fold_counts = fold_selections.sum(axis=0)
    
    # Select features appearing in enough folds
    min_folds = int(cv_folds * threshold)
    selected_mask = fold_counts >= min_folds
    selected_indices = np.where(selected_mask)[0]
    
    logger.info(f"Final selection: {len(selected_indices)} features "
                f"(appeared in >= {min_folds} folds)")
    
    return selected_indices, fold_counts


def get_stable_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_features: int = 1000,
    threshold: float = 0.7,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Get stable feature names after selection.
    
    Convenience function that returns feature names instead of indices.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        max_features: Maximum features to return
        threshold: Selection threshold
        random_state: Random seed
        
    Returns:
        Tuple of (selected_feature_names, selection_frequencies)
    """
    selected_idx, frequencies = stability_selection(
        X, y,
        threshold=threshold,
        random_state=random_state,
    )
    
    # Sort by frequency and take top features
    sorted_idx = selected_idx[np.argsort(frequencies[selected_idx])[::-1]]
    
    if len(sorted_idx) > max_features:
        sorted_idx = sorted_idx[:max_features]
        logger.info(f"Limiting to top {max_features} features")
    
    selected_names = [feature_names[i] for i in sorted_idx]
    selected_frequencies = frequencies[sorted_idx]
    
    return selected_names, selected_frequencies
