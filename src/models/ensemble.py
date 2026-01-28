"""
Ensemble methods for combining multiple models.
"""

from typing import List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class StackedEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacked ensemble of regression models.

    Uses cross-validated predictions from base learners as features
    for a meta-learner, reducing overfitting through model diversity.

    Attributes:
        base_learners: List of (name, model) tuples
        meta_learner: Model to combine base predictions
        cv: Number of cross-validation folds

    Example:
        >>> from sklearn.linear_model import Ridge, Lasso
        >>> base = [("ridge", Ridge()), ("lasso", Lasso())]
        >>> ensemble = StackedEnsemble(base, Ridge(alpha=100.0))
        >>> ensemble.fit(X_train, y_train)
        >>> y_pred = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_learners: List[Tuple[str, BaseEstimator]],
        meta_learner: BaseEstimator = None,
        cv: int = 5
    ):
        """
        Initialize stacked ensemble.

        Args:
            base_learners: List of (name, model) tuples for base models
            meta_learner: Model for combining predictions (default: Ridge(alpha=100))
            cv: Number of CV folds for generating meta-features
        """
        self.base_learners = base_learners
        self.meta_learner = meta_learner or Ridge(alpha=100.0)
        self.cv = cv

    def fit(self, X, y):
        """
        Fit the stacked ensemble.

        1. Generate cross-validated predictions from base learners
        2. Train meta-learner on base predictions
        3. Retrain base learners on full training set

        Args:
            X: Training features
            y: Training targets

        Returns:
            self
        """
        logger.info(f"Training stacked ensemble with {len(self.base_learners)} base learners")

        n_samples = X.shape[0]
        n_base = len(self.base_learners)

        # Generate meta-features via cross-validation
        meta_features = np.zeros((n_samples, n_base))

        for i, (name, model) in enumerate(self.base_learners):
            logger.debug(f"Generating CV predictions for {name}")

            # Cross-validated predictions to avoid overfitting
            cv_preds = cross_val_predict(model, X, y, cv=self.cv)
            meta_features[:, i] = cv_preds

            # Retrain on full data
            model.fit(X, y)

        # Train meta-learner
        logger.debug("Training meta-learner")
        self.meta_learner.fit(meta_features, y)

        logger.info("Stacked ensemble training complete")

        return self

    def predict(self, X):
        """
        Make predictions using the stacked ensemble.

        Args:
            X: Features to predict

        Returns:
            Array of predictions
        """
        n_base = len(self.base_learners)
        meta_features = np.zeros((X.shape[0], n_base))

        # Get predictions from base learners
        for i, (name, model) in enumerate(self.base_learners):
            meta_features[:, i] = model.predict(X)

        # Combine with meta-learner
        predictions = self.meta_learner.predict(meta_features)

        return predictions

    def get_base_predictions(self, X):
        """
        Get individual predictions from all base learners.

        Useful for analysis and debugging.

        Args:
            X: Features to predict

        Returns:
            Dictionary mapping model names to predictions
        """
        predictions = {}
        for name, model in self.base_learners:
            predictions[name] = model.predict(X)
        return predictions


def create_stacked_ensemble(
    base_models: List[Tuple[str, BaseEstimator]],
    alpha: float = 100.0
) -> StackedEnsemble:
    """
    Create a stacked ensemble with Ridge meta-learner.

    Args:
        base_models: List of (name, model) tuples
        alpha: Regularization strength for meta-learner

    Returns:
        StackedEnsemble instance

    Example:
        >>> from sklearn.linear_model import Ridge, Lasso, ElasticNet
        >>> base = [
        ...     ("ridge", Ridge(alpha=100)),
        ...     ("lasso", Lasso(alpha=0.1)),
        ...     ("elasticnet", ElasticNet(alpha=0.1))
        ... ]
        >>> ensemble = create_stacked_ensemble(base, alpha=100.0)
    """
    logger.info(f"Creating stacked ensemble with {len(base_models)} base models")
    logger.info(f"Meta-learner: Ridge(alpha={alpha})")

    meta_learner = Ridge(alpha=alpha)
    return StackedEnsemble(base_models, meta_learner, cv=5)
