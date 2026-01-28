"""
Machine learning models for age prediction.
"""

from .linear_models import (
    create_ridge_model,
    create_lasso_model,
    create_elasticnet_model,
)

from .tree_models import (
    create_random_forest_model,
    create_xgboost_model,
)

from .neural_models import create_mlp_model

from .ensemble import (
    StackedEnsemble,
    create_stacked_ensemble,
)

__all__ = [
    # Linear models
    "create_ridge_model",
    "create_lasso_model",
    "create_elasticnet_model",
    # Tree models
    "create_random_forest_model",
    "create_xgboost_model",
    # Neural models
    "create_mlp_model",
    # Ensemble
    "StackedEnsemble",
    "create_stacked_ensemble",
]
