"""
Unit tests for model creation modules.
"""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import ModelConfig
from src.models.linear_models import (
    create_ridge_model,
    create_lasso_model,
    create_elasticnet_model
)
from src.models.tree_models import create_random_forest_model, create_xgboost_model
from src.models.neural_models import create_mlp_model


def test_create_ridge_model():
    """Test Ridge model creation."""
    config = ModelConfig(ridge_alpha=100.0)
    model = create_ridge_model(config)

    assert isinstance(model, Pipeline)
    assert 'scaler' in model.named_steps
    assert 'model' in model.named_steps
    assert isinstance(model.named_steps['model'], Ridge)
    assert model.named_steps['model'].alpha == 100.0


def test_create_lasso_model():
    """Test Lasso model creation."""
    config = ModelConfig(lasso_alpha=0.5)
    model = create_lasso_model(config)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps['model'], Lasso)
    assert model.named_steps['model'].alpha == 0.5


def test_create_elasticnet_model():
    """Test ElasticNet model creation."""
    config = ModelConfig(elasticnet_alpha=0.2, elasticnet_l1_ratio=0.7)
    model = create_elasticnet_model(config)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps['model'], ElasticNet)
    assert model.named_steps['model'].alpha == 0.2
    assert model.named_steps['model'].l1_ratio == 0.7


def test_create_random_forest_model():
    """Test Random Forest model creation."""
    config = ModelConfig(rf_n_estimators=100, rf_max_depth=10)
    model = create_random_forest_model(config)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 100
    assert model.max_depth == 10


def test_create_xgboost_model():
    """Test XGBoost model creation."""
    pytest.importorskip("xgboost")  # Skip if xgboost not installed

    config = ModelConfig(
        xgboost_n_estimators=200,
        xgboost_learning_rate=0.1,
        xgboost_reg_alpha=2.0,
        xgboost_reg_lambda=5.0
    )
    model = create_xgboost_model(config)

    assert model.n_estimators == 200
    assert model.learning_rate == 0.1
    assert model.reg_alpha == 2.0
    assert model.reg_lambda == 5.0


def test_create_mlp_model():
    """Test MLP model creation."""
    config = ModelConfig(mlp_hidden_layers=(64, 32), mlp_alpha=0.01)
    model = create_mlp_model(config)

    assert isinstance(model, Pipeline)
    assert model.named_steps['model'].hidden_layer_sizes == (64, 32)
    assert model.named_steps['model'].alpha == 0.01


def test_models_can_fit_predict(sample_features_and_targets):
    """Test that created models can fit and predict."""
    X, y = sample_features_and_targets
    config = ModelConfig()

    models = [
        create_ridge_model(config),
        create_lasso_model(config),
        create_elasticnet_model(config),
    ]

    for model in models:
        # Fit
        model.fit(X, y)

        # Predict
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()
        assert len(predictions) == len(y)
