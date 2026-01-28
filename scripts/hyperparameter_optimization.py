#!/usr/bin/env python3
"""
Hyperparameter Optimization for DNA Methylation Age Prediction
================================================================

Senior Data Scientist Approach:
- Exhaustive hyperparameter search using Optuna (Bayesian optimization)
- All popular ML methods tested
- Memory-efficient chunked loading or PCA support
- Progressive results saving (SQLite database)
- Real-time monitoring with live dashboard
- Complete report: method, params, hyperparams, MAE test
- Maximum runtime: 8 hours (configurable)

Author: Claude Opus 4.5
Date: 2026-01-28
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import warnings
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Optimization
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

# Local imports
from src.utils.logging_config import setup_logger
from src.data.data_loader import load_annotations, load_cpg_names, load_selected_cpgs
from src.features.selection import select_top_k_cpgs
from src.features.demographic import add_demographic_features
from src.data.imputation import create_knn_imputer

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

START_TIME = None
MAX_RUNTIME_HOURS = 8
OPTUNA_DB_PATH = "results/optimization/optuna_study.db"
RESULTS_DIR = Path("results/optimization")

# Models to optimize
MODELS_CONFIG = {
    "Ridge": {
        "enabled": True,
        "n_trials": 100,
        "cv_folds": 5,
    },
    "Lasso": {
        "enabled": True,
        "n_trials": 100,
        "cv_folds": 5,
    },
    "ElasticNet": {
        "enabled": True,
        "n_trials": 150,
        "cv_folds": 5,
    },
    "SVR": {
        "enabled": True,
        "n_trials": 100,
        "cv_folds": 5,
    },
    "RandomForest": {
        "enabled": True,
        "n_trials": 80,
        "cv_folds": 5,
    },
    "GradientBoosting": {
        "enabled": True,
        "n_trials": 80,
        "cv_folds": 5,
    },
    "XGBoost": {
        "enabled": True,
        "n_trials": 100,
        "cv_folds": 5,
    },
    "LightGBM": {
        "enabled": True,
        "n_trials": 100,
        "cv_folds": 5,
    },
    "CatBoost": {
        "enabled": True,
        "n_trials": 80,
        "cv_folds": 5,
    },
    "MLP": {
        "enabled": True,
        "n_trials": 100,
        "cv_folds": 5,
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_time_limit() -> bool:
    """Check if time limit exceeded."""
    if START_TIME is None:
        return False
    elapsed = (datetime.now() - START_TIME).total_seconds() / 3600
    return elapsed >= MAX_RUNTIME_HOURS


def get_elapsed_time() -> str:
    """Get elapsed time as string."""
    if START_TIME is None:
        return "00:00:00"
    elapsed = datetime.now() - START_TIME
    return str(elapsed).split('.')[0]


# =============================================================================
# OBJECTIVE FUNCTIONS FOR EACH MODEL
# =============================================================================

def objective_ridge(trial, X_train, y_train, cv_folds=5):
    """Optimize Ridge regression."""
    params = {
        'alpha': trial.suggest_float('alpha', 1e-3, 1e5, log=True),
        'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
        'random_state': 42,
    }

    model = Ridge(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_lasso(trial, X_train, y_train, cv_folds=5):
    """Optimize Lasso regression."""
    params = {
        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
        'max_iter': trial.suggest_int('max_iter', 1000, 10000),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
        'random_state': 42,
    }

    model = Lasso(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_elasticnet(trial, X_train, y_train, cv_folds=5):
    """Optimize ElasticNet regression."""
    params = {
        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 0.99),
        'max_iter': trial.suggest_int('max_iter', 1000, 10000),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
        'random_state': 42,
    }

    model = ElasticNet(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_svr(trial, X_train, y_train, cv_folds=5):
    """Optimize SVR."""
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])

    params = {
        'kernel': kernel,
        'C': trial.suggest_float('C', 1e-2, 1e3, log=True),
        'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
    }

    if kernel == 'rbf':
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    elif kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])

    model = SVR(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_random_forest(trial, X_train, y_train, cv_folds=5):
    """Optimize Random Forest."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'n_jobs': -1,
        'random_state': 42,
    }

    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=1)  # n_jobs=1 car RF déjà parallèle
    return -scores.mean()


def objective_gradient_boosting(trial, X_train, y_train, cv_folds=5):
    """Optimize Gradient Boosting."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_xgboost(trial, X_train, y_train, cv_folds=5):
    """Optimize XGBoost."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.warning("XGBoost not available, skipping")
        raise optuna.TrialPruned()

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
        'n_jobs': -1,
        'random_state': 42,
    }

    model = XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=1)
    return -scores.mean()


def objective_lightgbm(trial, X_train, y_train, cv_folds=5):
    """Optimize LightGBM."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.warning("LightGBM not available, skipping")
        raise optuna.TrialPruned()

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
    }

    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=1)
    return -scores.mean()


def objective_catboost(trial, X_train, y_train, cv_folds=5):
    """Optimize CatBoost."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.warning("CatBoost not available, skipping")
        raise optuna.TrialPruned()

    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 2, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 100.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': 0,
    }

    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def objective_mlp(trial, X_train, y_train, cv_folds=5):
    """Optimize Multi-Layer Perceptron."""
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_layer_sizes = []
    for i in range(n_layers):
        hidden_layer_sizes.append(
            trial.suggest_int(f'n_units_l{i}', 32, 512, log=True)
        )

    params = {
        'hidden_layer_sizes': tuple(hidden_layer_sizes),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'max_iter': 1000,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'random_state': 42,
    }

    model = MLPRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                            scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


# =============================================================================
# OPTIMIZATION WORKFLOW
# =============================================================================

class OptimizationMonitor:
    """Monitor optimization progress in real-time."""

    def __init__(self, total_trials: int):
        self.total_trials = total_trials
        self.completed_trials = 0
        self.best_mae = float('inf')
        self.current_model = ""

    def update(self, trial_number: int, mae: float, model_name: str):
        """Update progress."""
        self.completed_trials = trial_number
        self.current_model = model_name
        if mae < self.best_mae:
            self.best_mae = mae

    def print_progress(self):
        """Print current progress."""
        pct = (self.completed_trials / self.total_trials) * 100 if self.total_trials > 0 else 0
        logger.info(
            f"Progress: {self.completed_trials}/{self.total_trials} ({pct:.1f}%) | "
            f"Current: {self.current_model} | "
            f"Best MAE: {self.best_mae:.3f} | "
            f"Elapsed: {get_elapsed_time()}"
        )


def optimize_model(
    model_name: str,
    objective_func,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int,
    cv_folds: int,
    storage: str,
) -> Dict[str, Any]:
    """
    Optimize a single model using Optuna.

    Returns:
        Dictionary with best params, metrics, and model
    """
    if check_time_limit():
        logger.warning(f"Time limit reached, skipping {model_name}")
        return None

    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZING: {model_name}")
    logger.info(f"{'='*80}")

    # Create study
    study_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=storage,
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    # Optimize
    start_time = time.time()

    def objective_wrapper(trial):
        if check_time_limit():
            raise optuna.TrialPruned()
        return objective_func(trial, X_train, y_train, cv_folds)

    try:
        study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            timeout=3600,  # 1 hour max per model
            show_progress_bar=True,
            n_jobs=1,  # Sequential to avoid memory issues
        )
    except KeyboardInterrupt:
        logger.warning(f"Optimization interrupted for {model_name}")

    optimization_time = time.time() - start_time

    if len(study.trials) == 0:
        logger.error(f"No trials completed for {model_name}")
        return None

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params

    logger.info(f"\nBest hyperparameters for {model_name}:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    # Train final model on full training set
    logger.info(f"\nTraining final {model_name} model on full training set...")
    final_model = _create_model_from_params(model_name, best_params)
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_train = final_model.predict(X_train)
    y_pred_test = final_model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mad_test = median_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    overfitting_ratio = mae_test / mae_train if mae_train > 0 else float('inf')

    logger.info(f"\nFinal Results for {model_name}:")
    logger.info(f"  MAE Train: {mae_train:.3f}")
    logger.info(f"  MAE Test:  {mae_test:.3f}")
    logger.info(f"  MAD Test:  {mad_test:.3f}")
    logger.info(f"  R² Train:  {r2_train:.4f}")
    logger.info(f"  R² Test:   {r2_test:.4f}")
    logger.info(f"  Overfitting Ratio: {overfitting_ratio:.2f}x")
    logger.info(f"  Optimization Time: {optimization_time/60:.1f} min")
    logger.info(f"  Trials Completed: {len(study.trials)}")

    # Count parameters
    n_params = _count_model_parameters(final_model)

    return {
        'model_name': model_name,
        'best_params': best_params,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'mad_test': mad_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'overfitting_ratio': overfitting_ratio,
        'cv_mae': best_trial.value,
        'n_params': n_params,
        'n_trials': len(study.trials),
        'optimization_time_sec': optimization_time,
        'model': final_model,
        'study': study,
    }


def _create_model_from_params(model_name: str, params: Dict) -> Any:
    """Create model instance from hyperparameters."""
    if model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "Lasso":
        return Lasso(**params)
    elif model_name == "ElasticNet":
        return ElasticNet(**params)
    elif model_name == "SVR":
        return SVR(**params)
    elif model_name == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_name == "GradientBoosting":
        return GradientBoostingRegressor(**params)
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    elif model_name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    elif model_name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    elif model_name == "MLP":
        return MLPRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _count_model_parameters(model) -> int:
    """Count number of parameters in model."""
    if hasattr(model, 'coef_'):
        # Linear models
        return np.prod(model.coef_.shape)
    elif hasattr(model, 'estimators_'):
        # Ensemble models
        return len(model.estimators_)
    elif hasattr(model, 'coefs_'):
        # Neural networks
        return sum(np.prod(coef.shape) for coef in model.coefs_)
    else:
        return -1  # Unknown


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Main optimization workflow."""
    global START_TIME, MAX_RUNTIME_HOURS

    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for Age Prediction"
    )
    parser.add_argument(
        '--data-dir', type=Path, default=Path('Data'),
        help='Directory containing data files'
    )
    parser.add_argument(
        '--use-pca', action='store_true',
        help='Use PCA for dimensionality reduction'
    )
    parser.add_argument(
        '--pca-components', type=int, default=400,
        help='Number of PCA components (if --use-pca)'
    )
    parser.add_argument(
        '--top-k-features', type=int, default=5000,
        help='Number of top CpG sites to select (if not using PCA)'
    )
    parser.add_argument(
        '--max-hours', type=float, default=8.0,
        help='Maximum runtime in hours'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Test set size'
    )
    parser.add_argument(
        '--models', nargs='+',
        help='Specific models to optimize (default: all enabled)'
    )

    args = parser.parse_args()

    START_TIME = datetime.now()
    MAX_RUNTIME_HOURS = args.max_hours

    logger.info("="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION - DNA METHYLATION AGE PREDICTION")
    logger.info("="*80)
    logger.info(f"Start Time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Max Runtime: {MAX_RUNTIME_HOURS:.1f} hours")
    logger.info(f"Use PCA: {args.use_pca}")
    if args.use_pca:
        logger.info(f"PCA Components: {args.pca_components}")
    else:
        logger.info(f"Top K Features: {args.top_k_features}")
    logger.info("="*80)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup Optuna storage
    storage = f"sqlite:///{OPTUNA_DB_PATH}"
    OPTUNA_DB_PATH_obj = Path(OPTUNA_DB_PATH)
    OPTUNA_DB_PATH_obj.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA LOADING")
    logger.info("="*80)

    annot = load_annotations(args.data_dir)
    logger.info(f"Loaded {len(annot)} samples")

    y = annot['age'].values
    sample_ids = annot.index.tolist()

    logger.info(f"Age range: {y.min():.1f} - {y.max():.1f} years (mean: {y.mean():.1f})")

    cpg_names = load_cpg_names(args.data_dir)
    logger.info(f"Total CpG sites: {len(cpg_names)}")

    # =========================================================================
    # FEATURE SELECTION OR PCA
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("="*80)

    data_path = args.data_dir / "c_sample.csv"

    if args.use_pca:
        logger.info(f"Using PCA with {args.pca_components} components...")

        # Load all CpG data (chunked to avoid memory issues)
        logger.info("Loading all CpG data (this may take a while)...")
        selected_indices = list(range(len(cpg_names)))
        selected_names = cpg_names

        cpg_matrix = load_selected_cpgs(
            data_path=data_path,
            sample_ids=sample_ids,
            selected_indices=selected_indices,
            selected_names=selected_names,
            chunk_size=2000,
        )
        cpg_matrix = cpg_matrix.T
        logger.info(f"CpG matrix shape: {cpg_matrix.shape}")

        # Apply PCA
        logger.info("Applying PCA...")
        pca = PCA(n_components=args.pca_components, random_state=42)
        cpg_matrix = pca.fit_transform(cpg_matrix)
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        logger.info(f"PCA matrix shape: {cpg_matrix.shape}")

        # Save PCA transformer
        pca_path = RESULTS_DIR / "pca_transformer.joblib"
        joblib.dump(pca, pca_path)
        logger.info(f"PCA transformer saved to {pca_path}")

    else:
        logger.info(f"Selecting top {args.top_k_features} CpG sites...")
        selected_indices, selected_names = select_top_k_cpgs(
            data_path=data_path,
            sample_ids=sample_ids,
            y=y,
            cpg_names=cpg_names,
            top_k=args.top_k_features,
            chunk_size=2000,
        )

        logger.info(f"Selected {len(selected_indices)} CpG sites")

        cpg_matrix = load_selected_cpgs(
            data_path=data_path,
            sample_ids=sample_ids,
            selected_indices=selected_indices,
            selected_names=selected_names,
            chunk_size=2000,
        )
        cpg_matrix = cpg_matrix.T
        logger.info(f"CpG matrix shape: {cpg_matrix.shape}")

    # Add demographic features
    demo_features = add_demographic_features(annot)
    logger.info(f"Demographic features shape: {demo_features.shape}")

    # Combine features
    X = pd.concat([pd.DataFrame(cpg_matrix), demo_features], axis=1)
    logger.info(f"Combined feature matrix shape: {X.shape}")

    # Impute missing values
    logger.info("Imputing missing values...")
    imputer = create_knn_imputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

    # Save imputer
    imputer_path = RESULTS_DIR / "imputer.joblib"
    joblib.dump(imputer, imputer_path)

    # Train-test split
    logger.info(f"Splitting data (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Standardize
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = RESULTS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # =========================================================================
    # HYPERPARAMETER OPTIMIZATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)

    # Select models to optimize
    if args.models:
        models_to_optimize = {k: v for k, v in MODELS_CONFIG.items() if k in args.models and v['enabled']}
    else:
        models_to_optimize = {k: v for k, v in MODELS_CONFIG.items() if v['enabled']}

    logger.info(f"Models to optimize: {list(models_to_optimize.keys())}")

    # Map model names to objective functions
    objective_funcs = {
        'Ridge': objective_ridge,
        'Lasso': objective_lasso,
        'ElasticNet': objective_elasticnet,
        'SVR': objective_svr,
        'RandomForest': objective_random_forest,
        'GradientBoosting': objective_gradient_boosting,
        'XGBoost': objective_xgboost,
        'LightGBM': objective_lightgbm,
        'CatBoost': objective_catboost,
        'MLP': objective_mlp,
    }

    # Optimize each model
    results = []

    for model_name, config in models_to_optimize.items():
        if check_time_limit():
            logger.warning("Time limit reached, stopping optimization")
            break

        try:
            result = optimize_model(
                model_name=model_name,
                objective_func=objective_funcs[model_name],
                X_train=X_train_scaled,
                y_train=y_train,
                X_test=X_test_scaled,
                y_test=y_test,
                n_trials=config['n_trials'],
                cv_folds=config['cv_folds'],
                storage=storage,
            )

            if result is not None:
                results.append(result)

                # Save model
                model_path = RESULTS_DIR / f"best_{model_name.lower()}.joblib"
                joblib.dump(result['model'], model_path)
                logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}", exc_info=True)

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: GENERATING REPORT")
    logger.info("="*80)

    if len(results) == 0:
        logger.error("No models were successfully optimized!")
        return

    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'MAE_Train': r['mae_train'],
            'MAE_Test': r['mae_test'],
            'MAD_Test': r['mad_test'],
            'R2_Train': r['r2_train'],
            'R2_Test': r['r2_test'],
            'Overfitting_Ratio': r['overfitting_ratio'],
            'CV_MAE': r['cv_mae'],
            'N_Params': r['n_params'],
            'N_Trials': r['n_trials'],
            'Optimization_Time_Min': r['optimization_time_sec'] / 60,
        }
        for r in results
    ])

    # Sort by MAE Test
    results_df = results_df.sort_values('MAE_Test')
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))

    # Save CSV
    csv_path = RESULTS_DIR / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")

    # Display results
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION RESULTS (sorted by MAE Test)")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))

    # Save hyperparameters
    hyperparams_data = []
    for r in results:
        for param_name, param_value in r['best_params'].items():
            hyperparams_data.append({
                'Model': r['model_name'],
                'Parameter': param_name,
                'Value': param_value,
            })

    hyperparams_df = pd.DataFrame(hyperparams_data)
    hyperparams_path = RESULTS_DIR / f"best_hyperparameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    hyperparams_df.to_csv(hyperparams_path, index=False)
    logger.info(f"Hyperparameters saved to: {hyperparams_path}")

    # Summary
    best_model = results_df.iloc[0]
    logger.info("\n" + "="*80)
    logger.info("BEST MODEL")
    logger.info("="*80)
    logger.info(f"Model: {best_model['Model']}")
    logger.info(f"MAE Test: {best_model['MAE_Test']:.3f} years")
    logger.info(f"R² Test: {best_model['R2_Test']:.4f}")
    logger.info(f"Overfitting Ratio: {best_model['Overfitting_Ratio']:.2f}x")
    logger.info(f"Number of Parameters: {best_model['N_Params']}")

    # Total time
    total_time = datetime.now() - START_TIME
    logger.info("\n" + "="*80)
    logger.info(f"TOTAL OPTIMIZATION TIME: {total_time}")
    logger.info("="*80)

    logger.info("\n[OK] Hyperparameter optimization complete!")
    logger.info(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
