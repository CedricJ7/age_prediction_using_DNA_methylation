#!/usr/bin/env python3
"""
Complete Hyperparameter Optimization with PCA Grid Search
==========================================================

EXHAUSTIVE APPROACH:
- Loads ALL available CpG data (max memory capacity)
- Tests PCA with multiple n_components: [50, 100, 150, 200, 250, 300, 350, 400]
- For EACH PCA configuration, optimizes ALL models
- Finds absolute global minimum MAE across all configurations
- Comprehensive report comparing PCA choices and models

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
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import joblib
import json

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Optimization
import optuna

# Local imports
from src.utils.logging_config import setup_logger
from src.data.data_loader import load_annotations, load_cpg_names
from src.features.demographic import add_demographic_features
from src.data.imputation import create_knn_imputer

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

START_TIME = None
MAX_RUNTIME_HOURS = 8
RESULTS_DIR = Path("results/optimization_complete")

# PCA configurations to test (variance explained)
PCA_CONFIGS = [50, 100, 150, 200, 250, 300, 350, 400]

# Models with reduced trials for exhaustive search
MODELS_CONFIG = {
    "Ridge": {"enabled": True, "n_trials": 50, "cv_folds": 5},
    "Lasso": {"enabled": True, "n_trials": 50, "cv_folds": 5},
    "ElasticNet": {"enabled": True, "n_trials": 75, "cv_folds": 5},
    "RandomForest": {"enabled": True, "n_trials": 40, "cv_folds": 5},
    "GradientBoosting": {"enabled": True, "n_trials": 40, "cv_folds": 5},
    "XGBoost": {"enabled": True, "n_trials": 50, "cv_folds": 5},
    "LightGBM": {"enabled": True, "n_trials": 50, "cv_folds": 5},
    "CatBoost": {"enabled": True, "n_trials": 40, "cv_folds": 5},
    "MLP": {"enabled": True, "n_trials": 50, "cv_folds": 5},
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
# DATA LOADING WITH MEMORY OPTIMIZATION
# =============================================================================

def load_all_cpg_data_chunked(
    data_path: Path,
    sample_ids: List[str],
    chunk_size: int = 1000
) -> pd.DataFrame:
    """
    Load ALL CpG data using chunked reading to avoid memory issues.

    Returns:
        DataFrame of shape (n_samples, n_cpg_sites)
    """
    logger.info("Loading ALL CpG methylation data (chunked reading)...")

    # Read in chunks and accumulate
    chunks = []
    total_sites = 0

    # First, read header to check if sample_ids are in columns
    header_df = pd.read_csv(data_path, nrows=1)

    # Check if sample_ids are in columns (expected format)
    if not all(sid in header_df.columns for sid in sample_ids[:5]):
        logger.error(f"Sample IDs not found in CSV columns!")
        logger.error(f"Expected: {sample_ids[:3]}")
        logger.error(f"Found columns: {list(header_df.columns[:5])}")
        raise ValueError("Sample IDs mismatch with CSV structure")

    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        # Filter to our samples (columns)
        try:
            chunk_filtered = chunk[sample_ids]
            chunks.append(chunk_filtered)
            total_sites += len(chunk_filtered)

            if total_sites % 10000 == 0:
                logger.info(f"  Loaded {total_sites} CpG sites...")
        except KeyError as e:
            logger.warning(f"Some samples not found in chunk, skipping: {e}")
            continue

    # Concatenate all chunks
    logger.info(f"  Concatenating {len(chunks)} chunks...")
    cpg_matrix = pd.concat(chunks, axis=0)

    logger.info(f"  Total CpG sites loaded: {cpg_matrix.shape[0]}")
    logger.info(f"  Matrix shape: {cpg_matrix.shape}")
    logger.info(f"  Memory usage: {cpg_matrix.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return cpg_matrix.T  # Transpose to (samples x features)


# =============================================================================
# OBJECTIVE FUNCTIONS (same as before but copied for completeness)
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
                            scoring='neg_mean_absolute_error', n_jobs=1)
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


# Map model names to objectives
OBJECTIVE_FUNCS = {
    'Ridge': objective_ridge,
    'Lasso': objective_lasso,
    'ElasticNet': objective_elasticnet,
    'RandomForest': objective_random_forest,
    'GradientBoosting': objective_gradient_boosting,
    'XGBoost': objective_xgboost,
    'LightGBM': objective_lightgbm,
    'CatBoost': objective_catboost,
    'MLP': objective_mlp,
}


# =============================================================================
# MODEL CREATION AND PARAMETER COUNTING
# =============================================================================

def create_model_from_params(model_name: str, params: Dict) -> Any:
    """Create model instance from hyperparameters."""
    if model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "Lasso":
        return Lasso(**params)
    elif model_name == "ElasticNet":
        return ElasticNet(**params)
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


def count_model_parameters(model) -> int:
    """Count number of parameters in model."""
    if hasattr(model, 'coef_'):
        return np.prod(model.coef_.shape)
    elif hasattr(model, 'estimators_'):
        return len(model.estimators_)
    elif hasattr(model, 'coefs_'):
        return sum(np.prod(coef.shape) for coef in model.coefs_)
    else:
        return -1


# =============================================================================
# OPTIMIZATION FOR SINGLE PCA CONFIGURATION
# =============================================================================

def optimize_single_pca_config(
    n_components: int,
    X_pca_train: np.ndarray,
    X_pca_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    variance_explained: float,
    config_index: int,
    total_configs: int,
) -> List[Dict]:
    """
    Optimize all models for a single PCA configuration.

    Returns:
        List of result dictionaries for each model
    """
    logger.info("\n" + "="*80)
    logger.info(f"PCA CONFIG {config_index}/{total_configs}: n_components={n_components}")
    logger.info(f"Variance Explained: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
    logger.info("="*80)

    results = []

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_pca_train)
    X_test_scaled = scaler.transform(X_pca_test)

    # Optimize each model
    for model_name, model_config in MODELS_CONFIG.items():
        if not model_config['enabled']:
            continue

        if check_time_limit():
            logger.warning("Time limit reached, stopping optimization")
            break

        logger.info(f"\n{'─'*80}")
        logger.info(f"Optimizing {model_name} (PCA={n_components})...")
        logger.info(f"{'─'*80}")

        try:
            # Create Optuna study
            study_name = f"{model_name}_PCA{n_components}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                study_name=study_name,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            )

            # Optimize
            objective_func = OBJECTIVE_FUNCS[model_name]

            def objective_wrapper(trial):
                if check_time_limit():
                    raise optuna.TrialPruned()
                return objective_func(trial, X_train_scaled, y_train, model_config['cv_folds'])

            study.optimize(
                objective_wrapper,
                n_trials=model_config['n_trials'],
                timeout=1800,  # 30 min max per model
                show_progress_bar=False,
                n_jobs=1,
            )

            if len(study.trials) == 0:
                logger.warning(f"No trials completed for {model_name}")
                continue

            # Train final model
            best_params = study.best_trial.params
            final_model = create_model_from_params(model_name, best_params)
            final_model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred_train = final_model.predict(X_train_scaled)
            y_pred_test = final_model.predict(X_test_scaled)

            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            mad_test = median_absolute_error(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            overfitting_ratio = mae_test / mae_train if mae_train > 0 else float('inf')
            n_params = count_model_parameters(final_model)

            logger.info(f"  MAE Test: {mae_test:.3f} | R²: {r2_test:.4f} | Overfitting: {overfitting_ratio:.2f}x")

            # Store results
            result = {
                'pca_n_components': n_components,
                'pca_variance_explained': variance_explained,
                'model_name': model_name,
                'best_params': best_params,
                'mae_train': mae_train,
                'mae_test': mae_test,
                'mad_test': mad_test,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'overfitting_ratio': overfitting_ratio,
                'cv_mae': study.best_trial.value,
                'n_params': n_params,
                'n_trials': len(study.trials),
            }
            results.append(result)

            # Save model
            model_dir = RESULTS_DIR / f"pca_{n_components}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{model_name.lower()}.joblib"
            joblib.dump({'model': final_model, 'scaler': scaler}, model_path)

        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}", exc_info=True)

    return results


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Main exhaustive optimization workflow."""
    global START_TIME, MAX_RUNTIME_HOURS

    parser = argparse.ArgumentParser(
        description="Complete Hyperparameter Optimization with PCA Grid Search"
    )
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--max-hours', type=float, default=8.0)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument(
        '--pca-configs', nargs='+', type=int, default=PCA_CONFIGS,
        help='PCA n_components to test (default: 50 100 150 200 250 300 350 400)'
    )

    args = parser.parse_args()

    START_TIME = datetime.now()
    MAX_RUNTIME_HOURS = args.max_hours

    logger.info("="*80)
    logger.info("COMPLETE HYPERPARAMETER OPTIMIZATION - PCA GRID SEARCH")
    logger.info("="*80)
    logger.info(f"Start Time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Max Runtime: {MAX_RUNTIME_HOURS:.1f} hours")
    logger.info(f"PCA Configs: {args.pca_configs}")
    logger.info("="*80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PHASE 1: LOAD ALL DATA
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA LOADING (ALL CPG SITES)")
    logger.info("="*80)

    annot = load_annotations(args.data_dir)
    logger.info(f"Loaded {len(annot)} samples")

    y = annot['age'].values
    sample_ids = annot.index.tolist()

    # Load ALL CpG data (chunked to manage memory)
    data_path = args.data_dir / "c_sample.csv"
    cpg_matrix = load_all_cpg_data_chunked(data_path, sample_ids, chunk_size=2000)

    logger.info(f"Full CpG matrix: {cpg_matrix.shape}")

    # Add demographic features
    demo_features = add_demographic_features(annot)
    X = pd.concat([cpg_matrix, demo_features], axis=1)
    logger.info(f"Combined matrix with demographics: {X.shape}")

    # Impute missing values
    logger.info("Imputing missing values...")
    imputer = create_knn_imputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

    # Train-test split (BEFORE PCA to avoid data leakage)
    logger.info(f"Train-test split (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Save imputer
    joblib.dump(imputer, RESULTS_DIR / "imputer.joblib")

    # =========================================================================
    # PHASE 2: PCA GRID SEARCH
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: PCA CONFIGURATIONS & OPTIMIZATION")
    logger.info("="*80)

    all_results = []
    pca_transformers = {}

    for i, n_components in enumerate(args.pca_configs, 1):
        if check_time_limit():
            logger.warning("Time limit reached, stopping PCA grid search")
            break

        logger.info(f"\n{'='*80}")
        logger.info(f"PCA Transformation: n_components={n_components}")
        logger.info(f"{'='*80}")

        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca_train = pca.fit_transform(X_train)
        X_pca_test = pca.transform(X_test)

        variance_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"Variance explained: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
        logger.info(f"PCA train shape: {X_pca_train.shape}")

        # Save PCA transformer
        pca_transformers[n_components] = pca
        pca_dir = RESULTS_DIR / f"pca_{n_components}"
        pca_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, pca_dir / "pca_transformer.joblib")

        # Optimize all models for this PCA config
        config_results = optimize_single_pca_config(
            n_components=n_components,
            X_pca_train=X_pca_train,
            X_pca_test=X_pca_test,
            y_train=y_train,
            y_test=y_test,
            variance_explained=variance_explained,
            config_index=i,
            total_configs=len(args.pca_configs),
        )

        all_results.extend(config_results)

        # Save intermediate results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_DIR / "results_intermediate.csv", index=False)

    # =========================================================================
    # PHASE 3: FINAL ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: FINAL ANALYSIS & REPORT")
    logger.info("="*80)

    if len(all_results) == 0:
        logger.error("No results obtained!")
        return

    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mae_test')
    results_df.insert(0, 'rank', range(1, len(results_df) + 1))

    # Save complete results
    final_path = RESULTS_DIR / f"complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(final_path, index=False)
    logger.info(f"\nResults saved to: {final_path}")

    # Display top 10
    logger.info("\n" + "="*80)
    logger.info("TOP 10 CONFIGURATIONS (Global Minimum MAE)")
    logger.info("="*80)
    top10 = results_df.head(10)[['rank', 'pca_n_components', 'model_name', 'mae_test', 'r2_test', 'overfitting_ratio', 'pca_variance_explained']]
    logger.info("\n" + top10.to_string(index=False))

    # Best overall
    best = results_df.iloc[0]
    logger.info("\n" + "="*80)
    logger.info("GLOBAL BEST CONFIGURATION")
    logger.info("="*80)
    logger.info(f"PCA n_components: {int(best['pca_n_components'])}")
    logger.info(f"Variance Explained: {best['pca_variance_explained']:.4f} ({best['pca_variance_explained']*100:.2f}%)")
    logger.info(f"Model: {best['model_name']}")
    logger.info(f"MAE Test: {best['mae_test']:.3f} years")
    logger.info(f"R² Test: {best['r2_test']:.4f}")
    logger.info(f"Overfitting Ratio: {best['overfitting_ratio']:.2f}x")

    # Summary by PCA
    logger.info("\n" + "="*80)
    logger.info("BEST MODEL PER PCA CONFIGURATION")
    logger.info("="*80)

    for n_comp in args.pca_configs:
        subset = results_df[results_df['pca_n_components'] == n_comp]
        if len(subset) > 0:
            best_for_pca = subset.iloc[0]
            logger.info(
                f"PCA {n_comp:3d}: {best_for_pca['model_name']:15s} | "
                f"MAE {best_for_pca['mae_test']:.3f} | "
                f"R² {best_for_pca['r2_test']:.4f} | "
                f"Var {best_for_pca['pca_variance_explained']:.3f}"
            )

    # Total time
    total_time = datetime.now() - START_TIME
    logger.info("\n" + "="*80)
    logger.info(f"TOTAL RUNTIME: {total_time}")
    logger.info("="*80)

    logger.info("\n[OK] Complete optimization finished!")
    logger.info(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
