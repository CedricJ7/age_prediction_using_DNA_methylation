#!/usr/bin/env python3
"""
Complete Hyperparameter Optimization — PCA + Top-N Feature Selection
=====================================================================

EXHAUSTIVE APPROACH with memory optimization:
- Loads ALL CpG data as float32 (~1.4 GB vs 2.8 GB)
- In-place mean imputation (no memory copy)
- Strategy A: PCA dimensionality reduction [50-300 components]
- Strategy B: Top-N features by |correlation with age| [100 to all]
- For EACH configuration, optimizes models with Optuna
- Finds global minimum MAE across all strategies and configs
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import gc
import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
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
from src.data.data_loader import load_annotations
from src.features.demographic import add_demographic_features

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

START_TIME = None
MAX_RUNTIME_HOURS = 8
RESULTS_DIR = Path("results/optimization_complete")

# PCA configurations (capped dynamically at n_train - 1)
PCA_CONFIGS = [50, 100, 150, 200, 250, 300]

# Top-N configs: small N → all 9 models, large N → linear models only
TOPN_ALL_MODELS = [100, 500, 1000, 5000, 10000]
TOPN_LINEAR_ONLY = [50000, 100000]  # "all" added dynamically

LINEAR_MODELS = {"Ridge", "Lasso", "ElasticNet"}

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
    if START_TIME is None:
        return False
    elapsed = (datetime.now() - START_TIME).total_seconds() / 3600
    return elapsed >= MAX_RUNTIME_HOURS


def get_elapsed_time() -> str:
    if START_TIME is None:
        return "00:00:00"
    elapsed = datetime.now() - START_TIME
    return str(elapsed).split('.')[0]


def get_memory_mb() -> float:
    """Current process RSS in MB (Linux)."""
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1


def get_available_ram_mb() -> float:
    """Available system RAM in MB (Linux)."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1


def log_memory(label: str = ""):
    rss = get_memory_mb()
    avail = get_available_ram_mb()
    if rss > 0:
        logger.info(f"  [MEM] {label}: process={rss:.0f} MB, available={avail:.0f} MB")


# =============================================================================
# DATA LOADING — FLOAT32, MEMORY-OPTIMIZED
# =============================================================================

def load_cpg_data_float32(
    data_path: Path,
    sample_ids: List[str],
    chunk_size: int = 2000,
) -> np.ndarray:
    """
    Load ALL CpG data as float32 numpy array (n_samples, n_cpg_sites).
    Uses chunked reading and converts to float32 immediately.
    """
    logger.info("Loading ALL CpG data as float32 (chunked)...")
    log_memory("before loading")

    # Validate header
    header_df = pd.read_csv(data_path, nrows=0)
    available_cols = set(header_df.columns)
    valid_ids = [sid for sid in sample_ids if sid in available_cols]
    if len(valid_ids) < len(sample_ids):
        logger.warning(f"  {len(sample_ids) - len(valid_ids)} sample IDs not found in CSV")
    if not valid_ids:
        raise ValueError("No matching sample IDs found in CSV columns")

    arrays = []
    total_sites = 0

    for chunk in pd.read_csv(data_path, usecols=valid_ids, chunksize=chunk_size):
        arr = chunk[valid_ids].values.astype(np.float32)
        arrays.append(arr)
        total_sites += arr.shape[0]
        if total_sites % 50000 == 0:
            logger.info(f"  Loaded {total_sites} CpG sites...")

    logger.info(f"  Concatenating {len(arrays)} chunks...")
    matrix = np.vstack(arrays)  # (n_cpg_sites, n_samples)
    del arrays
    gc.collect()

    matrix = matrix.T  # (n_samples, n_cpg_sites)
    mem_mb = matrix.nbytes / 1024**2
    logger.info(f"  Shape: {matrix.shape}, dtype: {matrix.dtype}, memory: {mem_mb:.1f} MB")
    log_memory("after loading")
    return matrix


# =============================================================================
# IN-PLACE MEAN IMPUTATION
# =============================================================================

def impute_mean_inplace(X: np.ndarray) -> None:
    """Replace NaN with column means, in-place. Zero extra memory."""
    logger.info("Imputing missing values (mean, in-place)...")
    nan_count = 0
    all_nan_cols = 0
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = np.isnan(col)
        if mask.any():
            nan_count += int(mask.sum())
            if mask.all():
                col[:] = 0.0
                all_nan_cols += 1
            else:
                col[mask] = np.nanmean(col)
    logger.info(f"  Imputed {nan_count} NaN values ({all_nan_cols} all-NaN columns set to 0)")


# =============================================================================
# CORRELATION-BASED FEATURE RANKING
# =============================================================================

def compute_abs_correlations(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Compute |Pearson correlation| between each feature and target.
    Processes in batches to limit temporary memory.
    Returns array of shape (n_features,).
    """
    logger.info("Computing |correlation| with age on train set...")
    n_features = X_train.shape[1]

    y_c = (y_train - y_train.mean()).astype(np.float64)
    y_den = np.sqrt(np.sum(y_c ** 2))

    batch_size = 50000
    abs_corr = np.zeros(n_features, dtype=np.float32)

    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        X_batch = X_train[:, start:end].astype(np.float64)
        X_c = X_batch - X_batch.mean(axis=0, keepdims=True)
        num = X_c.T @ y_c
        den = np.sqrt(np.sum(X_c ** 2, axis=0)) * y_den
        corr = np.divide(num, den, out=np.zeros(end - start, dtype=np.float64), where=den != 0)
        abs_corr[start:end] = np.abs(corr).astype(np.float32)

    logger.info(f"  Max |corr|={abs_corr.max():.4f}, median={np.median(abs_corr):.4f}")
    return abs_corr


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def objective_ridge(trial, X_train, y_train, cv_folds=5):
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
    if hasattr(model, 'coef_'):
        return int(np.prod(model.coef_.shape))
    elif hasattr(model, 'estimators_'):
        return len(model.estimators_)
    elif hasattr(model, 'coefs_'):
        return sum(int(np.prod(c.shape)) for c in model.coefs_)
    return -1


# =============================================================================
# GENERALIZED OPTIMIZATION FOR ANY FEATURE CONFIGURATION
# =============================================================================

def optimize_config(
    strategy: str,
    n_features: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config_index: int,
    total_configs: int,
    extra_info: Optional[Dict] = None,
    models_to_run: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Optimize models for one feature configuration.

    Args:
        strategy: "PCA" or "TopN"
        n_features: number of input features
        models_to_run: restrict to these model names, None = all enabled
    """
    info = extra_info or {}

    logger.info("\n" + "=" * 80)
    logger.info(f"CONFIG {config_index}/{total_configs}: {strategy}, n_features={n_features}")
    if 'variance_explained' in info:
        logger.info(f"  Variance explained: {info['variance_explained']:.4f} "
                     f"({info['variance_explained']*100:.2f}%)")
    if 'min_correlation' in info:
        logger.info(f"  Min |correlation| in selection: {info['min_correlation']:.6f}")
    logger.info("=" * 80)
    log_memory(f"{strategy}_{n_features} start")

    results = []

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    for model_name, model_config in MODELS_CONFIG.items():
        if not model_config['enabled']:
            continue
        if models_to_run is not None and model_name not in models_to_run:
            continue
        if check_time_limit():
            logger.warning("Time limit reached, stopping")
            break

        logger.info(f"\n{'─' * 80}")
        logger.info(f"Optimizing {model_name} ({strategy}={n_features}) [{get_elapsed_time()}]")
        logger.info(f"{'─' * 80}")

        try:
            study_name = (f"{model_name}_{strategy}{n_features}_"
                          f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            study = optuna.create_study(
                study_name=study_name,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=3),
            )

            objective_func = OBJECTIVE_FUNCS[model_name]

            def objective_wrapper(trial, _fn=objective_func, _cfg=model_config):
                if check_time_limit():
                    raise optuna.TrialPruned()
                return _fn(trial, X_train_s, y_train, _cfg['cv_folds'])

            study.optimize(
                objective_wrapper,
                n_trials=model_config['n_trials'],
                timeout=1800,
                show_progress_bar=False,
                n_jobs=1,
            )

            if not study.trials:
                logger.warning(f"No trials completed for {model_name}")
                continue

            best_params = study.best_trial.params
            final_model = create_model_from_params(model_name, best_params)
            final_model.fit(X_train_s, y_train)

            y_pred_train = final_model.predict(X_train_s)
            y_pred_test = final_model.predict(X_test_s)

            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            mad_test = median_absolute_error(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            overfitting = mae_test / mae_train if mae_train > 0 else float('inf')
            n_params = count_model_parameters(final_model)

            logger.info(f"  MAE Test: {mae_test:.3f} | R²: {r2_test:.4f} | "
                         f"Overfitting: {overfitting:.2f}x")

            result = {
                'strategy': strategy,
                'n_features': n_features,
                'variance_explained': info.get('variance_explained', np.nan),
                'min_correlation': info.get('min_correlation', np.nan),
                'model_name': model_name,
                'best_params': json.dumps(best_params),
                'mae_train': mae_train,
                'mae_test': mae_test,
                'mad_test': mad_test,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'overfitting_ratio': overfitting,
                'cv_mae': study.best_trial.value,
                'n_params': n_params,
                'n_trials': len(study.trials),
            }
            results.append(result)

            config_dir = RESULTS_DIR / f"{strategy.lower()}_{n_features}"
            config_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {'model': final_model, 'scaler': scaler},
                config_dir / f"{model_name.lower()}.joblib",
            )

        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}", exc_info=True)

    return results


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    global START_TIME, MAX_RUNTIME_HOURS

    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization: PCA + Top-N Feature Selection"
    )
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--max-hours', type=float, default=8.0)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument(
        '--pca-configs', nargs='+', type=int, default=PCA_CONFIGS,
        help='PCA n_components to test (default: 50 100 150 200 250 300)')
    parser.add_argument(
        '--topn-configs', nargs='+', type=int, default=TOPN_ALL_MODELS,
        help='Top-N configs with all models (default: 100 500 1000 5000 10000)')
    parser.add_argument(
        '--topn-large', nargs='+', type=int, default=TOPN_LINEAR_ONLY,
        help='Top-N configs with linear models only (default: 50000 100000)')

    args = parser.parse_args()

    START_TIME = datetime.now()
    MAX_RUNTIME_HOURS = args.max_hours

    logger.info("=" * 80)
    logger.info("COMPLETE HYPERPARAMETER OPTIMIZATION")
    logger.info("  Strategy A: PCA dimensionality reduction")
    logger.info("  Strategy B: Top-N features by |correlation with age|")
    logger.info("=" * 80)
    logger.info(f"Start: {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Max runtime: {MAX_RUNTIME_HOURS:.1f} hours")
    logger.info(f"PCA configs: {args.pca_configs}")
    logger.info(f"TopN configs (all models): {args.topn_configs}")
    logger.info(f"TopN configs (linear only): {args.topn_large} + [all]")
    logger.info("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PRE-FLIGHT CHECK
    # =========================================================================
    avail_mb = get_available_ram_mb()
    logger.info(f"\n[PRE-FLIGHT] Available RAM: {avail_mb:.0f} MB")
    log_memory("startup")

    if avail_mb > 0 and avail_mb < 4000:
        logger.warning("Less than 4 GB RAM available — risk of OOM!")

    # =========================================================================
    # PHASE 1: LOAD & PREPARE DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: DATA LOADING (float32, memory-optimized)")
    logger.info("=" * 80)

    annot = load_annotations(args.data_dir)
    y = annot['age'].values.astype(np.float32)
    sample_ids = annot.index.tolist()
    logger.info(f"Loaded {len(annot)} samples")

    # Load CpG data as float32
    data_path = args.data_dir / "c_sample.csv"
    X_cpg = load_cpg_data_float32(data_path, sample_ids, chunk_size=2000)
    n_cpg = X_cpg.shape[1]
    logger.info(f"CpG features: {n_cpg}")

    # Add demographic features
    demo_df = add_demographic_features(annot)
    demo_array = demo_df.values.astype(np.float32)
    n_demo = demo_array.shape[1]
    demo_names = list(demo_df.columns)
    logger.info(f"Demographic features ({n_demo}): {demo_names}")

    # Combine into single numpy array
    X = np.concatenate([X_cpg, demo_array], axis=1)
    del X_cpg, demo_array, demo_df
    gc.collect()
    logger.info(f"Combined matrix: {X.shape} ({X.nbytes / 1024**2:.1f} MB)")
    log_memory("after combine")

    # In-place mean imputation
    impute_mean_inplace(X)
    log_memory("after imputation")

    # Train-test split
    logger.info(f"\nTrain-test split (test_size={args.test_size})...")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42
    )
    X_train_full = X[train_idx]
    X_test_full = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    del X
    gc.collect()

    n_train = len(X_train_full)
    n_test = len(X_test_full)
    logger.info(f"Train: {n_train}, Test: {n_test}")
    logger.info(f"Train matrix: {X_train_full.shape} ({X_train_full.nbytes / 1024**2:.1f} MB)")
    log_memory("after split")

    all_results = []

    # Count total configs for progress tracking
    max_pca = min(max(args.pca_configs), n_train - 1)
    valid_pca = [c for c in args.pca_configs if c <= max_pca]
    topn_large_dedup = sorted(set(args.topn_large + [n_cpg]))
    total_configs = len(valid_pca) + len(args.topn_configs) + len(topn_large_dedup)
    config_idx = 0

    # =========================================================================
    # STRATEGY A: PCA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STRATEGY A: PCA DIMENSIONALITY REDUCTION")
    logger.info(f"  Configs: {valid_pca} (max possible: {min(n_train - 1, X_train_full.shape[1])})")
    logger.info("=" * 80)

    if valid_pca:
        # Fit PCA once with max components, then slice for each config
        max_comp = max(valid_pca)
        logger.info(f"Fitting PCA with n_components={max_comp} (single fit)...")
        log_memory("before PCA fit")

        pca = PCA(n_components=max_comp, random_state=42)
        X_pca_train = pca.fit_transform(X_train_full)
        X_pca_test = pca.transform(X_test_full)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        logger.info(f"PCA fit done. Total variance at {max_comp} components: "
                     f"{cumvar[-1]:.4f} ({cumvar[-1]*100:.2f}%)")
        log_memory("after PCA fit")

        # Save PCA transformer
        joblib.dump(pca, RESULTS_DIR / "pca_transformer.joblib")

        for n_comp in valid_pca:
            if check_time_limit():
                logger.warning("Time limit reached, stopping PCA phase")
                break

            config_idx += 1
            var_exp = float(cumvar[n_comp - 1])

            results = optimize_config(
                strategy="PCA",
                n_features=n_comp,
                X_train=X_pca_train[:, :n_comp],
                X_test=X_pca_test[:, :n_comp],
                y_train=y_train,
                y_test=y_test,
                config_index=config_idx,
                total_configs=total_configs,
                extra_info={'variance_explained': var_exp},
            )
            all_results.extend(results)

            # Save intermediate
            pd.DataFrame(all_results).to_csv(
                RESULTS_DIR / "results_intermediate.csv", index=False)

        del X_pca_train, X_pca_test, pca
        gc.collect()
        log_memory("after PCA phase cleanup")

    # =========================================================================
    # STRATEGY B: TOP-N FEATURES BY CORRELATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY B: TOP-N FEATURES BY |CORRELATION WITH AGE|")
    logger.info("=" * 80)

    # Compute correlations on CpG features of training set only (no leakage)
    abs_corr = compute_abs_correlations(X_train_full[:, :n_cpg], y_train)
    sorted_cpg_indices = np.argsort(abs_corr)[::-1]  # descending by |corr|

    # Demographic column indices (always included)
    demo_col_indices = np.arange(n_cpg, n_cpg + n_demo)

    # --- Small N: all 9 models ---
    for n_top in args.topn_configs:
        if check_time_limit():
            logger.warning("Time limit reached, stopping TopN phase")
            break
        n_top = min(n_top, n_cpg)
        config_idx += 1

        selected_cpg = sorted_cpg_indices[:n_top]
        all_cols = np.concatenate([selected_cpg, demo_col_indices])
        min_corr = float(abs_corr[sorted_cpg_indices[n_top - 1]])

        results = optimize_config(
            strategy="TopN",
            n_features=n_top,
            X_train=X_train_full[:, all_cols],
            X_test=X_test_full[:, all_cols],
            y_train=y_train,
            y_test=y_test,
            config_index=config_idx,
            total_configs=total_configs,
            extra_info={'min_correlation': min_corr},
        )
        all_results.extend(results)
        pd.DataFrame(all_results).to_csv(
            RESULTS_DIR / "results_intermediate.csv", index=False)

    # --- Large N: linear models only ---
    linear_list = sorted(LINEAR_MODELS)
    for n_top in topn_large_dedup:
        if check_time_limit():
            logger.warning("Time limit reached, stopping TopN-large phase")
            break
        n_top = min(n_top, n_cpg)
        config_idx += 1

        if n_top == n_cpg:
            # Use all features — avoid copy
            X_tr = X_train_full
            X_te = X_test_full
        else:
            selected_cpg = sorted_cpg_indices[:n_top]
            all_cols = np.concatenate([selected_cpg, demo_col_indices])
            X_tr = X_train_full[:, all_cols]
            X_te = X_test_full[:, all_cols]

        min_corr = float(abs_corr[sorted_cpg_indices[min(n_top - 1, len(sorted_cpg_indices) - 1)]])

        logger.info(f"\n  [TopN-large] n_top={n_top} → linear models only "
                     f"({', '.join(linear_list)})")

        results = optimize_config(
            strategy="TopN",
            n_features=n_top,
            X_train=X_tr,
            X_test=X_te,
            y_train=y_train,
            y_test=y_test,
            config_index=config_idx,
            total_configs=total_configs,
            extra_info={'min_correlation': min_corr},
            models_to_run=linear_list,
        )
        all_results.extend(results)
        pd.DataFrame(all_results).to_csv(
            RESULTS_DIR / "results_intermediate.csv", index=False)

    del X_train_full, X_test_full, abs_corr, sorted_cpg_indices
    gc.collect()
    log_memory("after all optimization")

    # =========================================================================
    # PHASE 3: FINAL ANALYSIS & REPORT
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL ANALYSIS & REPORT")
    logger.info("=" * 80)

    if not all_results:
        logger.error("No results obtained!")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mae_test')
    results_df.insert(0, 'rank', range(1, len(results_df) + 1))

    final_path = RESULTS_DIR / f"complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(final_path, index=False)
    logger.info(f"Results saved to: {final_path}")

    # --- Top 10 ---
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 CONFIGURATIONS (Global Minimum MAE)")
    logger.info("=" * 80)
    cols = ['rank', 'strategy', 'n_features', 'model_name', 'mae_test',
            'r2_test', 'overfitting_ratio']
    logger.info("\n" + results_df.head(10)[cols].to_string(index=False))

    # --- Global best ---
    best = results_df.iloc[0]
    logger.info("\n" + "=" * 80)
    logger.info("GLOBAL BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Strategy: {best['strategy']}")
    logger.info(f"N features: {int(best['n_features'])}")
    if best['strategy'] == 'PCA':
        logger.info(f"Variance explained: {best['variance_explained']:.4f}")
    else:
        logger.info(f"Min |correlation|: {best['min_correlation']:.6f}")
    logger.info(f"Model: {best['model_name']}")
    logger.info(f"MAE Test: {best['mae_test']:.3f} years")
    logger.info(f"R² Test: {best['r2_test']:.4f}")
    logger.info(f"Overfitting: {best['overfitting_ratio']:.2f}x")

    # --- Best per PCA config ---
    pca_results = results_df[results_df['strategy'] == 'PCA']
    if len(pca_results) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("BEST MODEL PER PCA CONFIGURATION")
        logger.info("=" * 80)
        for n_comp in sorted(pca_results['n_features'].unique()):
            sub = pca_results[pca_results['n_features'] == n_comp]
            if len(sub) > 0:
                b = sub.iloc[0]
                logger.info(
                    f"PCA {int(n_comp):>5d}: {b['model_name']:15s} | "
                    f"MAE {b['mae_test']:.3f} | R² {b['r2_test']:.4f} | "
                    f"Var {b['variance_explained']:.3f}"
                )

    # --- Best per TopN config ---
    topn_results = results_df[results_df['strategy'] == 'TopN']
    if len(topn_results) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("BEST MODEL PER TOP-N CONFIGURATION")
        logger.info("=" * 80)
        for n_feat in sorted(topn_results['n_features'].unique()):
            sub = topn_results[topn_results['n_features'] == n_feat]
            if len(sub) > 0:
                b = sub.iloc[0]
                logger.info(
                    f"Top-{int(n_feat):>6d}: {b['model_name']:15s} | "
                    f"MAE {b['mae_test']:.3f} | R² {b['r2_test']:.4f} | "
                    f"|corr| >= {b['min_correlation']:.6f}"
                )

    # --- Strategy comparison ---
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 80)
    for strat in ['PCA', 'TopN']:
        sub = results_df[results_df['strategy'] == strat]
        if len(sub) > 0:
            logger.info(f"  {strat}: best MAE = {sub['mae_test'].min():.3f}, "
                         f"configs tested = {sub['n_features'].nunique()}, "
                         f"models tested = {len(sub)}")

    total_time = datetime.now() - START_TIME
    logger.info("\n" + "=" * 80)
    logger.info(f"TOTAL RUNTIME: {total_time}")
    logger.info("=" * 80)
    logger.info(f"\n[OK] Complete optimization finished!")
    logger.info(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
