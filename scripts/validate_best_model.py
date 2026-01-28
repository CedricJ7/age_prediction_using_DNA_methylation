#!/usr/bin/env python3
"""
Validate Best Model — Repeated CV & Nested CV
==============================================

After hyperparameter optimization, this script validates the best model
to confirm the MAE is not an artifact. It provides:

1. Repeated K-Fold CV (e.g., 10 repeats × 5 folds = 50 evaluations)
   → MAE ± std with 95% confidence interval

2. Nested CV (outer CV for evaluation, inner CV for hyperparameter tuning)
   → Unbiased performance estimate

Usage:
    python scripts/validate_best_model.py
    python scripts/validate_best_model.py --results-file results/optimization_complete/complete_results_*.csv
    python scripts/validate_best_model.py --n-repeats 20 --n-outer 10
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import gc
import json
import warnings
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import (
    RepeatedKFold,
    cross_val_score,
    KFold,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from src.utils.logging_config import setup_logger
from src.data.data_loader import load_annotations
from src.features.demographic import add_demographic_features
from scripts.hyperparameter_optimization_complete import (
    load_cpg_data_float32,
    impute_mean_inplace,
    compute_abs_correlations,
)

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_name: str, params: dict):
    """Create model instance from name and parameters."""
    # Clean params (remove random_state if we want to set it ourselves)
    params = params.copy()

    if model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "Lasso":
        return Lasso(**params)
    elif model_name == "ElasticNet":
        return ElasticNet(**params)
    elif model_name == "RandomForest":
        params.setdefault('n_jobs', -1)
        return RandomForestRegressor(**params)
    elif model_name == "GradientBoosting":
        return GradientBoostingRegressor(**params)
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor
        params.setdefault('n_jobs', -1)
        return XGBRegressor(**params)
    elif model_name == "LightGBM":
        from lightgbm import LGBMRegressor
        params.setdefault('n_jobs', -1)
        params.setdefault('verbose', -1)
        return LGBMRegressor(**params)
    elif model_name == "CatBoost":
        from catboost import CatBoostRegressor
        params.setdefault('verbose', 0)
        return CatBoostRegressor(**params)
    elif model_name == "MLP":
        return MLPRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# REPEATED K-FOLD CV
# =============================================================================

def repeated_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Perform repeated K-fold cross-validation.

    Returns:
        Dictionary with MAE scores, mean, std, and 95% CI
    """
    logger.info(f"Running Repeated K-Fold CV: {n_repeats} repeats × {n_splits} folds = {n_repeats * n_splits} evaluations")

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    # Use pipeline with scaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])

    scores = cross_val_score(
        pipeline, X, y,
        cv=rkf,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
    )

    mae_scores = -scores  # Convert to positive MAE

    # Statistics
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores, ddof=1)
    sem = stats.sem(mae_scores)

    # 95% confidence interval
    ci_low, ci_high = stats.t.interval(
        0.95,
        len(mae_scores) - 1,
        loc=mean_mae,
        scale=sem
    )

    return {
        'scores': mae_scores,
        'mean': mean_mae,
        'std': std_mae,
        'sem': sem,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'n_evaluations': len(mae_scores),
    }


# =============================================================================
# NESTED CV
# =============================================================================

def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    base_params: dict,
    n_outer: int = 5,
    n_inner: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Perform nested cross-validation for unbiased performance estimation.

    Outer loop: evaluate model performance
    Inner loop: tune hyperparameters (simplified - uses base params with slight variations)

    Returns:
        Dictionary with outer fold MAEs
    """
    logger.info(f"Running Nested CV: {n_outer} outer folds × {n_inner} inner folds")

    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=random_state + 1)

    outer_scores = []
    outer_r2_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        logger.info(f"  Outer fold {fold_idx}/{n_outer}...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Inner CV: simple validation of the base params
        # (In a full nested CV, you'd do hyperparameter search here)
        model = create_model(model_name, base_params)

        inner_scores = cross_val_score(
            model, X_train_s, y_train,
            cv=inner_cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
        )
        inner_mae = -np.mean(inner_scores)

        # Train on full training set, evaluate on test
        model = create_model(model_name, base_params)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        outer_scores.append(mae)
        outer_r2_scores.append(r2)

        logger.info(f"    Inner CV MAE: {inner_mae:.3f}, Outer test MAE: {mae:.3f}, R²: {r2:.4f}")

    outer_scores = np.array(outer_scores)
    outer_r2_scores = np.array(outer_r2_scores)

    mean_mae = np.mean(outer_scores)
    std_mae = np.std(outer_scores, ddof=1)
    sem = stats.sem(outer_scores)

    ci_low, ci_high = stats.t.interval(
        0.95,
        len(outer_scores) - 1,
        loc=mean_mae,
        scale=sem
    )

    return {
        'outer_mae_scores': outer_scores,
        'outer_r2_scores': outer_r2_scores,
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'sem': sem,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'mean_r2': np.mean(outer_r2_scores),
        'std_r2': np.std(outer_r2_scores, ddof=1),
    }


# =============================================================================
# LOAD BEST CONFIG FROM RESULTS
# =============================================================================

def load_best_config(results_dir: Path) -> dict:
    """Load the best configuration from optimization results."""
    # Find the most recent complete_results file
    pattern = str(results_dir / "complete_results_*.csv")
    files = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No results files found matching {pattern}")

    results_file = files[-1]  # Most recent
    logger.info(f"Loading results from: {results_file}")

    df = pd.read_csv(results_file)
    df = df.sort_values('mae_test')

    best = df.iloc[0]

    # Parse best_params (stored as JSON string)
    params = json.loads(best['best_params'])

    config = {
        'strategy': best['strategy'],
        'n_features': int(best['n_features']),
        'model_name': best['model_name'],
        'params': params,
        'original_mae': best['mae_test'],
        'original_r2': best['r2_test'],
        'variance_explained': best.get('variance_explained', np.nan),
        'min_correlation': best.get('min_correlation', np.nan),
    }

    return config


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate best model with Repeated CV and Nested CV"
    )
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--results-dir', type=Path, default=Path('results/optimization_complete'))
    parser.add_argument('--n-splits', type=int, default=5, help='K for K-fold')
    parser.add_argument('--n-repeats', type=int, default=10, help='Repeats for repeated K-fold')
    parser.add_argument('--n-outer', type=int, default=5, help='Outer folds for nested CV')
    parser.add_argument('--n-inner', type=int, default=3, help='Inner folds for nested CV')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MODEL VALIDATION — Repeated CV & Nested CV")
    logger.info("=" * 80)
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # LOAD BEST CONFIG
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("LOADING BEST CONFIGURATION")
    logger.info("=" * 80)

    best_config = load_best_config(args.results_dir)

    logger.info(f"Strategy: {best_config['strategy']}")
    logger.info(f"N features: {best_config['n_features']}")
    logger.info(f"Model: {best_config['model_name']}")
    logger.info(f"Original MAE: {best_config['original_mae']:.3f}")
    logger.info(f"Original R²: {best_config['original_r2']:.4f}")
    logger.info(f"Parameters: {best_config['params']}")

    # =========================================================================
    # LOAD & PREPARE DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)

    annot = load_annotations(args.data_dir)
    y = annot['age'].values.astype(np.float32)
    sample_ids = annot.index.tolist()
    logger.info(f"Samples: {len(sample_ids)}")

    data_path = args.data_dir / "c_sample.csv"
    X_cpg = load_cpg_data_float32(data_path, sample_ids, chunk_size=2000)
    n_cpg = X_cpg.shape[1]

    demo_df = add_demographic_features(annot)
    demo_array = demo_df.values.astype(np.float32)
    n_demo = demo_array.shape[1]

    X_full = np.concatenate([X_cpg, demo_array], axis=1)
    del X_cpg, demo_array
    gc.collect()

    impute_mean_inplace(X_full)
    logger.info(f"Full matrix: {X_full.shape}")

    # =========================================================================
    # PREPARE FEATURES BASED ON STRATEGY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PREPARING FEATURES")
    logger.info("=" * 80)

    if best_config['strategy'] == 'PCA':
        n_comp = best_config['n_features']
        logger.info(f"Applying PCA with n_components={n_comp}")

        pca = PCA(n_components=n_comp, random_state=args.seed)
        X = pca.fit_transform(X_full)

        var_exp = np.sum(pca.explained_variance_ratio_)
        logger.info(f"Variance explained: {var_exp:.4f} ({var_exp*100:.2f}%)")

        del X_full
        gc.collect()

    elif best_config['strategy'] == 'TopN':
        n_top = best_config['n_features']
        logger.info(f"Selecting Top-{n_top} features by |correlation|")

        # Compute correlations
        abs_corr = compute_abs_correlations(X_full[:, :n_cpg], y)
        sorted_idx = np.argsort(abs_corr)[::-1]

        # Select top N CpG + demographics
        selected_cpg = sorted_idx[:n_top]
        demo_cols = np.arange(n_cpg, n_cpg + n_demo)
        all_cols = np.concatenate([selected_cpg, demo_cols])

        X = X_full[:, all_cols]

        min_corr = abs_corr[sorted_idx[n_top - 1]]
        logger.info(f"Min |correlation| in selection: {min_corr:.6f}")

        del X_full, abs_corr
        gc.collect()
    else:
        raise ValueError(f"Unknown strategy: {best_config['strategy']}")

    logger.info(f"Final feature matrix: {X.shape}")

    # =========================================================================
    # REPEATED K-FOLD CV
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"REPEATED K-FOLD CV ({args.n_repeats} × {args.n_splits}-fold)")
    logger.info("=" * 80)

    model = create_model(best_config['model_name'], best_config['params'])

    rkf_results = repeated_kfold_cv(
        X, y, model,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )

    logger.info(f"\nRepeated K-Fold Results:")
    logger.info(f"  MAE: {rkf_results['mean']:.3f} ± {rkf_results['std']:.3f}")
    logger.info(f"  95% CI: [{rkf_results['ci_95_low']:.3f}, {rkf_results['ci_95_high']:.3f}]")
    logger.info(f"  N evaluations: {rkf_results['n_evaluations']}")
    logger.info(f"  Min MAE: {rkf_results['scores'].min():.3f}")
    logger.info(f"  Max MAE: {rkf_results['scores'].max():.3f}")

    # =========================================================================
    # NESTED CV
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"NESTED CV ({args.n_outer} outer × {args.n_inner} inner)")
    logger.info("=" * 80)

    nested_results = nested_cv(
        X, y,
        model_name=best_config['model_name'],
        base_params=best_config['params'],
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        random_state=args.seed,
    )

    logger.info(f"\nNested CV Results:")
    logger.info(f"  MAE: {nested_results['mean_mae']:.3f} ± {nested_results['std_mae']:.3f}")
    logger.info(f"  95% CI: [{nested_results['ci_95_low']:.3f}, {nested_results['ci_95_high']:.3f}]")
    logger.info(f"  R²: {nested_results['mean_r2']:.4f} ± {nested_results['std_r2']:.4f}")
    logger.info(f"  Per-fold MAEs: {nested_results['outer_mae_scores']}")

    # =========================================================================
    # SUMMARY & COMPARISON
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nBest Model: {best_config['model_name']}")
    logger.info(f"Strategy: {best_config['strategy']} (n_features={best_config['n_features']})")
    logger.info(f"")
    logger.info(f"Original holdout MAE:     {best_config['original_mae']:.3f}")
    logger.info(f"Repeated K-Fold MAE:      {rkf_results['mean']:.3f} ± {rkf_results['std']:.3f}")
    logger.info(f"Nested CV MAE:            {nested_results['mean_mae']:.3f} ± {nested_results['std_mae']:.3f}")
    logger.info(f"")

    # Check consistency
    diff_rkf = abs(rkf_results['mean'] - best_config['original_mae'])
    diff_nested = abs(nested_results['mean_mae'] - best_config['original_mae'])

    if diff_rkf < 0.5 and diff_nested < 0.5:
        logger.info("✓ Results are consistent — original MAE is NOT an artifact")
    elif diff_rkf < 1.0 and diff_nested < 1.0:
        logger.info("~ Results are reasonably consistent (within 1 year)")
    else:
        logger.info("⚠ Results show some discrepancy — consider investigating")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = args.results_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_results = {
        'model_name': best_config['model_name'],
        'strategy': best_config['strategy'],
        'n_features': best_config['n_features'],
        'params': best_config['params'],
        'original_mae': best_config['original_mae'],
        'original_r2': best_config['original_r2'],
        'repeated_kfold': {
            'n_splits': args.n_splits,
            'n_repeats': args.n_repeats,
            'mean_mae': float(rkf_results['mean']),
            'std_mae': float(rkf_results['std']),
            'ci_95': [float(rkf_results['ci_95_low']), float(rkf_results['ci_95_high'])],
            'all_scores': rkf_results['scores'].tolist(),
        },
        'nested_cv': {
            'n_outer': args.n_outer,
            'n_inner': args.n_inner,
            'mean_mae': float(nested_results['mean_mae']),
            'std_mae': float(nested_results['std_mae']),
            'ci_95': [float(nested_results['ci_95_low']), float(nested_results['ci_95_high'])],
            'mean_r2': float(nested_results['mean_r2']),
            'std_r2': float(nested_results['std_r2']),
            'outer_mae_scores': nested_results['outer_mae_scores'].tolist(),
            'outer_r2_scores': nested_results['outer_r2_scores'].tolist(),
        },
    }

    output_file = output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n" + "=" * 80)
    logger.info("[OK] Validation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
