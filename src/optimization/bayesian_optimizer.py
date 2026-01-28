"""
Bayesian optimization using Optuna for hyperparameter tuning.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Try to import optuna, provide fallback if not installed
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


def check_optuna_available() -> bool:
    """Check if Optuna is available."""
    return OPTUNA_AVAILABLE


def optimize_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 100,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optimize Ridge regression hyperparameters using Bayesian optimization.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary with best parameters and study results
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available")
        return {"error": "Optuna not installed"}
    
    logger.info(f"Starting Ridge optimization with {n_trials} trials")
    
    def objective(trial):
        alpha = trial.suggest_float("alpha", 10.0, 10000.0, log=True)
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha))
        ])
        
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        
        return -scores.mean()
    
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # Suppress Optuna logs during optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best Ridge alpha: {study.best_params['alpha']:.2f}")
    logger.info(f"Best MAE: {study.best_value:.3f}")
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study,
    }


def optimize_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 100,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optimize ElasticNet hyperparameters using Bayesian optimization.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary with best parameters and study results
    """
    if not OPTUNA_AVAILABLE:
        return {"error": "Optuna not installed"}
    
    logger.info(f"Starting ElasticNet optimization with {n_trials} trials")
    
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=50000,
                tol=1e-4,
            ))
        ])
        
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        
        return -scores.mean()
    
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best ElasticNet params: {study.best_params}")
    logger.info(f"Best MAE: {study.best_value:.3f}")
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study,
    }


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Bayesian optimization.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary with best parameters and study results
    """
    if not OPTUNA_AVAILABLE:
        return {"error": "Optuna not installed"}
    
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {"error": "XGBoost not installed"}
    
    logger.info(f"Starting XGBoost optimization with {n_trials} trials")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                **params,
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
            ))
        ])
        
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="neg_mean_absolute_error",
            n_jobs=1,  # XGBoost already uses n_jobs internally
        )
        
        return -scores.mean()
    
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best XGBoost params: {study.best_params}")
    logger.info(f"Best MAE: {study.best_value:.3f}")
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study,
    }


def optimize_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize hyperparameters for all supported models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of trials per model
        cv: Cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary mapping model names to optimization results
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available for optimization")
        return {}
    
    results = {}
    
    # Ridge
    logger.info("=" * 50)
    logger.info("Optimizing Ridge")
    results["ridge"] = optimize_ridge(X_train, y_train, n_trials, cv, random_state)
    
    # ElasticNet
    logger.info("=" * 50)
    logger.info("Optimizing ElasticNet")
    results["elasticnet"] = optimize_elasticnet(X_train, y_train, n_trials, cv, random_state)
    
    # XGBoost
    logger.info("=" * 50)
    logger.info("Optimizing XGBoost")
    results["xgboost"] = optimize_xgboost(X_train, y_train, n_trials // 2, cv, random_state)
    
    # Summary
    logger.info("=" * 50)
    logger.info("OPTIMIZATION SUMMARY")
    for name, res in results.items():
        if "best_value" in res:
            logger.info(f"  {name}: MAE = {res['best_value']:.3f}")
    
    return results


def get_optimized_config(optimization_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Convert optimization results to configuration format.
    
    Args:
        optimization_results: Results from optimize_all_models
        
    Returns:
        Configuration dictionary with optimized parameters
    """
    config = {}
    
    if "ridge" in optimization_results and "best_params" in optimization_results["ridge"]:
        config["ridge_alpha"] = optimization_results["ridge"]["best_params"]["alpha"]
    
    if "elasticnet" in optimization_results and "best_params" in optimization_results["elasticnet"]:
        config["elasticnet_alpha"] = optimization_results["elasticnet"]["best_params"]["alpha"]
        config["elasticnet_l1_ratio"] = optimization_results["elasticnet"]["best_params"]["l1_ratio"]
    
    if "xgboost" in optimization_results and "best_params" in optimization_results["xgboost"]:
        xgb_params = optimization_results["xgboost"]["best_params"]
        config["xgboost_n_estimators"] = xgb_params.get("n_estimators", 400)
        config["xgboost_max_depth"] = xgb_params.get("max_depth", 6)
        config["xgboost_learning_rate"] = xgb_params.get("learning_rate", 0.05)
        config["xgboost_reg_alpha"] = xgb_params.get("reg_alpha", 1.0)
        config["xgboost_reg_lambda"] = xgb_params.get("reg_lambda", 10.0)
    
    return config
