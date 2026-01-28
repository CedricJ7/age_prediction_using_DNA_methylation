"""
Train age prediction models using modular architecture.

This is the refactored entry point that uses the new src/ modules.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Import from new modular structure
from src.utils.config import Config
from src.utils.logging_config import setup_logger
from src.data.data_loader import load_annotations, load_cpg_names, load_selected_cpgs
from src.features.selection import select_top_k_cpgs
from src.features.demographic import add_demographic_features
from src.data.imputation import create_knn_imputer
from src.models.linear_models import create_ridge_model, create_lasso_model, create_elasticnet_model
from src.models.tree_models import create_random_forest_model, create_xgboost_model
from src.models.neural_models import create_mlp_model
from src.models.ensemble import create_stacked_ensemble
from src.evaluation.metrics import evaluate_model, compare_models

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DNA methylation age prediction models"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/model_config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for output files"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train stacked ensemble in addition to individual models"
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("DNA METHYLATION AGE PREDICTION - TRAINING PIPELINE")
    logger.info("=" * 80)

    # Load configuration
    if args.config.exists():
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # Override output directory
    config.output_dir = args.output_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert data_dir to Path for consistency
    data_dir = Path(config.data.data_dir)

    # =========================================================================
    # PHASE 1: DATA LOADING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: DATA LOADING")
    logger.info("=" * 80)

    # Load annotations
    annot = load_annotations(data_dir)
    logger.info(f"Loaded {len(annot)} samples")

    # Extract target variable
    y = annot["age"].values
    sample_ids = annot.index.tolist()

    logger.info(f"Age range: {y.min():.1f} - {y.max():.1f} years (mean: {y.mean():.1f})")

    # Load CpG names
    cpg_names = load_cpg_names(data_dir)
    logger.info(f"Total CpG sites: {len(cpg_names)}")

    # =========================================================================
    # PHASE 2: FEATURE SELECTION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: FEATURE SELECTION")
    logger.info("=" * 80)

    data_path = data_dir / "c_sample.csv"

    logger.info(f"Selecting top {config.data.top_k_features} CpG sites...")
    selected_indices, selected_names = select_top_k_cpgs(
        data_path=data_path,
        sample_ids=sample_ids,
        y=y,
        cpg_names=cpg_names,
        top_k=config.data.top_k_features,
        chunk_size=config.data.chunk_size,
    )

    logger.info(f"Selected {len(selected_indices)} CpG sites")

    # Load selected CpG data
    logger.info("Loading selected CpG methylation data...")
    cpg_matrix = load_selected_cpgs(
        data_path=data_path,
        sample_ids=sample_ids,
        selected_indices=selected_indices,
        selected_names=selected_names,
        chunk_size=config.data.chunk_size,
    )

    cpg_matrix = cpg_matrix.T  # Transpose to samples x features
    logger.info(f"CpG matrix shape: {cpg_matrix.shape}")

    # =========================================================================
    # PHASE 3: DATA PREPROCESSING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: DATA PREPROCESSING")
    logger.info("=" * 80)

    # Add demographic features
    demo_features = add_demographic_features(annot)
    logger.info(f"Demographic features shape: {demo_features.shape}")

    # Combine CpG and demographic features
    X = pd.concat([cpg_matrix, demo_features], axis=1)
    logger.info(f"Combined feature matrix shape: {X.shape}")

    # Impute missing values
    logger.info("Imputing missing values...")
    imputer = create_knn_imputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

    # Train-test split
    logger.info(f"Splitting data (test_size={config.data.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.data.test_size,
        random_state=config.optimization.random_state
    )

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # =========================================================================
    # PHASE 4: MODEL TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: MODEL TRAINING")
    logger.info("=" * 80)

    models = []
    results = []

    # Define models to train
    model_definitions = [
        ("Ridge", lambda: create_ridge_model(config.models)),
        ("Lasso", lambda: create_lasso_model(config.models)),
        ("ElasticNet", lambda: create_elasticnet_model(config.models)),
        ("RandomForest", lambda: create_random_forest_model(config.models)),
        ("AltumAge", lambda: create_mlp_model(config.models)),
    ]

    # Add XGBoost if available
    try:
        model_definitions.append(
            ("XGBoost", lambda: create_xgboost_model(config.models))
        )
    except ImportError:
        logger.warning("XGBoost not available, skipping")

    # Train each model
    for name, model_factory in model_definitions:
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Training {name}...")
        logger.info(f"{'─' * 80}")

        start_time = time.time()

        try:
            # Create and train model
            model = model_factory()
            model.fit(X_train, y_train)

            # Evaluate
            metrics = evaluate_model(
                model, X_train, X_test, y_train, y_test,
                cv=config.optimization.cv_folds
            )

            fit_time = time.time() - start_time

            # Save model
            model_path = config.output_dir / f"models/{name.lower()}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")

            # Store results
            models.append((name, model))
            results.append({
                "model": name,
                "fit_time_sec": fit_time,
                "n_features": X_train.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                **metrics
            })

            logger.info(f"✓ {name} completed in {fit_time:.2f}s")

        except Exception as e:
            logger.error(f"✗ {name} failed: {e}", exc_info=True)

    # =========================================================================
    # PHASE 5: ENSEMBLE (Optional)
    # =========================================================================
    if args.ensemble and len(models) >= 3:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5: STACKED ENSEMBLE")
        logger.info("=" * 80)

        # Select top 3 base learners for ensemble
        results_df = pd.DataFrame(results)
        top_models = results_df.nsmallest(3, 'mae')['model'].tolist()

        base_learners = [(name, model) for name, model in models if name in top_models]
        logger.info(f"Base learners: {[name for name, _ in base_learners]}")

        try:
            ensemble = create_stacked_ensemble(base_learners, alpha=100.0)

            start_time = time.time()
            ensemble.fit(X_train, y_train)

            metrics = evaluate_model(
                ensemble, X_train, X_test, y_train, y_test,
                cv=config.optimization.cv_folds
            )

            fit_time = time.time() - start_time

            # Save ensemble
            ensemble_path = config.output_dir / "models/ensemble.joblib"
            joblib.dump(ensemble, ensemble_path)

            results.append({
                "model": "Ensemble",
                "fit_time_sec": fit_time,
                "n_features": X_train.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                **metrics
            })

            logger.info(f"✓ Ensemble completed in {fit_time:.2f}s")

        except Exception as e:
            logger.error(f"✗ Ensemble failed: {e}", exc_info=True)

    # =========================================================================
    # PHASE 6: RESULTS SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: RESULTS SUMMARY")
    logger.info("=" * 80)

    results_df = pd.DataFrame(results)

    # Sort and display
    comparison = compare_models(results_df)

    # Save results
    results_path = config.output_dir / "metrics.csv"
    comparison.to_csv(results_path, index=False)
    logger.info(f"\n✓ Results saved to {results_path}")

    # Display top 3
    logger.info("\nTop 3 Models:")
    for idx, row in comparison.head(3).iterrows():
        logger.info(
            f"  {row['rank']}. {row['model']}: "
            f"MAE={row['mae']:.3f}, R²={row['r2']:.4f}, "
            f"Overfitting={row['overfitting_ratio']:.2f}x"
        )

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
