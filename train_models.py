"""
Pipeline d'entraînement pour la prédiction d'âge basée sur la méthylation de l'ADN.

Ce script implémente plusieurs modèles de machine learning pour prédire l'âge 
chronologique à partir des profils de méthylation de l'ADN.
Utilise des données pré-imputées et NE contient PAS de features démographiques.
"""

import argparse
import warnings
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_annotations(data_dir: Path) -> pd.DataFrame:
    """Charge les annotations des échantillons."""
    annot = pd.read_csv(data_dir / "annot_projet.csv")
    annot = annot.dropna(subset=["age", "Sample_description"]).copy()
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    return annot

def load_cpg_names(data_dir: Path) -> list[str]:
    """Charge la liste des noms de CpG."""
    cpg = pd.read_csv(data_dir / "cpg_names_projet.csv", usecols=["cpg_names"])
    return cpg["cpg_names"].astype(str).tolist()

def load_imputed_data(data_path: Path, sample_ids: list[str], chunk_size: int = 2000) -> pd.DataFrame:
    """Charge les données CpG depuis le fichier imputé."""
    print(f"Loading data from {data_path}...")
    rows = []
    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        rows.append(chunk)
    
    selected = pd.concat(rows, axis=0)
    return selected

# =============================================================================
# MODEL BUILDING FUNCTIONS
# =============================================================================

def build_models() -> list[tuple[str, object]]:
    """Construit la liste des modèles à entraîner."""
    models: list[tuple[str, object]] = []

    # ElasticNet
    models.append((
        "ElasticNet",
        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000, tol=1e-4, random_state=42),
    ))

    # Lasso
    models.append((
        "Lasso",
        Lasso(alpha=0.1, max_iter=50000, tol=1e-4, random_state=42),
    ))

    # Ridge
    models.append((
        "Ridge",
        Ridge(alpha=100.0, random_state=42),
    ))

    # Random Forest
    models.append((
        "RandomForest",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        ),
    ))

    # AltumAge (MLP)
    models.append((
        "AltumAge",
        MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    ))

    return models

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement des modèles de prédiction d'âge DNAm.")
    parser.add_argument("--data-dir", default="Data", help="Répertoire des données")
    parser.add_argument("--output-dir", default="results", help="Répertoire de sortie")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion du test set")
    parser.add_argument("--random-state", type=int, default=42, help="Graine aléatoire")
    args = parser.parse_args()
    
    run_start = perf_counter()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des données
    print("\n[1/4] Chargement des données...")
    annot = load_annotations(data_dir)
    sample_ids = annot.index.tolist()
    y = annot["age"].astype(float).values

    # Find the imputed file
    data_files = list(data_dir.glob("*impute*.csv"))
    if not data_files:
        raise FileNotFoundError("No imputed data file found in Data directory.")
    data_path = data_files[0]
    
    X_df = load_imputed_data(data_path, sample_ids)
    
    # Handle possible length mismatch
    cpg_names = load_cpg_names(data_dir)
    if len(cpg_names) != len(X_df):
        cpg_names = cpg_names[:len(X_df)]
    
    X_df.index = cpg_names
    X = X_df.T
    print(f"  → Shape finale : {X.shape}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 3. Entraînement
    print("\n[2/4] Entraînement des modèles...")
    models = build_models()
    
    metrics_rows = []
    for name, model in models:
        print(f"  → Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics_rows.append({
            "model": name,
            "mae": mae,
            "r2": r2
        })
        print(f"    MAE: {mae:.3f} | R2: {r2:.3f}")
        
        # Save model
        import joblib
        joblib.dump(model, output_dir / f"{name.lower()}.joblib")

    # 4. Sauvegarde
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    
    print(f"\nTemps total : {perf_counter() - run_start:.2f}s")

if __name__ == "__main__":
    main()
