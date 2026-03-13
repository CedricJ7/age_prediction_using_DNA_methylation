#!/usr/bin/env python3
"""
ElasticNet Hyperparameter Optimization for DNA Methylation Age Prediction
========================================================================

Standalone script for optimizing an ElasticNet model using Optuna.
Uses pre-imputed data, CpG beta-values only (no demographic features).
Sélection des top-k CpG par corrélation (TRAIN only, pas de leakage).
Compare k=500, 1000, 1500, 2000. No StandardScaler.
"""

import sys
from pathlib import Path
import argparse
import heapq
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.linear_model import ElasticNet

# Optimization
import optuna

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

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

def load_cpg_names(data_dir: Path) -> list:
    """Charge la liste des noms de CpG."""
    cpg = pd.read_csv(data_dir / "cpg_names_projet.csv", usecols=["cpg_names"])
    return cpg["cpg_names"].astype(str).tolist()

def load_imputed_data(data_path: Path, sample_ids: list, chunk_size: int = 2000) -> pd.DataFrame:
    """Charge les données CpG depuis le fichier imputé."""
    print(f"  Loading data from {data_path}...")
    rows = []
    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        rows.append(chunk)
    selected = pd.concat(rows, axis=0)
    return selected

# =============================================================================
# FEATURE SELECTION BY CORRELATION
# =============================================================================

def select_top_k_cpgs(data_path: Path, train_ids: list, y_train: np.ndarray,
                      cpg_names: list, top_k: int, chunk_size: int = 5000):
    """
    Sélectionne les top-k CpG les plus corrélées avec l'âge.
    Calculé sur TRAIN uniquement (pas de leakage).
    Utilise un min-heap pour efficacité mémoire.
    """
    print(f"  Sélection des top {top_k} CpG par corrélation (TRAIN only)...")
    t0 = time.time()

    y_centered = y_train - y_train.mean()
    y_den = np.sqrt(np.sum(y_centered ** 2))

    best = []  # min-heap: (abs_corr, global_index)
    start = 0

    for chunk in pd.read_csv(data_path, usecols=train_ids, chunksize=chunk_size):
        x = chunk.to_numpy(dtype=np.float32, copy=False)

        # Imputation NaN par moyenne de la ligne
        if np.isnan(x).any():
            row_means = np.nanmean(x, axis=1, keepdims=True)
            row_means = np.where(np.isnan(row_means), 0, row_means)
            x = np.where(np.isnan(x), row_means, x)

        # Pearson correlation
        x_centered = x - x.mean(axis=1, keepdims=True)
        num = x_centered @ y_centered
        den = np.sqrt(np.sum(x_centered ** 2, axis=1)) * y_den
        corr = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        abs_corr = np.abs(corr)

        for i, c in enumerate(abs_corr):
            idx = start + i
            if len(best) < top_k:
                heapq.heappush(best, (c, idx))
            elif c > best[0][0]:
                heapq.heapreplace(best, (c, idx))

        start += len(chunk)

    # Trier par corrélation décroissante
    best_sorted = sorted(best, key=lambda t: t[0], reverse=True)
    indices = [idx for _, idx in best_sorted]
    names = [cpg_names[i] if i < len(cpg_names) else f"cpg_{i}" for i in indices]

    t1 = time.time()
    print(f"  -> {len(indices)} CpG sélectionnées en {t1 - t0:.1f}s")
    print(f"  -> Corrélation : max={best_sorted[0][0]:.4f}, min={best_sorted[-1][0]:.4f}")

    return indices, names

def load_selected_cpgs(data_path: Path, sample_ids: list, selected_indices: list,
                       selected_names: list, chunk_size: int = 5000) -> pd.DataFrame:
    """Charge uniquement les CpG sélectionnées pour tous les samples."""
    indices = np.array(sorted(selected_indices))
    rows = []
    start = 0

    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        end = start + len(chunk)
        pos_start = np.searchsorted(indices, start)
        pos_end = np.searchsorted(indices, end)
        local = indices[pos_start:pos_end] - start

        if len(local) > 0:
            rows.append(chunk.iloc[local])
        start = end

    selected = pd.concat(rows, axis=0)
    selected.index = selected_names
    return selected

# =============================================================================
# CONFIGURATION
# =============================================================================

OPTUNA_DB_PATH = "results/optimization_elasticnet/optuna_study_elasticnet.db"
RESULTS_DIR = Path("results/optimization_elasticnet")
START_TIME = datetime.now()

# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective_elasticnet(trial, X_train, y_train, cv_folds=5):
    """Optimize ElasticNet regression."""
    params = {
        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 0.99),
        'max_iter': trial.suggest_int('max_iter', 1000, 10000),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
        'random_state': RANDOM_STATE,
    }

    model = ElasticNet(**params)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv_folds,
        scoring='neg_mean_absolute_error', n_jobs=1
    )
    return -scores.mean()

# =============================================================================
# TRAIN + EVALUATE FOR A GIVEN K
# =============================================================================

def run_optimization_for_k(k, X_train_full, X_test_full, y_train, y_test,
                           n_trials, storage_base):
    """Lance l'optimisation Optuna pour un k donné et retourne les métriques."""
    print(f"\n{'#'*60}")
    print(f"# OPTIMISATION AVEC k={k} FEATURES")
    print(f"{'#'*60}")

    # Slice des k premières colonnes (déjà triées par corrélation décroissante)
    X_train_k = X_train_full[:, :k]
    X_test_k = X_test_full[:, :k]
    print(f"  X_train : {X_train_k.shape}, X_test : {X_test_k.shape}")

    # Optuna
    storage = f"sqlite:///{storage_base}_k{k}.db"
    Path(f"{storage_base}_k{k}.db").parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction='minimize',
        storage=storage,
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    print(f"  Lancement de {n_trials} trials...")
    study.optimize(
        lambda trial: objective_elasticnet(trial, X_train_k, y_train, 5),
        n_trials=n_trials,
        timeout=3600,
        show_progress_bar=True,
        n_jobs=1,
    )

    # Modèle final
    best_params = study.best_trial.params
    print(f"  Best params: {best_params}")

    final_model = ElasticNet(**best_params)
    final_model.fit(X_train_k, y_train)

    # Évaluation
    y_pred_train = final_model.predict(X_train_k)
    y_pred_test = final_model.predict(X_test_k)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test_val = r2_score(y_test, y_pred_test)
    mad_test = median_absolute_error(y_test, y_pred_test)
    overfit = mae_test / mae_train if mae_train > 0 else np.nan
    n_coef_nonzero = int(np.sum(final_model.coef_ != 0))

    results = {
        "k": k,
        "best_params": best_params,
        "cv_mae": study.best_value,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "r2_test": r2_test_val,
        "mad_test": mad_test,
        "overfitting": overfit,
        "n_coef_nonzero": n_coef_nonzero,
    }

    print(f"\n  Résultats k={k} :")
    print(f"    MAE CV       : {study.best_value:.4f}")
    print(f"    MAE train    : {mae_train:.4f}")
    print(f"    MAE test     : {mae_test:.4f}")
    print(f"    R² test      : {r2_test_val:.4f}")
    print(f"    MAD test     : {mad_test:.4f}")
    print(f"    Overfitting  : {overfit:.2f}x")
    print(f"    Coefs ≠ 0    : {n_coef_nonzero} / {k}")

    return results, final_model

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ElasticNet Optimization (Standalone)")
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    K_VALUES = [500, 1000, 1500, 2000]
    MAX_K = max(K_VALUES)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Chargement des annotations
    # =========================================================================
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Phase 1: Chargement des annotations...")
    annot = load_annotations(args.data_dir)
    print(f"  Annotations : {annot.shape[0]} échantillons")

    # =========================================================================
    # 2. Intersection échantillons annotations / méthylation
    # =========================================================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 2: Identification des échantillons...")
    data_files = list(args.data_dir.glob("*impute*.csv"))
    if not data_files:
        raise FileNotFoundError("No imputed data file found in Data directory.")
    data_path = data_files[0]
    print(f"  Fichier : {data_path}")

    sample_header = pd.read_csv(data_path, nrows=0)
    all_sample_ids = list(sample_header.columns)

    common_ids = [s for s in all_sample_ids if s in annot.index]
    ages = annot.loc[common_ids, 'age'].values.astype(np.float32)
    sample_ids = list(annot.loc[common_ids].index)
    print(f"  Échantillons en commun : {len(common_ids)}")
    print(f"  Âge moyen : {ages.mean():.1f} ± {ages.std():.1f} ans")

    # =========================================================================
    # 3. Split train/test AVANT toute sélection (pas de leakage)
    # =========================================================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 3: Split train/test...")

    train_idx, test_idx = train_test_split(
        np.arange(len(sample_ids)),
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    train_ids = [sample_ids[i] for i in train_idx]
    test_ids = [sample_ids[i] for i in test_idx]
    y_train = ages[train_idx]
    y_test = ages[test_idx]

    print(f"  Train : {len(train_ids)} échantillons")
    print(f"  Test  : {len(test_ids)} échantillons")
    print(f"  Âge train : {y_train.mean():.1f} ± {y_train.std():.1f}")
    print(f"  Âge test  : {y_test.mean():.1f} ± {y_test.std():.1f}")

    # =========================================================================
    # 4. Sélection des top MAX_K CpG par corrélation (TRAIN only)
    # =========================================================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 4: Sélection features par corrélation...")
    cpg_names = load_cpg_names(args.data_dir)
    selected_indices, selected_names = select_top_k_cpgs(
        data_path, train_ids, y_train, cpg_names, top_k=MAX_K
    )

    # =========================================================================
    # 5. Chargement des CpG sélectionnées (train + test, une seule lecture)
    # =========================================================================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 5: Chargement des {MAX_K} CpG sélectionnées...")
    all_ordered_ids = train_ids + test_ids
    X_all_raw = load_selected_cpgs(data_path, all_ordered_ids, selected_indices, selected_names)
    X_all_df = X_all_raw.T.astype(np.float32)  # (n_samples, n_features)

    X_train_full = X_all_df.loc[train_ids].values
    X_test_full = X_all_df.loc[test_ids].values
    print(f"  X_train_full : {X_train_full.shape}")
    print(f"  X_test_full  : {X_test_full.shape}")
    print(f"  NaN : train={np.isnan(X_train_full).sum()}, test={np.isnan(X_test_full).sum()}")

    del X_all_raw, X_all_df

    # =========================================================================
    # 6. Optimisation pour chaque k et comparaison
    # =========================================================================
    storage_base = str(RESULTS_DIR / "optuna_study_elasticnet")
    all_results = []

    for k in K_VALUES:
        results, model = run_optimization_for_k(
            k, X_train_full, X_test_full, y_train, y_test,
            args.n_trials, storage_base
        )
        results["n_train"] = len(train_ids)
        results["n_test"] = len(test_ids)
        results["timestamp"] = datetime.now().isoformat()
        all_results.append(results)

        # Sauvegarder le modèle
        joblib.dump(model, RESULTS_DIR / f"best_elasticnet_k{k}.joblib")
        joblib.dump(results, RESULTS_DIR / f"best_elasticnet_metrics_k{k}.joblib")

    # =========================================================================
    # 7. Tableau comparatif
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"COMPARAISON FINALE")
    print(f"{'='*60}")
    print(f"{'k':>6s} | {'MAE CV':>8s} | {'MAE train':>9s} | {'MAE test':>9s} | {'R² test':>8s} | {'Overfit':>8s} | {'Coefs≠0':>8s}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in all_results:
        print(f"{r['k']:>6d} | {r['cv_mae']:>8.4f} | {r['mae_train']:>9.4f} | {r['mae_test']:>9.4f} | "
              f"{r['r2_test']:>8.4f} | {r['overfitting']:>7.2f}x | {r['n_coef_nonzero']:>8d}")

    # Meilleur k
    best = min(all_results, key=lambda r: r['mae_test'])
    print(f"\n  -> Meilleur : k={best['k']} avec MAE test = {best['mae_test']:.4f}")

    # Sauvegarder le comparatif
    comp_df = pd.DataFrame(all_results)
    comp_df.to_csv(RESULTS_DIR / "comparison_k_values.csv", index=False)

    print(f"\n  Résultats sauvés dans {RESULTS_DIR}")
    print(f"  Temps total : {datetime.now() - START_TIME}")

if __name__ == "__main__":
    main()
