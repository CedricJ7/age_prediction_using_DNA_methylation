#!/usr/bin/env python3
"""
Évaluation par Nested Cross-Validation avec ElasticNet et MICE Optimisé.
========================================================================
1. Split Train/Test (Outer CV) avant tout.
2. Imputation MICE (IterativeImputer) optimisée intra-fold.
3. Sélection supervisée (variance + corrélation) intra-fold.
4. Optimisation ElasticNet (Inner CV) intra-fold.
5. Évaluation finale sur le pli Test.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from time import time

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

def load_annotations(data_dir: Path):
    annot = pd.read_csv(data_dir / "annot_projet.csv")
    annot = annot.dropna(subset=["age", "Sample_description"]).copy()
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    return annot

def get_top_indices(data_path, train_ids, y_train, n_top_var=10000, n_top_corr=1000):
    print(f"  Sélection des caractéristiques ({n_top_corr}) sur le Train...")
    y_c = y_train - y_train.mean()
    y_den = np.sqrt(np.sum(y_c ** 2))
    all_vars = []
    all_corrs = []
    
    for chunk in pd.read_csv(data_path, usecols=train_ids, chunksize=5000):
        X_chunk = chunk.to_numpy(dtype=np.float32).T
        all_vars.append(np.nanvar(X_chunk, axis=0))
        # Imputation rapide pour calcul de corrélation
        X_tmp = np.nan_to_num(X_chunk, nan=np.nanmean(X_chunk, axis=0))
        x_c = X_tmp - X_tmp.mean(axis=0, keepdims=True)
        num = x_c.T @ y_c
        den = np.sqrt(np.sum(x_c ** 2, axis=0)) * y_den
        all_corrs.append(np.abs(np.divide(num, den, out=np.zeros_like(num), where=den != 0)))
    
    idx_var = np.argsort(np.concatenate(all_vars))[::-1][:n_top_var]
    idx_corr = idx_var[np.argsort(np.concatenate(all_corrs)[idx_var])[::-1][:n_top_corr]]
    return np.sort(idx_corr)

def load_selected_data(data_path, sample_ids, indices):
    rows = []
    start = 0
    indices_to_load = np.sort(indices)
    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=10000):
        end = start + len(chunk)
        pos_s = np.searchsorted(indices_to_load, start)
        pos_e = np.searchsorted(indices_to_load, end)
        local = indices_to_load[pos_s:pos_e] - start
        if len(local) > 0:
            rows.append(chunk.iloc[local].values)
        start = end
    return np.vstack(rows).T.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    raw_path = data_dir / "c_sample.csv"
    
    # 1. Annotations et IDs
    annot = load_annotations(data_dir)
    sample_header = pd.read_csv(raw_path, nrows=0)
    common_ids = np.array([s for s in sample_header.columns if s in annot.index])
    y = annot.loc[common_ids, "age"].values.astype(np.float32)
    
    print(f"Démarrage Nested CV : {len(common_ids)} échantillons, {args.outer_folds} folds.")
    
    outer_kf = KFold(n_splits=args.outer_folds, shuffle=True, random_state=42)
    results = []
    
    # Imputer MICE optimisé
    mice = IterativeImputer(
        estimator=BayesianRidge(),
        n_nearest_features=100,
        max_iter=10,
        random_state=42,
        skip_complete=True
    )

    for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(common_ids)):
        t_start = time()
        print(f"\n--- Fold Externe {fold_idx + 1}/{args.outer_folds} ---")
        
        train_ids = common_ids[train_idx]
        test_ids = common_ids[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 2. Sélection de caractéristiques (Sur Train uniquement)
        indices = get_top_indices(raw_path, train_ids.tolist(), y_train)
        X_train = load_selected_data(raw_path, train_ids.tolist(), indices)
        X_test = load_selected_data(raw_path, test_ids.tolist(), indices)
        
        # 3. Optimisation ElasticNet (Inner CV)
        print(f"  Optimisation ElasticNet...")
        pipeline = Pipeline([
            ('imputer', mice),
            ('scaler', StandardScaler()),
            ('model', ElasticNet(max_iter=5000, random_state=42))
        ])
        
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0],
            'model__l1_ratio': [0.1, 0.5, 0.9]
        }
        
        search = GridSearchCV(pipeline, param_grid, cv=args.inner_folds, 
                              scoring='neg_mean_absolute_error', n_jobs=-1)
        search.fit(X_train, y_train)
        
        # 4. Évaluation sur Test
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'fold': fold_idx + 1,
            'mae': mae,
            'r2': r2,
            'best_params': search.best_params_
        })
        
        print(f"  Fold {fold_idx + 1} terminé en {time()-t_start:.1f}s. MAE: {mae:.3f}")

    # 5. Synthèse finale
    results_df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("RÉSULTATS FINAUX (NESTED CV - ELASTICNET)")
    print("="*40)
    print(f"MAE Moyenne: {results_df['mae'].mean():.3f} ± {results_df['mae'].std():.3f}")
    print(f"R2 Moyen:   {results_df['r2'].mean():.3f} ± {results_df['r2'].std():.3f}")
    
    # Sauvegarde
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "nested_cv_elasticnet_results.csv", index=False)
    print(f"\nRésultats sauvegardés dans results/nested_cv_elasticnet_results.csv")

if __name__ == "__main__":
    main()
