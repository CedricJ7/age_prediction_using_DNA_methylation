#!/usr/bin/env python3
"""
Optimisation de l'imputation MICE (IterativeImputer) sans Data Leakage.
Compare les configurations optimisées à la configuration par défaut.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir: Path):
    annot = pd.read_csv(data_dir / "annot_projet.csv")
    annot = annot.dropna(subset=["age", "Sample_description"]).copy()
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    
    raw_path = data_dir / "c_sample.csv"
    sample_header = pd.read_csv(raw_path, nrows=0)
    common_ids = [s for s in sample_header.columns if s in annot.index]
    y = annot.loc[common_ids, "age"].values.astype(np.float32)
    return raw_path, common_ids, y

def get_top_features_train(data_path, train_ids, y_train, n_var=5000, n_corr=500):
    """Sélection supervisée sur le train uniquement."""
    print(f"  Sélection de {n_corr} features sur le train...")
    vars_list = []
    corrs_list = []
    y_c = y_train - y_train.mean()
    y_den = np.sqrt(np.sum(y_c ** 2))

    for chunk in pd.read_csv(data_path, usecols=train_ids, chunksize=5000):
        X_chunk = chunk.to_numpy(dtype=np.float32).T
        vars_list.append(np.nanvar(X_chunk, axis=0))
        X_tmp = np.nan_to_num(X_chunk, nan=np.nanmean(X_chunk, axis=0))
        x_c = X_tmp - X_tmp.mean(axis=0, keepdims=True)
        num = x_c.T @ y_c
        den = np.sqrt(np.sum(x_c ** 2, axis=0)) * y_den
        corrs_list.append(np.abs(np.divide(num, den, out=np.zeros_like(num), where=den != 0)))

    variances = np.concatenate(vars_list)
    correlations = np.concatenate(corrs_list)
    
    idx_var = np.argsort(variances)[::-1][:n_var]
    idx_corr = idx_var[np.argsort(correlations[idx_var])[::-1][:n_corr]]
    return np.sort(idx_corr)

def load_selected_matrix(data_path, sample_ids, indices):
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
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    raw_path, common_ids, y = load_data(data_dir)
    
    X_ids_train, X_ids_test, y_train, y_test = train_test_split(
        common_ids, y, test_size=0.2, random_state=42
    )
    
    top_idx = get_top_features_train(raw_path, X_ids_train, y_train)
    X_train_raw = load_selected_matrix(raw_path, X_ids_train, top_idx)
    X_test_raw = load_selected_matrix(raw_path, X_ids_test, top_idx)
    
    print(f"Matrices chargées. NaNs dans train: {np.isnan(X_train_raw).sum()}")

    # Grille de test pour MICE
    configs = [
        # 1. Configuration par défaut (Scikit-learn)
        {'name': 'MICE Default (BayesianRidge)', 'params': {'max_iter': 10, 'random_state': 42}},
        
        # 2. Configurations optimisées (n_nearest_features)
        {'name': 'MICE + BR + 50 neighbors', 'params': {'estimator': BayesianRidge(), 'n_nearest_features': 50, 'max_iter': 10, 'random_state': 42}},
        {'name': 'MICE + BR + 100 neighbors', 'params': {'estimator': BayesianRidge(), 'n_nearest_features': 100, 'max_iter': 10, 'random_state': 42}},
        
        # 3. Changement d'estimateur interne
        {'name': 'MICE + DecisionTree + 50 neighbors', 'params': {'estimator': DecisionTreeRegressor(max_features='sqrt', random_state=42), 'n_nearest_features': 50, 'max_iter': 10, 'random_state': 42}},
        {'name': 'MICE + DecisionTree + 100 neighbors', 'params': {'estimator': DecisionTreeRegressor(max_features='sqrt', random_state=42), 'n_nearest_features': 100, 'max_iter': 10, 'random_state': 42}},
        
        {'name': 'MICE + KNN + 50 neighbors', 'params': {'estimator': KNeighborsRegressor(n_neighbors=5), 'n_nearest_features': 50, 'max_iter': 10, 'random_state': 42}},
        {'name': 'MICE + KNN + 100 neighbors', 'params': {'estimator': KNeighborsRegressor(n_neighbors=5), 'n_nearest_features': 100, 'max_iter': 10, 'random_state': 42}},
    ]
    
    best_mae = float('inf')
    best_config = {}

    print("\n" + "="*80)
    print(f"{'Configuration':<40} | {'Nearest':<8} | {'MAE':<8} | {'Time':<8}")
    print("-" * 80)

    for cfg in configs:
        t0 = time()
        mice = IterativeImputer(**cfg['params'])
        
        try:
            X_train_imp = mice.fit_transform(X_train_raw)
            X_test_imp = mice.transform(X_test_raw)
            
            # Évaluation rapide avec ElasticNet
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            model.fit(X_train_imp, y_train)
            preds = model.predict(X_test_imp)
            mae = mean_absolute_error(y_test, preds)
            
            duration = time() - t0
            n_near = cfg['params'].get('n_nearest_features', 'All')
            print(f"{cfg['name']:<40} | {str(n_near):<8} | {mae:<8.3f} | {duration:<8.1f}s")
            
            if mae < best_mae:
                best_mae = mae
                best_config = {
                    'config_name': cfg['name'],
                    'mae': mae,
                    'params': str(cfg['params'])
                }
        except Exception as e:
            print(f"{cfg['name']:<40} | Error    | -")

    print("="*80)
    print(f"\nMEILLEURE CONFIGURATION MICE :")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    # Sauvegarde des résultats
    res_df = pd.DataFrame([best_config])
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(results_dir / "best_mice_config.csv", index=False)

if __name__ == "__main__":
    main()
