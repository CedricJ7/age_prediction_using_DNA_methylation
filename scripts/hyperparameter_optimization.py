#!/usr/bin/env python3
"""
Optimisation des hyperparamètres sans Data Leakage.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna

def get_top_features(data_path, train_ids, y_train, n_var=10000, n_corr=1000):
    y_c = y_train - y_train.mean()
    y_den = np.sqrt(np.sum(y_c ** 2))
    vars_list = []
    corrs_list = []
    for chunk in pd.read_csv(data_path, usecols=train_ids, chunksize=5000):
        X_chunk = chunk.to_numpy(dtype=np.float32).T
        vars_list.append(np.nanvar(X_chunk, axis=0))
        X_tmp = np.nan_to_num(X_chunk, nan=np.nanmean(X_chunk, axis=0))
        x_c = X_tmp - X_tmp.mean(axis=0, keepdims=True)
        num = x_c.T @ y_c
        den = np.sqrt(np.sum(x_c ** 2, axis=0)) * y_den
        corrs_list.append(np.abs(np.divide(num, den, out=np.zeros_like(num), where=den != 0)))
    idx_var = np.argsort(np.concatenate(vars_list))[::-1][:n_var]
    idx_corr = idx_var[np.argsort(np.concatenate(corrs_list)[idx_var])[::-1][:n_corr]]
    return np.sort(idx_corr)

def load_matrix(data_path, sample_ids, indices):
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

def objective(trial, X_train, y_train):
    alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=5000))
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--n-trials', type=int, default=30)
    args = parser.parse_args()

    annot = pd.read_csv(args.data_dir / "annot_projet.csv").dropna(subset=["age", "Sample_description"])
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    
    raw_path = args.data_dir / "c_sample.csv"
    sample_header = pd.read_csv(raw_path, nrows=0)
    common_ids = [s for s in sample_header.columns if s in annot.index]
    y = annot.loc[common_ids, "age"].values.astype(np.float32)

    X_ids_train, X_ids_test, y_train, y_test = train_test_split(
        common_ids, y, test_size=0.2, random_state=42
    )

    top_idx = get_top_features(raw_path, X_ids_train, y_train)
    X_train = load_matrix(raw_path, X_ids_train, top_idx)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, X_train, y_train), n_trials=args.n_trials)

    print(f"Meilleurs paramètres : {study.best_params}")
    results_dir = Path("results/optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([study.best_params]).to_csv(results_dir / "best_params_robust.csv", index=False)

if __name__ == "__main__":
    main()
