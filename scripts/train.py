#!/usr/bin/env python3
"""
Entraînement final du modèle sans Data Leakage.
==============================================
Utilise les données brutes et effectue le split AVANT l'imputation et la sélection.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

def get_top_features(data_path, train_ids, y_train, n_var=10000, n_corr=2000):
    """Sélection de features sur le train uniquement."""
    print(f"Sélection des {n_corr} meilleures features sur le train...")
    y_c = y_train - y_train.mean()
    y_den = np.sqrt(np.sum(y_c ** 2))
    
    vars_list = []
    corrs_list = []
    
    for chunk in pd.read_csv(data_path, usecols=train_ids, chunksize=5000):
        X_chunk = chunk.to_numpy(dtype=np.float32).T
        vars_list.append(np.nanvar(X_chunk, axis=0))
        # Imputation rapide pour calcul de corrélation
        X_tmp = np.nan_to_num(X_chunk, nan=np.nanmean(X_chunk, axis=0))
        x_c = X_tmp - X_tmp.mean(axis=0, keepdims=True)
        num = x_c.T @ y_c
        den = np.sqrt(np.sum(x_c ** 2, axis=0)) * y_den
        corrs_list.append(np.abs(np.divide(num, den, out=np.zeros_like(num), where=den != 0)))
        
    idx_var = np.argsort(np.concatenate(vars_list))[::-1][:n_var]
    corrs = np.concatenate(corrs_list)
    idx_corr = idx_var[np.argsort(corrs[idx_var])[::-1][:n_corr]]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--output', type=Path, default=Path('results/models/best_model_robust.joblib'))
    args = parser.parse_args()

    # 1. Chargement annotations et IDs
    annot = pd.read_csv(args.data_dir / "annot_projet.csv").dropna(subset=["age", "Sample_description"])
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    
    raw_path = args.data_dir / "c_sample.csv"
    sample_header = pd.read_csv(raw_path, nrows=0)
    common_ids = [s for s in sample_header.columns if s in annot.index]
    y = annot.loc[common_ids, "age"].values.astype(np.float32)

    # 2. Split AVANT toute transformation
    X_ids_train, X_ids_test, y_train, y_test = train_test_split(
        common_ids, y, test_size=0.2, random_state=42
    )

    # 3. Sélection de features sur Train
    top_idx = get_top_features(raw_path, X_ids_train, y_train)
    
    # 4. Chargement des matrices
    X_train = load_matrix(raw_path, X_ids_train, top_idx)
    X_test = load_matrix(raw_path, X_ids_test, top_idx)

    # 5. Pipeline d'entraînement
    # On utilise SimpleImputer ici, mais on pourrait utiliser MICE avec les paramètres optimisés
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42))
    ])

    print("Entraînement du modèle...")
    pipeline.fit(X_train, y_train)
    
    # 6. Évaluation
    preds = pipeline.predict(X_test)
    print(f"MAE sur Test: {mean_absolute_error(y_test, preds):.4f}")
    print(f"R2 sur Test: {r2_score(y_test, preds):.4f}")

    # Sauvegarde
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"Modèle sauvegardé dans {args.output}")

if __name__ == "__main__":
    main()
