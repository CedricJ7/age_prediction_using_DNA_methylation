#!/usr/bin/env python3
"""
Validate Best Model (Standalone)
==============================================
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
import joblib

# =============================================================================
# HELPERS
# =============================================================================

def load_annotations(data_dir: Path) -> pd.DataFrame:
    annot = pd.read_csv(data_dir / "annot_projet.csv")
    annot = annot.dropna(subset=["age", "Sample_description"]).copy()
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    return annot

def load_imputed_data(data_path: Path, sample_ids: list, chunk_size: int = 2000) -> pd.DataFrame:
    rows = []
    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        rows.append(chunk)
    return pd.concat(rows, axis=0)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Model (Standalone)")
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--model-path', type=Path, help='Path to model joblib')
    args = parser.parse_args()

    # 1. Load Data
    annot = load_annotations(args.data_dir)
    y = annot['age'].values
    sample_ids = annot.index.tolist()

    data_files = list(args.data_dir.glob("*impute*.csv"))
    if not data_files:
        raise FileNotFoundError("No imputed data file found.")
    data_path = data_files[0]
    
    X_df = load_imputed_data(data_path, sample_ids)
    X = X_df.T.values

    # 2. Load or Create Model
    if args.model_path and args.model_path.exists():
        model = joblib.load(args.model_path)
    else:
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

    # 3. Validate
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    print(f"Mean MAE: {-np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

if __name__ == "__main__":
    main()
