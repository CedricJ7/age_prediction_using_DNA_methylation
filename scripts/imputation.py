#!/usr/bin/env python3
"""
Robust Imputation Pipeline (Standalone)
====================
- Preserves ALL rows even if they are entirely NaN.
"""

import gc
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

warnings.filterwarnings('ignore')

def load_cpg_data_float32(data_path: Path):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading CpG data from {data_path}...")
    header_df = pd.read_csv(data_path, nrows=0)
    all_columns = list(header_df.columns)
    
    # Read in chunks to save memory during loading
    arrays = []
    for chunk in pd.read_csv(data_path, chunksize=100000):
        arrays.append(chunk.values.astype(np.float32))
    matrix = np.vstack(arrays)
    print(f"DEBUG: Matrix shape loaded: {matrix.shape}")
    return matrix.T, all_columns

def impute_data(X: np.ndarray) -> np.ndarray:
    print(f"DEBUG: X shape before imputation: {X.shape}")
    
    # Identify columns (CpG sites) that are ALL NaN
    all_nan_cols = np.isnan(X).all(axis=0)
    num_all_nan = np.sum(all_nan_cols)
    if num_all_nan > 0:
        print(f"Found {num_all_nan} CpG sites with ALL missing values. Filling with 0.0...")
        X[:, all_nan_cols] = 0.0
        
    nan_count = int(np.isnan(X).sum())
    if nan_count == 0:
        print("No remaining missing values.")
        return X
    
    print(f"Remaining missing values: {nan_count}. Imputing with Mean...")
    # Use SimpleImputer with keep_empty_features=True to be safe
    try:
        imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
        X = imputer.fit_transform(X)
    except TypeError:
        # Older scikit-learn doesn't have keep_empty_features
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    print(f"DEBUG: X shape after imputation: {X.shape}")
    return X.astype(np.float32)

def main():
    start = datetime.now()
    data_dir = Path("Data")
    input_path = data_dir / "c_sample.csv"
    output_path = data_dir / "complete_impute_cpg.csv"

    X_cpg, column_names = load_cpg_data_float32(input_path)
    X_cpg = impute_data(X_cpg)
    
    X_output = X_cpg.T
    print(f"DEBUG: X_output shape to save: {X_output.shape}")
    
    with open(output_path, 'w') as f:
        f.write(','.join(column_names) + '\n')

    # Save in chunks to be memory efficient
    n_sites = X_output.shape[0]
    chunk_size = 100000
    for i in range(0, n_sites, chunk_size):
        end = min(i + chunk_size, n_sites)
        pd.DataFrame(X_output[i:end], columns=column_names).to_csv(
            output_path, mode='a', header=False, index=False
        )
        print(f"  Saved {end}/{n_sites} sites...")

    print(f"Done. Final shape: {X_output.shape}")
    print(f"Total time: {datetime.now() - start}")

if __name__ == "__main__":
    main()
