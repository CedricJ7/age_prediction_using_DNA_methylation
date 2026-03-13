"""
Evaluation des performances sur données imputées.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(data_dir: Path, sample_ids: list, chunk_size: int = 2000):
    """Charge les données CpG depuis le fichier imputé."""
    data_files = list(data_dir.glob("*impute*.csv"))
    if not data_files:
        raise FileNotFoundError("No imputed data file found.")
    data_path = data_files[0]
    
    rows = []
    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        rows.append(chunk)
    
    return pd.concat(rows, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate on imputed DNAm data.")
    parser.add_argument("--data-dir", default="Data", help="Path to data directory.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio.")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Annotations
    annot = pd.read_csv(data_dir / "annot_projet.csv")
    annot = annot.dropna(subset=["age", "Sample_description"]).copy()
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    
    y = annot["age"].astype(float).values
    sample_ids = annot.index.tolist()
    
    # Load data
    X_df = load_data(data_dir, sample_ids)
    X = X_df.T.values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    # Train model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.3f}")
    print(f"R2: {r2:.3f}")

if __name__ == "__main__":
    main()
