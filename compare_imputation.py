"""
Comparaison des méthodes d'imputation pour les données de méthylation ADN.

Ce script compare différentes stratégies d'imputation des valeurs manquantes
et évalue leur impact sur la prédiction de l'âge.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(data_dir: Path, top_k: int = 5000, chunk_size: int = 2000):
    """Charge les données et sélectionne les top-k CpG."""
    
    # Annotations
    annot = pd.read_csv(data_dir / "annot_projet.csv")
    annot = annot.dropna(subset=["age", "Sample_description"]).copy()
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    
    # CpG names
    cpg_names = pd.read_csv(data_dir / "cpg_names_projet.csv", usecols=["cpg_names"])
    cpg_names = cpg_names["cpg_names"].astype(str).tolist()
    
    # Data file
    data_file = data_dir / "c_sample.csv"
    header_cols = pd.read_csv(data_file, nrows=0).columns.tolist()
    sample_ids = [sid for sid in annot.index.tolist() if sid in header_cols]
    annot = annot.loc[sample_ids]
    y = annot["age"].astype(float).to_numpy()
    
    # Select top-k CpG by correlation with age
    print(f"Sélection des {top_k} CpG les plus corrélés avec l'âge...")
    y_centered = y - y.mean()
    y_den = np.sqrt(np.sum(y_centered**2))
    
    correlations = []
    start = 0
    for chunk in pd.read_csv(data_file, usecols=sample_ids, chunksize=chunk_size):
        x = chunk.to_numpy(dtype=np.float32, copy=False)
        if np.isnan(x).any():
            row_means = np.nanmean(x, axis=1, keepdims=True)
            x = np.where(np.isnan(x), row_means, x)
        x_centered = x - x.mean(axis=1, keepdims=True)
        num = x_centered @ y_centered
        den = np.sqrt(np.sum(x_centered**2, axis=1)) * y_den
        corr = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        correlations.extend(list(zip(range(start, start + len(corr)), np.abs(corr))))
        start += len(chunk)
    
    # Get top-k indices
    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in correlations[:top_k]]
    selected_names = [cpg_names[i] if i < len(cpg_names) else f"cpg_{i}" for i in selected_indices]
    
    # Load selected CpG data
    print("Chargement des données CpG sélectionnées...")
    indices = np.array(sorted(selected_indices))
    rows = []
    start = 0
    for chunk in pd.read_csv(data_file, usecols=sample_ids, chunksize=chunk_size):
        end = start + len(chunk)
        pos_start = np.searchsorted(indices, start)
        pos_end = np.searchsorted(indices, end)
        local = indices[pos_start:pos_end] - start
        if len(local) > 0:
            rows.append(chunk.iloc[local])
        start = end
    
    selected = pd.concat(rows, axis=0)
    selected.index = selected_names
    X = selected[sample_ids].T
    
    return X, annot["age"].astype(float), annot


def get_imputers():
    """Retourne un dictionnaire des méthodes d'imputation à comparer."""
    return {
        "Mean": SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
        "Most Frequent": SimpleImputer(strategy="most_frequent"),
        "KNN (k=5)": KNNImputer(n_neighbors=5),
        "KNN (k=10)": KNNImputer(n_neighbors=10),
        "KNN (k=20)": KNNImputer(n_neighbors=20),
        "Iterative (BayesianRidge)": IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=42
        ),
        "Iterative (ElasticNet)": IterativeImputer(
            estimator=ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
            max_iter=10,
            random_state=42
        ),
    }


def evaluate_imputation(X, y, imputer_name, imputer, test_size=0.2, random_state=42, cv=5):
    """Évalue une méthode d'imputation avec cross-validation."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Apply imputation
    start_time = perf_counter()
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    imputation_time = perf_counter() - start_time
    
    # Train model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000))
    ])
    
    model.fit(X_train_imp, y_train)
    y_pred = model.predict(X_test_imp)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_imp, y_train, cv=cv, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Missing values stats
    n_missing_before = X_train.isna().sum().sum()
    missing_rate = n_missing_before / (X_train.shape[0] * X_train.shape[1]) * 100
    
    return {
        "method": imputer_name,
        "mae_test": mae,
        "r2_test": r2,
        "mae_cv": cv_mae,
        "mae_cv_std": cv_std,
        "imputation_time_sec": imputation_time,
        "missing_rate_pct": missing_rate,
        "n_missing": n_missing_before,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare imputation methods for DNAm data.")
    parser.add_argument("--data-dir", default="Data", help="Path to data directory.")
    parser.add_argument("--output-dir", default="results", help="Path to output directory.")
    parser.add_argument("--top-k", type=int, default=5000, help="Number of CpG sites to use.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size for reading data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds.")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COMPARAISON DES MÉTHODES D'IMPUTATION")
    print("=" * 60)
    
    # Load data
    X, y, annot = load_data(data_dir, args.top_k, args.chunk_size)
    print(f"\nDonnées chargées: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"Taux de valeurs manquantes: {X.isna().mean().mean()*100:.2f}%")
    
    # Get imputers
    imputers = get_imputers()
    
    # Evaluate each imputer
    results = []
    for name, imputer in imputers.items():
        print(f"\nÉvaluation: {name}...")
        try:
            result = evaluate_imputation(
                X, y, name, imputer,
                test_size=args.test_size,
                random_state=args.random_state,
                cv=args.cv
            )
            results.append(result)
            print(f"  MAE test: {result['mae_test']:.3f} | MAE CV: {result['mae_cv']:.3f} ± {result['mae_cv_std']:.3f}")
        except Exception as e:
            print(f"  ERREUR: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values("mae_test")
    
    # Save results
    results_df.to_csv(output_dir / "imputation_comparison.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RÉSULTATS - COMPARAISON DES MÉTHODES D'IMPUTATION")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Best method
    best = results_df.iloc[0]
    print(f"\n*** MEILLEURE MÉTHODE: {best['method']} ***")
    print(f"    MAE test: {best['mae_test']:.3f} années")
    print(f"    R² test: {best['r2_test']:.4f}")
    print(f"    MAE CV: {best['mae_cv']:.3f} ± {best['mae_cv_std']:.3f}")
    print(f"    Temps d'imputation: {best['imputation_time_sec']:.2f}s")
    
    # Generate report
    report = f"""# Comparaison des Méthodes d'Imputation

## Contexte
Les données de méthylation de l'ADN contiennent des valeurs manquantes qui doivent être imputées
avant l'entraînement des modèles. Ce rapport compare différentes stratégies d'imputation.

## Données
- Échantillons: {X.shape[0]}
- Features (CpG): {X.shape[1]}
- Taux de valeurs manquantes: {X.isna().mean().mean()*100:.2f}%

## Méthodes Comparées
1. **Simple Imputer (Mean/Median/Most Frequent)**: Remplacement par statistique simple
2. **KNN Imputer**: Utilise les k plus proches voisins
3. **Iterative Imputer**: Imputation itérative basée sur un modèle

## Résultats

{results_df.to_markdown(index=False)}

## Recommandation

La méthode **{best['method']}** offre les meilleures performances avec:
- MAE: {best['mae_test']:.3f} années
- R²: {best['r2_test']:.4f}

Cette méthode est recommandée pour l'imputation des données de méthylation.
"""
    
    (output_dir / "imputation_report.md").write_text(report)
    print(f"\nRapport sauvegardé: {output_dir / 'imputation_report.md'}")


if __name__ == "__main__":
    main()
