"""
Pipeline d'entraînement pour la prédiction d'âge basée sur la méthylation de l'ADN.

Ce script implémente plusieurs modèles de machine learning avec optimisation
des hyperparamètres pour prédire l'âge chronologique à partir des profils
de méthylation de l'ADN (données EPICv2).

Modèles implémentés:
- ElasticNet (régression linéaire régularisée)
- Random Forest (bagging)
- Gradient Boosting (XGBoost, HistGradientBoosting)
- Bagging Regressor
- Support Vector Regression
- Multi-Layer Perceptron (AltumAge-inspired)

Auteur: DNAm Age Prediction Benchmark
"""

import argparse
import heapq
import warnings
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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


def load_clock_cpgs(path: Optional[str]) -> set[str]:
    """Charge une liste de CpG prédéfinie (ex: Horvath clock)."""
    if not path:
        return set()
    cpgs = pd.read_csv(path, header=None)
    return set(cpgs.iloc[:, 0].astype(str).tolist())


# =============================================================================
# FEATURE SELECTION FUNCTIONS
# =============================================================================

def select_top_k_cpgs(
    data_path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    cpg_names: list[str],
    top_k: int,
    chunk_size: int,
) -> tuple[list[int], list[str]]:
    """
    Sélectionne les top-k CpG les plus corrélés avec l'âge.
    
    Utilise la corrélation de Pearson pour identifier les CpG
    dont les niveaux de méthylation sont les plus prédictifs de l'âge.
    """
    y_centered = y - y.mean()
    y_den = np.sqrt(np.sum(y_centered**2))
    best: list[tuple[float, int]] = []

    start = 0
    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        x = chunk.to_numpy(dtype=np.float32, copy=False)
        if np.isnan(x).any():
            row_means = np.zeros((x.shape[0], 1), dtype=x.dtype)
            valid_rows = ~np.isnan(x).all(axis=1)
            if np.any(valid_rows):
                row_means[valid_rows] = np.nanmean(x[valid_rows], axis=1, keepdims=True)
            x = np.where(np.isnan(x), row_means, x)
        x_centered = x - x.mean(axis=1, keepdims=True)
        num = x_centered @ y_centered
        den = np.sqrt(np.sum(x_centered**2, axis=1)) * y_den
        corr = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        abs_corr = np.abs(corr)

        for i, c in enumerate(abs_corr):
            idx = start + i
            if len(best) < top_k:
                heapq.heappush(best, (c, idx))
            elif c > best[0][0]:
                heapq.heapreplace(best, (c, idx))
        start += len(chunk)

    best_sorted = sorted(best, key=lambda t: t[0], reverse=True)
    indices = [idx for _, idx in best_sorted]
    names = [cpg_names[i] if i < len(cpg_names) else f"cpg_{i}" for i in indices]
    return indices, names


def load_selected_cpgs(
    data_path: Path,
    sample_ids: list[str],
    selected_indices: list[int],
    selected_names: list[str],
    chunk_size: int,
) -> pd.DataFrame:
    """Charge les données CpG sélectionnées."""
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


def compute_pca_scores_streaming(
    data_path: Path,
    sample_ids: list[str],
    n_components: int,
    chunk_size: int,
    max_missing_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les scores PCA en streaming pour gérer la haute dimensionnalité."""
    n_samples = len(sample_ids)
    gram = np.zeros((n_samples, n_samples), dtype=np.float64)

    for chunk in pd.read_csv(data_path, usecols=sample_ids, chunksize=chunk_size):
        x = chunk.to_numpy(dtype=np.float32, copy=False)

        if max_missing_rate > 0:
            missing_rate = np.isnan(x).mean(axis=1)
            keep = missing_rate <= max_missing_rate
            x = x[keep]

        if x.size == 0:
            continue

        if np.isnan(x).any():
            row_means = np.zeros((x.shape[0], 1), dtype=x.dtype)
            valid_rows = ~np.isnan(x).all(axis=1)
            if np.any(valid_rows):
                row_means[valid_rows] = np.nanmean(x[valid_rows], axis=1, keepdims=True)
            x = np.where(np.isnan(x), row_means, x)

        x = x - x.mean(axis=1, keepdims=True)
        gram += x.T @ x

    eigvals, eigvecs = np.linalg.eigh(gram)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = min(n_components, n_samples)
    eigvals_k = np.maximum(eigvals[:k], 0.0)
    scores = eigvecs[:, :k] * np.sqrt(eigvals_k)
    explained = eigvals_k / eigvals.sum() if eigvals.sum() > 0 else np.zeros(k)
    return scores, explained


def filter_cpgs_by_missing_rate(matrix: pd.DataFrame, max_missing_rate: float) -> pd.DataFrame:
    """Filtre les CpG avec trop de valeurs manquantes."""
    if max_missing_rate <= 0:
        return matrix
    missing_rate = matrix.isna().mean(axis=1)
    kept = missing_rate <= max_missing_rate
    return matrix.loc[kept]


def clean_ethnicity(ethnicity: str) -> str:
    """
    Nettoie et regroupe les catégories d'ethnicité.
    
    Regroupe 'Unavailable', 'Declined', 'Other' en 'Inconnu'.
    """
    if pd.isna(ethnicity):
        return "Inconnu"
    eth = str(ethnicity).strip()
    if eth.lower() in ["unavailable", "declined", "other", ""]:
        return "Inconnu"
    return eth


def add_demographic_features(annot: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les features démographiques (genre et ethnicité).
    
    - Genre: encodage binaire (1=Femme, 0=Homme)
    - Ethnicité: encodage one-hot (avec regroupement des catégories rares)
    """
    demo_features = pd.DataFrame(index=annot.index)
    
    if "female" in annot.columns:
        demo_features["is_female"] = annot["female"].apply(
            lambda x: 1 if str(x).lower() == "true" else (0 if str(x).lower() == "false" else np.nan)
        )
        print(f"  → Genre: {demo_features['is_female'].notna().sum()} valeurs valides")
    
    if "ethnicity" in annot.columns:
        # Nettoyer et regrouper les catégories d'ethnicité
        ethnicity_clean = annot["ethnicity"].apply(clean_ethnicity)
        print(f"  → Ethnicité (après nettoyage): {ethnicity_clean.value_counts().to_dict()}")
        
        ethnicity_dummies = pd.get_dummies(ethnicity_clean, prefix="eth", dummy_na=False)
        demo_features = pd.concat([demo_features, ethnicity_dummies], axis=1)
        print(f"  → Features ethnicité: {list(ethnicity_dummies.columns)}")
    
    return demo_features


# =============================================================================
# MODEL BUILDING FUNCTIONS
# =============================================================================

def get_param_distributions():
    """
    Retourne les distributions de paramètres pour l'optimisation bayésienne.
    
    Ces distributions sont utilisées par RandomizedSearchCV pour explorer
    efficacement l'espace des hyperparamètres.
    """
    return {
        "ElasticNet": {
            "model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "Ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10],
        },
        "HistGradientBoosting": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_leaf": [10, 20, 30],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
        "XGBoost": {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8, 10],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [1.0, 2.0, 5.0],
        },
        "Bagging": {
            "n_estimators": [10, 20, 50],
            "max_samples": [0.5, 0.7, 1.0],
            "max_features": [0.5, 0.7, 1.0],
        },
        "SVR": {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__epsilon": [0.01, 0.1, 0.5],
            "model__gamma": ["scale", "auto"],
        },
        "MLP": {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64), (64, 32, 16)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01],
        },
    }


def build_models(optimize: bool = False) -> tuple[list[tuple[str, object]], list[str]]:
    """
    Construit la liste des modèles à entraîner.
    
    Inclut:
    - Modèles linéaires régularisés (ElasticNet, Ridge)
    - Méthodes d'ensemble (Random Forest, Bagging)
    - Boosting (Gradient Boosting, XGBoost, HistGradientBoosting)
    - Deep Learning (MLP)
    """
    models: list[tuple[str, object]] = []
    skipped: list[str] = []

    # ElasticNet - Régression linéaire avec régularisation L1+L2
    models.append((
        "ElasticNet",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000, tol=1e-4)),
        ]),
    ))

    # Ridge - Régression linéaire avec régularisation L2
    models.append((
        "Ridge",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
    ))

    # Random Forest - Bagging d'arbres de décision
    models.append((
        "RandomForest",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
    ))

    # Bagging avec ElasticNet comme estimateur de base
    models.append((
        "BaggingElasticNet",
        BaggingRegressor(
            estimator=Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000)),
            ]),
            n_estimators=20,
            max_samples=0.8,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        ),
    ))

    # Gradient Boosting (sklearn)
    models.append((
        "GradientBoosting",
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_split=5,
            random_state=42,
        ),
    ))

    # HistGradientBoosting (plus rapide, gère les NaN)
    models.append((
        "HistGradientBoosting",
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_depth=10,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        ),
    ))

    # XGBoost
    try:
        from xgboost import XGBRegressor
        models.append((
            "XGBoost",
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=2.0,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=42,
            ),
        ))
    except ImportError:
        skipped.append("XGBoost")

    # MLP (AltumAge-inspired)
    models.append((
        "AltumAge",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            )),
        ]),
    ))

    return models, skipped


def optimize_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_dist: dict,
    cv: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
) -> tuple[object, dict, float]:
    """
    Optimise les hyperparamètres d'un modèle avec RandomizedSearchCV.
    
    Returns:
        best_model: Modèle avec les meilleurs paramètres
        best_params: Dictionnaire des meilleurs paramètres
        best_score: Meilleur score CV (MAE négatif)
    """
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv_splitter,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    
    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_, -search.best_score_


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: int = 5,
) -> dict:
    """
    Évalue un modèle sur les données de test et en cross-validation.
    
    Métriques calculées:
    - MAE (Mean Absolute Error)
    - MAD (Median Absolute Deviation)
    - RMSE (Root Mean Squared Error)
    - R² (Coefficient of Determination)
    - Corrélation de Pearson
    - Scores CV
    """
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métriques sur le test set
    mae = mean_absolute_error(y_test, y_pred_test)
    mad = float(np.median(np.abs(y_test - y_pred_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    corr, _ = stats.pearsonr(y_test, y_pred_test)
    
    # Métriques sur le train set (pour détecter l'overfitting)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
    except Exception:
        cv_mae = np.nan
        cv_std = np.nan
    
    return {
        "mae": mae,
        "mad": mad,
        "rmse": rmse,
        "r2": r2,
        "correlation": corr,
        "mae_train": mae_train,
        "r2_train": r2_train,
        "cv_mae": cv_mae,
        "cv_std": cv_std,
        "overfitting_ratio": mae / mae_train if mae_train > 0 else np.nan,
    }


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_model(output_dir: Path, name: str, model) -> None:
    """Sauvegarde un modèle entraîné."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    
    if name == "XGBoost":
        model.save_model(str(output_dir / f"{safe_name}.json"))
        return
    
    try:
        import joblib
        joblib.dump(model, output_dir / f"{safe_name}.joblib")
    except Exception as exc:
        print(f"  ⚠ Could not save {name}: {exc}")


def save_plots(output_dir: Path, y_true, y_pred, model_name: str) -> None:
    """Génère les plots de diagnostic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["error"] = df["y_pred"] - df["y_true"]

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    sns.scatterplot(data=df, x="y_true", y="y_pred", s=25, alpha=0.7, ax=ax1)
    min_val = min(df["y_true"].min(), df["y_pred"].min())
    max_val = max(df["y_true"].max(), df["y_pred"].max())
    ax1.plot([min_val, max_val], [min_val, max_val], "--", color="red", linewidth=1, label="Idéal")
    ax1.set_title(f"Prédictions vs Âge réel — {model_name}")
    ax1.set_xlabel("Âge réel (années)")
    ax1.set_ylabel("Âge prédit (années)")
    ax1.legend()

    ax2 = axes[1]
    sns.histplot(df["error"], bins=30, kde=True, color="#4C72B0", ax=ax2)
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax2.set_title(f"Distribution des erreurs — {model_name}")
    ax2.set_xlabel("Erreur (prédit - réel)")
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"diagnostic_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def generate_summary_report(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    run_time: float,
) -> str:
    """Génère un rapport markdown de synthèse."""
    
    best = metrics_df.iloc[0]
    
    report = f"""# Rapport d'Entraînement — Prédiction d'Âge DNAm

## Configuration

| Paramètre | Valeur |
|-----------|--------|
| Mode de features | {args.feature_mode} |
| Top-k CpG | {args.top_k if args.feature_mode == 'topk' else 'N/A'} |
| Composantes PCA | {args.pca_components if args.feature_mode == 'pca' else 'N/A'} |
| Taux missing max | {args.max_missing_rate} |
| Test size | {args.test_size} |
| Random state | {args.random_state} |
| Optimisation | {'Oui' if args.optimize else 'Non'} |
| Temps total | {run_time:.2f} secondes |

## Résultats

### Classement des Modèles (par MAE)

{metrics_df[['model', 'mae', 'mad', 'r2', 'correlation', 'cv_mae']].to_markdown(index=False, floatfmt='.3f')}

### Meilleur Modèle: {best['model']}

- **MAE**: {best['mae']:.2f} années
- **MAD**: {best['mad']:.2f} années
- **R²**: {best['r2']:.4f}
- **Corrélation**: {best['correlation']:.4f}
- **MAE CV**: {best['cv_mae']:.2f} ± {best['cv_std']:.2f}

### Analyse de l'Overfitting

| Modèle | MAE Train | MAE Test | Ratio |
|--------|-----------|----------|-------|
"""
    
    for _, row in metrics_df.iterrows():
        report += f"| {row['model']} | {row['mae_train']:.2f} | {row['mae']:.2f} | {row['overfitting_ratio']:.2f} |\n"
    
    report += """
## Interprétation

- Un ratio proche de 1.0 indique un bon équilibre biais-variance
- Un ratio > 1.5 suggère de l'overfitting (ajouter de la régularisation)
- Un ratio < 1.0 est inhabituel (vérifier les données)

## Fichiers Générés

- `metrics.csv` : Métriques détaillées de tous les modèles
- `predictions.csv` : Prédictions sur le test set
- `annot_predictions.csv` : Annotations avec prédictions pour tous les modèles
- `models/` : Modèles sauvegardés
- `plots/` : Graphiques de diagnostic
"""
    
    return report


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement des modèles de prédiction d'âge DNAm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="Data", help="Répertoire des données")
    parser.add_argument("--output-dir", default="results", help="Répertoire de sortie")
    parser.add_argument("--top-k", type=int, default=10000, help="Nombre de CpG à sélectionner")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Taille des chunks de lecture")
    parser.add_argument("--max-missing-rate", type=float, default=0.05, help="Taux max de valeurs manquantes")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion du test set")
    parser.add_argument("--random-state", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--clock-cpgs", default=None, help="Fichier de CpG prédéfinis")
    parser.add_argument("--pca-components", type=int, default=400, help="Composantes PCA")
    parser.add_argument("--feature-mode", choices=["topk", "pca"], default="topk", help="Mode de sélection des features")
    parser.add_argument("--optimize", action="store_true", help="Activer l'optimisation des hyperparamètres")
    parser.add_argument("--cv", type=int, default=5, help="Nombre de folds pour la CV")
    parser.add_argument("--n-iter", type=int, default=20, help="Itérations pour RandomizedSearchCV")
    args = parser.parse_args()
    
    run_start = perf_counter()
    
    print("=" * 70)
    print("PIPELINE DE PRÉDICTION D'ÂGE PAR MÉTHYLATION DE L'ADN")
    print("=" * 70)
    
    # Validation
    if args.feature_mode == "topk" and args.top_k <= 0 and not args.clock_cpgs:
        raise ValueError("Avec feature-mode=topk, définir --top-k > 0 ou --clock-cpgs")
    if args.feature_mode == "pca" and args.pca_components <= 0:
        raise ValueError("Avec feature-mode=pca, définir --pca-components > 0")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # ÉTAPE 1: Chargement des données
    # ==========================================================================
    print("\n[1/5] Chargement des annotations...")
    annot = load_annotations(data_dir)
    sample_ids = annot.index.tolist()
    y = annot["age"].astype(float).to_numpy()
    print(f"  → {len(sample_ids)} échantillons chargés")

    data_file = data_dir / "c_sample.csv"
    header_cols = pd.read_csv(data_file, nrows=0).columns.tolist()
    sample_ids = [sid for sid in sample_ids if sid in header_cols]
    annot = annot.loc[sample_ids]
    y = annot["age"].astype(float).to_numpy()
    print(f"  → {len(sample_ids)} échantillons avec données de méthylation")

    cpg_names = load_cpg_names(data_dir)
    clock_cpgs = load_clock_cpgs(args.clock_cpgs)

    # ==========================================================================
    # ÉTAPE 2: Préparation des features
    # ==========================================================================
    print("\n[2/5] Préparation des features...")
    
    # Features démographiques
    print("  Ajout des features démographiques:")
    demo_features = add_demographic_features(annot)

    if args.feature_mode == "topk":
        print(f"  Sélection des top-{args.top_k} CpG par corrélation...")
        if clock_cpgs:
            selected_indices = [i for i, name in enumerate(cpg_names) if name in clock_cpgs]
            selected_names = [cpg_names[i] for i in selected_indices]
            if not selected_indices:
                print("  ⚠ Aucun CpG du fichier clock trouvé, fallback sur corrélation")
                selected_indices, selected_names = select_top_k_cpgs(
                    data_file, sample_ids, y, cpg_names, args.top_k, args.chunk_size
                )
        else:
            selected_indices, selected_names = select_top_k_cpgs(
                data_file, sample_ids, y, cpg_names, args.top_k, args.chunk_size
            )

        pd.DataFrame({"cpg": selected_names}).to_csv(output_dir / "selected_cpgs.csv", index=False)
        print(f"  → {len(selected_names)} CpG sélectionnés")

        selected_matrix = load_selected_cpgs(
            data_file, sample_ids, selected_indices, selected_names, args.chunk_size
        )
        selected_matrix = filter_cpgs_by_missing_rate(selected_matrix, args.max_missing_rate)
        X = selected_matrix[sample_ids].T
        y_series = annot["age"].astype(float)
        X = X.loc[y_series.index]

        # Ajout des features démographiques
        X = pd.concat([X, demo_features.loc[X.index]], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_series, test_size=args.test_size, random_state=args.random_state, shuffle=True
        )

        # Imputation KNN (k=5) - meilleure performance que la moyenne
        imputer = KNNImputer(n_neighbors=5)
        X_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)
        print(f"  → Imputation KNN (k=5) appliquée")

    else:  # PCA mode
        print(f"  Calcul PCA avec {args.pca_components} composantes...")
        scores, explained = compute_pca_scores_streaming(
            data_file, sample_ids, args.pca_components, args.chunk_size, args.max_missing_rate
        )
        X = pd.DataFrame(scores, index=sample_ids, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
        y_series = annot["age"].astype(float)
        X = X.loc[y_series.index]

        # Ajout des features démographiques
        X = pd.concat([X, demo_features.loc[X.index]], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_series, test_size=args.test_size, random_state=args.random_state, shuffle=True
        )

        pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(scores.shape[1])],
            "explained_variance_ratio": explained,
        }).to_csv(output_dir / "pca_variance.csv", index=False)

    print(f"  → Features totales: {X_train.shape[1]}")
    print(f"  → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ==========================================================================
    # ÉTAPE 3: Construction et entraînement des modèles
    # ==========================================================================
    print("\n[3/5] Entraînement des modèles...")
    
    models, skipped = build_models(optimize=args.optimize)
    param_dists = get_param_distributions()
    
    if skipped:
        print(f"  ⚠ Modèles ignorés (dépendances manquantes): {skipped}")

    metrics_rows = []
    predictions_rows = []
    trained_models = []

    for name, model in models:
        print(f"\n  → {name}...")
        start_fit = perf_counter()
        
        # Optimisation optionnelle
        if args.optimize and name in param_dists:
            print(f"    Optimisation des hyperparamètres...")
            model, best_params, best_cv_score = optimize_model(
                name, model, X_train, y_train,
                param_dists[name], cv=args.cv, n_iter=args.n_iter
            )
            print(f"    Meilleurs params: {best_params}")
            print(f"    MAE CV: {best_cv_score:.3f}")
        else:
            model.fit(X_train, y_train)
        
        fit_time = perf_counter() - start_fit
        
        # Évaluation
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, cv=args.cv)
        metrics["model"] = name
        metrics["fit_time_sec"] = fit_time
        metrics["n_features"] = X_train.shape[1]
        metrics["n_train"] = X_train.shape[0]
        metrics["n_test"] = X_test.shape[0]
        metrics_rows.append(metrics)
        
        print(f"    MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.4f} | Corr: {metrics['correlation']:.4f}")
        
        # Prédictions
        preds_test = model.predict(X_test)
        for sample_id, y_true_val, y_pred_val in zip(X_test.index.tolist(), y_test.tolist(), preds_test.tolist()):
            predictions_rows.append({
                "model": name,
                "sample_id": sample_id,
                "y_true": y_true_val,
                "y_pred": y_pred_val,
            })
        
        # Sauvegarde
        save_model(output_dir / "models", name, model)
        trained_models.append((name, model))

    # ==========================================================================
    # ÉTAPE 4: Sauvegarde des résultats
    # ==========================================================================
    print("\n[4/5] Sauvegarde des résultats...")
    
    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="mae")
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(predictions_rows).to_csv(output_dir / "predictions.csv", index=False)

    # Annotations avec prédictions pour tous les modèles
    split_info = pd.Series("non_test", index=X_train.index)
    split_info = split_info.reindex(y_series.index)
    split_info.loc[X_test.index] = "test"

    annot_rows = []
    for model_name, model in trained_models:
        model_preds_train = model.predict(X_train)
        model_preds_test = model.predict(X_test)

        all_preds = pd.Series(index=y_series.index, dtype=float)
        all_preds.loc[X_train.index] = model_preds_train
        all_preds.loc[X_test.index] = model_preds_test

        annot_model = annot.copy()
        annot_model["split"] = split_info
        annot_model["age_pred"] = all_preds
        annot_model["model"] = model_name
        annot_rows.append(annot_model)

    annot_all = pd.concat(annot_rows, axis=0)
    annot_all.to_csv(output_dir / "annot_predictions.csv", index=True, index_label="Sample_description")

    # Plots pour le meilleur modèle
    best_model_name = metrics_df.iloc[0]["model"]
    best_preds = pd.DataFrame(predictions_rows)
    best_preds = best_preds[best_preds["model"] == best_model_name]
    save_plots(output_dir, best_preds["y_true"], best_preds["y_pred"], best_model_name)

    # Coefficients pour modèles linéaires
    for name, model in trained_models:
        if name in {"ElasticNet", "Ridge"}:
            try:
                linear_model = model.named_steps["model"]
                coef = pd.DataFrame({
                    "feature": X_train.columns,
                    "coef": linear_model.coef_
                }).assign(abs_coef=lambda df: df["coef"].abs())
                coef = coef.sort_values("abs_coef", ascending=False).head(100)
                coef.to_csv(output_dir / f"coefficients_{name.lower()}.csv", index=False)
            except Exception:
                pass

    # ==========================================================================
    # ÉTAPE 5: Génération du rapport
    # ==========================================================================
    print("\n[5/5] Génération du rapport...")
    
    run_time = perf_counter() - run_start
    report = generate_summary_report(metrics_df, output_dir, args, run_time)
    (output_dir / "report.md").write_text(report)
    
    if skipped:
        (output_dir / "skipped_models.txt").write_text(
            "Modèles ignorés (dépendances manquantes):\n" + "\n".join(skipped)
        )

    # Résumé final
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"\nTemps total: {run_time:.2f} secondes\n")
    print(metrics_df[["model", "mae", "r2", "correlation", "cv_mae"]].to_string(index=False))
    print(f"\n✓ Meilleur modèle: {best_model_name} (MAE = {metrics_df.iloc[0]['mae']:.2f} ans)")
    print(f"✓ Résultats sauvegardés dans: {output_dir}")


if __name__ == "__main__":
    main()
