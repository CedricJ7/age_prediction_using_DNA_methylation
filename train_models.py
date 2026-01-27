# from numba import njit
# pour optimiser les performances, on peut utiliser numba pour les fonctions critiques

# (le modèle basé sur 200 caractéristique post pca est meilleur que 10000 ktop)
import argparse
import heapq
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def load_annotations(data_dir: Path) -> pd.DataFrame:
    annot = pd.read_csv(data_dir / "annot_projet.csv")
    annot = annot.dropna(subset=["age", "Sample_description"]).copy()
    annot["Sample_description"] = annot["Sample_description"].astype(str)
    annot = annot.set_index("Sample_description")
    return annot


def load_cpg_names(data_dir: Path) -> list[str]:
    cpg = pd.read_csv(data_dir / "cpg_names_projet.csv", usecols=["cpg_names"])
    return cpg["cpg_names"].astype(str).tolist()


def load_clock_cpgs(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    cpgs = pd.read_csv(path, header=None)
    return set(cpgs.iloc[:, 0].astype(str).tolist())


def select_top_k_cpgs(
    data_path: Path,
    sample_ids: list[str],
    y: np.ndarray,
    cpg_names: list[str],
    top_k: int,
    chunk_size: int,
) -> tuple[list[int], list[str]]:
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


def filter_cpgs_by_missing_rate(
    matrix: pd.DataFrame, max_missing_rate: float
) -> pd.DataFrame:
    if max_missing_rate <= 0:
        return matrix
    missing_rate = matrix.isna().mean(axis=1)
    kept = missing_rate <= max_missing_rate
    return matrix.loc[kept]


def save_model(output_dir: Path, name: str, model) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    if name == "TabNet":
        model.save_model(str(output_dir / f"{safe_name}.zip"))
        return
    if name == "XGBoost":
        model.save_model(str(output_dir / f"{safe_name}.json"))
        return
    try:
        import joblib

        joblib.dump(model, output_dir / f"{safe_name}.joblib")
    except Exception as exc:
        print(f"Could not save {name}: {exc}")


def save_plots(output_dir: Path, y_true, y_pred, model_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["error"] = df["y_pred"] - df["y_true"]

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="y_true", y="y_pred", s=25, alpha=0.7)
    min_val = min(df["y_true"].min(), df["y_pred"].min())
    max_val = max(df["y_true"].max(), df["y_pred"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="gray", linewidth=1)
    plt.title(f"Predictions vs âge réel — {model_name}")
    plt.xlabel("Âge réel")
    plt.ylabel("Âge prédit")
    plt.tight_layout()
    plt.savefig(plots_dir / f"scatter_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["error"], bins=30, kde=True, color="#4C72B0")
    plt.title(f"Distribution des erreurs — {model_name}")
    plt.xlabel("Erreur (prédit - réel)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"errors_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def build_models() -> tuple[list[tuple[str, object]], list[str]]:
    models: list[tuple[str, object]] = []
    skipped: list[str] = []

    models.append(
        (
            "ElasticNet",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)),
                ]
            ),
        )
    )
    models.append(("RandomForest", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)))
    models.append(
        (
            "AltumAge",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(32, 32, 32, 32, 32),
                            activation="relu",
                            max_iter=500,
                            early_stopping=True,
                        ),
                    ),
                ]
            ),
        )
    )

    try:
        from xgboost import XGBRegressor

        models.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        )
    except Exception:
        skipped.append("XGBoost")

    return models, skipped


def optimize_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_iter: int,
    cv: int,
) -> tuple[list[tuple[str, object]], pd.DataFrame]:
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    candidates: list[tuple[str, object, dict]] = [
        (
            "ElasticNet",
            Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=10000))]),
            {"model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5], "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        ),
    ]

    try:
        from xgboost import XGBRegressor

        candidates.append(
            (
                "XGBoost",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=random_state,
                ),
                {
                    "n_estimators": [200, 400, 600],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [4, 6, 8],
                    "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.7, 0.85, 1.0],
                },
            )
        )
    except Exception:
        pass

    optimized = []
    rows = []
    for name, model, params in candidates:
        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv_splitter,
            n_jobs=-1,
            random_state=random_state,
        )
        search.fit(X_train, y_train)
        optimized.append((name, search.best_estimator_))
        rows.append(
            {
                "model": name,
                "best_cv_mae": -search.best_score_,
                "best_params": search.best_params_,
            }
        )

    return optimized, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DNAm age prediction models.")
    parser.add_argument("--data-dir", default="Data", help="Path to data directory.")
    parser.add_argument("--output-dir", default="results", help="Path to output directory.")
    parser.add_argument("--top-k", type=int, default=10000, help="Number of CpG sites to keep.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size when reading CpG matrix.")
    parser.add_argument("--max-missing-rate", type=float, default=0.05, help="Max missing rate per CpG.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--clock-cpgs", default=None, help="Path to CpG list from known clocks.")
    parser.add_argument("--pca-components", type=int, default=400, help="PCA components.")
    parser.add_argument(
        "--feature-mode",
        choices=["topk", "pca"],
        default="topk",
        help="Feature mode: topk (CpG selection) or pca (all CpG -> PCA).",
    )
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization (minimize MAE).")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for optimization.")
    parser.add_argument("--n-iter", type=int, default=20, help="Randomized search iterations.")
    args = parser.parse_args()
    run_start = perf_counter()

    if args.feature_mode == "topk" and args.top_k <= 0 and not args.clock_cpgs:
        raise ValueError("With feature-mode=topk, set --top-k > 0 or provide --clock-cpgs.")
    if args.feature_mode == "pca" and args.pca_components <= 0:
        raise ValueError("With feature-mode=pca, set --pca-components > 0.")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annot = load_annotations(data_dir)
    sample_ids = annot.index.tolist()
    y = annot.loc[sample_ids, "age"].astype(float).to_numpy()

    data_file = data_dir / "c_sample.csv"
    header_cols = pd.read_csv(data_file, nrows=0).columns.tolist()
    sample_ids = [sid for sid in sample_ids if sid in header_cols]
    annot = annot.loc[sample_ids]
    y = annot["age"].astype(float).to_numpy()

    cpg_names = load_cpg_names(data_dir)
    clock_cpgs = load_clock_cpgs(args.clock_cpgs)

    if args.feature_mode == "topk":
        if clock_cpgs:
            selected_indices = [i for i, name in enumerate(cpg_names) if name in clock_cpgs]
            selected_names = [cpg_names[i] for i in selected_indices]
            if not selected_indices:
                print("No CpG from clock list found in cpg_names_projet.csv; fallback to correlation.")
                selected_indices, selected_names = select_top_k_cpgs(
                    data_file, sample_ids, y, cpg_names, args.top_k, args.chunk_size
                )
        else:
            selected_indices, selected_names = select_top_k_cpgs(
                data_file, sample_ids, y, cpg_names, args.top_k, args.chunk_size
            )

        pd.DataFrame({"cpg": selected_names}).to_csv(output_dir / "selected_cpgs.csv", index=False)

        selected_matrix = load_selected_cpgs(
            data_file, sample_ids, selected_indices, selected_names, args.chunk_size
        )
        selected_matrix = filter_cpgs_by_missing_rate(selected_matrix, args.max_missing_rate)
        X = selected_matrix[sample_ids].T
        y_series = annot["age"].astype(float)
        X = X.loc[y_series.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_series, test_size=args.test_size, random_state=args.random_state, shuffle=True
        )

        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)

    else:
        scores, explained = compute_pca_scores_streaming(
            data_file, sample_ids, args.pca_components, args.chunk_size, args.max_missing_rate
        )
        X = pd.DataFrame(scores, index=sample_ids, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
        y_series = annot["age"].astype(float)
        X = X.loc[y_series.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_series, test_size=args.test_size, random_state=args.random_state, shuffle=True
        )

        pd.DataFrame(
            {
                "component": [f"PC{i+1}" for i in range(scores.shape[1])],
                "explained_variance_ratio": explained,
            }
        ).to_csv(output_dir / "pca_variance.csv", index=False)

    if args.optimize:
        models, opt_df = optimize_models(X_train, y_train, args.random_state, args.n_iter, args.cv)
        opt_df.to_csv(output_dir / "optimization.csv", index=False)
        skipped = []
    else:
        models, skipped = build_models()
    metrics_rows = []
    predictions_rows = []

    for name, model in models:
        print(f"Training {name}...")
        start_fit = perf_counter()
        if name == "TabNet":
            model.fit(X_train.values.astype(np.float32), y_train.values.reshape(-1, 1))
        else:
            model.fit(X_train, y_train)
        fit_time = perf_counter() - start_fit

        start_pred = perf_counter()
        if name == "TabNet":
            preds_test = model.predict(X_test.values.astype(np.float32)).reshape(-1)
            preds_train = model.predict(X_train.values.astype(np.float32)).reshape(-1)
        else:
            preds_test = model.predict(X_test)
            preds_train = model.predict(X_train)
        pred_time = perf_counter() - start_pred

        mae = mean_absolute_error(y_test, preds_test)
        mad = float(np.median(np.abs(y_test - preds_test)))
        r2 = r2_score(y_test, preds_test)
        metrics_rows.append(
            {
                "model": name,
                "mae": mae,
                "mad": mad,
                "r2": r2,
                "fit_time_sec": fit_time,
                "predict_time_sec": pred_time,
                "n_features": X_train.shape[1],
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
            }
        )

        for sample_id, y_true, y_pred in zip(
            X_test.index.tolist(), y_test.tolist(), preds_test.tolist()
        ):
            predictions_rows.append(
                {"model": name, "sample_id": sample_id, "y_true": y_true, "y_pred": y_pred}
            )

        if name in {"ElasticNet", "Lasso", "Ridge"}:
            linear_model = model.named_steps["model"]
            coef = pd.DataFrame(
                {"cpg": X_train.columns, "coef": linear_model.coef_}
            ).assign(abs_coef=lambda df: df["coef"].abs())
            coef = coef.sort_values("abs_coef", ascending=False).head(100)
            coef.to_csv(output_dir / f"coefficients_{name.lower()}.csv", index=False)

        save_model(output_dir / "models", name, model)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="mae")
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(predictions_rows).to_csv(output_dir / "predictions.csv", index=False)

    # Annot + split + predictions (best model)
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = next(m for m in models if m[0] == best_model_name)[1]
    if best_model_name == "TabNet":
        best_preds_train = best_model.predict(X_train.values.astype(np.float32)).reshape(-1)
        best_preds_test = best_model.predict(X_test.values.astype(np.float32)).reshape(-1)
    else:
        best_preds_train = best_model.predict(X_train)
        best_preds_test = best_model.predict(X_test)

    split_info = pd.Series("non_test", index=X_train.index)
    split_info = split_info.reindex(y_series.index)
    split_info.loc[X_test.index] = "test"

    all_preds = pd.Series(index=y_series.index, dtype=float)
    all_preds.loc[X_train.index] = best_preds_train
    all_preds.loc[X_test.index] = best_preds_test

    annot_out = annot.copy()
    annot_out["split"] = split_info
    annot_out["age_pred"] = all_preds
    annot_out["model"] = best_model_name
    annot_out.to_csv(output_dir / "annot_predictions.csv", index=True, index_label="Sample_description")

    best_preds = pd.DataFrame(predictions_rows)
    best_preds = best_preds[best_preds["model"] == best_model_name]
    save_plots(output_dir, best_preds["y_true"], best_preds["y_pred"], best_model_name)

    if metrics_df.shape[0] > 0:
        report_path = output_dir / "report.md"
        selected_path = output_dir / "selected_cpgs.csv"
        top_cpgs = pd.read_csv(selected_path).head(20) if selected_path.exists() else None
        run_time_sec = perf_counter() - run_start
        report_lines = [
            "# Rapport — prédiction d'âge DNAm",
            "",
            "## Paramètres",
            f"- top_k CpG: {args.top_k}",
            f"- max_missing_rate: {args.max_missing_rate}",
            f"- test_size: {args.test_size}",
            f"- pca_components: {args.pca_components}",
            f"- temps_total_sec: {run_time_sec:.2f}",
            "",
            "## Résultats",
            metrics_df.to_markdown(index=False),
        ]
        if top_cpgs is not None:
            report_lines += [
                "",
                "## Top CpG (sélection corrélation)",
                top_cpgs.to_markdown(index=False),
            ]
        report_path.write_text("\n".join(report_lines))

    try:
        best_row = metrics_df.iloc[0]
        best_name = best_row["model"]
        best_model = next(m for m in models if m[0] == best_name)[1]
        if best_name not in {"TabNet"} and X_test.shape[1] <= 5000:
            result = permutation_importance(
                best_model, X_test, y_test, n_repeats=5, random_state=args.random_state, n_jobs=-1
            )
            importances = pd.DataFrame(
                {"cpg": X_test.columns, "importance": result.importances_mean}
            ).sort_values("importance", ascending=False)
            importances.to_csv(output_dir / "feature_importance.csv", index=False)
    except Exception as exc:
        print(f"Permutation importance skipped: {exc}")

    if skipped:
        skipped_path = output_dir / "skipped_models.txt"
        skipped_path.write_text("Models skipped due to missing dependencies:\n" + "\n".join(skipped))

    total_time_sec = perf_counter() - run_start
    print(f"Total training time: {total_time_sec:.2f} sec")

    print(metrics_df)


if __name__ == "__main__":
    main()
