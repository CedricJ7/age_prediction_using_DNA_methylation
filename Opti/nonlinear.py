#!/usr/bin/env python3
"""
Comparaison de modèles NON-LINÉAIRES pour la prédiction d'âge (M-values).
Sans data leakage, même pipeline que comparaison_m_values.ipynb.

Modèles :
  0. ElasticNetCV (baseline linéaire)
  1. ElasticNetCV + Transformation de Horvath (cible transformée)
  2. ElasticNetCV + Features Polynomiales (x²) sur top-50
  3. SVR RBF (sur top-100, noyau radial)
  4. GAM Splines (sur top-20, modèle additif généralisé)
  5. Horvath + Poly (combinaison 1+2)

Pipeline intra-fold (aucun leakage) :
  1. Sélection top-K CpG par corrélation (train only, NaN-safe)
  2. MICE imputation : fit train, transform test
  3. StandardScaler : fit train, transform test
  4. Entraînement / évaluation de chaque modèle
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from time import time

from sklearn.linear_model import ElasticNet, ElasticNetCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
from pygam import LinearGAM, s

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# CONSTANTES (identiques à comparaison_m_values.ipynb)
# ══════════════════════════════════════════════════════════════════════
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path('Data')
TOP_K = 500
N_FOLDS = 5
CHUNK_SIZE = 5000
BETA_EPS = 1e-6

MICE_PARAMS = {
    'estimator': BayesianRidge(),
    'n_nearest_features': 100,
    'max_iter': 10,
    'random_state': RANDOM_STATE,
}

# Sous-sélections pour modèles coûteux
TOP_K_POLY = 50    # Polynomial features : top 50 -> 1275 features (C(50,2) + 50)
TOP_K_SVR = 100    # SVR RBF : top 100 (noyau = N×N, indépendant de p)
TOP_K_GAM = 20     # GAM Splines : top 20 (une spline par feature)


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def beta_to_m(beta):
    """B-value -> M-value : M = log2(beta / (1 - beta)). Préserve NaN."""
    b = np.clip(beta, BETA_EPS, 1.0 - BETA_EPS)
    b = np.where(np.isnan(beta), np.nan, b)
    return np.log2(b / (1.0 - b))


def select_top_k(X_train, y_train, k):
    """Top-k features par corrélation absolue avec y (NaN-safe, train only)."""
    y_c = y_train - y_train.mean()
    y_den = np.sqrt(np.sum(y_c ** 2))
    col_means = np.nanmean(X_train, axis=0, keepdims=True)
    X_filled = np.where(np.isnan(X_train), col_means, X_train)
    x_c = X_filled - X_filled.mean(axis=0, keepdims=True)
    num = x_c.T @ y_c
    den = np.sqrt(np.sum(x_c ** 2, axis=0)) * y_den
    corrs = np.abs(np.divide(num, den, out=np.zeros_like(num), where=den != 0))
    return np.argsort(corrs)[::-1][:k]


# ── 1. Transformation de Horvath ──

def horvath_transform(age):
    """Transformation de l'âge selon Horvath (2013).
    Logarithmique avant adult_age, linéaire après.
    F(age) = log(age+1) - log(adult_age+1)  si age <= adult_age
           = (age - adult_age) / (adult_age+1) sinon
    """
    adult_age = 20
    return np.where(
        age <= adult_age,
        np.log(age + 1) - np.log(adult_age + 1),
        (age - adult_age) / (adult_age + 1),
    )


def horvath_inverse(transformed):
    """Inverse de la transformation de Horvath."""
    adult_age = 20
    return np.where(
        transformed <= 0,
        np.exp(transformed + np.log(adult_age + 1)) - 1,
        transformed * (adult_age + 1) + adult_age,
    )


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """Charge annotations + M-values (identique à comparaison_m_values.ipynb)."""
    # Annotations
    ind = pd.read_csv(DATA_DIR / 'annot_projet.csv')
    ind = ind.dropna(subset=['age', 'Sample_description']).copy()
    ind['Sample_description'] = ind['Sample_description'].astype(str)
    ind = ind.set_index('Sample_description')

    # Données brutes
    data_path = DATA_DIR / 'c_sample.csv'
    sample_header = pd.read_csv(data_path, nrows=0)
    all_sample_ids = list(sample_header.columns)
    common_ids = [s for s in all_sample_ids if s in ind.index]
    y = ind.loc[common_ids, 'age'].values.astype(np.float32)

    print(f'Échantillons : {len(common_ids)}, âge : {y.mean():.1f} +/- {y.std():.1f}')

    # Charger tous les CpG + conversion M-values
    print('Chargement + conversion M-values...')
    t0 = time()
    rows = []
    for chunk in pd.read_csv(data_path, usecols=common_ids, chunksize=CHUNK_SIZE):
        rows.append(beta_to_m(chunk.to_numpy(dtype=np.float32)))

    X_all = np.vstack(rows).T.astype(np.float32)
    print(f'  {time()-t0:.0f}s | shape={X_all.shape} | NaN={np.isnan(X_all).mean()*100:.2f}%')
    return X_all, y


# ══════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════

def main():
    X_all, y = load_data()
    n_cpg_total = X_all.shape[1]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    model_names = [
        '0. ElasticNetCV (baseline)',
        '1. ENet + Horvath',
        '2. ENet + Poly (top50)',
        '3. SVR RBF (top100)',
        '4. GAM Splines (top20)',
        '5. ENet + Horvath + Poly',
    ]
    scores = {n: {'mae': [], 'r2': []} for n in model_names}

    print('\n' + '=' * 75)
    print(f'  NON-LINEAR MODELS | {N_FOLDS}-fold CV | M-values')
    print(f'  {n_cpg_total} CpG -> top {TOP_K} (corr intra-fold)')
    print(f'  MICE: BayesianRidge, n_nearest=100 | StandardScaler')
    print('=' * 75)

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(y)):
        print(f'\n{"=" * 60}')
        print(f'  FOLD {fold_i+1}/{N_FOLDS}  (train={len(train_idx)}, test={len(test_idx)})')
        print(f'{"=" * 60}')

        y_train, y_test = y[train_idx], y[test_idx]
        X_train_all = X_all[train_idx]
        X_test_all = X_all[test_idx]

        # ── Sélection supervisée INTRA-FOLD (train only) ──
        t0 = time()
        top_idx = select_top_k(X_train_all, y_train, TOP_K)
        X_tr_raw = X_train_all[:, top_idx]
        X_te_raw = X_test_all[:, top_idx]
        print(f'  Feature selection: {TOP_K} CpG ({time()-t0:.1f}s)')

        # ── MICE imputation : fit train, transform test ──
        t0 = time()
        mice = IterativeImputer(**MICE_PARAMS)
        X_tr_imp = mice.fit_transform(X_tr_raw)
        X_te_imp = mice.transform(X_te_raw)
        print(f'  MICE: {time()-t0:.0f}s')

        # ── StandardScaler : fit train, transform test ──
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_imp)
        X_te = scaler.transform(X_te_imp)

        # ────────────────────────────────────────────────────
        # 0. Baseline : ElasticNetCV classique
        # ────────────────────────────────────────────────────
        enet = ElasticNetCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1)
        enet.fit(X_tr, y_train)
        y_pred_0 = enet.predict(X_te)
        scores['0. ElasticNetCV (baseline)']['mae'].append(mean_absolute_error(y_test, y_pred_0))
        scores['0. ElasticNetCV (baseline)']['r2'].append(r2_score(y_test, y_pred_0))
        print(f"  0. Baseline            MAE={scores['0. ElasticNetCV (baseline)']['mae'][-1]:.3f}")

        # ────────────────────────────────────────────────────
        # 1. ElasticNetCV + Transformation de Horvath
        #    Entraîne sur F(age), prédit F^{-1}(y_hat)
        # ────────────────────────────────────────────────────
        y_tr_horv = horvath_transform(y_train)
        enet_h = ElasticNetCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1)
        enet_h.fit(X_tr, y_tr_horv)
        y_pred_1 = horvath_inverse(enet_h.predict(X_te))
        scores['1. ENet + Horvath']['mae'].append(mean_absolute_error(y_test, y_pred_1))
        scores['1. ENet + Horvath']['r2'].append(r2_score(y_test, y_pred_1))
        print(f"  1. ENet + Horvath      MAE={scores['1. ENet + Horvath']['mae'][-1]:.3f}")

        # ────────────────────────────────────────────────────
        # 2. ElasticNetCV + Features Polynomiales (degré 2)
        #    Sur les top-50 features les plus corrélées
        #    (parmi les 500 déjà sélectionnées)
        # ────────────────────────────────────────────────────
        # Re-calculer corrélations sur les 500 features IMPUTÉES + NORMALISÉES (train only)
        corrs_500 = np.abs(np.corrcoef(X_tr.T, y_train)[-1, :-1])
        idx_top50 = np.argsort(corrs_500)[::-1][:TOP_K_POLY]

        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_tr_poly = poly.fit_transform(X_tr[:, idx_top50])
        X_te_poly = poly.transform(X_te[:, idx_top50])

        enet_p = ElasticNetCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1, max_iter=10000)
        enet_p.fit(X_tr_poly, y_train)
        y_pred_2 = enet_p.predict(X_te_poly)
        scores['2. ENet + Poly (top50)']['mae'].append(mean_absolute_error(y_test, y_pred_2))
        scores['2. ENet + Poly (top50)']['r2'].append(r2_score(y_test, y_pred_2))
        print(f"  2. ENet + Poly         MAE={scores['2. ENet + Poly (top50)']['mae'][-1]:.3f}  ({X_tr_poly.shape[1]} features)")

        # ────────────────────────────────────────────────────
        # 3. SVR RBF (sur top-100)
        #    GridSearchCV intra-fold pour C et epsilon
        # ────────────────────────────────────────────────────
        idx_top100 = np.argsort(corrs_500)[::-1][:TOP_K_SVR]
        X_tr_svr = X_tr[:, idx_top100]
        X_te_svr = X_te[:, idx_top100]

        svr_grid = GridSearchCV(
            SVR(kernel='rbf'),
            param_grid={
                'C': [1.0, 10.0, 50.0, 100.0],
                'epsilon': [0.05, 0.1, 0.5],
                'gamma': ['scale', 'auto'],
            },
            cv=3, scoring='neg_mean_absolute_error', n_jobs=-1,
        )
        svr_grid.fit(X_tr_svr, y_train)
        y_pred_3 = svr_grid.predict(X_te_svr)
        scores['3. SVR RBF (top100)']['mae'].append(mean_absolute_error(y_test, y_pred_3))
        scores['3. SVR RBF (top100)']['r2'].append(r2_score(y_test, y_pred_3))
        print(f"  3. SVR RBF             MAE={scores['3. SVR RBF (top100)']['mae'][-1]:.3f}  (C={svr_grid.best_params_['C']}, eps={svr_grid.best_params_['epsilon']})")

        # ────────────────────────────────────────────────────
        # 4. GAM Splines (sur top-20)
        #    Un terme spline s(i) par feature
        # ────────────────────────────────────────────────────
        idx_top20 = np.argsort(corrs_500)[::-1][:TOP_K_GAM]
        X_tr_gam = X_tr[:, idx_top20]
        X_te_gam = X_te[:, idx_top20]

        gam_terms = s(0)
        for i in range(1, TOP_K_GAM):
            gam_terms += s(i)

        gam = LinearGAM(gam_terms)
        gam.gridsearch(X_tr_gam, y_train, progress=False)
        y_pred_4 = gam.predict(X_te_gam)
        scores['4. GAM Splines (top20)']['mae'].append(mean_absolute_error(y_test, y_pred_4))
        scores['4. GAM Splines (top20)']['r2'].append(r2_score(y_test, y_pred_4))
        print(f"  4. GAM Splines         MAE={scores['4. GAM Splines (top20)']['mae'][-1]:.3f}")

        # ────────────────────────────────────────────────────
        # 5. Horvath + Polynomial (combinaison 1 + 2)
        #    Transformation de la cible + features quadratiques
        # ────────────────────────────────────────────────────
        enet_hp = ElasticNetCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1, max_iter=10000)
        enet_hp.fit(X_tr_poly, y_tr_horv)
        y_pred_5 = horvath_inverse(enet_hp.predict(X_te_poly))
        scores['5. ENet + Horvath + Poly']['mae'].append(mean_absolute_error(y_test, y_pred_5))
        scores['5. ENet + Horvath + Poly']['r2'].append(r2_score(y_test, y_pred_5))
        print(f"  5. Horvath + Poly      MAE={scores['5. ENet + Horvath + Poly']['mae'][-1]:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # RÉSULTATS
    # ══════════════════════════════════════════════════════════════════
    t_crit = stats.t.ppf(0.975, df=N_FOLDS - 1)

    print('\n' + '=' * 85)
    print(f'  COMPARAISON NON-LINÉAIRE ({N_FOLDS}-fold CV, M-values, IC 95%)')
    print(f'  {n_cpg_total} CpG -> top {TOP_K} | MICE fit train / transform test | StandardScaler')
    print('=' * 85)
    print(f"  {'Modèle':<30s} | {'MAE':>6s} {'± CI':>8s} | {'R²':>6s} {'± CI':>8s}")
    print(f"  {'-'*30}-+-{'-'*15}-+-{'-'*15}")

    summary_rows = []
    for name in model_names:
        s_val = scores[name]
        mae_arr = np.array(s_val['mae'])
        r2_arr = np.array(s_val['r2'])
        mae_ci = t_crit * mae_arr.std(ddof=1) / np.sqrt(N_FOLDS)
        r2_ci = t_crit * r2_arr.std(ddof=1) / np.sqrt(N_FOLDS)
        print(f"  {name:<30s} | {mae_arr.mean():6.3f} {mae_ci:>7.3f}  | {r2_arr.mean():6.3f} {r2_ci:>7.3f}")
        summary_rows.append({
            'Modèle': name,
            'MAE_mean': mae_arr.mean(), 'MAE_ci95': mae_ci,
            'R2_mean': r2_arr.mean(), 'R2_ci95': r2_ci,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values('MAE_mean')
    best = df_summary.iloc[0]
    print(f"\n  -> Meilleur : {best['Modèle']}")
    print(f"     MAE = {best['MAE_mean']:.3f} +/- {best['MAE_ci95']:.3f}")
    print(f"     R²  = {best['R2_mean']:.3f} +/- {best['R2_ci95']:.3f}")

    # Sauvegarde CSV
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(results_dir / 'nonlinear_comparison_results.csv', index=False)
    print(f'\n  Sauvegardé : results/nonlinear_comparison_results.csv')

    # ══════════════════════════════════════════════════════════════════
    # VISUALISATION
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    models = df_summary['Modèle'].tolist()
    colors = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    # MAE
    ax = axes[0]
    mae_vals = df_summary['MAE_mean'].values
    mae_cis = df_summary['MAE_ci95'].values
    bars = ax.barh(range(len(models)), mae_vals, xerr=mae_cis, capsize=4,
                   color=colors[:len(models)], alpha=0.85, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split('. ', 1)[1] for m in models], fontsize=10)
    ax.set_xlabel('MAE (ans)')
    ax.set_title(f'MAE +/- IC 95% ({N_FOLDS}-fold CV, M-values)')
    ax.invert_yaxis()
    for i, (m, c) in enumerate(zip(mae_vals, mae_cis)):
        ax.text(m + c + 0.05, i, f'{m:.3f}', va='center', fontsize=9, fontweight='bold')

    # R²
    ax = axes[1]
    r2_vals = df_summary['R2_mean'].values
    r2_cis = df_summary['R2_ci95'].values
    bars = ax.barh(range(len(models)), r2_vals, xerr=r2_cis, capsize=4,
                   color=colors[:len(models)], alpha=0.85, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split('. ', 1)[1] for m in models], fontsize=10)
    ax.set_xlabel('R²')
    ax.set_title(f'R² +/- IC 95% ({N_FOLDS}-fold CV, M-values)')
    ax.invert_yaxis()
    for i, (m, c) in enumerate(zip(r2_vals, r2_cis)):
        ax.text(m + c + 0.003, i, f'{m:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / 'nonlinear_comparison_metrics.png', dpi=150, bbox_inches='tight')
    print('  Figure : results/nonlinear_comparison_metrics.png')

    # MAE par fold
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(N_FOLDS)
    width = 0.12
    for i, name in enumerate(model_names):
        mae_f = scores[name]['mae']
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, mae_f, width, label=name.split('. ', 1)[1],
               color=colors[i], alpha=0.85, edgecolor='white')
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE (ans)')
    ax.set_title('MAE par Fold (modèles non-linéaires)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(N_FOLDS)])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(results_dir / 'nonlinear_comparison_per_fold.png', dpi=150, bbox_inches='tight')
    print('  Figure : results/nonlinear_comparison_per_fold.png')

    # ══════════════════════════════════════════════════════════════════
    # CONCLUSION
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 75)
    print('  RÉSUMÉ - MODÈLES NON-LINÉAIRES (sans data leakage)')
    print('=' * 75)
    print(f'  Données       : B-values brutes -> M-values, {n_cpg_total} CpG')
    print(f'  Sélection     : top {TOP_K} CpG par corrélation (intra-fold, train only)')
    print(f'  Imputation    : MICE BayesianRidge n_nearest=100 (fit train, transform test)')
    print(f'  Normalisation : StandardScaler (fit train, transform test)')
    print(f'  CV            : {N_FOLDS}-fold (shuffle, seed={RANDOM_STATE})')
    print(f'  IC 95%        : t-Student (df={N_FOLDS-1})')
    print()
    print(f'  Approches non-linéaires :')
    print(f'    1. Horvath   : transformation log-linéaire de la cible (adulte=20 ans)')
    print(f'    2. Poly      : features quadratiques sur top-{TOP_K_POLY} CpG')
    print(f'    3. SVR RBF   : noyau radial sur top-{TOP_K_SVR} CpG (GridSearchCV intra-fold)')
    print(f'    4. GAM       : splines additives sur top-{TOP_K_GAM} CpG (gridsearch interne)')
    print(f'    5. Horv+Poly : combinaison transformation cible + features quadratiques')
    print()
    print(f'  Data leakage : AUCUN')
    print(f'    - Feature selection  : train only')
    print(f'    - MICE               : fit train, transform test')
    print(f'    - StandardScaler     : fit train, transform test')
    print(f'    - PolynomialFeatures : fit train, transform test')
    print(f'    - SVR GridSearchCV   : CV interne sur train uniquement')
    print(f'    - GAM gridsearch     : sur train uniquement')
    print()
    print(df_summary.to_string(index=False))
    print(f"\n  Meilleur : {best['Modèle']}")
    print(f"  MAE = {best['MAE_mean']:.3f} +/- {best['MAE_ci95']:.3f}")
    print('=' * 75)


if __name__ == '__main__':
    main()
