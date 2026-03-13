"""
Génère les prédictions des meilleurs modèles (5-Fold CV) dans le format attendu par l'app Dash.

Modèles :
  1. Residual Learning  (ElasticNetCV + XGBoost résidus) -> MAE ~3.28
  2. ElasticNetCV       (top-500 CpG par corrélation)   -> MAE ~3.57
  3. Stack Optuna       (StackingRegressor, params pré-optimisés) -> MAE ~3.54

Sortie :
  results/metrics.csv         - métriques globales par modèle
  results/predictions.csv     - prédictions par échantillon
  results/annot_predictions.csv - prédictions + infos démographiques
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from time import time

from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

SEED = 42
np.random.seed(SEED)
DATA_DIR = Path('Data')
RESULTS_DIR = Path('results')
TOP_K = 500
N_FOLDS = 5
CHUNK_SIZE = 5000


# ── Params Optuna pré-optimisés (par fold) ─────────────────────────────────
OPTUNA_PARAMS = [
    {'enet_alpha': 0.1111, 'enet_l1': 0.2434, 'ridge_alpha': 0.1497, 'svr_C': 17.7405,
     'knn_n': 4, 'rf_n': 300, 'rf_depth': 10, 'xgb_n': 300, 'xgb_depth': 4,
     'xgb_lr': 0.0431, 'lgbm_n': 400, 'lgbm_depth': 8, 'lgbm_lr': 0.2142,
     'meta_alpha': 25.544, 'passthrough': True},
    {'enet_alpha': 1.9335, 'enet_l1': 0.3999, 'ridge_alpha': 0.3903, 'svr_C': 3.1735,
     'knn_n': 11, 'rf_n': 150, 'rf_depth': 8, 'xgb_n': 200, 'xgb_depth': 7,
     'xgb_lr': 0.0771, 'lgbm_n': 150, 'lgbm_depth': 6, 'lgbm_lr': 0.0502,
     'meta_alpha': 99.6794, 'passthrough': True},
    {'enet_alpha': 0.08, 'enet_l1': 0.3692, 'ridge_alpha': 0.0024, 'svr_C': 25.2494,
     'knn_n': 19, 'rf_n': 150, 'rf_depth': 4, 'xgb_n': 550, 'xgb_depth': 7,
     'xgb_lr': 0.1974, 'lgbm_n': 500, 'lgbm_depth': 7, 'lgbm_lr': 0.2066,
     'meta_alpha': 0.7164, 'passthrough': False},
    {'enet_alpha': 0.029, 'enet_l1': 0.6232, 'ridge_alpha': 15.2078, 'svr_C': 7.5337,
     'knn_n': 17, 'rf_n': 200, 'rf_depth': 9, 'xgb_n': 500, 'xgb_depth': 8,
     'xgb_lr': 0.1499, 'lgbm_n': 350, 'lgbm_depth': 4, 'lgbm_lr': 0.0341,
     'meta_alpha': 0.8066, 'passthrough': False},
    {'enet_alpha': 1.0387, 'enet_l1': 0.7379, 'ridge_alpha': 0.0025, 'svr_C': 4.6404,
     'knn_n': 8, 'rf_n': 300, 'rf_depth': 5, 'xgb_n': 250, 'xgb_depth': 4,
     'xgb_lr': 0.1908, 'lgbm_n': 200, 'lgbm_depth': 9, 'lgbm_lr': 0.0311,
     'meta_alpha': 98.9044, 'passthrough': True},
]


# ── Helpers ────────────────────────────────────────────────────────────────

def select_top_k(X_train, y_train, k):
    y_c = y_train - y_train.mean()
    y_den = np.sqrt(np.sum(y_c ** 2))
    col_means = np.nanmean(X_train, axis=0, keepdims=True)
    X_filled = np.where(np.isnan(X_train), col_means, X_train)
    x_c = X_filled - X_filled.mean(axis=0, keepdims=True)
    num = x_c.T @ y_c
    den = np.sqrt(np.sum(x_c ** 2, axis=0)) * y_den
    corrs = np.abs(np.divide(num, den, out=np.zeros_like(num), where=den != 0))
    return np.argsort(corrs)[::-1][:k]


def preprocess(X_train_raw, X_test_raw):
    """MICE (fit train) + StandardScaler (fit train)."""
    mice = IterativeImputer(
        estimator=BayesianRidge(), n_nearest_features=100,
        max_iter=10, random_state=SEED
    )
    X_tr = mice.fit_transform(X_train_raw)
    X_te = mice.transform(X_test_raw)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te


def build_stacking(p):
    estimators = [
        ('enet', ElasticNet(alpha=p['enet_alpha'], l1_ratio=p['enet_l1'], max_iter=10000, random_state=SEED)),
        ('ridge', Ridge(alpha=p['ridge_alpha'], random_state=SEED)),
        ('svr', SVR(C=p['svr_C'], kernel='rbf')),
        ('knn', KNeighborsRegressor(n_neighbors=p['knn_n'], weights='distance')),
        ('rf', RandomForestRegressor(n_estimators=p['rf_n'], max_depth=p['rf_depth'], random_state=SEED, n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=p['xgb_n'], max_depth=p['xgb_depth'], learning_rate=p['xgb_lr'],
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)),
        ('lgbm', LGBMRegressor(n_estimators=p['lgbm_n'], max_depth=p['lgbm_depth'], learning_rate=p['lgbm_lr'],
                               subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=p['meta_alpha'], random_state=SEED),
        cv=3, passthrough=p.get('passthrough', True), n_jobs=-1
    )


def compute_metrics(model_name, y_true, y_pred, fit_time, n_features):
    mae  = mean_absolute_error(y_true, y_pred)
    mad  = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return {
        'model': model_name,
        'fit_time_sec': round(fit_time, 2),
        'n_features': n_features,
        'n_train': int(len(y_true) * 0.8),
        'n_test': int(len(y_true) * 0.2),
        'mae': mae, 'mad': mad, 'rmse': rmse,
        'r2': r2, 'correlation': corr,
        'mae_train': np.nan, 'r2_train': np.nan,
        'cv_mae': mae, 'cv_std': np.nan,
        'overfitting_ratio': np.nan,
    }


# ── Chargement des données ─────────────────────────────────────────────────

print('=' * 60)
print('  Génération des prédictions (meilleurs modèles, 5-Fold CV)')
print('=' * 60)

ind = pd.read_csv(DATA_DIR / 'annot_projet.csv')
ind = ind.dropna(subset=['age', 'Sample_description']).copy()
ind['Sample_description'] = ind['Sample_description'].astype(str)
ind = ind.set_index('Sample_description')

data_path = DATA_DIR / 'c_sample.csv'
sample_header = pd.read_csv(data_path, nrows=0)
common_ids = [s for s in sample_header.columns if s in ind.index]

y = ind.loc[common_ids, 'age'].values.astype(np.float32)
sex = ind.loc[common_ids, 'female'].apply(
    lambda x: 1.0 if str(x).lower() == 'true' else (0.0 if str(x).lower() == 'false' else np.nan)
).values.astype(np.float32)
sex = np.where(np.isnan(sex), np.nanmean(sex), sex)

print(f'Chargement de {X_all.shape[1] if "X_all" in dir() else "?"} CpG bruts...')
t0 = time()
rows = []
for chunk in pd.read_csv(data_path, usecols=common_ids, chunksize=CHUNK_SIZE):
    rows.append(chunk.to_numpy(dtype=np.float32))
X_all = np.vstack(rows).T.astype(np.float32)
print(f'  {len(common_ids)} echantillons, {X_all.shape[1]} CpG, NaN={np.isnan(X_all).mean()*100:.2f}% ({time()-t0:.1f}s)')


# ── 5-Fold CV ─────────────────────────────────────────────────────────────

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Stockage : {model_name -> {idx: y_pred}}
all_preds = {
    'Residual Learning': np.full(len(y), np.nan),
    'ElasticNetCV': np.full(len(y), np.nan),
    'Stack Optuna': np.full(len(y), np.nan),
}
fit_times = {k: 0.0 for k in all_preds}

for fold_i, (train_idx, test_idx) in enumerate(kf.split(y)):
    print(f'\n{"="*50}')
    print(f'  Fold {fold_i+1}/{N_FOLDS}  (train={len(train_idx)}, test={len(test_idx)})')
    print(f'{"="*50}')

    y_train, y_test = y[train_idx], y[test_idx]
    params = OPTUNA_PARAMS[fold_i]

    # 1. Sélection top-K (train only)
    top_idx = select_top_k(X_all[train_idx], y_train, TOP_K)
    X_tr_raw = X_all[train_idx][:, top_idx]
    X_te_raw = X_all[test_idx][:, top_idx]

    # 2. Préprocessing (MICE + Scaler)
    t0 = time()
    X_tr, X_te = preprocess(X_tr_raw, X_te_raw)
    print(f'  Preprocessing: {time()-t0:.1f}s')

    # ── Modèle 1 : ElasticNetCV ──────────────────────────────────────────
    t0 = time()
    enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], n_alphas=50,
                        cv=3, max_iter=50000, random_state=SEED, n_jobs=-1)
    enet.fit(X_tr, y_train)
    all_preds['ElasticNetCV'][test_idx] = enet.predict(X_te)
    dt = time() - t0
    fit_times['ElasticNetCV'] += dt
    print(f'  ElasticNetCV: {dt:.1f}s, MAE={mean_absolute_error(y_test, all_preds["ElasticNetCV"][test_idx]):.3f}')

    # ── Modèle 2 : Residual Learning ─────────────────────────────────────
    t0 = time()
    # Sous-ensemble de validation interne pour XGBoost early stopping
    val_size = max(20, len(train_idx) // 5)
    val_idx_local = np.random.choice(len(y_train), val_size, replace=False)
    tr_idx_local = np.setdiff1d(np.arange(len(y_train)), val_idx_local)

    enet_rl = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], n_alphas=50,
                           cv=3, max_iter=50000, random_state=SEED, n_jobs=-1)
    enet_rl.fit(X_tr[tr_idx_local], y_train[tr_idx_local])
    P_lin_tr = enet_rl.predict(X_tr)
    residuals = y_train - P_lin_tr

    xgb_rl = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5, reg_alpha=10.0, reg_lambda=50.0,
        min_child_weight=10, early_stopping_rounds=30,
        objective='reg:squarederror', eval_metric='mae',
        n_jobs=-1, random_state=SEED, verbosity=0,
    )
    P_val_lin = enet_rl.predict(X_tr[val_idx_local])
    xgb_rl.fit(
        X_tr, residuals,
        eval_set=[(X_tr[val_idx_local], y_train[val_idx_local] - P_val_lin)],
        verbose=False
    )

    P_lin_te = enet_rl.predict(X_te)
    R_hat_te = xgb_rl.predict(X_te)
    P_lin_all = enet_rl.predict(X_tr)
    R_hat_all = xgb_rl.predict(X_tr)

    bias = LinearRegression()
    bias.fit(np.column_stack([P_lin_all, xgb_rl.predict(X_tr)]), y_train)
    y_pred_rl = bias.predict(np.column_stack([P_lin_te, R_hat_te]))
    all_preds['Residual Learning'][test_idx] = y_pred_rl

    dt = time() - t0
    fit_times['Residual Learning'] += dt
    print(f'  Residual Learning: {dt:.1f}s, MAE={mean_absolute_error(y_test, y_pred_rl):.3f}')

    # ── Modèle 3 : Stack Optuna ──────────────────────────────────────────
    t0 = time()
    stacking = build_stacking(params)
    stacking.fit(X_tr, y_train)
    y_pred_stk = stacking.predict(X_te)
    all_preds['Stack Optuna'][test_idx] = y_pred_stk
    dt = time() - t0
    fit_times['Stack Optuna'] += dt
    print(f'  Stack Optuna: {dt:.1f}s, MAE={mean_absolute_error(y_test, y_pred_stk):.3f}')


# ── Sauvegarde ─────────────────────────────────────────────────────────────

print('\n' + '=' * 60)
print('  Sauvegarde des résultats')
print('=' * 60)

# --- metrics.csv ---
metrics_rows = []
for rank, (model_name, y_pred_oof) in enumerate(all_preds.items(), start=1):
    m = compute_metrics(model_name, y, y_pred_oof, fit_times[model_name], TOP_K)
    m['rank'] = rank
    metrics_rows.append(m)
    print(f'  {model_name}: MAE={m["mae"]:.3f}, R2={m["r2"]:.3f}')

df_metrics = pd.DataFrame(metrics_rows)
cols_order = ['rank', 'model', 'fit_time_sec', 'n_features', 'n_train', 'n_test',
              'mae', 'mad', 'rmse', 'r2', 'correlation',
              'mae_train', 'r2_train', 'cv_mae', 'cv_std', 'overfitting_ratio']
df_metrics = df_metrics[cols_order].sort_values('mae').reset_index(drop=True)
df_metrics['rank'] = range(1, len(df_metrics) + 1)
df_metrics.to_csv(RESULTS_DIR / 'metrics.csv', index=False)
print(f'  -> metrics.csv sauvegardé')

# --- predictions.csv ---
pred_rows = []
for model_name, y_pred_oof in all_preds.items():
    for i, sid in enumerate(common_ids):
        pred_rows.append({
            'model': model_name,
            'sample_id': sid,
            'y_true': float(y[i]),
            'y_pred': float(y_pred_oof[i]),
        })
df_preds = pd.DataFrame(pred_rows)
df_preds.to_csv(RESULTS_DIR / 'predictions.csv', index=False)
print(f'  -> predictions.csv sauvegardé ({len(df_preds)} lignes)')

# --- annot_predictions.csv ---
annot_rows = []
ind_reset = ind.reset_index()
for model_name, y_pred_oof in all_preds.items():
    for i, sid in enumerate(common_ids):
        row_ann = ind_reset[ind_reset['Sample_description'] == sid].iloc[0]
        annot_rows.append({
            'Sample_description': sid,
            'Sample_Name': row_ann.get('Sample_Name', ''),
            'female': row_ann.get('female', ''),
            'ethnicity': row_ann.get('ethnicity', ''),
            'age': float(y[i]),
            'split': 'test',   # CV : tous les echantillons passent en test
            'age_pred': float(y_pred_oof[i]),
            'model': model_name,
        })
df_annot = pd.DataFrame(annot_rows)
df_annot.to_csv(RESULTS_DIR / 'annot_predictions.csv', index=False)
print(f'  -> annot_predictions.csv sauvegardé ({len(df_annot)} lignes)')

print('\nTerminé.')
