# Hyperparameter Optimization - Guide Complet

## 🎯 Objectif

Trouver les **meilleurs modèles prédictifs** pour l'âge épigénétique en testant exhaustivement toutes les méthodes ML populaires avec optimisation bayésienne des hyperparamètres.

**Approche Senior Data Scientist** :
- ✅ Optuna (optimisation bayésienne TPE + élagage médian)
- ✅ 10 algorithmes testés (Ridge, Lasso, ElasticNet, SVR, RF, GBM, XGBoost, LightGBM, CatBoost, MLP)
- ✅ 50-150 trials par modèle selon complexité
- ✅ Cross-validation 5-fold pour robustesse
- ✅ Gestion mémoire optimisée (PCA ou sélection features)
- ✅ Sauvegarde progressive (SQLite)
- ✅ Suivi temps réel
- ✅ Rapport détaillé (CSV + logs)

---

## 📦 Installation

### Dépendances Requises

```bash
# Installer toutes les bibliothèques d'optimisation
pip install optuna lightgbm catboost

# Ou réinstaller depuis requirements.txt
pip install -r requirements.txt
```

### Vérifier Installation

```bash
python -c "import optuna, lightgbm, catboost; print('OK')"
```

---

## 🚀 Utilisation

### Option 1: Mode Standard (5000 features sélectionnées)

```bash
python scripts/hyperparameter_optimization.py \
    --top-k-features 5000 \
    --max-hours 8.0 \
    --test-size 0.2
```

**Temps estimé** : 6-8 heures
**Mémoire** : ~4-8 GB RAM

### Option 2: Mode PCA (réduction dimensionnalité)

```bash
python scripts/hyperparameter_optimization.py \
    --use-pca \
    --pca-components 400 \
    --max-hours 6.0 \
    --test-size 0.2
```

**Temps estimé** : 4-6 heures
**Mémoire** : ~2-4 GB RAM
**Avantage** : Plus rapide, moins de mémoire

### Option 3: Optimiser Modèles Spécifiques

```bash
# Seulement Ridge, Lasso, ElasticNet
python scripts/hyperparameter_optimization.py \
    --models Ridge Lasso ElasticNet \
    --max-hours 2.0

# Seulement les boosting methods
python scripts/hyperparameter_optimization.py \
    --models XGBoost LightGBM CatBoost \
    --max-hours 4.0
```

### Option 4: Mode Rapide (Test)

```bash
python scripts/hyperparameter_optimization.py \
    --top-k-features 1000 \
    --max-hours 1.0 \
    --models Ridge XGBoost
```

**Temps estimé** : ~1 heure
**Usage** : Test rapide pour vérifier fonctionnement

---

## 📊 Modèles Optimisés

### 1. **Modèles Linéaires** (rapides, interprétables)

#### Ridge Regression
- **Hyperparamètres** : alpha (1e-3 à 1e5), solver
- **Trials** : 100
- **Temps** : ~10-15 min
- **Avantages** : Robuste, stable, bon avec high-dimensional data

#### Lasso Regression
- **Hyperparamètres** : alpha (1e-5 à 10), max_iter, selection
- **Trials** : 100
- **Temps** : ~10-15 min
- **Avantages** : Feature selection automatique (sparse)

#### ElasticNet
- **Hyperparamètres** : alpha, l1_ratio, max_iter, selection
- **Trials** : 150
- **Temps** : ~15-20 min
- **Avantages** : Meilleur des deux mondes (L1 + L2)

---

### 2. **Support Vector Machine**

#### SVR (Support Vector Regressor)
- **Hyperparamètres** : kernel (linear/rbf/poly), C, epsilon, gamma, degree
- **Trials** : 100
- **Temps** : ~30-45 min
- **Avantages** : Capture non-linéarités complexes
- **Inconvénient** : Lent sur grands datasets

---

### 3. **Modèles Ensembles Classiques**

#### Random Forest
- **Hyperparamètres** : n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap
- **Trials** : 80
- **Temps** : ~20-30 min
- **Avantages** : Robuste aux outliers, peu d'overfitting

#### Gradient Boosting (scikit-learn)
- **Hyperparamètres** : n_estimators, learning_rate, max_depth, min_samples_split, subsample, max_features
- **Trials** : 80
- **Temps** : ~30-45 min
- **Avantages** : Généralement excellentes performances

---

### 4. **Gradient Boosting Moderne** (SOTA)

#### XGBoost
- **Hyperparamètres** : n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma
- **Trials** : 100
- **Temps** : ~30-45 min
- **Avantages** : Très performant, régularisation forte, GPU support

#### LightGBM
- **Hyperparamètres** : n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
- **Trials** : 100
- **Temps** : ~20-30 min
- **Avantages** : TRÈS RAPIDE, économe en mémoire, excellent pour high-dimensional

#### CatBoost
- **Hyperparamètres** : iterations, learning_rate, depth, l2_leaf_reg, border_count
- **Trials** : 80
- **Temps** : ~40-60 min
- **Avantages** : Bon avec données catégorielles, robuste, peu de tuning nécessaire

---

### 5. **Réseaux de Neurones**

#### MLP (Multi-Layer Perceptron)
- **Hyperparamètres** : n_layers (1-4), n_units_per_layer (32-512), activation, alpha, learning_rate, learning_rate_init
- **Trials** : 100
- **Temps** : ~45-60 min
- **Avantages** : Capture relations très non-linéaires
- **Inconvénient** : Peut overfitter, moins interprétable

---

## 📈 Espace de Recherche Hyperparamètres

### Ridge
```python
{
    'alpha': [1e-3, 1e5] (log-scale),  # Régularisation L2
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}
```

### XGBoost (exemple détaillé)
```python
{
    'n_estimators': [50, 500],           # Nombre d'arbres
    'learning_rate': [1e-3, 0.3] (log),  # Taux d'apprentissage
    'max_depth': [2, 12],                # Profondeur arbres
    'min_child_weight': [1, 10],         # Min samples leaf
    'subsample': [0.5, 1.0],             # Row sampling
    'colsample_bytree': [0.5, 1.0],      # Column sampling
    'reg_alpha': [1e-3, 100] (log),      # L1 regularization
    'reg_lambda': [1e-3, 100] (log),     # L2 regularization
    'gamma': [1e-3, 10] (log)            # Min split loss
}
```

### LightGBM (optimisé vitesse)
```python
{
    'n_estimators': [50, 500],
    'learning_rate': [1e-3, 0.3] (log),
    'num_leaves': [10, 200],             # Complexité arbre
    'max_depth': [2, 12],
    'min_child_samples': [5, 50],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.5, 1.0],
    'reg_alpha': [1e-3, 100] (log),
    'reg_lambda': [1e-3, 100] (log)
}
```

---

## 🔍 Stratégie d'Optimisation

### Optuna TPE Sampler
- **Tree-structured Parzen Estimator** (TPE)
- Optimisation bayésienne intelligente
- Apprend des trials précédents
- Concentre la recherche sur zones prometteuses

### Pruning
- **MedianPruner** avec `n_startup_trials=10`
- Arrête les trials non prometteurs tôt
- Économise du temps de calcul
- Basé sur médiane des CV scores

### Cross-Validation
- **5-fold CV** par défaut
- Robuste, évite overfitting
- MAE moyen utilisé comme métrique

---

## 📁 Résultats Générés

### Structure des Fichiers

```
results/optimization/
├── optuna_study.db                          # Base SQLite (tous trials)
├── optimization_results_YYYYMMDD_HHMMSS.csv # Tableau comparatif
├── best_hyperparameters_YYYYMMDD_HHMMSS.csv # Tous hyperparamètres
├── scaler.joblib                            # StandardScaler fitted
├── imputer.joblib                           # KNN Imputer fitted
├── pca_transformer.joblib                   # PCA (si --use-pca)
├── best_ridge.joblib                        # Meilleur modèle Ridge
├── best_xgboost.joblib                      # Meilleur modèle XGBoost
├── best_lightgbm.joblib                     # Meilleur modèle LightGBM
└── ...
```

### Format CSV Résultats

```csv
Rank,Model,MAE_Train,MAE_Test,MAD_Test,R2_Train,R2_Test,Overfitting_Ratio,CV_MAE,N_Params,N_Trials,Optimization_Time_Min
1,LightGBM,2.145,3.234,2.876,0.9821,0.9678,1.51,3.156,350,100,25.4
2,XGBoost,2.223,3.298,2.934,0.9803,0.9665,1.48,3.201,400,100,32.1
3,Ridge,2.567,3.412,3.012,0.9745,0.9634,1.33,3.389,5002,100,12.3
...
```

### Colonnes Expliquées

- **Rank** : Classement (1 = meilleur)
- **Model** : Nom du modèle
- **MAE_Train** : MAE sur ensemble d'entraînement
- **MAE_Test** : MAE sur ensemble de test (MÉTRIQUE CLEF)
- **MAD_Test** : Median Absolute Deviation (robuste aux outliers)
- **R2_Train** : R² sur train
- **R2_Test** : R² sur test
- **Overfitting_Ratio** : MAE_Test / MAE_Train (< 2.0 excellent)
- **CV_MAE** : MAE cross-validation (moyenne)
- **N_Params** : Nombre de paramètres du modèle
- **N_Trials** : Nombre de trials Optuna exécutés
- **Optimization_Time_Min** : Temps d'optimisation (minutes)

---

## ⏱️ Temps d'Exécution Estimés

### Configuration Minimale (1000 features, 2h max)
- Ridge: ~5 min
- Lasso: ~5 min
- ElasticNet: ~7 min
- XGBoost: ~15 min
- LightGBM: ~10 min
- **Total**: ~1h

### Configuration Standard (5000 features, 8h max)
- Ridge: ~12 min
- Lasso: ~12 min
- ElasticNet: ~18 min
- SVR: ~40 min
- RandomForest: ~25 min
- GradientBoosting: ~35 min
- XGBoost: ~35 min
- LightGBM: ~25 min
- CatBoost: ~50 min
- MLP: ~55 min
- **Total**: ~6-7h

### Configuration PCA (400 components, 6h max)
- Tous modèles ~30% plus rapides
- **Total**: ~4-5h

---

## 💡 Conseils d'Utilisation

### Pour Minimiser le Temps

1. **Utiliser PCA** : Réduit dimensionnalité drastiquement
   ```bash
   --use-pca --pca-components 200
   ```

2. **Sélectionner moins de features**
   ```bash
   --top-k-features 2000
   ```

3. **Optimiser seulement top modèles**
   ```bash
   --models XGBoost LightGBM Ridge
   ```

4. **Réduire max_hours**
   ```bash
   --max-hours 4.0
   ```

### Pour Maximiser la Performance

1. **Plus de features** (si mémoire suffisante)
   ```bash
   --top-k-features 10000
   ```

2. **Tous les modèles** (laisser tourner)
   ```bash
   --max-hours 8.0
   ```

3. **Test size plus petit** (plus de données train)
   ```bash
   --test-size 0.15
   ```

### Gestion Mémoire

**Si RAM < 8 GB** :
- Utiliser `--use-pca --pca-components 200`
- Ou `--top-k-features 2000`

**Si RAM >= 16 GB** :
- Peut utiliser `--top-k-features 10000` sans problème

**Si RAM >= 32 GB** :
- Peut charger toutes les features avec PCA

---

## 🔬 Analyse des Résultats

### 1. Charger les Résultats

```python
import pandas as pd
import joblib

# Charger tableau comparatif
results = pd.read_csv('results/optimization/optimization_results_*.csv')
print(results.sort_values('MAE_Test').head(5))

# Charger meilleur modèle
best_model = joblib.load('results/optimization/best_lightgbm.joblib')
scaler = joblib.load('results/optimization/scaler.joblib')
```

### 2. Prédire sur Nouvelles Données

```python
import numpy as np

# Charger transformers
scaler = joblib.load('results/optimization/scaler.joblib')
imputer = joblib.load('results/optimization/imputer.joblib')

# Optionnel: PCA si utilisé
pca = joblib.load('results/optimization/pca_transformer.joblib')

# Préparer données
X_new = ...  # Vos nouvelles données
X_new = imputer.transform(X_new)
if pca:
    X_new = pca.transform(X_new)
X_new = scaler.transform(X_new)

# Prédire
predictions = best_model.predict(X_new)
print(f"Ages prédits: {predictions}")
```

### 3. Explorer Base Optuna

```python
import optuna

# Charger study
storage = 'sqlite:///results/optimization/optuna_study.db'
study_name = 'XGBoost_20260128_120000'  # Adapter
study = optuna.load_study(study_name=study_name, storage=storage)

# Best trial
print(f"Best MAE: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")

# Historique
df = study.trials_dataframe()
print(df[['number', 'value', 'state']].head(10))

# Visualisations
from optuna.visualization import plot_optimization_history
fig = plot_optimization_history(study)
fig.show()
```

---

## 🎯 Critères de Succès

### Excellent Modèle
- ✅ MAE Test < 3.5 ans
- ✅ R² Test > 0.95
- ✅ Overfitting Ratio < 2.0x
- ✅ MAD Test ≈ MAE Test (pas d'outliers)

### Bon Modèle
- ✅ MAE Test < 4.5 ans
- ✅ R² Test > 0.92
- ✅ Overfitting Ratio < 3.0x

### Acceptable
- ⚠️ MAE Test < 6.0 ans
- ⚠️ R² Test > 0.85
- ⚠️ Overfitting Ratio < 5.0x

### Problématique
- ❌ MAE Test > 6.0 ans
- ❌ Overfitting Ratio > 10x

---

## 🐛 Troubleshooting

### Erreur: Out of Memory

**Solution** :
```bash
# Utiliser PCA
python scripts/hyperparameter_optimization.py --use-pca --pca-components 200

# Ou réduire features
python scripts/hyperparameter_optimization.py --top-k-features 1000
```

### Erreur: Import Error (LightGBM/CatBoost)

**Solution** :
```bash
pip install lightgbm catboost

# Ou optimiser sans ces modèles
python scripts/hyperparameter_optimization.py --models Ridge Lasso ElasticNet XGBoost
```

### Processus Trop Lent

**Solution** :
```bash
# Réduire nombre de trials (modifier dans le script)
# Ou limiter modèles
python scripts/hyperparameter_optimization.py --models XGBoost LightGBM --max-hours 2
```

### Interrompre et Reprendre

**L'optimisation peut être interrompue** (Ctrl+C) à tout moment :
- ✅ Résultats déjà complétés sont sauvegardés
- ✅ Base Optuna préservée
- ⚠️ Modèle en cours perdu (mais pas grave)

**Relancer** reprend avec nouveaux modèles (ne reprend PAS trials précédents car nouveaux study names).

---

## 📚 Références Scientifiques

### Optimisation Bayésienne
- Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD 2019*.

### Gradient Boosting
- Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.
- Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*.
- Prokhorenkova et al. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS 2018*.

---

## ✅ Checklist Utilisation

- [ ] Installer dépendances (`pip install -r requirements.txt`)
- [ ] Vérifier mémoire disponible (au moins 4 GB RAM recommandé)
- [ ] Choisir stratégie: PCA ou sélection features
- [ ] Lancer optimisation avec paramètres adaptés
- [ ] Surveiller logs en temps réel
- [ ] Attendre fin (ou interrompre si satisfait)
- [ ] Analyser `optimization_results_*.csv`
- [ ] Charger meilleur modèle et tester
- [ ] Utiliser meilleur modèle pour prédictions futures

---

**Date** : 2026-01-28
**Auteur** : Claude Opus 4.5
**Version** : 1.0
**Status** : ✅ Prêt pour production
