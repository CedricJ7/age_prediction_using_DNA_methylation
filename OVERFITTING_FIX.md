# 🔧 Correction de l'Overfitting - Analyse et Solutions

## 📊 Problème Détecté

Lors de l'entraînement précédent, **overfitting SÉVÈRE** détecté sur plusieurs modèles :

| Modèle | MAE Test | R² | Overfitting Ratio | Statut |
|--------|----------|-----|-------------------|--------|
| **Ridge** | 3.415 | 0.961 | **39.3x** ⚠️ | CRITIQUE |
| **XGBoost** | 4.465 | 0.938 | **63.1x** ⚠️⚠️ | CRITIQUE |
| ElasticNet | 3.594 | 0.955 | 8.2x | Modéré |
| Lasso | 3.814 | 0.950 | 5.1x | Acceptable |
| RandomForest | 5.619 | 0.903 | 2.6x | ✅ Bon |
| AltumAge | 15.305 | 0.290 | 1.6x | ✅ Bon (mais mauvaise perf) |

### 🎯 Objectif
Réduire l'overfitting à **< 5x** pour tous les modèles.

---

## ✅ Solutions Implémentées

### 1️⃣ **Réduction du nombre de features**
**Avant :** `top_k_features: 10000`
**Après :** `top_k_features: 5000` ✅

**Justification :**
- 10,000 features pour 320 samples d'entraînement = ratio 31:1
- 5,000 features = ratio 16:1 (meilleur)
- Moins de features = moins de risque de surapprendre les patterns spurieux

---

### 2️⃣ **Augmentation de la régularisation Ridge**
**Avant :** `ridge_alpha: 100.0`
**Après :** `ridge_alpha: 5000.0` ✅ (50x augmentation)

**Justification :**
- Ridge avait overfitting 39.3x (CRITIQUE)
- Régularisation L2 beaucoup plus forte
- Pénalise davantage les coefficients élevés
- Cible : ramener overfitting < 5x

**Formule Ridge :**
```
Loss = MSE + alpha * ||β||²
```
Plus alpha est grand, plus les coefficients sont contraints.

---

### 3️⃣ **Renforcement de la régularisation XGBoost**

**Avant :**
```yaml
xgboost_reg_alpha: 1.0    # L1
xgboost_reg_lambda: 10.0  # L2
```

**Après :**
```yaml
xgboost_reg_alpha: 10.0   # L1 (10x augmentation)
xgboost_reg_lambda: 50.0  # L2 (5x augmentation)
```

**Justification :**
- XGBoost avait overfitting 63.1x (CRITIQUE++)
- Régularisation L1 (sparsity) + L2 (shrinkage)
- Pénalise les arbres trop complexes

---

### 4️⃣ **Réduction de la complexité XGBoost**

**Avant :**
```yaml
xgboost_n_estimators: 400
xgboost_max_depth: 6
```

**Après :**
```yaml
xgboost_n_estimators: 200  # Moitié moins d'arbres
xgboost_max_depth: 4       # Arbres moins profonds
```

**Justification :**
- Moins d'arbres = moins de capacité à mémoriser
- max_depth 4 au lieu de 6 = arbres plus simples
- Réduit le risque de surapprendre les détails

---

### 5️⃣ **Early Stopping pour XGBoost** 🆕

**Nouveau paramètre :**
```yaml
xgboost_early_stopping_rounds: 20
```

**Implémentation :**
```python
# Split train en train/validation (85/15)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15)

# Fit avec eval_set
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

**Justification :**
- Arrête l'entraînement quand la validation ne s'améliore plus
- Évite de surentraîner au-delà du point optimal
- Détecte automatiquement le nombre d'itérations optimal

**Fonctionnement :**
Si la MAE sur validation ne s'améliore pas pendant 20 itérations consécutives → STOP.

---

### 6️⃣ **Réduction de la complexité Random Forest**

**Avant :** `rf_max_depth: 20`
**Après :** `rf_max_depth: 10` ✅

**Justification :**
- Random Forest avait overfitting 2.6x (déjà acceptable)
- Réduction préventive pour sécurité
- Arbres moins profonds = moins de mémorisation

---

## 📋 Résumé des Changements

### Fichiers modifiés :

#### 1. `config/model_config.yaml`
```diff
data:
- top_k_features: 10000
+ top_k_features: 5000

models:
- ridge_alpha: 100.0
+ ridge_alpha: 5000.0

- xgboost_n_estimators: 400
+ xgboost_n_estimators: 200

- xgboost_max_depth: 6
+ xgboost_max_depth: 4

- xgboost_reg_alpha: 1.0
+ xgboost_reg_alpha: 10.0

- xgboost_reg_lambda: 10.0
+ xgboost_reg_lambda: 50.0

+ xgboost_early_stopping_rounds: 20  # NEW

- rf_max_depth: 20
+ rf_max_depth: 10
```

#### 2. `src/utils/config.py`
- Ajout du paramètre `xgboost_early_stopping_rounds` dans `ModelConfig`
- Mis à jour dans `to_yaml()` pour sérialisation

#### 3. `src/models/tree_models.py`
- Ajout de `early_stopping_rounds` dans XGBRegressor
- Ajout de `eval_metric="mae"` pour early stopping

#### 4. `scripts/train.py`
- Détection spéciale pour XGBoost
- Split train/val (85/15) pour early stopping
- Passage de `eval_set` lors du fit
- Log de l'itération optimale

---

## 🎯 Résultats Attendus

### Ridge
- **Avant :** Overfitting 39.3x, MAE 3.415
- **Après (cible) :** Overfitting < 5x, MAE ~4.0-4.5
- **Compromis :** Légère augmentation MAE pour éliminer l'overfitting

### XGBoost
- **Avant :** Overfitting 63.1x, MAE 4.465
- **Après (cible) :** Overfitting < 5x, MAE ~5.0-6.0
- **Bénéfices :**
  - Early stopping = trouve itération optimale
  - Moins d'arbres = entraînement 2x plus rapide
  - Meilleure généralisation

### ElasticNet & Lasso
- **Statut :** Déjà acceptables (< 10x)
- **Impact :** Minime, bénéficient de la réduction des features

### RandomForest
- **Avant :** Overfitting 2.6x ✅
- **Après :** Devrait rester stable, peut-être légèrement meilleur

---

## 📊 Comment vérifier

```bash
# Ré-entraîner avec les nouveaux paramètres
python scripts/train.py --config config/model_config.yaml

# Vérifier les ratios d'overfitting dans les logs
grep "Overfitting:" | grep -E "(Ridge|XGBoost)"

# Objectif : Tous < 5.0x
```

**Logs à surveiller :**
```
Ridge - Overfitting: X.XXx      # Doit être < 5.0
XGBoost - Overfitting: X.XXx    # Doit être < 5.0
Early stopping at iteration XX  # XGBoost doit s'arrêter tôt
```

---

## 🧪 Validation

### Test 1 : Overfitting Ratios
✅ **PASS** si tous les modèles ont overfitting < 5x

### Test 2 : Performance Test Set
✅ **PASS** si MAE test ≤ MAE train × 5

### Test 3 : Généralisation
✅ **PASS** si R² test > 0.85 (performances acceptables maintenues)

---

## 🔬 Principes Appliqués

### 1. **Bias-Variance Tradeoff**
- ↑ Régularisation → ↑ Bias, ↓ Variance
- Accepter légère ↑ bias (MAE train) pour ↓↓ variance (overfitting)

### 2. **Occam's Razor**
- Modèles plus simples généralisent mieux
- ↓ Features, ↓ Depth, ↓ Estimators

### 3. **Early Stopping**
- Arrêt avant convergence complète = régularisation implicite
- Basé sur validation = meilleure indication de généralisation

### 4. **Regularization**
- **L1 (Lasso, alpha)** : Sparsity, sélection features
- **L2 (Ridge, lambda)** : Shrinkage, petits coefficients
- **Elastic Net** : Combinaison L1+L2

---

## 📖 Références

### Pourquoi l'overfitting est mauvais ?
- **En recherche :** Résultats non reproductibles
- **En production :** Prédictions catastrophiques sur nouvelles données
- **En clinique :** Diagnostics erronés, patients mal traités

### Ratio acceptable
- **< 2x** : Excellent (RandomForest actuel : 2.6x)
- **< 5x** : Acceptable (Lasso actuel : 5.1x)
- **< 10x** : Limite (ElasticNet actuel : 8.2x)
- **> 10x** : ⚠️ Problématique (Ridge : 39x, XGBoost : 63x)

---

## ✅ Checklist

- [x] Réduction features (10k → 5k)
- [x] Augmentation Ridge alpha (100 → 5000)
- [x] Augmentation XGBoost reg_alpha (1 → 10)
- [x] Augmentation XGBoost reg_lambda (10 → 50)
- [x] Réduction XGBoost n_estimators (400 → 200)
- [x] Réduction XGBoost max_depth (6 → 4)
- [x] Réduction RF max_depth (20 → 10)
- [x] Implémentation early stopping XGBoost
- [x] Ajout config parameter xgboost_early_stopping_rounds
- [x] Modification train.py pour XGBoost validation split
- [x] Documentation complète

---

## 🚀 Prochaines Étapes

1. **Ré-entraîner :**
   ```bash
   python scripts/train.py --config config/model_config.yaml
   ```

2. **Vérifier logs :** Overfitting ratios < 5x ?

3. **Si toujours > 5x :**
   - Ridge : ↑ alpha à 10000
   - XGBoost : ↑ reg_lambda à 100

4. **Si < 5x mais MAE trop haute :**
   - Équilibre trouvé ! ✅
   - Trade-off acceptable

5. **Analyser avec l'app :**
   ```bash
   python app.py
   # Vérifier graphiques de généralisation
   ```

---

**Date :** 2026-01-28
**Auteur :** Claude Opus 4.5
**Status :** ✅ Solutions implémentées, prêt pour ré-entraînement
