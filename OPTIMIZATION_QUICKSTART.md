# Hyperparameter Optimization - Guide Démarrage Rapide ⚡

## 🚀 Lancement en 3 Étapes

### 1️⃣ Installation (2 minutes)

```bash
# Installer les dépendances nécessaires
pip install optuna lightgbm catboost

# Vérifier installation
python -c "import optuna, lightgbm, catboost; print('✓ OK')"
```

### 2️⃣ Lancer l'Optimisation (6-8 heures)

#### Option Recommandée: Mode Standard

```bash
python scripts/hyperparameter_optimization.py
```

**Ce qui va se passer:**
- ✅ Charge vos données (Data/)
- ✅ Sélectionne 5000 meilleurs sites CpG
- ✅ Teste 10 algorithmes ML différents
- ✅ 50-150 trials par algorithme
- ✅ Sauvegarde progressive dans `results/optimization/`
- ✅ Temps estimé: 6-8 heures

**Surveiller la progression:**
```bash
# Dans un autre terminal
tail -f logs/*.log  # ou regarder la sortie console
```

#### Option Rapide: Mode PCA (4-6 heures)

```bash
python scripts/hyperparameter_optimization.py --use-pca --pca-components 400
```

Plus rapide, utilise moins de mémoire!

### 3️⃣ Analyser les Résultats (1 minute)

```bash
python scripts/analyze_optimization.py
```

**Génère:**
- 📊 Graphiques comparatifs
- 📋 Tableau détaillé
- 📈 Analyse overfitting
- ⏱️ Statistiques temps

---

## 📊 Interpréter les Résultats

### Fichier Principal: `optimization_results_*.csv`

```csv
Rank,Model,MAE_Test,R2_Test,Overfitting_Ratio,...
1,LightGBM,3.234,0.9678,1.51,...
2,XGBoost,3.298,0.9665,1.48,...
3,Ridge,3.412,0.9634,1.33,...
```

**Colonnes clés:**
- **MAE_Test**: Erreur absolue moyenne (PLUS BAS = MEILLEUR) ⭐
- **R2_Test**: Variance expliquée (PLUS HAUT = MEILLEUR)
- **Overfitting_Ratio**: MAE_Test/MAE_Train (< 2.0 = excellent)

### Quel est le Meilleur Modèle?

**Cherchez:**
1. ✅ **MAE_Test le plus bas** (rang 1)
2. ✅ **Overfitting_Ratio < 2.0** (bon équilibre)
3. ✅ **R2_Test > 0.95** (excellente précision)

**Exemple:**
```
Rank 1: LightGBM
  MAE Test: 3.234 ans  ← Erreur moyenne de prédiction
  R² Test: 0.9678      ← Explique 96.78% de la variance
  Overfitting: 1.51x   ← Excellente généralisation!
```

---

## 🎯 Utiliser le Meilleur Modèle

### Charger et Prédire

```python
import joblib
import pandas as pd
import numpy as np

# 1. Charger le meilleur modèle (exemple: LightGBM)
model = joblib.load('results/optimization/best_lightgbm.joblib')
scaler = joblib.load('results/optimization/scaler.joblib')
imputer = joblib.load('results/optimization/imputer.joblib')

# 2. Préparer nouvelles données
X_new = pd.read_csv('mes_nouvelles_donnees.csv')  # Vos données
X_new = imputer.transform(X_new)
X_new = scaler.transform(X_new)

# 3. Prédire
ages_predits = model.predict(X_new)
print(f"Ages prédits: {ages_predits}")
```

---

## 💡 Options Avancées

### Optimiser Seulement Certains Modèles

```bash
# Seulement les 3 plus rapides
python scripts/hyperparameter_optimization.py --models Ridge Lasso ElasticNet --max-hours 2

# Seulement gradient boosting (les meilleurs généralement)
python scripts/hyperparameter_optimization.py --models XGBoost LightGBM CatBoost --max-hours 4
```

### Ajuster le Budget Temps

```bash
# Test rapide (1 heure)
python scripts/hyperparameter_optimization.py --max-hours 1 --top-k-features 1000

# Overnight (12 heures max)
python scripts/hyperparameter_optimization.py --max-hours 12
```

### Changer Nombre de Features

```bash
# Plus de features (meilleure précision, plus lent)
python scripts/hyperparameter_optimization.py --top-k-features 10000 --max-hours 10

# Moins de features (plus rapide)
python scripts/hyperparameter_optimization.py --top-k-features 2000 --max-hours 4
```

---

## ❓ FAQ

### Q: Combien de temps ça prend vraiment?

**R:** Dépend de votre configuration:
- **Mode rapide** (PCA 400, 3 modèles): ~2h
- **Mode standard** (5000 features, tous modèles): ~6-8h
- **Mode complet** (10000 features, tous modèles): ~10-12h

### Q: Combien de RAM nécessaire?

**R:**
- **Mode PCA**: 4 GB minimum, 8 GB recommandé
- **Mode standard (5000 features)**: 8 GB minimum, 16 GB recommandé
- **Mode complet (10000+ features)**: 16 GB minimum, 32 GB recommandé

### Q: Puis-je interrompre et reprendre?

**R:** Oui et non:
- ✅ **Vous pouvez interrompre** (Ctrl+C) à tout moment
- ✅ **Résultats complétés sont sauvegardés** (SQLite + joblib)
- ❌ **Impossible de reprendre exactement** où vous étiez
- ✅ **Mais vous pouvez relancer** et ignorer modèles déjà faits

### Q: Quel modèle choisir si plusieurs similaires?

**R:** Critères de choix:
1. **Performance proche** → Choisir le plus **simple** (Ridge > XGBoost en complexité)
2. **Même performance** → Choisir le plus **rapide** (LightGBM > CatBoost)
3. **Production** → Choisir le plus **stable** (Random Forest très stable)
4. **Interprétabilité** → Choisir **linéaire** (Ridge, Lasso, ElasticNet)

### Q: MAE de 3-4 ans, c'est bien?

**R:**
- **< 3 ans**: Excellent! État de l'art
- **3-4 ans**: Très bon! Comparable à Horvath/Hannum
- **4-5 ans**: Bon, acceptable
- **> 6 ans**: Problématique, revoir approche

### Q: Overfitting ratio, qu'est-ce qui est bon?

**R:**
- **< 1.5x**: Excellent, parfait équilibre
- **1.5-2.5x**: Très bon, généralisation acceptable
- **2.5-5.0x**: Limite, surveiller
- **> 5.0x**: Problème, overfitting sévère

---

## 🔧 Troubleshooting Rapide

### ❌ Erreur: "Out of Memory"

```bash
# Solution 1: Utiliser PCA
python scripts/hyperparameter_optimization.py --use-pca --pca-components 200

# Solution 2: Moins de features
python scripts/hyperparameter_optimization.py --top-k-features 1000
```

### ❌ Erreur: "Module 'lightgbm' not found"

```bash
pip install lightgbm catboost optuna
```

### ⚠️ Warning: "Trial pruned"

C'est normal! Optuna arrête les trials non prometteurs pour gagner du temps.

### 🐌 C'est trop lent!

```bash
# Option 1: Moins de modèles
python scripts/hyperparameter_optimization.py --models XGBoost LightGBM

# Option 2: PCA
python scripts/hyperparameter_optimization.py --use-pca --pca-components 200

# Option 3: Moins de features
python scripts/hyperparameter_optimization.py --top-k-features 2000
```

---

## 📈 Exemple Complet de Bout en Bout

```bash
# Terminal 1: Lancer optimisation
python scripts/hyperparameter_optimization.py

# [Attendre 6-8 heures... ☕]

# Terminal 2: Analyser résultats
python scripts/analyze_optimization.py

# Python: Utiliser meilleur modèle
python
>>> import joblib
>>> model = joblib.load('results/optimization/best_lightgbm.joblib')
>>> # Prédire avec nouvelles données...
```

---

## 🎓 Pour Aller Plus Loin

**Lire la documentation complète:**
```bash
cat HYPERPARAMETER_OPTIMIZATION.md  # Documentation détaillée
```

**Explorer base Optuna:**
```python
import optuna
storage = 'sqlite:///results/optimization/optuna_study.db'
studies = optuna.study.get_all_study_names(storage)
print(studies)  # Liste toutes les études
```

**Visualisations interactives Optuna:**
```python
import optuna
from optuna.visualization import plot_optimization_history

storage = 'sqlite:///results/optimization/optuna_study.db'
study = optuna.load_study(study_name='XGBoost_...', storage=storage)

# Historique optimisation
fig = plot_optimization_history(study)
fig.show()

# Importance paramètres
fig = plot_param_importances(study)
fig.show()
```

---

**Bonne optimisation! 🚀**

*Temps total lecture: 5 minutes*
*Temps total setup: 2 minutes*
*Temps total optimisation: 6-8 heures*
*Temps total analyse: 1 minute*

**Total: ~8 heures (dont 7h45 automatisé) pour trouver le MEILLEUR modèle possible!**
