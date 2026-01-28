# Scripts d'Optimisation - Guide de Choix

## 📋 Deux Scripts Disponibles

### 1. `hyperparameter_optimization.py` - Optimisation Standard

**Usage**: Optimisation rapide avec configuration fixe

```bash
python scripts/hyperparameter_optimization.py --use-pca --pca-components 200
```

**Caractéristiques**:
- ✅ Teste 1 seule configuration PCA
- ✅ Optimise tous les modèles
- ✅ Plus rapide (2-4h)
- ✅ Bon pour tests rapides

**Quand l'utiliser?**
- Vous savez déjà quel PCA utiliser
- Vous voulez un résultat rapide
- Vous testez le système

---

### 2. `hyperparameter_optimization_complete.py` - Optimisation Exhaustive ⭐

**Usage**: Recherche exhaustive du minimum GLOBAL

```bash
python scripts/hyperparameter_optimization_complete.py
```

**Caractéristiques**:
- ✅ Charge TOUTES les données CpG (~400,000 sites)
- ✅ Teste PLUSIEURS PCA: [50, 100, 150, 200, 250, 300, 350, 400]
- ✅ Pour CHAQUE PCA, optimise TOUS les modèles
- ✅ Trouve le minimum global absolu
- ✅ Comparaison complète PCA vs Modèles
- ⚠️ Plus long (6-8h)

**Quand l'utiliser?**
- Vous voulez LA meilleure configuration possible
- Vous ne savez pas quel PCA choisir
- Vous avez le temps (laissez tourner overnight)
- C'est pour votre projet final / publication

---

## 🎯 Recommandation

### Pour la Demande Initiale

Utilisez **`hyperparameter_optimization_complete.py`** car vous vouliez:
- ✅ "Maximum de données en entrée" → Charge tout
- ✅ "Tester PCA avec 50, 100, 150, ..., 400" → Grid search PCA
- ✅ "Trouver minimum global MAE" → Compare toutes configs
- ✅ "Rapport complet" → CSV détaillé avec PCA + modèle + métriques

### Comparaison Rapide

| Critère | Standard | Complete ⭐ |
|---------|----------|------------|
| **Données chargées** | 5000 features | TOUTES (~400k) |
| **Configs PCA testées** | 1 | 8 |
| **Modèles optimisés** | 9 | 9 × 8 = 72 |
| **Temps estimé** | 2-4h | 6-8h |
| **Trouve minimum global** | Non | Oui |
| **Mémoire requise** | 4-8 GB | 8-16 GB |

---

## 🚀 Lancement

### Script Complete (Recommandé pour Vous)

```bash
# Installation dépendances
pip install optuna lightgbm catboost

# Lancer optimisation exhaustive
python scripts/hyperparameter_optimization_complete.py

# Options disponibles:
python scripts/hyperparameter_optimization_complete.py \
    --data-dir Data \
    --max-hours 8 \
    --test-size 0.2 \
    --pca-configs 50 100 150 200 250 300 350 400
```

### Résultats

```
results/optimization_complete/
├── complete_results_YYYYMMDD_HHMMSS.csv  # TOUTES les configs
├── results_intermediate.csv               # Sauvegarde progressive
├── imputer.joblib
├── pca_50/
│   ├── pca_transformer.joblib
│   ├── ridge.joblib
│   ├── xgboost.joblib
│   └── ...
├── pca_100/
│   └── ...
├── pca_200/  ← Souvent le meilleur
│   └── ...
└── ...
```

### Analyse Résultats

```python
import pandas as pd

# Charger résultats complets
results = pd.read_csv('results/optimization_complete/complete_results_*.csv')

# Rank 1 = MINIMUM GLOBAL
best = results.iloc[0]
print(f"Meilleure config: PCA={best['pca_n_components']}, Modèle={best['model_name']}")
print(f"MAE Test: {best['mae_test']:.3f} ans")

# Analyser par PCA
by_pca = results.groupby('pca_n_components')['mae_test'].min()
print(by_pca)
```

---

## 📚 Documentation

- **Guide Junior** (pédagogique): `GUIDE_JUNIOR_COMPLET.md`
- **Documentation technique**: `HYPERPARAMETER_OPTIMIZATION.md`
- **Quick start**: `OPTIMIZATION_QUICKSTART.md`

---

## ✅ Résumé

**Votre demande** : "Prendre max données, tester PCA (50,100,...,400), trouver minimum global"

**Réponse** : Utilisez `hyperparameter_optimization_complete.py`

```bash
python scripts/hyperparameter_optimization_complete.py
```

Laissez tourner 6-8h et vous aurez **LA** meilleure configuration possible! 🎯
