# Documentation Complète — Prédiction d'Âge par Méthylation de l'ADN

## Table des Matières

1. [Vue d'Ensemble du Projet](#vue-densemble-du-projet)
2. [Architecture du Code](#architecture-du-code)
3. [Fichiers et leurs Rôles](#fichiers-et-leurs-rôles)
4. [Pipeline de Données](#pipeline-de-données)
5. [Modèles Implémentés](#modèles-implémentés)
6. [Métriques d'Évaluation](#métriques-dévaluation)
7. [Guide d'Utilisation](#guide-dutilisation)
8. [Pistes d'Amélioration](#pistes-damélioration)

---

## Vue d'Ensemble du Projet

### Objectif
Développer des **horloges épigénétiques** capables de prédire l'âge chronologique d'un individu à partir de son profil de méthylation de l'ADN (données EPICv2, ~900,000 sites CpG).

### Concept Biologique
La méthylation de l'ADN (ajout d'un groupe méthyle CH₃ sur les cytosines des sites CpG) évolue de manière prévisible avec l'âge. Cette propriété permet de construire des modèles prédictifs appelés "horloges épigénétiques".

### Stack Technique
- **Python 3.10+**
- **Scikit-learn** : Modèles ML, prétraitement, évaluation
- **XGBoost** : Gradient boosting optimisé
- **Pandas/NumPy** : Manipulation des données
- **Plotly/Dash** : Visualisation interactive
- **SciPy** : Tests statistiques

---

## Architecture du Code

```
age_prediction_using_DNA_methylation/
│
├── Data/                          # Données brutes (non versionnées)
│   ├── annot_projet.csv           # Annotations échantillons (âge, genre, ethnicité)
│   ├── cpg_names_projet.csv       # Liste des noms de CpG
│   └── c_sample.csv               # Matrice de méthylation (CpG × échantillons)
│
├── results/                       # Résultats générés
│   ├── metrics.csv                # Métriques de tous les modèles
│   ├── predictions.csv            # Prédictions sur le test set
│   ├── annot_predictions.csv      # Annotations + prédictions (tous modèles)
│   ├── selected_cpgs.csv          # CpG sélectionnés
│   ├── coefficients_*.csv         # Coefficients des modèles linéaires
│   ├── report.md                  # Rapport markdown
│   ├── rapport_complet.tex        # Rapport LaTeX
│   ├── models/                    # Modèles sauvegardés (.joblib, .json)
│   └── plots/                     # Graphiques de diagnostic
│
├── assets/                        # Assets pour l'application web
│   └── style.css                  # Styles CSS
│
├── train_models.py                # 🔴 PRINCIPAL: Pipeline d'entraînement
├── app.py                         # Application Dash interactive
├── compare_imputation.py          # Comparaison des méthodes d'imputation
├── generate_latex_report.py       # Génération de rapport LaTeX
├── requirements.txt               # Dépendances Python
└── DOCUMENTATION.md               # Ce fichier
```

---

## Fichiers et leurs Rôles

### 1. `train_models.py` — Pipeline d'Entraînement Principal

**Fonction principale**: Entraîner et évaluer plusieurs modèles de prédiction d'âge.

#### Flux d'exécution:

```
[1] Chargement des données
    └── load_annotations() → DataFrame avec âge, genre, ethnicité
    └── load_cpg_names() → Liste des 900k noms de CpG

[2] Préparation des features
    └── select_top_k_cpgs() → Sélection des k CpG les plus corrélés avec l'âge
    └── add_demographic_features() → Ajout genre (binaire) + ethnicité (one-hot)
    └── Imputation des valeurs manquantes (SimpleImputer, mean)

[3] Entraînement des modèles
    └── build_models() → Liste des modèles à entraîner
    └── optimize_model() → Optimisation des hyperparamètres (optionnel)
    └── model.fit(X_train, y_train)

[4] Évaluation
    └── evaluate_model() → MAE, MAD, R², Corrélation, CV scores

[5] Sauvegarde
    └── metrics.csv, predictions.csv, annot_predictions.csv
    └── Modèles (.joblib), Plots, Rapport
```

#### Paramètres clés:

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--top-k` | 10000 | Nombre de CpG à sélectionner |
| `--feature-mode` | topk | Mode: `topk` (corrélation) ou `pca` |
| `--test-size` | 0.2 | Proportion du test set |
| `--optimize` | False | Activer l'optimisation des hyperparamètres |
| `--cv` | 5 | Nombre de folds pour la cross-validation |

#### Modèles implémentés:

```python
models = [
    "ElasticNet",        # Régression L1+L2
    "Lasso",             # Régression L1
    "Ridge",             # Régression L2
    "RandomForest",      # Bagging d'arbres
    "XGBoost",           # Boosting optimisé
    "AltumAge",          # MLP (deep learning)
]
```

---

### 2. `app.py` — Application Web Interactive

**Fonction**: Interface Dash pour explorer les résultats et comparer les modèles.

#### Structure de l'interface:

```
┌─────────────────────────────────────────────────────────────────┐
│ TOPBAR: Logo + Bouton Export LaTeX                              │
├──────────────┬──────────────────────────────────────────────────┤
│              │  HERO: Titre + Description                       │
│   SIDEBAR    │──────────────────────────────────────────────────│
│              │  TABS:                                           │
│  - Dropdown  │    [Comparaison] [Échantillons] [Contexte] [Réf] │
│    modèle    │──────────────────────────────────────────────────│
│              │  CONTENU TAB:                                    │
│  - Légende   │    - KPIs (Corrélation, MAE, R², Écart)         │
│    métriques │    - Graphiques (barres, scatter, box, histo)   │
│              │    - Analyses stratifiées (genre, âge, batch)   │
└──────────────┴──────────────────────────────────────────────────┘
```

#### Callbacks principaux:

| Callback | Entrée | Sortie |
|----------|--------|--------|
| `update_charts` | model-dropdown | Tous les graphiques + KPIs |
| `update_samples_table` | model-dropdown | Tableau des échantillons |
| `export_report` | btn-export | Fichier LaTeX téléchargeable |

#### Graphiques générés:

1. **Métriques Cohorte**:
   - MAE par modèle (barres)
   - R² par modèle (barres)
   - Scatter tous modèles
   - Régression modèle sélectionné

2. **Métriques Individuelles**:
   - Delta Age vs Âge chronologique
   - Histogramme Age Acceleration
   - Box plot erreurs (tous modèles)
   - Histogramme Delta Age

3. **Analyses Stratifiées**:
   - Non-linéarité (erreur vs âge + polynôme)
   - Différence par genre (box plot)
   - Variabilité par batch/chip (box plot)

---

### 3. `compare_imputation.py` — Comparaison des Méthodes d'Imputation

**Fonction**: Évaluer l'impact des différentes stratégies d'imputation sur les performances.

#### Méthodes comparées:

| Méthode | Description |
|---------|-------------|
| Mean | Remplacement par la moyenne |
| Median | Remplacement par la médiane |
| Most Frequent | Remplacement par la valeur la plus fréquente |
| KNN (k=5,10,20) | K plus proches voisins |
| Iterative (BayesianRidge) | Imputation itérative avec régression bayésienne |
| Iterative (ElasticNet) | Imputation itérative avec ElasticNet |

#### Métriques de comparaison:

- MAE sur le test set
- R² sur le test set
- MAE en cross-validation (5 folds)
- Temps d'imputation

---

### 4. `generate_latex_report.py` — Génération de Rapport LaTeX

**Fonction**: Créer un rapport LaTeX complet avec toutes les analyses.

#### Sections du rapport:

1. **Introduction et Contexte**
   - Définition méthylation ADN
   - Concept d'âge biologique
   - Horloges épigénétiques
   - Lien avec le cancer

2. **Données et Matériel**
   - Description de la cohorte
   - Variables disponibles
   - Haute dimensionnalité
   - Défis techniques

3. **Méthodologie**
   - Pipeline d'analyse
   - Algorithmes (ElasticNet, RF, XGBoost, MLP)
   - Sélection de variables
   - Gestion des covariables

4. **Résultats**
   - Tableaux de métriques
   - Analyse cohorte (corrélation, MAE)
   - Analyse individuelle (Delta Age, Age Acceleration)

5. **Analyses Stratifiées**
   - Non-linéarité selon l'âge
   - Différences selon le genre
   - Variabilité technique

6. **Conclusion et Perspectives**

---

## Pipeline de Données

### Flux complet:

```
┌─────────────────┐
│ annot_projet.csv│ → 400 échantillons avec âge, genre, ethnicité
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ c_sample.csv    │ → Matrice 900k CpG × 400 échantillons (valeurs β: 0-1)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRÉTRAITEMENT                                                    │
│ 1. Filtrage des CpG avec trop de valeurs manquantes (>5%)       │
│ 2. Sélection top-k CpG par corrélation avec l'âge               │
│ 3. Imputation des valeurs manquantes (KNN, k=5)                  │
│ 4. Ajout features démographiques (genre, ethnicité)              │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ X: n×p matrice  │ → n=400 échantillons, p=~10000 features
│ y: vecteur âge  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ SPLIT TRAIN/TEST (80/20)                                         │
│ X_train: 320×p    X_test: 80×p                                   │
│ y_train: 320      y_test: 80                                     │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ ENTRAÎNEMENT                                                     │
│ Pour chaque modèle:                                              │
│   1. (Optionnel) Optimisation hyperparamètres (RandomizedSearchCV)│
│   2. Fit sur X_train, y_train                                    │
│   3. Prédiction sur X_test                                       │
│   4. Calcul métriques (MAE, R², Corrélation)                     │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ RÉSULTATS       │ → metrics.csv, predictions.csv, modèles sauvegardés
└─────────────────┘
```

---

## Modèles Implémentés

### 1. ElasticNet (Régression Linéaire Régularisée)

```python
# Combinaison L1 (Lasso) + L2 (Ridge)
# Objectif: min ||y - Xβ||² + α(ρ||β||₁ + (1-ρ)||β||²/2)

ElasticNet(
    alpha=0.1,      # Force de régularisation
    l1_ratio=0.5,   # Balance L1/L2 (0.5 = équilibre)
    max_iter=50000, # Itérations max
)
```

**Avantages**: Sélection de variables, interprétabilité, gère la multicolinéarité
**Inconvénients**: Suppose une relation linéaire

### 2. Random Forest (Bagging)

```python
# Ensemble d'arbres entraînés sur des bootstrap samples
RandomForestRegressor(
    n_estimators=300,      # Nombre d'arbres
    max_depth=20,          # Profondeur max
    min_samples_split=5,   # Min échantillons pour split
    max_features="sqrt",   # Features par arbre: √p
)
```

**Avantages**: Robuste, gère les non-linéarités, peu d'hyperparamètres
**Inconvénients**: Moins interprétable, peut overfitter

### 3. XGBoost

```python
# Boosting: arbres entraînés séquentiellement sur les résidus
XGBRegressor(
    n_estimators=400,       # Nombre d'itérations
    learning_rate=0.05,     # Taux d'apprentissage
    max_depth=6,            # Profondeur des arbres
    subsample=0.8,          # Fraction d'échantillons par arbre
    colsample_bytree=0.8,   # Fraction de features par arbre
    reg_alpha=0.1,          # Régularisation L1
    reg_lambda=2.0,         # Régularisation L2
)
```

**Avantages**: Excellentes performances, régularisation intégrée
**Inconvénients**: Risque d'overfitting, plus lent à entraîner

### 4. AltumAge (MLP)

```python
# Réseau de neurones multicouche
MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),  # Architecture
    activation="relu",                 # Fonction d'activation
    alpha=0.001,                       # Régularisation L2
    early_stopping=True,               # Arrêt précoce
)
```

**Avantages**: Peut capturer des relations complexes
**Inconvénients**: Nécessite plus de données, moins interprétable

---

## Métriques d'Évaluation

### Métriques de Performance

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **MAE** | mean(\|y - ŷ\|) | Erreur moyenne en années |
| **MAD** | median(\|y - ŷ\|) | Erreur médiane (robuste aux outliers) |
| **RMSE** | √mean((y - ŷ)²) | Erreur quadratique (pénalise les gros écarts) |
| **R²** | 1 - SS_res/SS_tot | Variance expliquée (0-1) |
| **Corrélation** | corr(y, ŷ) | Force de la relation linéaire |

### Métriques Biologiques

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Delta Age** | ŷ - y | Différence âge prédit - chronologique |
| **Age Acceleration** | résidu(ŷ ~ y) | Vieillissement relatif à la population |
| **Écart moyen** | mean(ŷ - y) | Biais systématique du modèle |

### Métriques de Validation

| Métrique | Description |
|----------|-------------|
| **CV MAE** | MAE moyenne sur k-folds |
| **CV std** | Écart-type du MAE sur k-folds |
| **Overfitting ratio** | MAE_test / MAE_train (idéal ≈ 1) |

---

## Guide d'Utilisation

### Installation

```bash
# Cloner le projet
git clone <repo_url>
cd age_prediction_using_DNA_methylation

# Installer les dépendances
pip install -r requirements.txt
```

### Entraînement des modèles

```bash
# Entraînement standard
python train_models.py --top-k 10000

# Avec optimisation des hyperparamètres
python train_models.py --top-k 10000 --optimize --n-iter 30

# Avec PCA au lieu de top-k
python train_models.py --feature-mode pca --pca-components 400
```

### Comparaison des imputations

```bash
python compare_imputation.py --top-k 5000
```

### Génération du rapport LaTeX

```bash
python generate_latex_report.py

# Compiler le PDF
cd results
pdflatex rapport_complet.tex
pdflatex rapport_complet.tex  # 2ème passe pour table des matières
```

### Lancer l'application web

```bash
python app.py
# Ouvrir http://127.0.0.1:8050 dans un navigateur
```

---

## Pistes d'Amélioration

### 1. Amélioration des Données

- [ ] **Normalisation des batch effects**: Appliquer ComBat ou SVA pour corriger la variabilité technique entre chips
- [ ] **Augmentation de données**: Si possible, inclure plus d'échantillons
- [ ] **Filtrage avancé**: Utiliser des critères biologiques pour la sélection des CpG (ex: CpG dans des promoteurs)

### 2. Amélioration des Features

- [ ] **Sélection de features avancée**: 
  - Recursive Feature Elimination (RFE)
  - Boruta algorithm
  - SHAP-based selection
- [ ] **Features dérivées**: 
  - Agrégations par îlots CpG
  - Scores de voies biologiques
- [ ] **Embeddings**: 
  - Autoencoders pour réduction de dimensionnalité
  - Word2Vec sur séquences CpG

### 3. Amélioration des Modèles

- [ ] **Architectures deep learning**:
  - Convolutional Neural Networks (CNN) sur les régions génomiques
  - Attention mechanisms
  - Transformers adaptés aux données omiques
- [ ] **Ensemble avancés**:
  - Stacking avec méta-learner
  - Voting avec poids optimisés
- [ ] **Modèles spécifiques**:
  - Modèles séparés par genre
  - Modèles par tranche d'âge

### 4. Amélioration de l'Évaluation

- [ ] **Validation externe**: Tester sur des cohortes indépendantes
- [ ] **Calibration**: Vérifier et corriger la calibration des prédictions
- [ ] **Intervalles de confiance**: Bootstrap pour estimer l'incertitude

### 5. Amélioration de l'Application

- [ ] **Export PDF natif**: Intégrer ReportLab ou WeasyPrint
- [ ] **Comparaison interactive**: Permettre de comparer 2 modèles côte à côte
- [ ] **Upload de données**: Permettre de charger de nouvelles données

### 6. Optimisation du Code

- [ ] **Parallélisation**: Utiliser Dask pour le traitement des gros fichiers
- [ ] **Caching**: Mettre en cache les features pré-calculées
- [ ] **GPU**: Utiliser RAPIDS pour accélérer le preprocessing

---

## Références

1. Horvath, S. (2013). DNA methylation age of human tissues. *Genome Biology*, 14(10), R115.
2. Hannum, G., et al. (2013). Genome-wide methylation profiles. *Molecular Cell*, 49(2), 359-367.
3. Levine, M. E., et al. (2018). PhenoAge biomarker. *Aging*, 10(4), 573-591.
4. Lu, A. T., et al. (2019). GrimAge predictor. *Aging*, 11(2), 303-327.
5. de Lima Camillo, L. P., et al. (2021). AltumAge. *Aging and Disease*.

---

*Documentation générée automatiquement — DNAm Age Prediction Benchmark*
