# Guide Complet d'Optimisation - Pour Juniors en Data Science 🎓

## 📚 Table des Matières

1. [Introduction - Comprendre le Problème](#1-introduction)
2. [Concepts Fondamentaux](#2-concepts-fondamentaux)
3. [Pourquoi Optimiser les Hyperparamètres?](#3-pourquoi-optimiser)
4. [PCA et Réduction de Dimensionnalité](#4-pca-expliqué)
5. [Les Modèles ML Expliqués](#5-modèles-ml)
6. [Fonctionnement du Script](#6-fonctionnement-script)
7. [Interpréter les Résultats](#7-interpréter-résultats)
8. [Exemples Pratiques](#8-exemples-pratiques)
9. [FAQ et Pièges Courants](#9-faq)

---

## 1. Introduction - Comprendre le Problème

### 🎯 Qu'est-ce qu'on essaie de faire?

Imaginez que vous avez un thermomètre cassé qui donne parfois la bonne température, parfois non. Votre travail est de **trouver le meilleur moyen de prédire la température réelle** à partir des lectures du thermomètre et d'autres informations.

Dans notre cas:
- **Thermomètre cassé** = Profils de méthylation de l'ADN
- **Température réelle** = Âge chronologique de la personne
- **Notre travail** = Trouver le meilleur modèle mathématique pour prédire l'âge

### 📊 Nos Données

```
Données d'entrée (X):
  - ~400,000 sites CpG (positions sur l'ADN où on mesure la méthylation)
  - Valeurs entre 0 et 1 (0 = pas méthylé, 1 = totalement méthylé)
  - Features démographiques (sexe, etc.)

Données de sortie (y):
  - Âge en années (ex: 25, 45, 67)

Objectif:
  - Créer une fonction f(X) = y qui prédit l'âge le plus précisément possible
```

---

## 2. Concepts Fondamentaux

### 🧠 Machine Learning - Les Bases

#### Qu'est-ce qu'un Modèle ML?

Un modèle ML est une **fonction mathématique** qui transforme des inputs en outputs:

```python
# Exemple simple
age_prédit = modèle.predict(profil_méthylation)
# 45.3 = f([0.23, 0.67, 0.12, ...])
```

**Analogie**: C'est comme une recette de cuisine
- **Ingrédients** = Vos données (X)
- **Recette** = L'algorithme ML
- **Plat final** = Prédiction (y)

#### Train vs Test - Pourquoi Séparer?

```
+------------------+      +------------------+
|   Données        |      |                  |
|   Totales        |      |                  |
|   (400 samples)  | ---> |  Train (320)     | ---> Entraîner modèle
|                  |      |                  |
|                  |      +------------------+
|                  |
|                  |      +------------------+
|                  |      |                  |
|                  | ---> |  Test (80)       | ---> Évaluer modèle
+------------------+      |                  |
                          +------------------+
```

**Pourquoi?**
- **Train (80%)**: Données que le modèle "voit" pour apprendre
- **Test (20%)**: Données "cachées" pour vérifier s'il a vraiment appris

**Analogie**:
- Train = Réviser avec les exercices du livre
- Test = Examen avec de nouveaux exercices jamais vus

### 📉 Overfitting (Sur-apprentissage)

#### Qu'est-ce que c'est?

**Overfitting** = Quand le modèle "apprend par cœur" les données d'entraînement au lieu de comprendre les vrais patterns.

**Analogie**: Un étudiant qui mémorise toutes les réponses du livre sans comprendre le cours. Résultat:
- ✅ Excellent sur les exercices du livre (MAE train = 0.5)
- ❌ Désastreux sur l'examen (MAE test = 10.0)

#### Comment le Détecter?

```python
MAE train = 2.0 ans  # Très bon sur données entraînement
MAE test = 8.0 ans   # Mauvais sur nouvelles données

Overfitting Ratio = MAE_test / MAE_train = 8.0 / 2.0 = 4.0x
```

**Critères**:
- **< 1.5x**: Excellent (modèle généralise bien)
- **1.5-3.0x**: Bon (léger overfitting acceptable)
- **3.0-5.0x**: Limite (attention!)
- **> 5.0x**: Problème sévère (modèle inutilisable)

### 📊 Métriques de Performance

#### MAE (Mean Absolute Error)

**Définition**: Erreur moyenne en valeur absolue.

```python
# Exemple
y_réel = [25, 40, 60]
y_prédit = [27, 38, 65]

erreurs = |25-27| + |40-38| + |60-65| = 2 + 2 + 5 = 9
MAE = 9 / 3 = 3.0 ans
```

**Interprétation**: En moyenne, le modèle se trompe de 3 ans.

**Bon ou mauvais?**
- MAE < 3 ans: Excellent
- MAE 3-5 ans: Très bon
- MAE 5-10 ans: Acceptable
- MAE > 10 ans: Mauvais

#### R² (Coefficient de Détermination)

**Définition**: Pourcentage de la variance expliquée par le modèle.

```python
R² = 0.95  # Le modèle explique 95% de la variabilité de l'âge
R² = 0.50  # Le modèle explique seulement 50% (pas terrible)
```

**Interprétation visuelle**:

```
R² = 0.95 (excellent)          R² = 0.50 (moyen)

Âge prédit                     Âge prédit
    |  ●                           | ●  ●
    | ●                            |●  ●
    |●                             |● ●
    +------ Âge réel               +------ Âge réel

Points très proches             Points dispersés
de la diagonale                 (prédictions imprécises)
```

**Bon ou mauvais?**
- R² > 0.95: Excellent
- R² 0.90-0.95: Très bon
- R² 0.80-0.90: Bon
- R² < 0.80: Pas terrible

---

## 3. Pourquoi Optimiser les Hyperparamètres?

### 🎛️ Paramètres vs Hyperparamètres

#### Paramètres (Appris Automatiquement)

Ce sont les "poids" que le modèle apprend pendant l'entraînement.

```python
# Régression linéaire: y = w1*x1 + w2*x2 + ... + b
# w1, w2, ... = Paramètres (appris automatiquement)
```

#### Hyperparamètres (Vous Devez Choisir)

Ce sont les "réglages" que VOUS devez configurer avant l'entraînement.

```python
# Exemples d'hyperparamètres
Ridge(alpha=100)           # Combien de régularisation?
RandomForest(n_estimators=300)  # Combien d'arbres?
XGBoost(learning_rate=0.1)      # Vitesse d'apprentissage?
```

**Analogie**:
- **Paramètres** = Ce qu'un piano apprend (les notes à jouer)
- **Hyperparamètres** = Réglages du piano (accordage, pédale, volume)

### 🔍 Recherche Manuelle vs Automatique

#### Recherche Manuelle (Mauvaise Idée)

```python
# Vous testez manuellement
model1 = Ridge(alpha=1)      # MAE = 5.2
model2 = Ridge(alpha=10)     # MAE = 4.8
model3 = Ridge(alpha=100)    # MAE = 3.9
model4 = Ridge(alpha=1000)   # MAE = 4.2
# ...

# Problème:
# - Très long (des heures de travail manuel)
# - Vous pouvez manquer la meilleure valeur
# - Impossible de tester toutes les combinaisons
```

#### Recherche Automatique avec Optuna (Bonne Idée)

```python
# Optuna teste intelligemment
study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Optuna va tester:
# Trial 1: alpha=50      MAE=4.0
# Trial 2: alpha=200     MAE=3.7  ✓ Mieux!
# Trial 3: alpha=180     MAE=3.6  ✓ Encore mieux!
# ...
# Trial 100: alpha=172.3 MAE=3.2  ✓ Optimal!

# Avantages:
# ✅ Teste 100+ configurations en quelques minutes
# ✅ Apprend des essais précédents (bayésien)
# ✅ Trouve le minimum global
```

### 🧪 Optimisation Bayésienne (Comment Optuna Fonctionne)

**Analogie**: Chercher le point le plus bas dans une vallée brumeuse.

**Méthode naive (Grid Search)**:
- Teste tous les points méthodiquement
- Très lent si beaucoup de dimensions

**Méthode intelligente (Bayésienne - Optuna)**:
1. Teste quelques points aléatoires
2. Construit un "modèle" de la vallée
3. Teste les endroits prometteurs
4. Met à jour le modèle
5. Répète jusqu'à trouver le fond

```
Itération 1:           Itération 10:          Itération 50:
   ?  ?                  ▼                        ▼
    ?    ?             ▼  ▼                      ● ●
  ?   ?                  ▼                      ● ● ●
                                                  ●

Teste partout       Focus sur zone        Trouve minimum
aléatoirement       prometteuse           global!
```

---

## 4. PCA et Réduction de Dimensionnalité

### 🤔 Le Problème de la Grande Dimensionnalité

Nous avons **~400,000 sites CpG** pour seulement **400 échantillons**.

**Problème**: 400,000 variables >> 400 échantillons = **Curse of Dimensionality**

**Conséquences**:
1. **Overfitting garanti**: Le modèle peut "mémoriser" parfaitement
2. **Lenteur**: Calculs très longs
3. **Mémoire**: Besoin de 100+ GB RAM

**Analogie**:
Imaginez décrire une personne avec 400,000 caractéristiques (couleur de chaque cheveu, position de chaque cellule...) alors que vous n'avez vu que 400 personnes. Impossible de généraliser!

### 🔬 PCA (Principal Component Analysis)

#### Qu'est-ce que PCA fait?

PCA trouve les **directions les plus importantes** dans vos données.

**Analogie**: Photographier un objet 3D
- Objet 3D = Données originales (400,000 dimensions)
- Photo 2D = Données réduites (200 dimensions)
- Vous perdez un peu d'information, mais gardez l'essentiel

**Exemple concret**:

```
Données originales (4 variables):
  - Taille (cm)
  - Poids (kg)
  - Tour de taille (cm)
  - IMC

PCA trouve que ces 4 variables sont corrélées!
→ PC1 (80% variance) = "Corpulence générale"
→ PC2 (15% variance) = "Forme du corps"
→ PC3 (4% variance) = Bruit
→ PC4 (1% variance) = Bruit

On garde PC1 + PC2 (95% variance) et on jette PC3, PC4
4 variables → 2 composantes principales
```

#### Variance Expliquée

**Définition**: Combien d'information vous gardez après réduction.

```python
PCA(n_components=50)   → Variance = 0.75 (75% info gardée)
PCA(n_components=100)  → Variance = 0.85 (85% info gardée)
PCA(n_components=200)  → Variance = 0.92 (92% info gardée)
PCA(n_components=400)  → Variance = 0.97 (97% info gardée)
```

**Trade-off**:
- **Peu de composantes** (ex: 50):
  - ✅ Rapide, peu de mémoire
  - ❌ Perd beaucoup d'information

- **Beaucoup de composantes** (ex: 400):
  - ✅ Garde presque toute l'information
  - ❌ Plus lent, plus de mémoire, risque overfitting

### 🎯 Pourquoi Tester Plusieurs Configurations PCA?

**On ne sait pas à l'avance** quel nombre de composantes est optimal!

```
PCA 50:  MAE = 4.2  (trop peu d'info)
PCA 100: MAE = 3.8  ✓
PCA 150: MAE = 3.5  ✓ Meilleur!
PCA 200: MAE = 3.7  (commence à overfitter)
PCA 400: MAE = 4.5  (trop de dimensions, overfitting)
```

**Notre stratégie**: Tester systématiquement [50, 100, 150, 200, 250, 300, 350, 400] et garder le meilleur!

---

## 5. Les Modèles ML Expliqués (Pour Juniors)

### 🎓 Modèles Linéaires

#### Ridge Regression

**Équation**:
```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

**Principe**: Trouve la meilleure droite (ou hyperplan) qui passe au milieu des points.

**Régularisation L2**: Pénalise les coefficients trop grands.
```python
Loss = MSE + alpha * (w1² + w2² + ... + wn²)
```

**Hyperparamètre principal**: `alpha`
- `alpha` petit (ex: 0.1) → Peu de régularisation → Risque overfitting
- `alpha` grand (ex: 1000) → Forte régularisation → Modèle simple

**Quand l'utiliser?**
- ✅ Beaucoup de features corrélées
- ✅ Veut un modèle stable et interprétable
- ❌ Relations très non-linéaires

#### Lasso Regression

Similaire à Ridge mais régularisation L1:
```python
Loss = MSE + alpha * (|w1| + |w2| + ... + |wn|)
```

**Particularité**: Met certains coefficients exactement à 0 → **Sélection automatique de features**

**Quand l'utiliser?**
- ✅ Veut identifier les features importantes
- ✅ Veut un modèle sparse (peu de features actives)

#### ElasticNet

Combine Ridge (L2) + Lasso (L1):
```python
Loss = MSE + alpha * (l1_ratio*|w| + (1-l1_ratio)*w²)
```

**Hyperparamètres**:
- `alpha`: Force de régularisation totale
- `l1_ratio`: Mélange entre L1 et L2 (0 = Ridge pur, 1 = Lasso pur)

---

### 🌳 Modèles à Base d'Arbres

#### Random Forest

**Principe**: Crée plein d'arbres de décision et fait voter.

**Analogie**: Comité d'experts
- Chaque arbre = Un expert qui donne son avis
- Prédiction finale = Moyenne des avis

```
Arbre 1: 45 ans
Arbre 2: 47 ans    → Moyenne = 46 ans
Arbre 3: 46 ans
```

**Hyperparamètres principaux**:
- `n_estimators`: Nombre d'arbres (ex: 100, 300, 500)
- `max_depth`: Profondeur max des arbres (ex: 10, 20, 30)
- `min_samples_split`: Combien d'échantillons min pour split

**Avantages**:
- ✅ Robuste, peu d'overfitting naturellement
- ✅ Capture relations non-linéaires
- ✅ Gère bien les données manquantes

**Inconvénients**:
- ❌ Moins interprétable que modèles linéaires
- ❌ Plus lent à entraîner

#### Gradient Boosting (XGBoost, LightGBM, CatBoost)

**Principe**: Construit des arbres séquentiellement, chaque arbre corrige les erreurs du précédent.

**Analogie**: Étudiant qui s'améliore
1. Premier arbre → Prédictions médiocres
2. Deuxième arbre → Apprend des erreurs du premier
3. Troisième arbre → Corrige les erreurs restantes
4. ...

```
Arbre 1: Prédit 40 (erreur = +5)
Arbre 2: Apprend à prédire cette erreur de +5
Arbre 3: Affine encore...
→ Prédiction finale = Somme de tous les arbres
```

**Hyperparamètres**:
- `n_estimators`: Nombre d'arbres
- `learning_rate`: Vitesse d'apprentissage (petit = plus prudent)
- `max_depth`: Profondeur des arbres
- `reg_alpha`, `reg_lambda`: Régularisation L1, L2

**Différences entre variantes**:
- **XGBoost**: Le plus populaire, très performant
- **LightGBM**: Plus rapide, économe en mémoire
- **CatBoost**: Bon avec données catégorielles

---

### 🧠 Réseaux de Neurones

#### MLP (Multi-Layer Perceptron)

**Principe**: Réseau de neurones artificiels organisés en couches.

```
Input Layer       Hidden Layers      Output Layer

   x1 ───┐
         ├──→ [Neuron] ───┐
   x2 ───┤                ├──→ [Neuron] ──→ y (âge)
         ├──→ [Neuron] ───┘
   x3 ───┘
```

**Hyperparamètres**:
- `hidden_layer_sizes`: Nombre et taille des couches (ex: (128, 64))
- `activation`: Fonction d'activation (relu, tanh, sigmoid)
- `alpha`: Régularisation L2
- `learning_rate_init`: Vitesse d'apprentissage

**Avantages**:
- ✅ Capture relations très complexes et non-linéaires
- ✅ Très flexible

**Inconvénients**:
- ❌ "Boîte noire" (difficile à interpréter)
- ❌ Risque d'overfitting si mal configuré
- ❌ Plus long à entraîner

---

## 6. Fonctionnement du Script

### 📋 Vue d'Ensemble du Workflow

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: Chargement Données                            │
│  - Charger TOUTES les données CpG (~400,000 sites)     │
│  - Ajouter features démographiques                      │
│  - Imputer valeurs manquantes                           │
│  - Split train/test (80/20)                            │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 2: PCA Grid Search                               │
│                                                         │
│  Pour n_components in [50, 100, 150, ..., 400]:       │
│    ├─ Appliquer PCA(n_components)                      │
│    ├─ Calculer variance_expliquée                      │
│    └─ Pour chaque modèle (Ridge, XGBoost, ...):       │
│         ├─ Optimiser hyperparamètres (Optuna)          │
│         ├─ Entraîner meilleur modèle                   │
│         ├─ Évaluer sur test                            │
│         └─ Sauvegarder résultats                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 3: Analyse Finale                                │
│  - Comparer TOUTES les configurations                  │
│  - Trouver le MINIMUM GLOBAL (MAE le plus bas)         │
│  - Générer rapport complet                             │
└─────────────────────────────────────────────────────────┘
```

### 🔍 Détails Techniques

#### Phase 1: Chargement Intelligent des Données

```python
def load_all_cpg_data_chunked(data_path, sample_ids, chunk_size=1000):
    """
    Charge toutes les données par morceaux pour éviter saturation mémoire.

    Pourquoi chunked?
    - Fichier CSV = ~10 GB
    - Charger tout d'un coup → Out of Memory
    - Charger par chunks de 1000 lignes → OK!
    """
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        chunks.append(chunk.loc[:, sample_ids])

    return pd.concat(chunks)
```

**Astuce Mémoire**:
```
400,000 sites × 400 samples × 8 bytes (float64) = 1.28 GB
+ Overhead pandas = ~2-3 GB en mémoire

Avec PCA 200 composantes:
200 × 400 × 8 bytes = 0.64 MB (!)
→ Réduction de mémoire de 2000x !
```

#### Phase 2: PCA et Optimisation

```python
# Pour chaque configuration PCA
for n_components in [50, 100, 150, 200, 250, 300, 350, 400]:

    # 1. Appliquer PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_train)

    # 2. Vérifier variance
    variance = pca.explained_variance_ratio_.sum()
    print(f"PCA {n_components}: {variance:.2%} variance expliquée")

    # 3. Optimiser chaque modèle
    for model_name in ["Ridge", "XGBoost", "LightGBM", ...]:

        # Optuna trouve les meilleurs hyperparamètres
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_fonction, n_trials=50)

        # Entraîner avec meilleurs params
        best_model = create_model(study.best_params)
        best_model.fit(X_pca, y_train)

        # Évaluer
        mae = evaluate(best_model, X_pca_test, y_test)

        # Sauvegarder
        results.append({
            'pca': n_components,
            'model': model_name,
            'mae': mae,
            ...
        })
```

#### Phase 3: Trouver le Minimum Global

```python
# Trier par MAE croissant
results_df = results_df.sort_values('mae_test')

# Le premier = MINIMUM GLOBAL
best = results_df.iloc[0]

print(f"Meilleure config:")
print(f"  PCA: {best['pca_n_components']}")
print(f"  Modèle: {best['model_name']}")
print(f"  MAE: {best['mae_test']:.3f}")
```

---

## 7. Interpréter les Résultats

### 📊 Format des Résultats

```csv
rank,pca_n_components,model_name,mae_test,r2_test,overfitting_ratio,pca_variance_explained
1,200,LightGBM,3.234,0.9678,1.51,0.9234
2,150,XGBoost,3.298,0.9665,1.48,0.8956
3,200,Ridge,3.412,0.9634,1.33,0.9234
```

### 🎯 Comment Choisir le Meilleur?

#### Étape 1: Regarder le Rank 1

Le modèle de rank 1 a la **MAE Test la plus basse** = Meilleure précision.

#### Étape 2: Vérifier l'Overfitting

```python
if overfitting_ratio < 2.0:
    print("✅ Excellent! Modèle généralise bien")
elif overfitting_ratio < 3.0:
    print("✓ Bon, léger overfitting acceptable")
else:
    print("⚠️ Attention, overfitting problématique")
```

#### Étape 3: Analyser PCA

```python
if pca_variance_explained > 0.90:
    print("✅ PCA garde beaucoup d'information")
elif pca_variance_explained > 0.80:
    print("✓ Acceptable")
else:
    print("⚠️ PCA jette trop d'information")
```

### 📈 Comparaison Multi-Critères

Ne regardez pas QUE le MAE! Équilibrez plusieurs critères:

```
Modèle A: MAE=3.2, R²=0.97, Overfitting=3.5x, PCA=200
Modèle B: MAE=3.3, R²=0.96, Overfitting=1.4x, PCA=150

Lequel choisir?
→ Modèle B! Légèrement moins précis mais bien meilleure généralisation
```

### 🔬 Analyse par PCA

Regardez la tendance:

```
PCA 50:  MAE = 4.5  (pas assez d'info)
PCA 100: MAE = 3.8
PCA 150: MAE = 3.4  ← Plateau commence
PCA 200: MAE = 3.3  ← Optimal
PCA 250: MAE = 3.5  (commence overfitting)
PCA 400: MAE = 4.2  (trop de dimensions)
```

**Insight**: Le "sweet spot" est souvent vers 150-250 composantes.

---

## 8. Exemples Pratiques

### 🎓 Exemple 1: Lire et Comprendre les Résultats

```python
import pandas as pd

# Charger résultats
results = pd.read_csv('results/optimization_complete/complete_results_*.csv')

# Top 5
print(results.head(5))
```

**Output**:
```
   rank  pca  model      mae_test  r2_test  overfit  variance
   1     200  LightGBM   3.234     0.9678   1.51     0.9234
   2     150  XGBoost    3.298     0.9665   1.48     0.8956
   3     200  Ridge      3.412     0.9634   1.33     0.9234
   4     150  ElasticNet 3.487     0.9612   1.62     0.8956
   5     250  LightGBM   3.523     0.9598   1.58     0.9401
```

**Interprétation**:
1. **Meilleur modèle**: LightGBM avec PCA 200
2. **MAE 3.234 ans**: Excellent! Erreur moyenne de ~3 ans
3. **R² 0.9678**: Explique 96.78% de la variance
4. **Overfitting 1.51x**: Excellente généralisation
5. **Variance 0.9234**: PCA garde 92% de l'information

### 🎓 Exemple 2: Utiliser le Meilleur Modèle

```python
import joblib

# 1. Charger le modèle gagnant
model_package = joblib.load('results/optimization_complete/pca_200/lightgbm.joblib')
model = model_package['model']
scaler = model_package['scaler']

# 2. Charger PCA et imputer
pca = joblib.load('results/optimization_complete/pca_200/pca_transformer.joblib')
imputer = joblib.load('results/optimization_complete/imputer.joblib')

# 3. Préparer nouvelles données
X_new = pd.read_csv('mes_nouvelles_donnees.csv')

# 4. Pipeline de transformation
X_new = imputer.transform(X_new)      # Imputer missing values
X_new = pca.transform(X_new)          # Réduire dimensions
X_new = scaler.transform(X_new)       # Standardiser

# 5. Prédire!
ages = model.predict(X_new)
print(f"Ages prédits: {ages}")
# [34.2, 56.8, 23.1, ...]
```

### 🎓 Exemple 3: Analyser Impact PCA

```python
# Grouper par PCA
pca_analysis = results.groupby('pca_n_components').agg({
    'mae_test': 'min',  # Meilleur MAE pour chaque PCA
    'model_name': 'first',  # Quel modèle?
    'pca_variance_explained': 'first'
})

print(pca_analysis)
```

**Output**:
```
pca   mae_min  best_model  variance
50    4.523    Ridge       0.7456
100   3.892    LightGBM    0.8512
150   3.298    XGBoost     0.8956
200   3.234    LightGBM    0.9234  ← Optimal!
250   3.523    LightGBM    0.9401
300   3.745    Ridge       0.9578
350   4.012    ElasticNet  0.9689
400   4.434    Ridge       0.9756
```

**Observation**:
- PCA 200 = Meilleur compromis variance/performance
- Au-delà de 200, on commence à overfitter
- En dessous de 150, on perd trop d'information

---

## 9. FAQ et Pièges Courants

### ❓ Questions Fréquentes

#### Q1: Combien de temps ça va prendre?

**R**: Dépend de votre config:

```
8 configs PCA × 9 modèles × 50 trials = 3600 optimisations

Estimation temps:
- Modèles linéaires (Ridge, Lasso): ~5 min × 8 PCA = 40 min
- Random Forest: ~15 min × 8 PCA = 2h
- XGBoost/LightGBM: ~10 min × 8 PCA = 1h20
- MLP: ~20 min × 8 PCA = 2h40

Total estimé: 6-8 heures
```

#### Q2: Combien de RAM minimum?

**R**:
- Chargement données complètes: ~3-4 GB
- Après PCA 200: ~500 MB
- **Minimum recommandé**: 8 GB RAM
- **Confortable**: 16 GB RAM

#### Q3: Pourquoi mon modèle a MAE=10+ ans?

**R**: Plusieurs causes possibles:

1. **Overfitting sévère** → Vérifier ratio
2. **Pas assez de données** → Vérifier train size
3. **Mauvais hyperparamètres** → Laisser Optuna optimiser plus longtemps
4. **Trop peu de composantes PCA** → Essayer PCA plus élevé

#### Q4: PCA 400 devrait être meilleur, non?

**R**: **NON!** Plus de composantes ≠ Mieux

```
PCA 400 signifie:
- 400 dimensions (très élevé pour 320 samples train)
- Risque d'overfitting élevé
- Le modèle peut "mémoriser" au lieu de généraliser

PCA 200 c'est souvent optimal:
- Assez de dimensions pour capturer l'information
- Pas trop pour éviter overfitting
```

### ⚠️ Pièges Courants

#### Piège #1: Regarder Seulement MAE Train

```python
❌ MAUVAIS:
Modèle A: MAE train = 1.2  ← "Wow c'est bon!"

✅ BON:
Modèle A: MAE train = 1.2, MAE test = 8.5  ← Overfitting!
Modèle B: MAE train = 3.5, MAE test = 3.8  ← Meilleur!
```

**Leçon**: Toujours vérifier MAE test et overfitting ratio.

#### Piège #2: Appliquer PCA AVANT Split Train/Test

```python
❌ MAUVAIS (Data Leakage):
pca.fit(X_complet)  # PCA voit les données test!
X_pca = pca.transform(X_complet)
X_train, X_test = split(X_pca)

✅ BON:
X_train, X_test = split(X_complet)
pca.fit(X_train)  # PCA ne voit QUE train
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

**Leçon**: Toujours fitter les transformations sur train uniquement!

#### Piège #3: Oublier de Standardiser

```python
❌ MAUVAIS:
model.fit(X_pca, y)  # Features ont échelles différentes

✅ BON:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
model.fit(X_scaled, y)
```

**Pourquoi?** Beaucoup de modèles (SVM, MLP, Ridge...) sont sensibles à l'échelle des features.

#### Piège #4: Comparer des Modèles sur Différents Splits

```python
❌ MAUVAIS:
# Jour 1
X_train1, X_test1 = split(X, random_state=42)
Model A sur X_train1 → MAE = 3.5

# Jour 2 (différent random_state!)
X_train2, X_test2 = split(X, random_state=123)
Model B sur X_train2 → MAE = 3.3

# ❌ Vous ne pouvez PAS comparer!

✅ BON:
# Même split pour tous
X_train, X_test = split(X, random_state=42)
Model A sur X_train → MAE = 3.5
Model B sur X_train → MAE = 3.3  # Comparable!
```

---

## 🎓 Résumé - Ce Que Vous Avez Appris

### Concepts Clés

✅ **Overfitting**: Modèle mémorise au lieu d'apprendre
✅ **Train/Test Split**: Séparer pour évaluer généralisation
✅ **MAE**: Erreur moyenne en années
✅ **R²**: % variance expliquée
✅ **PCA**: Réduction dimensions tout en gardant l'info
✅ **Hyperparamètres**: Réglages à optimiser
✅ **Optimisation Bayésienne**: Recherche intelligente avec Optuna
✅ **Variance Expliquée**: Combien d'info garde PCA

### Workflow Complet

```
1. Charger TOUTES les données
2. Pour chaque PCA config (50, 100, ..., 400):
     a. Réduire dimensions
     b. Pour chaque modèle (Ridge, XGBoost, ...):
          i. Optimiser hyperparamètres (Optuna)
          ii. Entraîner meilleur modèle
          iii. Évaluer sur test
          iv. Sauvegarder résultats
3. Comparer TOUTES les configs
4. Choisir celle avec MAE test minimal ET bon overfitting ratio
```

### Checklist Avant de Lancer

- [ ] J'ai au moins 8 GB RAM disponible
- [ ] J'ai installé toutes les dépendances (optuna, lightgbm, catboost)
- [ ] Je comprends ce qu'est l'overfitting
- [ ] Je sais interpréter MAE et R²
- [ ] Je comprends pourquoi on teste plusieurs PCA
- [ ] J'ai du temps (6-8h de calcul)

---

## 📚 Pour Aller Plus Loin

### Ressources Recommandées

**Livres**:
- "Hands-On Machine Learning" - Aurélien Géron (EXCELLENT pour débutants)
- "The Elements of Statistical Learning" - Hastie et al. (Plus avancé)

**Cours en ligne**:
- Andrew Ng - Machine Learning (Coursera) - GRATUIT
- Fast.ai - Practical Deep Learning - GRATUIT

**Documentation**:
- Scikit-learn User Guide: https://scikit-learn.org
- Optuna Documentation: https://optuna.readthedocs.io

---

**Auteur**: Claude Opus 4.5
**Date**: 2026-01-28
**Version**: 1.0
**Public**: Étudiants juniors en Data Science / ML
