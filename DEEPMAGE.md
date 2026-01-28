# DeepMAge - Deep Learning pour la Prédiction d'Âge Épigénétique

## 📚 Contexte Scientifique

**DeepMAge** est un modèle d'apprentissage profond inspiré de l'article de Galkin et al. (2021) qui utilise des réseaux de neurones profonds pour prédire l'âge biologique à partir de profils de méthylation de l'ADN.

### Références Clés

- **Galkin, F., et al. (2021).** "DeepMAge: A methylation aging clock developed with deep learning." *Aging and Disease*, 12(5), 1252-1262.
- Approche moderne utilisant PyTorch pour capturer les relations non-linéaires complexes dans les données de méthylation

---

## 🏗️ Architecture du Modèle

### Structure du Réseau

```
Input (5000 CpG sites)
    ↓
Linear(5000 → 512)
    ↓
BatchNorm1d(512)
    ↓
ReLU
    ↓
Dropout(p=0.3)
    ↓
Linear(512 → 1)
    ↓
Output (predicted age)
```

### Composants

1. **Couche d'entrée** : 5000 sites CpG (features sélectionnés)
2. **Couche cachée** : 512 neurones avec activation ReLU
3. **Batch Normalization** : Stabilise l'entraînement et accélère la convergence
4. **Dropout (30%)** : Régularisation pour prévenir l'overfitting
5. **Couche de sortie** : 1 neurone (âge prédit en années)

---

## ⚙️ Hyperparamètres

### Configuration par Défaut (`config/model_config.yaml`)

```yaml
models:
  deepmage_hidden_size: 512             # Nombre de neurones dans la couche cachée
  deepmage_dropout: 0.3                 # Probabilité de dropout
  deepmage_learning_rate: 0.001         # Learning rate pour Adam optimizer
  deepmage_batch_size: 32               # Taille des batchs
  deepmage_epochs: 100                  # Nombre maximum d'époques
  deepmage_early_stopping_patience: 10  # Patience pour early stopping
  deepmage_random_state: 42             # Seed pour reproductibilité
```

### Détails des Paramètres

- **hidden_size (512)** : Suffisamment large pour capturer les interactions complexes entre CpG sites
- **dropout (0.3)** : Régularisation modérée pour éviter l'overfitting sur petit dataset
- **learning_rate (0.001)** : Taux d'apprentissage standard pour Adam
- **batch_size (32)** : Compromis entre stabilité et vitesse d'entraînement
- **early_stopping_patience (10)** : Arrête si validation ne s'améliore pas pendant 10 époques

---

## 🔧 Implémentation Technique

### 1. Fichier Principal

**`src/models/deep_learning.py`** contient :
- `DeepMAge` : Classe PyTorch `nn.Module` définissant l'architecture
- `DeepMAgeRegressor` : Wrapper scikit-learn compatible pour intégration facile
- `create_deepmage_model()` : Factory function pour créer le modèle

### 2. Caractéristiques Clés

#### Initialisation des Poids
```python
# He initialization pour ReLU
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

#### Early Stopping
```python
# Arrête l'entraînement si validation ne s'améliore pas
if epoch_val_loss < best_val_loss:
    best_val_loss = epoch_val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= early_stopping_patience:
        break  # Stop training
```

#### Standardisation des Features
```python
# Standardise les features avant entraînement (μ=0, σ=1)
self.scaler_ = StandardScaler()
X_scaled = self.scaler_.fit_transform(X)
```

### 3. Détection Automatique GPU

Le modèle utilise automatiquement le GPU si disponible :
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Pour Ubuntu 24.04** :
```bash
# Vérifier disponibilité CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Si GPU NVIDIA disponible, installer PyTorch avec CUDA :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Utilisation

### Entraînement

```bash
# Le modèle DeepMAge est automatiquement inclus dans le pipeline
python scripts/train.py --config config/model_config.yaml

# Logs attendus :
# Training DeepMAge...
# Using validation set: train=272, val=48
# Epoch 10/100 - Train Loss: 45.2341 - Val Loss: 52.1234
# ...
# Early stopping at epoch 45
# Best epoch: 35 with val loss: 48.5678
```

### Prédiction

```python
import joblib
import numpy as np

# Charger le modèle entraîné
model = joblib.load("results/models/deepmage.joblib")

# Prédire sur nouvelles données
X_new = np.array([...])  # Shape: (n_samples, 5000)
ages_predicted = model.predict(X_new)
```

---

## 📊 Avantages de DeepMAge

### 1. Capture des Relations Non-Linéaires
- Les réseaux de neurones peuvent apprendre des interactions complexes entre sites CpG
- Contrairement aux modèles linéaires (Ridge, Lasso), pas besoin de spécifier manuellement les interactions

### 2. Regularisation Intégrée
- **Dropout** : Empêche le modèle de trop dépendre de certains neurones
- **Batch Normalization** : Réduit l'internal covariate shift
- **Early Stopping** : Arrête avant overfitting

### 3. Scalabilité
- Peut gérer des milliers de features efficacement
- Utilise GPU si disponible pour accélération

### 4. Performance Attendue
D'après la littérature (Galkin 2021) :
- **MAE** : ~3-4 ans sur données de validation
- **R²** : ~0.95-0.97
- **Meilleure performance** sur jeunes adultes (20-40 ans)

---

## 🔬 Comparaison avec Autres Approches

| Modèle | Type | Complexité | Interprétabilité | Performance Attendue |
|--------|------|------------|------------------|----------------------|
| **Horvath (2013)** | Elastic Net | Linéaire | ✅ Haute (coefficients) | MAE ~4 ans |
| **Hannum (2013)** | WGCNA + Linear | Linéaire | ✅ Haute | MAE ~4 ans |
| **PhenoAge (2018)** | Cox Regression | Semi-linéaire | ✅ Moyenne | Prédit mortalité |
| **DeepMAge (2021)** | Deep Neural Net | ⚠️ Non-linéaire | ❌ Faible (black box) | MAE ~3 ans |
| **Notre Ridge** | Ridge Regression | Linéaire | ✅ Haute | MAE ~3.4 ans |
| **Notre DeepMAge** | PyTorch DNN | ⚠️ Non-linéaire | ❌ Faible | MAE ~? ans (à tester) |

---

## ⚠️ Limitations et Considérations

### 1. Overfitting Risk
- **Dataset petit** : 400 samples seulement
- **Solution** : Early stopping + Dropout + Batch Normalization
- **Surveiller** : Train loss vs Validation loss

### 2. Interprétabilité
- Modèle "boîte noire"
- Difficile d'identifier quels sites CpG sont importants
- **Alternative pour interprétation** : Utiliser SHAP ou Integrated Gradients

### 3. Temps d'Entraînement
- Plus lent que Ridge/Lasso
- **CPU** : ~2-5 minutes
- **GPU** : ~30-60 secondes

### 4. Reproductibilité
- Nécessite seed fixe pour résultats identiques
- Variance due à initialisation aléatoire des poids

---

## 📈 Tuning des Hyperparamètres

### Scénarios d'Ajustement

#### Si Overfitting (Train Loss << Val Loss) :
```yaml
deepmage_dropout: 0.5                # Augmenter dropout
deepmage_early_stopping_patience: 5  # Réduire patience
deepmage_learning_rate: 0.0005       # Réduire learning rate
```

#### Si Underfitting (Train Loss et Val Loss hauts) :
```yaml
deepmage_hidden_size: 1024           # Plus de neurones
deepmage_epochs: 200                 # Plus d'époques
deepmage_learning_rate: 0.01         # Augmenter learning rate
```

#### Si Instabilité :
```yaml
deepmage_batch_size: 16              # Réduire batch size
deepmage_learning_rate: 0.0001       # Réduire learning rate
```

---

## 🧪 Validation

### Vérifier le Modèle Entraîné

```python
import joblib
import pandas as pd

# Charger modèle et résultats
model = joblib.load("results/models/deepmage.joblib")
metrics = pd.read_csv("results/metrics.csv")

# Afficher performance DeepMAge
deepmage_metrics = metrics[metrics["model"] == "DeepMAge"]
print(deepmage_metrics[["mae", "r2", "overfitting_ratio"]])

# Vérifier historique d'entraînement
if hasattr(model, 'training_history_'):
    history = pd.DataFrame(model.training_history_)
    print(history.tail(10))
```

### Critères de Succès

✅ **Bon modèle** si :
- MAE test < 5.0 ans
- R² > 0.90
- Overfitting ratio < 5.0x
- Early stopping avant époque 100 (indique bon réglage)

⚠️ **Problème** si :
- MAE test > 8.0 ans
- R² < 0.80
- Overfitting ratio > 10.0x
- Convergence après 100 époques (augmenter epochs)

---

## 🔄 Workflow Complet

### 1. Installation
```bash
# Installer PyTorch
pip install torch>=2.0.0

# Ou avec CUDA pour GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Entraînement
```bash
python scripts/train.py --config config/model_config.yaml
```

### 3. Visualisation
```bash
python app.py
# Ouvrir http://localhost:8050
# Sélectionner "DeepMAge" dans le dropdown
```

### 4. Export Rapport
- Cliquer sur "Export Report"
- Le rapport PDF incluera automatiquement DeepMAge dans les comparaisons

---

## 📚 Références Complètes

1. **Galkin, F., Mamoshina, P., Aliper, A., Putin, E., Moskalev, V., Gladyshev, V. N., & Zhavoronkov, A. (2021).** DeepMAge: A methylation aging clock developed with deep learning. *Aging and Disease*, 12(5), 1252-1262.

2. **Horvath, S. (2013).** DNA methylation age of human tissues and cell types. *Genome Biology*, 14(10), R115.

3. **Hannum, G., Guinney, J., Zhao, L., et al. (2013).** Genome-wide methylation profiles reveal quantitative views of human aging rates. *Molecular Cell*, 49(2), 359-367.

4. **Levine, M. E., Lu, A. T., Quach, A., et al. (2018).** An epigenetic biomarker of aging for lifespan and healthspan. *Aging*, 10(4), 573-591.

5. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** Deep Learning. MIT Press. *(Chapitre 7: Regularization)*

---

## 💡 Perspectives Futures

### Améliorations Possibles

1. **Architecture Plus Profonde**
   ```python
   Input → 512 → BatchNorm → ReLU → Dropout
         → 256 → BatchNorm → ReLU → Dropout
         → 128 → BatchNorm → ReLU → Dropout
         → 1
   ```

2. **Attention Mechanism**
   - Identifier automatiquement les sites CpG importants
   - Améliore interprétabilité

3. **Residual Connections**
   - Permet entraînement de réseaux plus profonds
   - Évite vanishing gradients

4. **Ensemble avec Modèles Linéaires**
   - Combiner DeepMAge + Ridge pour meilleure robustesse

---

**Date** : 2026-01-28
**Auteur** : Claude Opus 4.5
**Version** : 1.0
**Status** : ✅ Implémenté et prêt pour entraînement
