# Rapport d'Entraînement — Prédiction d'Âge DNAm

## Configuration

| Paramètre | Valeur |
|-----------|--------|
| Mode de features | topk |
| Top-k CpG | 10000 |
| Composantes PCA | N/A |
| Taux missing max | 0.05 |
| Test size | 0.2 |
| Random state | 42 |
| Optimisation | Non |
| Temps total | 588.55 secondes |

## Résultats

### Classement des Modèles (par MAE)

| model        |    mae |   mad |    r2 |   correlation |   cv_mae |
|:-------------|-------:|------:|------:|--------------:|---------:|
| Ridge        |  3.399 | 2.761 | 0.961 |         0.982 |    3.376 |
| ElasticNet   |  3.625 | 3.021 | 0.955 |         0.979 |    3.545 |
| Lasso        |  3.769 | 2.915 | 0.952 |         0.977 |    4.006 |
| XGBoost      |  4.201 | 3.059 | 0.936 |         0.969 |    4.300 |
| RandomForest |  5.420 | 4.514 | 0.910 |         0.970 |    5.468 |
| AltumAge     | 11.592 | 9.904 | 0.558 |         0.829 |   11.915 |

### Meilleur Modèle: Ridge

- **MAE**: 3.40 années
- **MAD**: 2.76 années
- **R²**: 0.9614
- **Corrélation**: 0.9823
- **MAE CV**: 3.38 ± 0.43

### Analyse de l'Overfitting

| Modèle | MAE Train | MAE Test | Ratio |
|--------|-----------|----------|-------|
| Ridge | 0.01 | 3.40 | 363.73 |
| ElasticNet | 0.44 | 3.63 | 8.21 |
| Lasso | 0.76 | 3.77 | 4.98 |
| XGBoost | 0.01 | 4.20 | 554.54 |
| RandomForest | 2.16 | 5.42 | 2.50 |
| AltumAge | 2.43 | 11.59 | 4.77 |

## Interprétation

- Un ratio proche de 1.0 indique un bon équilibre biais-variance
- Un ratio > 1.5 suggère de l'overfitting (ajouter de la régularisation)
- Un ratio < 1.0 est inhabituel (vérifier les données)

## Fichiers Générés

- `metrics.csv` : Métriques détaillées de tous les modèles
- `predictions.csv` : Prédictions sur le test set
- `annot_predictions.csv` : Annotations avec prédictions pour tous les modèles
- `models/` : Modèles sauvegardés
- `plots/` : Graphiques de diagnostic
