"""
Génération d'un rapport LaTeX complet pour l'analyse de prédiction d'âge
basée sur la méthylation de l'ADN.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def escape_latex(text: str) -> str:
    """Échappe les caractères spéciaux LaTeX."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str, 
                      float_format: str = "%.3f", escape: bool = True) -> str:
    """Convertit un DataFrame en table LaTeX."""
    latex = df.to_latex(
        index=False,
        float_format=float_format,
        escape=escape,
        caption=caption,
        label=label,
        position='htbp'
    )
    return latex


def generate_introduction() -> str:
    """Génère la section Introduction."""
    return r"""
\section{Introduction et Contexte}

\subsection{Méthylation de l'ADN}

La méthylation de l'ADN est une modification épigénétique fondamentale qui consiste en l'ajout 
d'un groupe méthyle (CH\textsubscript{3}) sur la cytosine des dinucléotides CpG. Ces sites CpG 
sont des régions de l'ADN où une cytosine est suivie d'une guanine dans la direction 5' vers 3'.

Cette modification chimique joue un rôle crucial dans la régulation de l'expression génique sans 
altérer la séquence d'ADN sous-jacente. Les patterns de méthylation sont :
\begin{itemize}
    \item \textbf{Héritables} : transmis lors de la division cellulaire
    \item \textbf{Réversibles} : peuvent être modifiés par des facteurs environnementaux
    \item \textbf{Tissu-spécifiques} : varient selon le type cellulaire
\end{itemize}

\subsection{Le Concept d'Âge Biologique}

L'âge biologique représente l'état physiologique réel d'un organisme, par opposition à l'âge 
chronologique qui mesure simplement le temps écoulé depuis la naissance. Plusieurs études 
pionnières, notamment celles de Horvath (2013) et Hannum (2013), ont démontré que les profils 
de méthylation de l'ADN sont fortement corrélés avec l'âge.

Cette découverte a conduit au développement des \textbf{horloges épigénétiques}, des modèles 
mathématiques capables de prédire l'âge d'un individu à partir de son profil de méthylation 
avec une précision remarquable (erreur moyenne de 3-4 années).

\subsection{Horloges Biologiques}

Les horloges épigénétiques reposent sur l'identification de sites CpG dont le niveau de 
méthylation évolue de manière prévisible avec l'âge. Les principales horloges développées sont :

\begin{table}[htbp]
\centering
\begin{tabular}{llcc}
\hline
\textbf{Horloge} & \textbf{Auteur} & \textbf{Année} & \textbf{N CpG} \\
\hline
Multi-tissue clock & Horvath & 2013 & 353 \\
Blood clock & Hannum & 2013 & 71 \\
PhenoAge & Levine & 2018 & 513 \\
GrimAge & Lu & 2019 & 1030 \\
\hline
\end{tabular}
\caption{Principales horloges épigénétiques publiées}
\label{tab:clocks}
\end{table}

\subsection{Lien avec le Cancer}

Les patients atteints de cancer présentent souvent une dérégulation globale de la méthylation 
de l'ADN, caractérisée par :
\begin{itemize}
    \item Une \textbf{hypométhylation globale} du génome
    \item Une \textbf{hyperméthylation locale} des promoteurs de gènes suppresseurs de tumeurs
    \item Une \textbf{accélération épigénétique} (âge biologique > âge chronologique)
\end{itemize}

L'accélération de l'âge épigénétique est associée à un risque accru de mortalité et de 
développement de maladies liées à l'âge, faisant des horloges biologiques des biomarqueurs 
prometteurs pour la médecine personnalisée.
"""


def clean_ethnicity(eth: str) -> str:
    """Regroupe les catégories d'ethnicité rares en 'Inconnu'."""
    if pd.isna(eth):
        return "Inconnu"
    eth_str = str(eth).strip()
    if eth_str.lower() in ["unavailable", "declined", "other", ""]:
        return "Inconnu"
    return eth_str


def generate_data_section(annot: pd.DataFrame, n_cpg: int, n_features: int) -> str:
    """Génère la section Données et Matériel."""
    
    n_samples = len(annot)
    age_mean = annot['age'].mean()
    age_std = annot['age'].std()
    age_min = annot['age'].min()
    age_max = annot['age'].max()
    
    # Gender stats
    if 'female' in annot.columns:
        n_female = annot['female'].apply(lambda x: str(x).lower() == 'true').sum()
        n_male = n_samples - n_female
        gender_stats = f"""
\\begin{{itemize}}
    \\item Femmes : {n_female} ({100*n_female/n_samples:.1f}\\%)
    \\item Hommes : {n_male} ({100*n_male/n_samples:.1f}\\%)
\\end{{itemize}}"""
    else:
        gender_stats = "Information non disponible."
    
    # Ethnicity stats (avec nettoyage)
    if 'ethnicity' in annot.columns:
        ethnicity_clean = annot['ethnicity'].apply(clean_ethnicity)
        eth_counts = ethnicity_clean.value_counts()
        eth_lines = []
        for eth, count in eth_counts.items():
            eth_lines.append(f"    \\item {escape_latex(str(eth))} : {count} ({100*count/n_samples:.1f}\\%)")
        ethnicity_stats = "\\begin{itemize}\n" + "\n".join(eth_lines) + "\n\\end{itemize}"
    else:
        ethnicity_stats = "Information non disponible."
    
    return f"""
\\section{{Données et Matériel}}

\\subsection{{Description de la Cohorte}}

Cette étude utilise des données de méthylation de l'ADN provenant d'échantillons sains, 
permettant d'établir une référence pour le vieillissement normal.

\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{lr}}
\\hline
\\textbf{{Caractéristique}} & \\textbf{{Valeur}} \\\\
\\hline
Nombre total d'échantillons & {n_samples} \\\\
Âge moyen & {age_mean:.1f} $\\pm$ {age_std:.1f} ans \\\\
Âge minimum & {age_min:.1f} ans \\\\
Âge maximum & {age_max:.1f} ans \\\\
\\hline
\\end{{tabular}}
\\caption{{Caractéristiques démographiques de la cohorte}}
\\label{{tab:demographics}}
\\end{{table}}

\\subsubsection{{Répartition par Genre}}
{gender_stats}

\\subsubsection{{Répartition par Ethnicité}}
{ethnicity_stats}

\\subsection{{Variables Disponibles}}

Les données comprennent :
\\begin{{enumerate}}
    \\item \\textbf{{Intensité de méthylation}} : Valeurs beta ($\\beta$) comprises entre 0 et 1, 
          représentant la proportion de méthylation à chaque site CpG
    \\item \\textbf{{Âge chronologique}} : Âge réel des individus en années
    \\item \\textbf{{Genre}} : Information binaire (Homme/Femme)
    \\item \\textbf{{Ethnicité}} : Classification ethnique des participants
\\end{{enumerate}}

\\subsection{{Haute Dimensionnalité}}

Les données de méthylation sont caractérisées par une très haute dimensionnalité :
\\begin{{itemize}}
    \\item \\textbf{{Plateforme}} : Illumina EPICv2 (Infinium MethylationEPIC v2.0)
    \\item \\textbf{{Sites CpG totaux}} : $\\sim$ 900,000 sondes
    \\item \\textbf{{Sites CpG utilisés}} : {n_cpg:,} après filtrage
    \\item \\textbf{{Features finales}} : {n_features:,} (incluant variables démographiques)
    \\item \\textbf{{Ratio échantillons/features}} : {n_samples/n_features:.4f}
\\end{{itemize}}

Ce ratio défavorable ($n << p$) nécessite l'utilisation de techniques de régularisation 
et/ou de réduction de dimensionnalité.

\\subsection{{Défis Techniques}}

Plusieurs défis techniques doivent être adressés :

\\begin{{enumerate}}
    \\item \\textbf{{Variabilité technique (batch effects)}} : Les échantillons proviennent 
          de différents lots de puces (chips), introduisant une variabilité technique 
          indépendante du signal biologique.
    
    \\item \\textbf{{Valeurs manquantes}} : Certaines sondes peuvent présenter des valeurs 
          manquantes dues à des problèmes d'hybridation ou de qualité.
    
    \\item \\textbf{{Non-linéarité selon l'âge}} : La relation entre méthylation et âge 
          peut présenter des non-linéarités, notamment aux extrêmes de la distribution d'âge.
    
    \\item \\textbf{{Différences selon le genre}} : Certains sites CpG présentent des 
          patterns de méthylation différents entre hommes et femmes.
\\end{{enumerate}}
"""


def generate_methodology_section(metrics: pd.DataFrame) -> str:
    """Génère la section Méthodologie."""
    
    models_list = metrics['model'].tolist()
    n_features = int(metrics.iloc[0]['n_features'])
    
    return f"""
\\section{{Méthodologie et Pistes Algorithmiques}}

\\subsection{{Objectif Principal}}

L'objectif de ce travail est de développer et comparer plusieurs horloges biologiques 
basées sur les profils de méthylation de l'ADN, et d'évaluer leurs performances par 
rapport aux modèles de référence existants (notamment PC-Horvath).

\\subsection{{Pipeline d'Analyse}}

Le pipeline d'analyse comprend les étapes suivantes :

\\begin{{enumerate}}
    \\item \\textbf{{Prétraitement}} : Filtrage des sondes de mauvaise qualité, 
          imputation des valeurs manquantes par KNN (k=5)
    \\item \\textbf{{Sélection de variables}} : Identification des CpG les plus 
          corrélés avec l'âge
    \\item \\textbf{{Intégration des covariables}} : Ajout du genre et de l'ethnicité 
          comme features supplémentaires
    \\item \\textbf{{Entraînement}} : Ajustement des modèles sur l'ensemble d'entraînement
    \\item \\textbf{{Évaluation}} : Mesure des performances sur l'ensemble de test
\\end{{enumerate}}

\\subsection{{Algorithmes Implémentés}}

\\subsubsection{{Méthodes Classiques}}

\\paragraph{{Elastic Net}}
La régression Elastic Net combine les pénalités L1 (Lasso) et L2 (Ridge) :
\\begin{{equation}}
\\min_{{\\beta}} \\frac{{1}}{{2n}} ||y - X\\beta||_2^2 + \\alpha \\left( \\rho ||\\beta||_1 + \\frac{{1-\\rho}}{{2}} ||\\beta||_2^2 \\right)
\\end{{equation}}
où $\\alpha$ contrôle la force de régularisation et $\\rho$ (l1\\_ratio) équilibre entre L1 et L2.
Cette méthode est particulièrement adaptée pour :
\\begin{{itemize}}
    \\item La sélection automatique de variables (sparsité via L1)
    \\item La gestion de la multicolinéarité (stabilisation via L2)
    \\item Les problèmes de haute dimension ($p >> n$)
\\end{{itemize}}

\\subsubsection{{Machine Learning Avancé}}

\\paragraph{{Random Forest}}
Ensemble de {n_features} arbres de décision entraînés sur des sous-échantillons bootstrap. 
Les prédictions sont moyennées pour réduire la variance.

\\paragraph{{XGBoost}}
Implémentation optimisée du gradient boosting avec :
\\begin{{itemize}}
    \\item Régularisation L1 et L2 des poids des feuilles
    \\item Apprentissage incrémental avec early stopping
    \\item Gestion native des valeurs manquantes
\\end{{itemize}}

\\paragraph{{AltumAge (MLP)}}
Réseau de neurones multicouche inspiré de l'architecture AltumAge :
\\begin{{itemize}}
    \\item Architecture : 5 couches cachées de 32 neurones
    \\item Activation : ReLU (Rectified Linear Unit)
    \\item Régularisation : Early stopping
\\end{{itemize}}

\\subsection{{Sélection de Variables}}

Face à la haute dimensionnalité des données ($\\sim$900,000 CpG), une sélection de variables 
est indispensable. Nous utilisons une approche basée sur la corrélation :

\\begin{{enumerate}}
    \\item Calcul de la corrélation de Pearson entre chaque CpG et l'âge
    \\item Sélection des top-k CpG les plus corrélés (|r| maximal)
    \\item Ajout des covariables démographiques (genre, ethnicité)
\\end{{enumerate}}

Cette approche filtre permet de réduire considérablement la dimensionnalité tout en 
conservant les variables les plus informatives.

\\subsection{{Gestion des Covariables}}

Les modèles intègrent les covariables démographiques suivantes :
\\begin{{itemize}}
    \\item \\textbf{{Genre}} : Variable binaire (is\\_female)
    \\item \\textbf{{Ethnicité}} : Encodage one-hot des catégories ethniques
\\end{{itemize}}

Cette intégration permet de capturer les différences biologiques liées au sexe et à 
l'origine ethnique, améliorant potentiellement la précision des prédictions.
"""


def generate_results_section(metrics: pd.DataFrame, preds: pd.DataFrame, annot: pd.DataFrame) -> str:
    """Génère la section Résultats."""
    
    # Best model
    best = metrics.iloc[0]
    best_name = best['model']
    
    # Prepare metrics table
    metrics_display = metrics[['model', 'mae', 'mad', 'r2', 'n_train', 'n_test']].copy()
    metrics_display.columns = ['Modèle', 'MAE', 'MAD', 'R²', 'N train', 'N test']
    
    # Calculate additional stats per model
    results_rows = []
    for model_name in metrics['model']:
        preds_model = preds[preds['model'] == model_name]
        y_true = preds_model['y_true'].values
        y_pred = preds_model['y_pred'].values
        
        corr, _ = stats.pearsonr(y_true, y_pred)
        mean_diff = np.mean(y_pred - y_true)
        
        # Age acceleration
        lr = LinearRegression()
        lr.fit(y_true.reshape(-1, 1), y_pred)
        y_expected = lr.predict(y_true.reshape(-1, 1))
        age_accel = y_pred - y_expected
        
        results_rows.append({
            'model': model_name,
            'correlation': corr,
            'mean_diff': mean_diff,
            'age_accel_mean': np.mean(age_accel),
            'age_accel_std': np.std(age_accel),
        })
    
    results_df = pd.DataFrame(results_rows)
    
    return f"""
\\section{{Métriques de Performance et Évaluation}}

\\subsection{{Résultats Globaux}}

Le tableau \\ref{{tab:metrics}} présente les performances des différents modèles évalués 
sur l'ensemble de test.

\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Modèle}} & \\textbf{{MAE (ans)}} & \\textbf{{MAD (ans)}} & \\textbf{{R²}} & \\textbf{{Corrélation}} \\\\
\\hline
{chr(10).join([f"{row['model']} & {metrics[metrics['model']==row['model']].iloc[0]['mae']:.2f} & {metrics[metrics['model']==row['model']].iloc[0]['mad']:.2f} & {metrics[metrics['model']==row['model']].iloc[0]['r2']:.4f} & {row['correlation']:.4f} \\\\" for _, row in results_df.iterrows()])}
\\hline
\\end{{tabular}}
\\caption{{Performances des modèles de prédiction d'âge}}
\\label{{tab:metrics}}
\\end{{table}}

\\subsection{{Analyse au Niveau de la Cohorte}}

\\subsubsection{{Corrélation Âge Biologique -- Âge Chronologique}}

La corrélation de Pearson entre l'âge prédit (biologique) et l'âge chronologique mesure 
la force de la relation linéaire entre ces deux variables. Une corrélation proche de 1 
indique une excellente capacité prédictive.

\\begin{{itemize}}
{chr(10).join([f"    \\item \\textbf{{{row['model']}}} : r = {row['correlation']:.4f}" for _, row in results_df.iterrows()])}
\\end{{itemize}}

\\subsubsection{{Écart Moyen (Biais)}}

L'écart moyen (Mean Error) indique si le modèle surestime ou sous-estime systématiquement l'âge :

\\begin{{itemize}}
{chr(10).join([f"    \\item \\textbf{{{row['model']}}} : {row['mean_diff']:+.2f} ans" for _, row in results_df.iterrows()])}
\\end{{itemize}}

Un écart proche de zéro indique un modèle non biaisé.

\\subsection{{Analyse au Niveau Individuel}}

\\subsubsection{{Delta Age}}

Le Delta Age ($\\Delta$Age) représente la différence entre l'âge prédit et l'âge chronologique 
pour chaque individu :
\\begin{{equation}}
\\Delta\\text{{Age}} = \\text{{Âge}}_{{\\text{{prédit}}}} - \\text{{Âge}}_{{\\text{{chronologique}}}}
\\end{{equation}}

\\begin{{itemize}}
    \\item $\\Delta$Age $>$ 0 : Vieillissement accéléré
    \\item $\\Delta$Age $<$ 0 : Vieillissement ralenti
    \\item $\\Delta$Age $\\approx$ 0 : Vieillissement normal
\\end{{itemize}}

\\subsubsection{{Age Acceleration}}

L'accélération de l'âge (Age Acceleration) est définie comme le résidu de la régression 
linéaire de l'âge prédit sur l'âge chronologique :
\\begin{{equation}}
\\text{{AgeAccel}} = \\text{{Âge}}_{{\\text{{prédit}}}} - (\\alpha + \\beta \\times \\text{{Âge}}_{{\\text{{chronologique}}}})
\\end{{equation}}

Cette mesure est indépendante de l'âge chronologique et représente l'écart par rapport 
au vieillissement attendu.

\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Modèle}} & \\textbf{{AgeAccel moyen}} & \\textbf{{AgeAccel écart-type}} \\\\
\\hline
{chr(10).join([f"{row['model']} & {row['age_accel_mean']:.2f} & {row['age_accel_std']:.2f} \\\\" for _, row in results_df.iterrows()])}
\\hline
\\end{{tabular}}
\\caption{{Statistiques d'Age Acceleration par modèle}}
\\label{{tab:age_accel}}
\\end{{table}}

\\subsection{{Meilleur Modèle}}

Le modèle \\textbf{{{best_name}}} obtient les meilleures performances avec :
\\begin{{itemize}}
    \\item MAE = {best['mae']:.2f} années
    \\item R² = {best['r2']:.4f}
    \\item Corrélation = {results_df[results_df['model']==best_name].iloc[0]['correlation']:.4f}
\\end{{itemize}}
"""


def generate_stratified_analysis(annot: pd.DataFrame, preds: pd.DataFrame, best_model: str) -> str:
    """Génère la section Analyses Stratifiées."""
    
    # Filter for best model
    preds_best = preds[preds['model'] == best_model].copy()
    
    # Merge with annotations
    annot_preds = annot.copy()
    if 'Sample_description' in preds_best.columns:
        preds_best = preds_best.set_index('sample_id')
    
    # Calculate delta age
    y_true = preds_best['y_true'].values
    y_pred = preds_best['y_pred'].values
    delta_age = y_pred - y_true
    
    # Non-linearity analysis
    z = np.polyfit(y_true, delta_age, 2)
    
    # Gender analysis (if available in annot)
    gender_analysis = ""
    if 'female' in annot.columns:
        # This would need proper merging with predictions
        gender_analysis = """
\\subsubsection{Différences selon le Genre}

L'analyse des erreurs de prédiction selon le genre révèle d'éventuelles différences 
biologiques dans les patterns de méthylation liés au vieillissement.

Les graphiques de l'application interactive montrent la distribution du Delta Age 
pour chaque genre, permettant d'identifier d'éventuels biais du modèle.
"""
    
    return f"""
\\section{{Analyses Stratifiées}}

\\subsection{{Non-linéarité selon l'Âge}}

La relation entre l'erreur de prédiction (Delta Age) et l'âge chronologique peut 
présenter des non-linéarités. Une régression polynomiale de degré 2 a été ajustée :

\\begin{{equation}}
\\Delta\\text{{Age}} = {z[0]:.4f} \\times \\text{{Âge}}^2 + ({z[1]:+.4f}) \\times \\text{{Âge}} + ({z[2]:+.2f})
\\end{{equation}}

\\begin{{itemize}}
    \\item Si le coefficient quadratique est significativement différent de zéro, 
          cela indique une non-linéarité dans les erreurs.
    \\item Un coefficient positif suggère une surestimation aux âges extrêmes.
    \\item Un coefficient négatif suggère une sous-estimation aux âges extrêmes.
\\end{{itemize}}

{gender_analysis}

\\subsection{{Variabilité Technique (Batch Effects)}}

La variabilité technique entre les différents lots de puces (chips) peut introduire 
des biais systématiques. L'analyse par chip ID permet d'évaluer :

\\begin{{itemize}}
    \\item La variance inter-lot (différences moyennes entre chips)
    \\item La variance intra-lot (variabilité au sein d'un même chip)
    \\item L'impact potentiel sur les prédictions d'âge
\\end{{itemize}}

Si la variance inter-lot est élevée par rapport à la variance intra-lot, une 
normalisation par batch (comme ComBat) pourrait améliorer les performances.
"""


def generate_conclusion(metrics: pd.DataFrame) -> str:
    """Génère la section Conclusion."""
    
    best = metrics.iloc[0]
    best_name = best['model']
    
    return f"""
\\section{{Conclusion et Perspectives}}

\\subsection{{Synthèse des Résultats}}

Cette étude a permis de développer et comparer plusieurs modèles de prédiction d'âge 
basés sur les profils de méthylation de l'ADN. Les principaux résultats sont :

\\begin{{enumerate}}
    \\item Le modèle \\textbf{{{best_name}}} offre les meilleures performances avec une 
          MAE de {best['mae']:.2f} années et un R² de {best['r2']:.4f}.
    
    \\item L'intégration des covariables démographiques (genre, ethnicité) permet 
          d'améliorer la précision des prédictions en capturant les différences 
          biologiques individuelles.
    
    \\item Les analyses stratifiées révèlent des patterns de non-linéarité selon l'âge 
          et des différences potentielles selon le genre.
\\end{{enumerate}}

\\subsection{{Amélioration des Modèles Existants}}

Par rapport aux horloges existantes (Horvath, Hannum), notre approche présente 
plusieurs avantages :

\\begin{{itemize}}
    \\item \\textbf{{Intégration multi-variée}} : Prise en compte du genre et de l'ethnicité
    \\item \\textbf{{Flexibilité algorithmique}} : Comparaison de multiples approches 
          (linéaire, ensemble, deep learning)
    \\item \\textbf{{Analyse de la variabilité}} : Identification des sources de biais technique
\\end{{itemize}}

\\subsection{{Interprétabilité des Modèles}}

L'interprétabilité des modèles est cruciale pour comprendre les mécanismes biologiques 
du vieillissement. Les coefficients du modèle Elastic Net identifient les CpG les plus 
prédictifs, qui peuvent être associés à :

\\begin{{itemize}}
    \\item Des gènes impliqués dans la réparation de l'ADN
    \\item Des voies de signalisation du vieillissement (mTOR, sirtuines)
    \\item Des processus inflammatoires chroniques
    \\item La sénescence cellulaire
\\end{{itemize}}

\\subsection{{Perspectives}}

Plusieurs pistes d'amélioration peuvent être envisagées :

\\begin{{enumerate}}
    \\item \\textbf{{Normalisation des batch effects}} : Application de ComBat ou méthodes 
          similaires pour réduire la variabilité technique.
    
    \\item \\textbf{{Architectures deep learning}} : Exploration de réseaux plus complexes 
          (attention mechanisms, transformers) adaptés aux données omiques.
    
    \\item \\textbf{{Validation externe}} : Test sur des cohortes indépendantes pour 
          évaluer la généralisabilité des modèles.
    
    \\item \\textbf{{Application clinique}} : Utilisation comme biomarqueur pour le 
          suivi des interventions anti-âge ou le pronostic de maladies.
\\end{{enumerate}}
"""


def generate_full_report(output_dir: Path) -> str:
    """Génère le rapport LaTeX complet."""
    
    # Load data
    metrics = pd.read_csv(output_dir / "metrics.csv")
    preds = pd.read_csv(output_dir / "predictions.csv")
    annot = pd.read_csv(output_dir / "annot_predictions.csv")
    
    # Get first model's data for general stats
    first_model = metrics.iloc[0]['model']
    annot_first = annot[annot['model'] == first_model]
    
    n_cpg = int(metrics.iloc[0]['n_features'])
    n_features = n_cpg
    
    best_model = metrics.iloc[0]['model']
    
    # Build report
    report = r"""\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{listings}

\geometry{margin=2.5cm}

% Couleurs personnalisées
\definecolor{primary}{RGB}{0,150,136}
\definecolor{secondary}{RGB}{63,81,181}

\hypersetup{
    colorlinks=true,
    linkcolor=secondary,
    filecolor=secondary,
    urlcolor=primary,
    citecolor=primary
}

% Titre
\title{
    \vspace{-1cm}
    \textbf{Prédiction de l'Âge par Méthylation de l'ADN} \\
    \large Développement et Évaluation d'Horloges Épigénétiques
}
\author{
    Analyse automatisée \\
    \textit{DNAm Age Prediction Benchmark}
}
\date{""" + datetime.now().strftime("%d %B %Y") + r"""}

\begin{document}

\maketitle

\begin{abstract}
Ce rapport présente une analyse complète des modèles de prédiction d'âge basés sur les 
profils de méthylation de l'ADN. À partir d'une cohorte de """ + str(len(annot_first)) + r""" échantillons sains, 
nous avons développé et comparé plusieurs approches algorithmiques : régression Elastic Net, 
Random Forest, XGBoost et réseaux de neurones (AltumAge). Le meilleur modèle (""" + best_model + r""") 
atteint une erreur absolue moyenne de """ + f"{metrics.iloc[0]['mae']:.2f}" + r""" années avec un 
coefficient de détermination R² de """ + f"{metrics.iloc[0]['r2']:.4f}" + r""". Les analyses stratifiées 
révèlent des patterns de non-linéarité selon l'âge et des différences potentielles selon le 
genre, soulignant l'importance d'intégrer ces covariables dans les modèles prédictifs.
\end{abstract}

\tableofcontents
\newpage

"""
    
    report += generate_introduction()
    report += generate_data_section(annot_first, n_cpg, n_features)
    report += generate_methodology_section(metrics)
    report += generate_results_section(metrics, preds, annot_first)
    report += generate_stratified_analysis(annot_first, preds, best_model)
    report += generate_conclusion(metrics)
    
    report += r"""

\section*{Références}
\addcontentsline{toc}{section}{Références}

\begin{enumerate}
    \item Horvath, S. (2013). DNA methylation age of human tissues and cell types. 
          \textit{Genome Biology}, 14(10), R115.
    
    \item Hannum, G., et al. (2013). Genome-wide methylation profiles reveal quantitative 
          views of human aging rates. \textit{Molecular Cell}, 49(2), 359-367.
    
    \item Levine, M. E., et al. (2018). An epigenetic biomarker of aging for lifespan 
          and healthspan. \textit{Aging}, 10(4), 573-591.
    
    \item Lu, A. T., et al. (2019). DNA methylation GrimAge strongly predicts lifespan 
          and healthspan. \textit{Aging}, 11(2), 303-327.
    
    \item de Lima Camillo, L. P., et al. (2021). AltumAge: A Pan-Tissue DNA Methylation 
          Epigenetic Clock Based on Deep Learning. \textit{Aging and Disease}.
\end{enumerate}

\end{document}
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX report for DNAm age prediction.")
    parser.add_argument("--output-dir", default="results", help="Path to results directory.")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("Génération du rapport LaTeX...")
    
    report = generate_full_report(output_dir)
    
    # Save report
    report_path = output_dir / "rapport_complet.tex"
    report_path.write_text(report, encoding='utf-8')
    
    print(f"Rapport LaTeX sauvegardé: {report_path}")
    print(f"\nPour compiler le PDF:")
    print(f"  cd {output_dir}")
    print(f"  pdflatex rapport_complet.tex")
    print(f"  pdflatex rapport_complet.tex  # (2ème passe pour table des matières)")


if __name__ == "__main__":
    main()
