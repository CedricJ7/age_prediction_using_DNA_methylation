"""
Génération de figures de qualité publication pour Nature.

Standards Nature:
- Largeur: 89mm (simple colonne) ou 183mm (double colonne)
- Police: Arial/Helvetica, 7-8pt pour labels, 8-9pt pour titres
- Résolution: 300-600 DPI
- Couleurs: palette accessible aux daltoniens
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

# =============================================================================
# CONFIGURATION NATURE
# =============================================================================

# Dimensions en pouces (Nature: 89mm = 3.5in, 183mm = 7.2in)
SINGLE_COL = 3.5
DOUBLE_COL = 7.2
FULL_PAGE_HEIGHT = 9.0

# Palette daltonien-friendly (Wong, 2011 - Nature Methods)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'pink': '#CC79A7',
    'lightblue': '#56B4E9',
    'yellow': '#F0E442',
    'red': '#D55E00',
    'black': '#000000',
    'grey': '#999999',
}

def setup_nature_style():
    """Configure matplotlib pour le style Nature."""
    plt.rcParams.update({
        # Police
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        
        # Axes
        'axes.linewidth': 0.5,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelpad': 4,
        
        # Ticks
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Légende
        'legend.fontsize': 7,
        'legend.frameon': False,
        
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Lignes
        'lines.linewidth': 1,
        'lines.markersize': 4,
        
        # Grille
        'axes.grid': False,
    })


def load_data():
    """Charge les données de prédiction."""
    results_dir = Path("results")
    
    metrics = pd.read_csv(results_dir / "metrics.csv")
    preds = pd.read_csv(results_dir / "predictions.csv")
    annot = pd.read_csv(results_dir / "annot_predictions.csv") if (results_dir / "annot_predictions.csv").exists() else None
    
    return metrics, preds, annot


def create_figure_1_scatter(preds: pd.DataFrame, model_name: str = None, save_path: str = None):
    """
    Figure 1: Scatter plot âge prédit vs âge chronologique.
    Style Nature - panneau unique.
    """
    setup_nature_style()
    
    if model_name is None:
        # Prendre le meilleur modèle (plus bas MAE)
        metrics, _, _ = load_data()
        model_name = metrics.loc[metrics['mae'].idxmin(), 'model']
    
    df = preds[preds['model'] == model_name].copy()
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    
    # Statistiques
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    corr, p_val = stats.pearsonr(y_true, y_pred)
    
    # Régression
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    y_line = lr.predict(x_line.reshape(-1, 1))
    
    # Figure
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    
    # Scatter avec transparence
    ax.scatter(y_true, y_pred, 
               c=COLORS['blue'], 
               s=15, 
               alpha=0.6, 
               edgecolors='none',
               rasterized=True)  # Rasterize pour PDF plus léger
    
    # Ligne identité (diagonale parfaite)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, '--', color=COLORS['grey'], linewidth=0.75, label='Identity', zorder=0)
    
    # Ligne de régression
    ax.plot(x_line, y_line, '-', color=COLORS['red'], linewidth=1, label='Regression')
    
    # Labels
    ax.set_xlabel('Chronological age (years)')
    ax.set_ylabel('Predicted age (years)')
    
    # Limites
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    
    # Statistiques en annotation (coin supérieur gauche)
    stats_text = f'r = {corr:.3f}\nMAE = {mae:.2f} years\nR² = {r2:.3f}'
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=7,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # N échantillons
    ax.text(0.95, 0.05, f'n = {len(df)}',
            transform=ax.transAxes,
            horizontalalignment='right',
            fontsize=7,
            color=COLORS['grey'])
    
    plt.tight_layout()
    
    if save_path:
        # Sauvegarder en plusieurs formats
        for fmt in ['pdf', 'png', 'svg']:
            fig.savefig(f"{save_path}.{fmt}", format=fmt, dpi=600 if fmt == 'png' else None)
        print(f"✓ Figure sauvegardée: {save_path}.[pdf|png|svg]")
    
    return fig, ax


def create_figure_2_multi_panel(preds: pd.DataFrame, metrics: pd.DataFrame, annot: pd.DataFrame = None, save_path: str = None):
    """
    Figure 2: Figure multi-panneaux pour Nature.
    (a) Scatter plot meilleur modèle
    (b) Comparaison MAE des modèles
    (c) Distribution des résidus
    (d) Résidus vs âge (biais)
    """
    setup_nature_style()
    
    best_model = metrics.loc[metrics['mae'].idxmin(), 'model']
    df = preds[preds['model'] == best_model].copy()
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    residuals = y_pred - y_true
    
    # Statistiques
    mae = np.mean(np.abs(residuals))
    corr, _ = stats.pearsonr(y_true, y_pred)
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
    
    # Figure 2x2
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    
    # Panel labels
    panel_labels = ['a', 'b', 'c', 'd']
    
    # ==========================================================================
    # (a) Scatter plot
    # ==========================================================================
    ax = axes[0, 0]
    
    ax.scatter(y_true, y_pred, c=COLORS['blue'], s=12, alpha=0.5, edgecolors='none', rasterized=True)
    
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, '--', color=COLORS['grey'], linewidth=0.5, zorder=0)
    
    # Régression
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_line, lr.predict(x_line.reshape(-1, 1)), '-', color=COLORS['red'], linewidth=0.8)
    
    ax.set_xlabel('Chronological age (years)')
    ax.set_ylabel('Predicted age (years)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    
    # Stats
    ax.text(0.05, 0.95, f'r = {corr:.3f}\nMAE = {mae:.1f} y',
            transform=ax.transAxes, va='top', fontsize=6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    
    # ==========================================================================
    # (b) Comparaison des modèles (bar chart horizontal)
    # ==========================================================================
    ax = axes[0, 1]
    
    metrics_sorted = metrics.sort_values('mae')
    models = metrics_sorted['model'].values
    maes = metrics_sorted['mae'].values
    
    # Couleurs: meilleur en couleur, autres en gris
    colors = [COLORS['blue'] if m == best_model else COLORS['grey'] for m in models]
    
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, maes, color=colors, height=0.6, edgecolor='none')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('MAE (years)')
    ax.set_xlim(0, max(maes) * 1.15)
    ax.invert_yaxis()
    
    # Valeurs sur les barres
    for i, (bar, mae_val) in enumerate(zip(bars, maes)):
        ax.text(mae_val + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{mae_val:.2f}', va='center', fontsize=6, color=COLORS['black'])
    
    # ==========================================================================
    # (c) Distribution des résidus (histogram + KDE)
    # ==========================================================================
    ax = axes[1, 0]
    
    # Histogram
    n, bins, patches = ax.hist(residuals, bins=25, density=True, 
                                color=COLORS['lightblue'], edgecolor='white', 
                                linewidth=0.3, alpha=0.7)
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    x_kde = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x_kde, kde(x_kde), color=COLORS['blue'], linewidth=1)
    
    # Ligne verticale à 0
    ax.axvline(0, color=COLORS['red'], linestyle='--', linewidth=0.5)
    
    # Ligne verticale à la moyenne
    mean_res = np.mean(residuals)
    ax.axvline(mean_res, color=COLORS['orange'], linestyle='-', linewidth=0.8)
    
    ax.set_xlabel('Prediction error (years)')
    ax.set_ylabel('Density')
    
    # Stats
    ax.text(0.95, 0.95, f'μ = {mean_res:.2f}\nσ = {np.std(residuals):.2f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    
    # ==========================================================================
    # (d) Résidus vs âge (Bland-Altman style)
    # ==========================================================================
    ax = axes[1, 1]
    
    ax.scatter(y_true, residuals, c=COLORS['blue'], s=12, alpha=0.5, edgecolors='none', rasterized=True)
    
    # Ligne à 0
    ax.axhline(0, color=COLORS['grey'], linestyle='-', linewidth=0.5)
    
    # Lignes ±1.96 SD (limites d'agrément)
    sd = np.std(residuals)
    ax.axhline(1.96 * sd, color=COLORS['red'], linestyle='--', linewidth=0.5)
    ax.axhline(-1.96 * sd, color=COLORS['red'], linestyle='--', linewidth=0.5)
    
    # Tendance (LOWESS ou polynomiale)
    z = np.polyfit(y_true, residuals, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_smooth, p(x_smooth), color=COLORS['orange'], linewidth=1)
    
    ax.set_xlabel('Chronological age (years)')
    ax.set_ylabel('Prediction error (years)')
    
    # Annotation des limites
    ax.text(y_true.max(), 1.96*sd, f'+1.96 SD', va='bottom', ha='right', fontsize=5, color=COLORS['red'])
    ax.text(y_true.max(), -1.96*sd, f'−1.96 SD', va='top', ha='right', fontsize=5, color=COLORS['red'])
    
    # ==========================================================================
    # Labels des panneaux (a, b, c, d)
    # ==========================================================================
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    if save_path:
        for fmt in ['pdf', 'png', 'svg']:
            fig.savefig(f"{save_path}.{fmt}", format=fmt, dpi=600 if fmt == 'png' else None)
        print(f"✓ Figure sauvegardée: {save_path}.[pdf|png|svg]")
    
    return fig, axes


def create_figure_3_gender_ethnicity(preds: pd.DataFrame, annot: pd.DataFrame, model_name: str = None, save_path: str = None):
    """
    Figure 3: Analyses stratifiées par sexe et ethnicité.
    """
    setup_nature_style()
    
    if model_name is None:
        metrics, _, _ = load_data()
        model_name = metrics.loc[metrics['mae'].idxmin(), 'model']
    
    if annot is None:
        print("⚠ Données d'annotation non disponibles")
        return None, None
    
    df = annot[annot['model'] == model_name].copy()
    df['error'] = df['age_pred'] - df['age']
    
    # Préparer les données de genre
    if 'female' in df.columns:
        df['Gender'] = df['female'].apply(
            lambda x: 'Female' if str(x).lower() == 'true' else ('Male' if str(x).lower() == 'false' else None)
        )
    
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 0.9))
    
    # ==========================================================================
    # (a) Par genre
    # ==========================================================================
    ax = axes[0]
    
    if 'Gender' in df.columns:
        df_gender = df[df['Gender'].notna()].copy()
        
        # Box plot
        gender_data = [df_gender[df_gender['Gender'] == g]['error'].values for g in ['Female', 'Male']]
        bp = ax.boxplot(gender_data, 
                        positions=[0, 1],
                        widths=0.5,
                        patch_artist=True,
                        medianprops=dict(color=COLORS['black'], linewidth=1),
                        whiskerprops=dict(linewidth=0.5),
                        capprops=dict(linewidth=0.5),
                        flierprops=dict(marker='o', markersize=2, markerfacecolor=COLORS['grey'], 
                                       markeredgecolor='none', alpha=0.5))
        
        colors_box = [COLORS['pink'], COLORS['lightblue']]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(COLORS['black'])
            patch.set_linewidth(0.5)
        
        ax.axhline(0, color=COLORS['grey'], linestyle='--', linewidth=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Female', 'Male'])
        ax.set_ylabel('Prediction error (years)')
        
        # Stats
        for i, (g, data) in enumerate(zip(['Female', 'Male'], gender_data)):
            mean_err = np.mean(data)
            ax.text(i, ax.get_ylim()[1] * 0.9, f'μ={mean_err:.1f}', 
                    ha='center', fontsize=6, color=COLORS['black'])
        
        # Test statistique
        if len(gender_data[0]) > 5 and len(gender_data[1]) > 5:
            t_stat, p_val = stats.ttest_ind(gender_data[0], gender_data[1])
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            ax.text(0.5, 1.02, sig, ha='center', transform=ax.transAxes, fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Gender data\nnot available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=8, color=COLORS['grey'])
    
    # ==========================================================================
    # (b) Par tranche d'âge
    # ==========================================================================
    ax = axes[1]
    
    # Créer des tranches d'âge
    bins = [0, 30, 50, 70, 100]
    labels = ['<30', '30-50', '50-70', '>70']
    df['Age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    
    age_groups = df.groupby('Age_group')['error'].apply(list).to_dict()
    
    positions = range(len(labels))
    data_to_plot = [age_groups.get(l, []) for l in labels]
    
    bp = ax.boxplot([d for d in data_to_plot if len(d) > 0],
                    positions=[i for i, d in enumerate(data_to_plot) if len(d) > 0],
                    widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color=COLORS['black'], linewidth=1),
                    whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5),
                    flierprops=dict(marker='o', markersize=2, markerfacecolor=COLORS['grey'],
                                   markeredgecolor='none', alpha=0.5))
    
    # Gradient de couleur bleu
    blues = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
    for patch, color in zip(bp['boxes'], blues[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor(COLORS['black'])
        patch.set_linewidth(0.5)
    
    ax.axhline(0, color=COLORS['grey'], linestyle='--', linewidth=0.5)
    ax.set_xticks([i for i, d in enumerate(data_to_plot) if len(d) > 0])
    ax.set_xticklabels([l for l, d in zip(labels, data_to_plot) if len(d) > 0])
    ax.set_xlabel('Age group (years)')
    ax.set_ylabel('Prediction error (years)')
    
    # Stats par groupe
    for i, (label, data) in enumerate(zip(labels, data_to_plot)):
        if len(data) > 0:
            mean_err = np.mean(data)
            ax.text(i, ax.get_ylim()[1] * 0.9, f'n={len(data)}', 
                    ha='center', fontsize=5, color=COLORS['grey'])
    
    # Panel labels
    for ax, label in zip(axes, ['a', 'b']):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout()
    
    if save_path:
        for fmt in ['pdf', 'png', 'svg']:
            fig.savefig(f"{save_path}.{fmt}", format=fmt, dpi=600 if fmt == 'png' else None)
        print(f"✓ Figure sauvegardée: {save_path}.[pdf|png|svg]")
    
    return fig, axes


def create_all_figures():
    """Génère toutes les figures de qualité publication."""
    print("=" * 60)
    print("GÉNÉRATION DES FIGURES - QUALITÉ NATURE")
    print("=" * 60)
    
    output_dir = Path("results/figures_publication")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics, preds, annot = load_data()
    
    print("\n[1/3] Figure 1: Scatter plot (panneau unique)...")
    create_figure_1_scatter(preds, save_path=str(output_dir / "figure_1_scatter"))
    
    print("\n[2/3] Figure 2: Multi-panneaux...")
    create_figure_2_multi_panel(preds, metrics, annot, save_path=str(output_dir / "figure_2_multipanel"))
    
    print("\n[3/3] Figure 3: Analyses stratifiées...")
    if annot is not None:
        create_figure_3_gender_ethnicity(preds, annot, save_path=str(output_dir / "figure_3_stratified"))
    else:
        print("  ⚠ Données d'annotation non disponibles, figure 3 ignorée")
    
    print("\n" + "=" * 60)
    print(f"✓ Figures sauvegardées dans: {output_dir}")
    print("  Formats: PDF (vecteur), PNG (600 DPI), SVG (vecteur)")
    print("=" * 60)


if __name__ == "__main__":
    create_all_figures()
