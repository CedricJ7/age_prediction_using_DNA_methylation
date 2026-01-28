"""
Visualisation Révolutionnaire - Horloge Épigénétique
Une interface qui incarne la révolution de la médecine prédictive.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import colorsys


def load_data():
    """Charge les données."""
    results_dir = Path("results")
    metrics = pd.read_csv(results_dir / "metrics.csv")
    preds = pd.read_csv(results_dir / "predictions.csv")
    annot = pd.read_csv(results_dir / "annot_predictions.csv") if (results_dir / "annot_predictions.csv").exists() else None
    return metrics, preds, annot


def create_gradient_colors(n, start_hue=0.55, end_hue=0.85):
    """Crée un dégradé de couleurs."""
    colors = []
    for i in range(n):
        hue = start_hue + (end_hue - start_hue) * (i / max(n-1, 1))
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    return colors


def create_revolutionary_dashboard():
    """Crée le dashboard révolutionnaire."""
    
    metrics, preds, annot = load_data()
    best_model = metrics.loc[metrics['mae'].idxmin(), 'model']
    df = preds[preds['model'] == best_model].copy()
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    residuals = y_pred - y_true
    
    # Statistiques
    mae = np.mean(np.abs(residuals))
    corr, _ = stats.pearsonr(y_true, y_pred)
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
    
    # Couleurs basées sur l'erreur (gradient cyan -> magenta)
    error_normalized = (np.abs(residuals) - np.abs(residuals).min()) / (np.abs(residuals).max() - np.abs(residuals).min() + 0.001)
    
    # Figure avec fond noir
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "scatter", "colspan": 2, "rowspan": 2}, None, {"type": "indicator"}],
            [None, None, {"type": "indicator"}],
        ],
        column_widths=[0.45, 0.25, 0.30],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=("", "", "")
    )
    
    # =========================================================================
    # SCATTER PRINCIPAL - Style néon futuriste
    # =========================================================================
    
    # Ligne diagonale avec glow
    diag_min, diag_max = min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5
    
    # Glow effect pour la diagonale (plusieurs lignes semi-transparentes)
    for width, opacity in [(12, 0.05), (8, 0.08), (4, 0.15), (2, 0.4)]:
        fig.add_trace(go.Scatter(
            x=[diag_min, diag_max],
            y=[diag_min, diag_max],
            mode='lines',
            line=dict(color='rgba(0, 255, 200, {})'.format(opacity), width=width),
            hoverinfo='skip',
            showlegend=False,
        ), row=1, col=1)
    
    # Points principaux avec couleur basée sur l'erreur
    colors = [f'hsla({180 + err * 120}, 100%, 65%, 0.85)' for err in error_normalized]
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=12,
            color=np.abs(residuals),
            colorscale=[
                [0, 'rgba(0, 255, 220, 0.9)'],      # Cyan pour faible erreur
                [0.3, 'rgba(100, 200, 255, 0.9)'],  # Bleu clair
                [0.6, 'rgba(180, 100, 255, 0.9)'],  # Violet
                [1, 'rgba(255, 50, 150, 0.9)'],     # Magenta pour haute erreur
            ],
            colorbar=dict(
                title=dict(text="Erreur<br>(années)", font=dict(color='white', size=11)),
                tickfont=dict(color='white', size=10),
                x=0.52,
                len=0.4,
                thickness=15,
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
            ),
            line=dict(width=0),
        ),
        text=[f"Chrono: {t:.1f}y<br>Prédit: {p:.1f}y<br>Δ: {r:+.1f}y" for t, p, r in zip(y_true, y_pred, residuals)],
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    
    # Glow autour des points (effet de halo)
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=20,
            color=np.abs(residuals),
            colorscale=[
                [0, 'rgba(0, 255, 220, 0.15)'],
                [0.5, 'rgba(150, 100, 255, 0.15)'],
                [1, 'rgba(255, 50, 150, 0.15)'],
            ],
            line=dict(width=0),
            showscale=False,
        ),
        hoverinfo='skip',
        showlegend=False,
    ), row=1, col=1)
    
    # Ligne de régression avec glow
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred)
    x_reg = np.linspace(diag_min, diag_max, 100)
    y_reg = lr.predict(x_reg.reshape(-1, 1))
    
    for width, opacity in [(8, 0.1), (4, 0.3), (2, 0.8)]:
        fig.add_trace(go.Scatter(
            x=x_reg,
            y=y_reg,
            mode='lines',
            line=dict(color=f'rgba(255, 180, 50, {opacity})', width=width),
            hoverinfo='skip',
            showlegend=False,
        ), row=1, col=1)
    
    # =========================================================================
    # INDICATEURS STYLE COCKPIT
    # =========================================================================
    
    # Gauge MAE
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=mae,
        number=dict(
            suffix=" ans",
            font=dict(size=32, color='white', family='Orbitron, sans-serif'),
        ),
        delta=dict(
            reference=5,
            increasing=dict(color='rgba(255, 100, 100, 0.8)'),
            decreasing=dict(color='rgba(0, 255, 200, 0.8)'),
            font=dict(size=14),
        ),
        title=dict(
            text="<b>ERREUR MOYENNE</b><br><span style='font-size:11px;color:#888'>Mean Absolute Error</span>",
            font=dict(size=14, color='white'),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 15],
                tickwidth=2,
                tickcolor='rgba(255,255,255,0.5)',
                tickfont=dict(color='white', size=10),
            ),
            bar=dict(color='rgba(0, 255, 200, 0.8)', thickness=0.75),
            bgcolor='rgba(0,0,0,0.3)',
            borderwidth=2,
            bordercolor='rgba(255,255,255,0.2)',
            steps=[
                dict(range=[0, 3], color='rgba(0, 255, 200, 0.15)'),
                dict(range=[3, 6], color='rgba(255, 220, 100, 0.15)'),
                dict(range=[6, 15], color='rgba(255, 100, 100, 0.15)'),
            ],
            threshold=dict(
                line=dict(color='white', width=3),
                thickness=0.8,
                value=mae,
            ),
        ),
    ), row=1, col=3)
    
    # Gauge Corrélation
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=corr,
        number=dict(
            font=dict(size=32, color='white', family='Orbitron, sans-serif'),
            valueformat=".3f",
        ),
        title=dict(
            text="<b>CORRÉLATION</b><br><span style='font-size:11px;color:#888'>Pearson r</span>",
            font=dict(size=14, color='white'),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 1],
                tickwidth=2,
                tickcolor='rgba(255,255,255,0.5)',
                tickfont=dict(color='white', size=10),
            ),
            bar=dict(color='rgba(180, 100, 255, 0.8)', thickness=0.75),
            bgcolor='rgba(0,0,0,0.3)',
            borderwidth=2,
            bordercolor='rgba(255,255,255,0.2)',
            steps=[
                dict(range=[0, 0.7], color='rgba(255, 100, 100, 0.15)'),
                dict(range=[0.7, 0.9], color='rgba(255, 220, 100, 0.15)'),
                dict(range=[0.9, 1], color='rgba(0, 255, 200, 0.15)'),
            ],
            threshold=dict(
                line=dict(color='white', width=3),
                thickness=0.8,
                value=corr,
            ),
        ),
    ), row=2, col=3)
    
    # =========================================================================
    # LAYOUT FUTURISTE
    # =========================================================================
    
    fig.update_layout(
        title=dict(
            text="<b>🧬 HORLOGE ÉPIGÉNÉTIQUE</b><br><sup style='color:#00ffd0;font-size:14px'>Prédiction de l'Âge Biologique par Méthylation de l'ADN</sup>",
            font=dict(size=28, color='white', family='Orbitron, Arial, sans-serif'),
            x=0.5,
            y=0.98,
        ),
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Arial, sans-serif', color='white'),
        height=700,
        width=1400,
        margin=dict(l=60, r=60, t=100, b=60),
        
        # Axes du scatter
        xaxis=dict(
            title=dict(text="<b>Âge Chronologique</b> (années)", font=dict(size=14, color='#00ffd0')),
            range=[diag_min, diag_max],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(color='rgba(255,255,255,0.7)', size=11),
            linecolor='rgba(255,255,255,0.2)',
            linewidth=2,
        ),
        yaxis=dict(
            title=dict(text="<b>Âge Prédit</b> (années)", font=dict(size=14, color='#00ffd0')),
            range=[diag_min, diag_max],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(color='rgba(255,255,255,0.7)', size=11),
            linecolor='rgba(255,255,255,0.2)',
            linewidth=2,
            scaleanchor="x",
            scaleratio=1,
        ),
        
        hoverlabel=dict(
            bgcolor='rgba(10, 10, 20, 0.95)',
            bordercolor='rgba(0, 255, 200, 0.5)',
            font=dict(color='white', size=12, family='JetBrains Mono, monospace'),
        ),
    )
    
    # Annotations
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f"<b>R² = {r2:.4f}</b><br>n = {len(y_true)} échantillons<br>Modèle: {best_model}",
        showarrow=False,
        font=dict(size=12, color='rgba(255,255,255,0.8)', family='JetBrains Mono'),
        align='left',
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='rgba(0, 255, 200, 0.3)',
        borderwidth=1,
        borderpad=10,
    )
    
    # Ligne parfaite légende
    fig.add_annotation(
        x=diag_max - 5, y=diag_max - 3,
        xref='x', yref='y',
        text="← Prédiction parfaite",
        showarrow=False,
        font=dict(size=10, color='rgba(0, 255, 200, 0.7)'),
    )
    
    return fig


def create_biological_clock_viz():
    """
    Crée une visualisation d'horloge biologique immersive.
    
    L'HISTOIRE: Chaque vie est un voyage à travers le temps. Cette horloge
    représente le passage du temps biologique - certains voyagent plus vite,
    d'autres plus lentement. Le centre représente la naissance, le bord 
    extérieur la fin du voyage. Entre les deux, l'épigénétique révèle
    les secrets de notre vieillissement.
    """
    
    metrics, preds, annot = load_data()
    best_model = metrics.loc[metrics['mae'].idxmin(), 'model']
    df = preds[preds['model'] == best_model].copy()
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    age_accel = y_pred - y_true
    
    # Tri par âge chronologique
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    age_accel_sorted = age_accel[sort_idx]
    
    n = len(y_true)
    mae = np.mean(np.abs(age_accel))
    corr, _ = stats.pearsonr(y_true, y_pred)
    
    # Angles pour chaque échantillon (spirale de la vie)
    angles = np.linspace(0, 340, n, endpoint=True)
    angles_rad = np.radians(angles)
    
    # Rayon basé sur l'âge - spirale qui s'élargit avec l'âge
    age_norm = (y_true_sorted - y_true_sorted.min()) / (y_true_sorted.max() - y_true_sorted.min() + 0.001)
    r_base = 0.25 + age_norm * 0.5
    
    # Perturbation du rayon pour l'âge prédit (différence visible)
    pred_norm = (y_pred_sorted - y_true_sorted.min()) / (y_true_sorted.max() - y_true_sorted.min() + 0.001)
    r_pred = 0.25 + pred_norm * 0.5
    
    # Coordonnées
    x_chrono = r_base * np.cos(angles_rad)
    y_chrono = r_base * np.sin(angles_rad)
    x_pred = r_pred * np.cos(angles_rad)
    y_pred_coord = r_pred * np.sin(angles_rad)
    
    fig = go.Figure()
    
    # =========================================================================
    # FOND: Cercles concentriques avec gradient bleu-violet
    # =========================================================================
    
    # Anneaux de décennies avec glow
    decades = [20, 30, 40, 50, 60, 70, 80, 90]
    for i, decade in enumerate(decades):
        if y_true_sorted.min() - 5 <= decade <= y_true_sorted.max() + 5:
            r_decade = 0.25 + ((decade - y_true_sorted.min()) / (y_true_sorted.max() - y_true_sorted.min() + 0.001)) * 0.5
            r_decade = max(0.2, min(0.8, r_decade))
            
            theta = np.linspace(0, 2*np.pi, 150)
            
            # Gradient bleu -> violet selon l'âge
            hue_start = 220  # Bleu
            hue_end = 280    # Violet
            hue = hue_start + (hue_end - hue_start) * (i / len(decades))
            
            # Glow extérieur
            for glow_width, glow_alpha in [(4, 0.03), (2, 0.08), (1, 0.15)]:
                fig.add_trace(go.Scatter(
                    x=r_decade * np.cos(theta),
                    y=r_decade * np.sin(theta),
                    mode='lines',
                    line=dict(color=f'hsla({hue}, 80%, 60%, {glow_alpha})', width=glow_width),
                    hoverinfo='skip',
                    showlegend=False,
                ))
            
            # Ligne principale
            fig.add_trace(go.Scatter(
                x=r_decade * np.cos(theta),
                y=r_decade * np.sin(theta),
                mode='lines',
                line=dict(color=f'hsla({hue}, 70%, 50%, 0.25)', width=0.5),
                hoverinfo='skip',
                showlegend=False,
            ))
    
    # =========================================================================
    # RAYONS: Marques temporelles comme une horloge
    # =========================================================================
    
    # 12 rayons comme les heures d'une horloge
    for hour in range(12):
        angle = hour * 30
        rad = np.radians(angle)
        
        # Gradient le long du rayon
        for seg in range(20):
            r_start = 0.15 + seg * 0.035
            r_end = r_start + 0.04
            alpha = 0.03 + (seg / 20) * 0.08
            hue = 220 + (seg / 20) * 60  # Bleu -> Violet
            
            fig.add_trace(go.Scatter(
                x=[r_start * np.cos(rad), r_end * np.cos(rad)],
                y=[r_start * np.sin(rad), r_end * np.sin(rad)],
                mode='lines',
                line=dict(color=f'hsla({hue}, 70%, 60%, {alpha})', width=1),
                hoverinfo='skip',
                showlegend=False,
            ))
    
    # =========================================================================
    # CONNEXIONS: Fils du destin entre chrono et prédit
    # =========================================================================
    
    for i in range(n):
        # Couleur basée sur l'accélération: bleu (jeune) -> violet (normal) -> magenta (vieux)
        accel_norm = (age_accel_sorted[i] + 15) / 30  # Normaliser entre -15 et +15
        accel_norm = max(0, min(1, accel_norm))
        
        # Gradient: cyan -> bleu -> violet -> magenta
        if accel_norm < 0.4:  # Plus jeune biologiquement
            hue = 180 + accel_norm * 100  # Cyan -> Bleu
            lightness = 60
        elif accel_norm < 0.6:  # Normal
            hue = 250  # Violet
            lightness = 55
        else:  # Plus vieux biologiquement
            hue = 280 + (accel_norm - 0.6) * 50  # Violet -> Magenta
            lightness = 50
        
        alpha = 0.4 if abs(age_accel_sorted[i]) > 5 else 0.2
        
        fig.add_trace(go.Scatter(
            x=[x_chrono[i], x_pred[i]],
            y=[y_chrono[i], y_pred_coord[i]],
            mode='lines',
            line=dict(color=f'hsla({hue}, 80%, {lightness}%, {alpha})', width=1.5),
            hoverinfo='skip',
            showlegend=False,
        ))
    
    # =========================================================================
    # TRAJECTOIRE: Chemin de vie (ligne chronologique)
    # =========================================================================
    
    # Chemin chronologique avec gradient
    for i in range(n - 1):
        progress = i / n
        hue = 220 + progress * 60  # Bleu -> Violet
        
        fig.add_trace(go.Scatter(
            x=x_chrono[i:i+2],
            y=y_chrono[i:i+2],
            mode='lines',
            line=dict(color=f'hsla({hue}, 70%, 55%, 0.6)', width=2),
            hoverinfo='skip',
            showlegend=False,
        ))
    
    # =========================================================================
    # POINTS: Moments de vie (âge chronologique)
    # =========================================================================
    
    # Halo des points chrono
    fig.add_trace(go.Scatter(
        x=x_chrono,
        y=y_chrono,
        mode='markers',
        marker=dict(
            size=12,
            color=[f'hsla({220 + (i/n)*60}, 70%, 60%, 0.2)' for i in range(n)],
            line=dict(width=0),
        ),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # Points principaux chrono
    chrono_colors = [220 + (i/n)*60 for i in range(n)]
    fig.add_trace(go.Scatter(
        x=x_chrono,
        y=y_chrono,
        mode='markers',
        marker=dict(
            size=5,
            color=chrono_colors,
            colorscale=[[0, 'hsl(220, 80%, 60%)'], [1, 'hsl(280, 80%, 60%)']],
            line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
        ),
        name='⏱️ Âge chronologique',
        text=[f"<b>Individu #{i+1}</b><br>Âge chronologique: {t:.1f} ans" for i, t in enumerate(y_true_sorted)],
        hovertemplate="%{text}<extra></extra>",
    ))
    
    # =========================================================================
    # POINTS: Âge biologique (prédit) - étoiles du destin
    # =========================================================================
    
    # Normalisation des couleurs pour l'accélération
    accel_colors = []
    for a in age_accel_sorted:
        norm = (a + 12) / 24  # Normaliser -12 à +12
        norm = max(0, min(1, norm))
        accel_colors.append(norm)
    
    # Halo des points prédits
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=y_pred_coord,
        mode='markers',
        marker=dict(
            size=22,
            color=accel_colors,
            colorscale=[
                [0.0, 'rgba(0, 200, 255, 0.15)'],    # Cyan - rajeunissement
                [0.3, 'rgba(100, 150, 255, 0.12)'],  # Bleu
                [0.5, 'rgba(180, 100, 255, 0.10)'],  # Violet - normal
                [0.7, 'rgba(220, 80, 200, 0.12)'],   # Magenta
                [1.0, 'rgba(255, 50, 150, 0.15)'],   # Rose - vieillissement
            ],
            line=dict(width=0),
        ),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # Points principaux prédits
    fig.add_trace(go.Scatter(
        x=x_pred,
        y=y_pred_coord,
        mode='markers',
        marker=dict(
            size=10,
            color=accel_colors,
            colorscale=[
                [0.0, 'rgb(0, 220, 255)'],     # Cyan vif
                [0.25, 'rgb(80, 160, 255)'],   # Bleu clair
                [0.5, 'rgb(160, 100, 255)'],   # Violet
                [0.75, 'rgb(220, 80, 220)'],   # Magenta
                [1.0, 'rgb(255, 60, 150)'],    # Rose vif
            ],
            colorbar=dict(
                title=dict(
                    text="<b>Accélération</b><br>épigénétique",
                    font=dict(color='white', size=11),
                ),
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['-12 ans', '-6 ans', '0', '+6 ans', '+12 ans'],
                tickfont=dict(color='rgba(255,255,255,0.8)', size=9),
                x=1.02,
                len=0.6,
                thickness=12,
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
            ),
            line=dict(width=1.5, color='rgba(255,255,255,0.4)'),
            symbol='diamond',
        ),
        name='🔮 Âge biologique',
        text=[
            f"<b>Individu #{i+1}</b><br>"
            f"Âge biologique: <b>{p:.1f} ans</b><br>"
            f"Âge chronologique: {t:.1f} ans<br>"
            f"Accélération: <b style='color:{'#ff6090' if a > 0 else '#00d4ff'}'>{a:+.1f} ans</b>"
            for i, (t, p, a) in enumerate(zip(y_true_sorted, y_pred_sorted, age_accel_sorted))
        ],
        hovertemplate="%{text}<extra></extra>",
    ))
    
    # =========================================================================
    # CENTRE: Cœur de l'horloge
    # =========================================================================
    
    # Cercles concentriques au centre avec pulsation simulée
    for r, alpha in [(0.18, 0.03), (0.14, 0.05), (0.10, 0.08), (0.06, 0.12)]:
        theta = np.linspace(0, 2*np.pi, 50)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta),
            y=r * np.sin(theta),
            mode='lines',
            fill='toself',
            fillcolor=f'rgba(160, 100, 255, {alpha})',
            line=dict(color=f'rgba(180, 120, 255, {alpha * 2})', width=1),
            hoverinfo='skip',
            showlegend=False,
        ))
    
    # Symbole central
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers+text',
        marker=dict(
            size=40,
            color='rgba(140, 80, 255, 0.3)',
            line=dict(width=2, color='rgba(180, 120, 255, 0.8)'),
        ),
        text=['🧬'],
        textfont=dict(size=22),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # =========================================================================
    # ANNOTATIONS: L'histoire
    # =========================================================================
    
    # Titre poétique en haut
    fig.add_annotation(
        x=0.5, y=1.12,
        xref='paper', yref='paper',
        text="<b>« Chaque vie est une spirale à travers le temps »</b>",
        showarrow=False,
        font=dict(size=11, color='rgba(180, 150, 255, 0.8)', family='Georgia, serif'),
    )
    
    # Légende narrative à gauche
    narrative_text = (
        "<b>LECTURE DE L'HORLOGE</b><br><br>"
        "<span style='color:#00d4ff'>●</span> Centre → Naissance<br>"
        "<span style='color:#a070ff'>●</span> Bord → Âge avancé<br><br>"
        "<span style='color:#00d4ff'>◆</span> Bleu/Cyan = Jeunesse biologique<br>"
        "<span style='color:#ff6090'>◆</span> Rose = Vieillissement accéléré"
    )
    
    fig.add_annotation(
        x=-0.02, y=0.98,
        xref='paper', yref='paper',
        text=narrative_text,
        showarrow=False,
        font=dict(size=9, color='rgba(255,255,255,0.75)', family='Inter'),
        align='left',
        bgcolor='rgba(10, 10, 30, 0.7)',
        bordercolor='rgba(160, 100, 255, 0.3)',
        borderwidth=1,
        borderpad=12,
    )
    
    # Statistiques en bas à droite
    stats_text = (
        f"<b>MÉTRIQUES</b><br>"
        f"n = {n} individus<br>"
        f"MAE = {mae:.2f} ans<br>"
        f"r = {corr:.3f}"
    )
    
    fig.add_annotation(
        x=1.02, y=0.02,
        xref='paper', yref='paper',
        text=stats_text,
        showarrow=False,
        font=dict(size=9, color='rgba(255,255,255,0.7)', family='JetBrains Mono'),
        align='left',
        bgcolor='rgba(10, 10, 30, 0.7)',
        bordercolor='rgba(100, 150, 255, 0.3)',
        borderwidth=1,
        borderpad=10,
    )
    
    # Labels des décennies
    for decade in [30, 50, 70]:
        if y_true_sorted.min() <= decade <= y_true_sorted.max():
            r_label = 0.25 + ((decade - y_true_sorted.min()) / (y_true_sorted.max() - y_true_sorted.min() + 0.001)) * 0.5
            r_label = max(0.25, min(0.78, r_label))
            
            fig.add_annotation(
                x=r_label + 0.03, y=0.03,
                text=f"<b>{decade}</b>",
                showarrow=False,
                font=dict(size=8, color='rgba(180, 150, 255, 0.6)'),
            )
    
    # =========================================================================
    # LAYOUT FINAL
    # =========================================================================
    
    fig.update_layout(
        title=dict(
            text="<b>⏱️ L'HORLOGE ÉPIGÉNÉTIQUE</b><br><sup style='color:#b080ff;font-size:13px'>Le Temps Biologique Révélé par la Méthylation de l'ADN</sup>",
            font=dict(size=26, color='white', family='Orbitron, sans-serif'),
            x=0.5,
            y=0.95,
        ),
        paper_bgcolor='#08080f',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter, sans-serif'),
        height=850,
        width=950,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.05, 1.15],
            fixedrange=True,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, 1],
            scaleanchor='x',
            fixedrange=True,
        ),
        showlegend=True,
        legend=dict(
            x=0.75, y=-0.05,
            orientation='h',
            bgcolor='rgba(10, 10, 30, 0.6)',
            bordercolor='rgba(160, 100, 255, 0.2)',
            borderwidth=1,
            font=dict(size=10, color='rgba(255,255,255,0.9)'),
        ),
        hoverlabel=dict(
            bgcolor='rgba(15, 15, 35, 0.95)',
            bordercolor='rgba(160, 100, 255, 0.6)',
            font=dict(color='white', size=11, family='Inter'),
        ),
        margin=dict(l=20, r=100, t=100, b=60),
    )
    
    return fig


def create_age_acceleration_wave():
    """Crée une visualisation en vagues de l'accélération de l'âge."""
    
    metrics, preds, annot = load_data()
    best_model = metrics.loc[metrics['mae'].idxmin(), 'model']
    df = preds[preds['model'] == best_model].copy()
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    age_accel = y_pred - y_true
    
    # Tri par âge
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    age_accel_sorted = age_accel[sort_idx]
    
    fig = go.Figure()
    
    # Zone de remplissage pour les accélérations positives et négatives
    fig.add_trace(go.Scatter(
        x=y_true_sorted,
        y=age_accel_sorted,
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 200, 0.2)',
        fillpattern=dict(shape=''),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # Ligne principale avec gradient
    # Créer plusieurs segments colorés
    n_segments = 50
    for i in range(n_segments):
        start_idx = int(i * len(y_true_sorted) / n_segments)
        end_idx = int((i + 1) * len(y_true_sorted) / n_segments) + 1
        
        segment_accel = np.mean(np.abs(age_accel_sorted[start_idx:end_idx]))
        hue = 180 - segment_accel * 10  # Cyan -> Rouge
        hue = max(0, min(180, hue))
        
        fig.add_trace(go.Scatter(
            x=y_true_sorted[start_idx:end_idx],
            y=age_accel_sorted[start_idx:end_idx],
            mode='lines',
            line=dict(
                color=f'hsla({hue}, 100%, 60%, 0.9)',
                width=3,
            ),
            hoverinfo='skip',
            showlegend=False,
        ))
    
    # Points avec glow
    fig.add_trace(go.Scatter(
        x=y_true_sorted,
        y=age_accel_sorted,
        mode='markers',
        marker=dict(
            size=8,
            color=age_accel_sorted,
            colorscale=[
                [0, 'rgba(0, 255, 200, 0.9)'],
                [0.5, 'rgba(255, 255, 100, 0.9)'],
                [1, 'rgba(255, 50, 100, 0.9)'],
            ],
            cmin=-10,
            cmax=10,
            line=dict(width=0),
            colorbar=dict(
                title=dict(text="Δ Age", font=dict(color='white')),
                tickfont=dict(color='white'),
            ),
        ),
        text=[f"Âge: {t:.0f} ans<br>Accélération: {a:+.1f} ans" for t, a in zip(y_true_sorted, age_accel_sorted)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    
    # Ligne zéro
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'))
    
    # Zones de référence
    fig.add_hrect(y0=-2, y1=2, fillcolor='rgba(0, 255, 200, 0.05)', line_width=0)
    fig.add_hrect(y0=5, y1=15, fillcolor='rgba(255, 100, 100, 0.05)', line_width=0)
    fig.add_hrect(y0=-15, y1=-5, fillcolor='rgba(100, 200, 255, 0.05)', line_width=0)
    
    fig.update_layout(
        title=dict(
            text="<b>🌊 VAGUES D'ACCÉLÉRATION ÉPIGÉNÉTIQUE</b><br><sup style='color:#00ffd0'>Déviation de l'âge biologique selon l'âge chronologique</sup>",
            font=dict(size=24, color='white', family='Orbitron, sans-serif'),
            x=0.5,
        ),
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        width=1200,
        xaxis=dict(
            title=dict(text="<b>Âge Chronologique</b> (années)", font=dict(color='#00ffd0', size=14)),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='rgba(255,255,255,0.7)'),
            linecolor='rgba(255,255,255,0.2)',
            linewidth=2,
        ),
        yaxis=dict(
            title=dict(text="<b>Accélération Épigénétique</b> (années)", font=dict(color='#00ffd0', size=14)),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='rgba(255,255,255,0.7)'),
            linecolor='rgba(255,255,255,0.2)',
            linewidth=2,
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor='rgba(10, 10, 20, 0.95)',
            bordercolor='rgba(0, 255, 200, 0.5)',
            font=dict(color='white', size=12),
        ),
    )
    
    # Annotations
    fig.add_annotation(
        x=y_true_sorted.max(), y=7,
        text="⚠️ Vieillissement<br>accéléré",
        showarrow=False,
        font=dict(size=11, color='rgba(255, 100, 100, 0.8)'),
        align='right',
    )
    fig.add_annotation(
        x=y_true_sorted.max(), y=-7,
        text="✨ Vieillissement<br>ralenti",
        showarrow=False,
        font=dict(size=11, color='rgba(100, 200, 255, 0.8)'),
        align='right',
    )
    
    return fig


def create_dna_strand_viz():
    """Crée une visualisation style double hélice ADN."""
    
    metrics, preds, annot = load_data()
    best_model = metrics.loc[metrics['mae'].idxmin(), 'model']
    df = preds[preds['model'] == best_model].copy()
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    
    # Tri
    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    y_pred_s = y_pred[sort_idx]
    error = y_pred_s - y_true_s
    
    n = len(y_true_s)
    t = np.linspace(0, 4 * np.pi, n)
    
    # Hélice 1 (âge chronologique)
    x1 = t
    y1 = np.sin(t) * 2
    z1 = y_true_s
    
    # Hélice 2 (âge prédit)
    x2 = t
    y2 = -np.sin(t) * 2
    z2 = y_pred_s
    
    fig = go.Figure()
    
    # Hélice chronologique
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines+markers',
        marker=dict(size=4, color='rgba(100, 200, 255, 0.8)'),
        line=dict(color='rgba(100, 200, 255, 0.6)', width=4),
        name='Âge chronologique',
    ))
    
    # Hélice prédite
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines+markers',
        marker=dict(
            size=5,
            color=error,
            colorscale='RdYlGn_r',
            cmin=-10,
            cmax=10,
            colorbar=dict(title='Erreur', x=1.1),
        ),
        line=dict(color='rgba(255, 150, 100, 0.6)', width=4),
        name='Âge prédit',
    ))
    
    # Connexions entre les hélices (ponts hydrogène style)
    for i in range(0, n, 3):
        fig.add_trace(go.Scatter3d(
            x=[x1[i], x2[i]],
            y=[y1[i], y2[i]],
            z=[(z1[i] + z2[i])/2, (z1[i] + z2[i])/2],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))
    
    fig.update_layout(
        title=dict(
            text="<b>🧬 DOUBLE HÉLICE ÉPIGÉNÉTIQUE</b>",
            font=dict(size=24, color='white', family='Orbitron, sans-serif'),
            x=0.5,
        ),
        paper_bgcolor='#0a0a0f',
        font=dict(color='white'),
        height=700,
        width=1000,
        scene=dict(
            xaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
            yaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
            zaxis=dict(
                title='Âge (années)',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=False,
                tickfont=dict(color='white'),
            ),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white'),
        ),
    )
    
    return fig


def create_wtf_data_art():
    """
    WTF IS THAT?! - Data Art Abstrait
    
    Une œuvre d'art générative basée sur les données épigénétiques.
    Chaque point de données devient une particule dans un univers visuel
    où le temps, l'âge et la biologie se mêlent en formes organiques.
    """
    
    metrics, preds, annot = load_data()
    best_model = metrics.loc[metrics['mae'].idxmin(), 'model']
    df = preds[preds['model'] == best_model].copy()
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    age_accel = y_pred - y_true
    n = len(y_true)
    
    # Normalisation
    age_norm = (y_true - y_true.min()) / (y_true.max() - y_true.min() + 0.001)
    accel_norm = (age_accel - age_accel.min()) / (age_accel.max() - age_accel.min() + 0.001)
    
    fig = go.Figure()
    
    # =========================================================================
    # LAYER 1: NÉBULEUSE DE FOND - Nuages de particules
    # =========================================================================
    
    np.random.seed(42)
    n_nebula = 2000
    
    # Générer des points de nébuleuse basés sur la distribution des données
    nebula_x = []
    nebula_y = []
    nebula_colors = []
    nebula_sizes = []
    
    for i in range(n):
        # Chaque point de données génère un nuage de particules autour de lui
        cx = y_true[i] / 10 - 4  # Centrer
        cy = age_accel[i] / 3
        
        n_particles = int(20 + accel_norm[i] * 30)
        
        for _ in range(n_particles):
            # Distribution gaussienne autour du point
            px = cx + np.random.randn() * (0.5 + accel_norm[i] * 0.5)
            py = cy + np.random.randn() * (0.3 + age_norm[i] * 0.3)
            
            nebula_x.append(px)
            nebula_y.append(py)
            
            # Couleur basée sur position
            hue = 200 + (px + 5) * 15 + np.random.randn() * 20
            nebula_colors.append(f'hsla({hue % 360}, 80%, 60%, {np.random.uniform(0.02, 0.08)})')
            nebula_sizes.append(np.random.uniform(3, 12))
    
    # Ajouter la nébuleuse
    fig.add_trace(go.Scatter(
        x=nebula_x,
        y=nebula_y,
        mode='markers',
        marker=dict(
            size=nebula_sizes,
            color=nebula_colors,
            line=dict(width=0),
        ),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # =========================================================================
    # LAYER 2: FLUX D'ÉNERGIE - Courbes de Bézier organiques
    # =========================================================================
    
    # Trier par âge
    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    y_pred_s = y_pred[sort_idx]
    accel_s = age_accel[sort_idx]
    
    # Créer des courbes fluides entre les points
    for i in range(0, n - 3, 2):
        # Points de contrôle
        x_points = y_true_s[i:i+4] / 10 - 4
        y_points = accel_s[i:i+4] / 3
        
        # Interpolation cubique pour courbe lisse
        t = np.linspace(0, 1, 50)
        
        # Bézier cubique
        x_curve = (1-t)**3 * x_points[0] + 3*(1-t)**2*t * x_points[1] + 3*(1-t)*t**2 * x_points[2] + t**3 * x_points[3]
        y_curve = (1-t)**3 * y_points[0] + 3*(1-t)**2*t * y_points[1] + 3*(1-t)*t**2 * y_points[2] + t**3 * y_points[3]
        
        # Ajouter du bruit organique
        noise_amp = 0.1 + np.mean(np.abs(accel_s[i:i+4])) * 0.02
        y_curve += np.sin(t * np.pi * 4) * noise_amp * np.random.uniform(0.5, 1.5)
        
        # Couleur basée sur l'accélération moyenne
        mean_accel = np.mean(accel_s[i:i+4])
        hue = 220 + mean_accel * 5
        
        # Glow effect
        for width, alpha in [(8, 0.05), (4, 0.1), (2, 0.3)]:
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=y_curve,
                mode='lines',
                line=dict(
                    color=f'hsla({hue}, 90%, 60%, {alpha})',
                    width=width,
                    shape='spline',
                ),
                hoverinfo='skip',
                showlegend=False,
            ))
    
    # =========================================================================
    # LAYER 3: FILAMENTS COSMIQUES - Connexions entre points proches
    # =========================================================================
    
    # Calculer les distances et connecter les points proches
    for i in range(n):
        x1 = y_true[i] / 10 - 4
        y1 = age_accel[i] / 3
        
        for j in range(i + 1, min(i + 5, n)):
            x2 = y_true[j] / 10 - 4
            y2 = age_accel[j] / 3
            
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if dist < 1.5:  # Seulement les points proches
                # Ligne ondulée
                t = np.linspace(0, 1, 30)
                x_line = x1 + (x2 - x1) * t
                y_line = y1 + (y2 - y1) * t + np.sin(t * np.pi * 3) * 0.1 * dist
                
                alpha = 0.15 * (1 - dist / 1.5)
                hue = 260 + (age_accel[i] + age_accel[j]) * 2
                
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color=f'hsla({hue}, 70%, 55%, {alpha})', width=1),
                    hoverinfo='skip',
                    showlegend=False,
                ))
    
    # =========================================================================
    # LAYER 4: ANNEAUX DE RÉSONANCE - Cercles concentriques aux points extrêmes
    # =========================================================================
    
    # Trouver les points avec accélération extrême
    extreme_idx = np.where(np.abs(age_accel) > np.percentile(np.abs(age_accel), 80))[0]
    
    for idx in extreme_idx:
        cx = y_true[idx] / 10 - 4
        cy = age_accel[idx] / 3
        
        # Anneaux concentriques
        for r in [0.2, 0.4, 0.6, 0.8]:
            theta = np.linspace(0, 2*np.pi, 60)
            
            # Déformation organique
            deform = 1 + 0.2 * np.sin(theta * 3 + age_accel[idx])
            
            x_ring = cx + r * deform * np.cos(theta)
            y_ring = cy + r * 0.6 * deform * np.sin(theta)
            
            alpha = 0.08 * (1 - r)
            hue = 300 if age_accel[idx] > 0 else 180
            
            fig.add_trace(go.Scatter(
                x=x_ring,
                y=y_ring,
                mode='lines',
                line=dict(color=f'hsla({hue}, 80%, 60%, {alpha})', width=1.5),
                hoverinfo='skip',
                showlegend=False,
            ))
    
    # =========================================================================
    # LAYER 5: ÉTOILES PRIMAIRES - Les points de données principaux
    # =========================================================================
    
    # Halo externe
    fig.add_trace(go.Scatter(
        x=y_true / 10 - 4,
        y=age_accel / 3,
        mode='markers',
        marker=dict(
            size=35,
            color=accel_norm,
            colorscale=[
                [0, 'rgba(0, 255, 255, 0.08)'],
                [0.5, 'rgba(150, 100, 255, 0.08)'],
                [1, 'rgba(255, 50, 200, 0.08)'],
            ],
            line=dict(width=0),
        ),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # Halo moyen
    fig.add_trace(go.Scatter(
        x=y_true / 10 - 4,
        y=age_accel / 3,
        mode='markers',
        marker=dict(
            size=20,
            color=accel_norm,
            colorscale=[
                [0, 'rgba(0, 255, 255, 0.2)'],
                [0.5, 'rgba(150, 100, 255, 0.2)'],
                [1, 'rgba(255, 50, 200, 0.2)'],
            ],
            line=dict(width=0),
        ),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # Points centraux brillants
    fig.add_trace(go.Scatter(
        x=y_true / 10 - 4,
        y=age_accel / 3,
        mode='markers',
        marker=dict(
            size=8,
            color=accel_norm,
            colorscale=[
                [0, 'rgb(100, 255, 255)'],
                [0.3, 'rgb(100, 180, 255)'],
                [0.5, 'rgb(180, 120, 255)'],
                [0.7, 'rgb(255, 100, 220)'],
                [1, 'rgb(255, 80, 180)'],
            ],
            line=dict(width=1, color='rgba(255,255,255,0.5)'),
        ),
        text=[
            f"<b>✧ Entité #{i+1}</b><br>"
            f"Temps vécu: {t:.0f} cycles<br>"
            f"Dérive temporelle: {a:+.1f}"
            for i, (t, a) in enumerate(zip(y_true, age_accel))
        ],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    
    # =========================================================================
    # LAYER 6: VORTEX CENTRAL - Point focal artistique
    # =========================================================================
    
    # Spirale centrale
    t_spiral = np.linspace(0, 6*np.pi, 200)
    r_spiral = 0.1 + t_spiral * 0.08
    x_spiral = r_spiral * np.cos(t_spiral) * 0.3
    y_spiral = r_spiral * np.sin(t_spiral) * 0.2
    
    # Gradient de couleur le long de la spirale
    for i in range(len(t_spiral) - 1):
        progress = i / len(t_spiral)
        hue = 200 + progress * 100
        alpha = 0.3 * (1 - progress * 0.5)
        
        fig.add_trace(go.Scatter(
            x=x_spiral[i:i+2],
            y=y_spiral[i:i+2],
            mode='lines',
            line=dict(color=f'hsla({hue}, 90%, 65%, {alpha})', width=3 - progress * 2),
            hoverinfo='skip',
            showlegend=False,
        ))
    
    # =========================================================================
    # LAYER 7: POUSSIÈRE D'ÉTOILES - Particules fines partout
    # =========================================================================
    
    n_dust = 500
    dust_x = np.random.uniform(-6, 6, n_dust)
    dust_y = np.random.uniform(-5, 5, n_dust)
    dust_sizes = np.random.uniform(1, 3, n_dust)
    dust_alphas = np.random.uniform(0.1, 0.4, n_dust)
    
    dust_colors = [f'hsla({220 + np.random.uniform(-30, 80)}, 70%, 70%, {a})' 
                   for a in dust_alphas]
    
    fig.add_trace(go.Scatter(
        x=dust_x,
        y=dust_y,
        mode='markers',
        marker=dict(
            size=dust_sizes,
            color=dust_colors,
            symbol='circle',
            line=dict(width=0),
        ),
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # =========================================================================
    # LAYER 8: TEXTE ARTISTIQUE - Poésie de données
    # =========================================================================
    
    # Citations flottantes
    quotes = [
        ("Le temps n'est qu'une illusion", -5, 4, 9, 0.15),
        ("biologique", -4.5, 3.5, 7, 0.1),
        ("DONNÉES", 4, -3, 20, 0.08),
        ("méthylation", 3, 3.5, 8, 0.12),
        ("∞", 0, 0, 40, 0.1),
        ("ADN", -3, -4, 12, 0.1),
        ("2024", 5, -4, 10, 0.08),
        ("n=" + str(n), 4.5, 4, 8, 0.15),
    ]
    
    for text, x, y, size, alpha in quotes:
        fig.add_annotation(
            x=x, y=y,
            text=text,
            showarrow=False,
            font=dict(
                size=size,
                color=f'rgba(180, 150, 255, {alpha})',
                family='Georgia, serif',
            ),
            textangle=np.random.uniform(-15, 15),
        )
    
    # =========================================================================
    # ANNOTATIONS PRINCIPALES
    # =========================================================================
    
    # Titre artistique
    fig.add_annotation(
        x=0.5, y=1.08,
        xref='paper', yref='paper',
        text="<b>「 CHRONOS FRAGMENTÉ 」</b>",
        showarrow=False,
        font=dict(size=32, color='rgba(200, 180, 255, 0.9)', family='Georgia, serif'),
    )
    
    fig.add_annotation(
        x=0.5, y=1.02,
        xref='paper', yref='paper',
        text="Une méditation visuelle sur le vieillissement épigénétique",
        showarrow=False,
        font=dict(size=12, color='rgba(150, 130, 200, 0.7)', family='Georgia, serif'),
    )
    
    # Signature artistique
    fig.add_annotation(
        x=0.98, y=0.02,
        xref='paper', yref='paper',
        text="data art • 2024",
        showarrow=False,
        font=dict(size=9, color='rgba(255,255,255,0.3)', family='Courier'),
    )
    
    # Légende poétique
    legend_text = (
        "<b>LÉGENDE DES ÂMES</b><br><br>"
        "<span style='color:#64ffff'>◉</span> Jeunesse éternelle<br>"
        "<span style='color:#b080ff'>◉</span> Équilibre temporel<br>"
        "<span style='color:#ff50c8'>◉</span> Temps accéléré<br><br>"
        "<i>Chaque point est une vie,<br>"
        "chaque ligne un destin.</i>"
    )
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=legend_text,
        showarrow=False,
        font=dict(size=9, color='rgba(255,255,255,0.7)', family='Inter'),
        align='left',
        bgcolor='rgba(10, 5, 20, 0.6)',
        bordercolor='rgba(150, 100, 255, 0.2)',
        borderwidth=1,
        borderpad=12,
    )
    
    # =========================================================================
    # LAYOUT FINAL
    # =========================================================================
    
    fig.update_layout(
        paper_bgcolor='#050208',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=900,
        width=1400,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-6.5, 6.5],
            fixedrange=True,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-5.5, 5.5],
            fixedrange=True,
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor='rgba(20, 10, 40, 0.9)',
            bordercolor='rgba(180, 150, 255, 0.5)',
            font=dict(color='white', size=11, family='Georgia'),
        ),
        margin=dict(l=20, r=20, t=120, b=40),
    )
    
    return fig


def save_all_visualizations():
    """Génère et sauvegarde toutes les visualisations."""
    
    output_dir = Path("results/revolutionary_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 GÉNÉRATION DES VISUALISATIONS RÉVOLUTIONNAIRES")
    print("=" * 70)
    
    # Dashboard principal
    print("\n[1/5] Dashboard futuriste...")
    fig1 = create_revolutionary_dashboard()
    fig1.write_html(str(output_dir / "dashboard_futuriste.html"))
    fig1.write_image(str(output_dir / "dashboard_futuriste.png"), scale=2)
    print("  ✓ dashboard_futuriste.html/png")
    
    # Horloge biologique
    print("\n[2/5] Horloge biologique radiale...")
    fig2 = create_biological_clock_viz()
    fig2.write_html(str(output_dir / "horloge_biologique.html"))
    fig2.write_image(str(output_dir / "horloge_biologique.png"), scale=2)
    print("  ✓ horloge_biologique.html/png")
    
    # Vagues d'accélération
    print("\n[3/5] Vagues d'accélération...")
    fig3 = create_age_acceleration_wave()
    fig3.write_html(str(output_dir / "vagues_acceleration.html"))
    fig3.write_image(str(output_dir / "vagues_acceleration.png"), scale=2)
    print("  ✓ vagues_acceleration.html/png")
    
    # Double hélice 3D
    print("\n[4/5] Double hélice ADN 3D...")
    fig4 = create_dna_strand_viz()
    fig4.write_html(str(output_dir / "double_helice_3d.html"))
    print("  ✓ double_helice_3d.html")
    
    # WTF Data Art
    print("\n[5/5] WTF Data Art - Chronos Fragmenté...")
    fig5 = create_wtf_data_art()
    fig5.write_html(str(output_dir / "chronos_fragmente.html"))
    fig5.write_image(str(output_dir / "chronos_fragmente.png"), scale=2)
    print("  ✓ chronos_fragmente.html/png")
    
    print("\n" + "=" * 70)
    print(f"✨ Visualisations sauvegardées dans: {output_dir}")
    print("=" * 70)
    
    return fig1, fig2, fig3, fig4, fig5


if __name__ == "__main__":
    save_all_visualizations()
