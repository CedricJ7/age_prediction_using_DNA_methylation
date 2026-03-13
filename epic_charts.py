#!/usr/bin/env python3
"""
EPIC Charts - Next-Level Visualizations for Age Prediction
===========================================================
Premium, publication-ready visualizations with stunning aesthetics.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "epic_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Premium color palette (Cyberpunk/Neon theme)
NEON = {
    'cyan': '#00fff5',
    'magenta': '#ff00ff',
    'yellow': '#ffff00',
    'orange': '#ff6b35',
    'pink': '#ff1493',
    'purple': '#9d4edd',
    'blue': '#00b4d8',
    'green': '#39ff14',
    'red': '#ff073a',
}

GRADIENT_COLORS = [
    [0, '#0d0221'],
    [0.2, '#0f084b'],
    [0.4, '#26408b'],
    [0.6, '#0ead69'],
    [0.8, '#f4e409'],
    [1, '#ff1654']
]


def load_data():
    """Load all datasets."""
    data = {}

    if (RESULTS_DIR / "metrics.csv").exists():
        data['metrics'] = pd.read_csv(RESULTS_DIR / "metrics.csv")

    if (RESULTS_DIR / "annot_predictions.csv").exists():
        data['predictions'] = pd.read_csv(RESULTS_DIR / "annot_predictions.csv")

    if (RESULTS_DIR / "optimization_complete" / "results_intermediate.csv").exists():
        data['optimization'] = pd.read_csv(RESULTS_DIR / "optimization_complete" / "results_intermediate.csv")

    if (RESULTS_DIR / "coefficients_elasticnet.csv").exists():
        data['coefficients'] = pd.read_csv(RESULTS_DIR / "coefficients_elasticnet.csv")

    return data


def create_3d_prediction_landscape(pred_df):
    """
    3D Surface showing prediction accuracy across age and samples.
    """
    print("Creating 3D prediction landscape...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['error'] = df['age_pred'] - df['age']
    df['abs_error'] = np.abs(df['error'])
    df = df.sort_values('age')

    # Create a grid for 3D surface
    age_bins = np.linspace(df['age'].min(), df['age'].max(), 30)
    sample_idx = np.arange(len(df))

    fig = go.Figure()

    # 3D Scatter of predictions
    fig.add_trace(go.Scatter3d(
        x=df['age'],
        y=df['age_pred'],
        z=df['abs_error'],
        mode='markers',
        marker=dict(
            size=4,
            color=df['abs_error'],
            colorscale='Turbo',
            opacity=0.8,
            colorbar=dict(
                title=dict(text='Error (years)', font=dict(color='white')),
                tickfont=dict(color='white')
            ),
            line=dict(width=0.5, color='white')
        ),
        text=[f"Age: {a:.1f}<br>Pred: {p:.1f}<br>Error: {e:.2f}"
              for a, p, e in zip(df['age'], df['age_pred'], df['abs_error'])],
        hoverinfo='text',
        name='Predictions'
    ))

    # Perfect prediction plane
    age_range = np.linspace(df['age'].min(), df['age'].max(), 20)
    xx, yy = np.meshgrid(age_range, age_range)
    zz = np.zeros_like(xx)

    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        opacity=0.3,
        colorscale=[[0, NEON['cyan']], [1, NEON['cyan']]],
        showscale=False,
        name='Perfect prediction'
    ))

    fig.update_layout(
        title=dict(
            text='🌌 3D Prediction Landscape',
            font=dict(size=28, color='white', family='Orbitron, sans-serif'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title='Chronological Age', backgroundcolor='#0a0a0a',
                      gridcolor='#333', color='white'),
            yaxis=dict(title='Predicted Age', backgroundcolor='#0a0a0a',
                      gridcolor='#333', color='white'),
            zaxis=dict(title='Absolute Error', backgroundcolor='#0a0a0a',
                      gridcolor='#333', color='white'),
            bgcolor='#0a0a0a',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=800,
        width=1200
    )

    fig.write_html(OUTPUT_DIR / "01_3d_prediction_landscape.html")
    print("  ✓ Saved 3D prediction landscape")
    return fig


def create_neon_gauge_dashboard(metrics_df):
    """
    Futuristic gauge dashboard with neon styling.
    """
    print("Creating neon gauge dashboard...")

    best = metrics_df.loc[metrics_df['mae'].idxmin()]

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
        ],
        vertical_spacing=0.3,
        horizontal_spacing=0.1
    )

    # Gauge configurations
    gauges = [
        {'value': best['mae'], 'title': 'MAE', 'suffix': ' years', 'range': [0, 15],
         'colors': ['#00fff5', '#ff00ff'], 'threshold': 5},
        {'value': best['r2'] * 100, 'title': 'R² Score', 'suffix': '%', 'range': [0, 100],
         'colors': ['#ff1493', '#39ff14'], 'threshold': 90},
        {'value': best['correlation'] * 100, 'title': 'Correlation', 'suffix': '%', 'range': [0, 100],
         'colors': ['#9d4edd', '#00b4d8'], 'threshold': 95},
        {'value': best['rmse'], 'title': 'RMSE', 'suffix': ' years', 'range': [0, 20],
         'colors': ['#ff6b35', '#ffff00'], 'threshold': 6},
        {'value': best['overfitting_ratio'], 'title': 'Overfitting', 'suffix': 'x', 'range': [0, 10],
         'colors': ['#39ff14', '#ff073a'], 'threshold': 3},
        {'value': best['fit_time_sec'], 'title': 'Training Time', 'suffix': 's', 'range': [0, 60],
         'colors': ['#00b4d8', '#ff1493'], 'threshold': 30},
    ]

    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]

    for (row, col), g in zip(positions, gauges):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=g['value'],
                number={'suffix': g['suffix'], 'font': {'size': 36, 'color': g['colors'][0]}},
                title={'text': g['title'], 'font': {'size': 18, 'color': 'white'}},
                delta={'reference': g['threshold'], 'relative': False},
                gauge=dict(
                    axis=dict(range=g['range'], tickcolor='white', tickfont=dict(color='white')),
                    bar=dict(color=g['colors'][0], thickness=0.75),
                    bgcolor='#1a1a2e',
                    borderwidth=2,
                    bordercolor=g['colors'][1],
                    steps=[
                        {'range': [0, g['range'][1]*0.5], 'color': 'rgba(0,255,245,0.1)'},
                        {'range': [g['range'][1]*0.5, g['range'][1]], 'color': 'rgba(255,0,255,0.1)'}
                    ],
                    threshold=dict(
                        line=dict(color=NEON['yellow'], width=4),
                        thickness=0.8,
                        value=g['threshold']
                    )
                )
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(
            text=f'⚡ {best["model"]} Performance Dashboard',
            font=dict(size=32, color=NEON['cyan'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        paper_bgcolor='#0a0a0a',
        font=dict(color='white', family='Rajdhani, sans-serif'),
        height=700,
        width=1200
    )

    fig.write_html(OUTPUT_DIR / "02_neon_gauge_dashboard.html")
    print("  ✓ Saved neon gauge dashboard")
    return fig


def create_animated_model_race(metrics_df):
    """
    Animated bar chart race showing model performance.
    """
    print("Creating animated model race...")

    df = metrics_df[metrics_df['model'] != 'AltumAge'].copy()

    # Create animation frames
    metrics_list = ['mae', 'rmse', 'r2', 'correlation']
    metric_names = ['MAE (years)', 'RMSE (years)', 'R² Score', 'Correlation']

    fig = go.Figure()

    # Colors for each model
    colors = [NEON['cyan'], NEON['magenta'], NEON['green'], NEON['orange'], NEON['purple']]

    for i, model in enumerate(df['model']):
        fig.add_trace(go.Bar(
            name=model,
            x=[model],
            y=[df[df['model'] == model]['mae'].values[0]],
            marker=dict(
                color=colors[i % len(colors)],
                line=dict(color='white', width=2)
            ),
            text=[f"{df[df['model'] == model]['mae'].values[0]:.2f}"],
            textposition='outside',
            textfont=dict(color='white', size=14)
        ))

    # Create frames for animation
    frames = []
    for metric, name in zip(metrics_list, metric_names):
        frame_data = []
        values = df[metric].values
        if metric in ['r2', 'correlation']:
            values = values * 100  # Convert to percentage

        for i, model in enumerate(df['model']):
            val = values[i]
            frame_data.append(go.Bar(
                name=model,
                x=[model],
                y=[val],
                marker=dict(color=colors[i % len(colors)], line=dict(color='white', width=2)),
                text=[f"{val:.2f}"],
                textposition='outside',
                textfont=dict(color='white', size=14)
            ))
        frames.append(go.Frame(data=frame_data, name=name))

    fig.frames = frames

    # Animation controls
    fig.update_layout(
        title=dict(
            text='🏁 Model Performance Race',
            font=dict(size=28, color=NEON['yellow'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, {'frame': {'duration': 1500, 'redraw': True},
                                     'fromcurrent': True,
                                     'transition': {'duration': 500, 'easing': 'cubic-in-out'}}]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate',
                                       'transition': {'duration': 0}}])
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16, 'color': 'white'},
                'prefix': 'Metric: ',
                'visible': True,
                'xanchor': 'center'
            },
            'transition': {'duration': 500, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': [
                {'args': [[name], {'frame': {'duration': 500, 'redraw': True},
                                   'mode': 'immediate',
                                   'transition': {'duration': 500}}],
                 'label': name,
                 'method': 'animate'} for name in metric_names
            ]
        }],
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#222', color='white'),
        yaxis=dict(gridcolor='#222', color='white', title='Score'),
        height=600,
        width=1000,
        showlegend=False
    )

    fig.write_html(OUTPUT_DIR / "03_animated_model_race.html")
    print("  ✓ Saved animated model race")
    return fig


def create_sunburst_analysis(pred_df):
    """
    Hierarchical sunburst chart for demographic analysis.
    """
    print("Creating sunburst analysis...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['gender'] = df['female'].map({True: 'Female', False: 'Male'})
    df['abs_error'] = np.abs(df['age_pred'] - df['age'])
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 55, 75, 100],
                             labels=['Young', 'Middle', 'Senior', 'Elderly'])

    # Aggregate data for sunburst
    agg = df.groupby(['gender', 'ethnicity', 'age_group']).agg({
        'abs_error': 'mean',
        'Sample_Name': 'count'
    }).reset_index()
    agg.columns = ['gender', 'ethnicity', 'age_group', 'mae', 'count']

    fig = px.sunburst(
        agg,
        path=['gender', 'ethnicity', 'age_group'],
        values='count',
        color='mae',
        color_continuous_scale='Turbo',
        title='🎯 Demographic Analysis Sunburst'
    )

    fig.update_traces(
        textfont=dict(size=14, color='white'),
        insidetextorientation='radial',
        marker=dict(line=dict(color='#0a0a0a', width=2))
    )

    fig.update_layout(
        title=dict(
            font=dict(size=28, color=NEON['cyan'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=800,
        width=900,
        coloraxis_colorbar=dict(
            title=dict(text='MAE (years)', font=dict(color='white')),
            tickfont=dict(color='white')
        )
    )

    fig.write_html(OUTPUT_DIR / "04_sunburst_demographics.html")
    print("  ✓ Saved sunburst demographics")
    return fig


def create_animated_scatter(pred_df):
    """
    Animated scatter plot with pulsing effects.
    """
    print("Creating animated scatter with trails...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['error'] = df['age_pred'] - df['age']
    df['abs_error'] = np.abs(df['error'])
    df['gender'] = df['female'].map({True: 'Female', False: 'Male'})
    df = df.sort_values('age')

    fig = go.Figure()

    # Background gradient effect
    for i, alpha in enumerate([0.05, 0.1, 0.15]):
        offset = (i + 1) * 2
        fig.add_trace(go.Scatter(
            x=[df['age'].min() - offset, df['age'].max() + offset,
               df['age'].max() + offset, df['age'].min() - offset],
            y=[df['age'].min() - offset, df['age'].min() - offset,
               df['age'].max() + offset, df['age'].max() + offset],
            fill='toself',
            fillcolor=f'rgba(99, 102, 241, {alpha})',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Perfect prediction line with glow
    for width, alpha in [(8, 0.2), (4, 0.5), (2, 1)]:
        fig.add_trace(go.Scatter(
            x=[df['age'].min(), df['age'].max()],
            y=[df['age'].min(), df['age'].max()],
            mode='lines',
            line=dict(color=f'rgba(0, 255, 245, {alpha})', width=width),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Main scatter with size based on error
    for gender, color in [('Female', NEON['magenta']), ('Male', NEON['cyan'])]:
        gdf = df[df['gender'] == gender]
        fig.add_trace(go.Scatter(
            x=gdf['age'],
            y=gdf['age_pred'],
            mode='markers',
            name=gender,
            marker=dict(
                size=gdf['abs_error'] * 2 + 5,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='white'),
                symbol='circle'
            ),
            text=[f"<b>{g}</b><br>Age: {a:.1f}<br>Pred: {p:.1f}<br>Error: {e:+.2f}y"
                  for g, a, p, e in zip(gdf['gender'], gdf['age'], gdf['age_pred'], gdf['error'])],
            hoverinfo='text'
        ))

    # Annotations for extreme cases
    worst = df.nlargest(3, 'abs_error')
    for _, row in worst.iterrows():
        fig.add_annotation(
            x=row['age'], y=row['age_pred'],
            text=f"⚠️ {row['abs_error']:.1f}y error",
            showarrow=True,
            arrowhead=2,
            arrowcolor=NEON['red'],
            font=dict(color=NEON['red'], size=10),
            bgcolor='rgba(10,10,10,0.8)',
            bordercolor=NEON['red']
        )

    fig.update_layout(
        title=dict(
            text='✨ Prediction Accuracy by Gender',
            font=dict(size=28, color='white', family='Orbitron, sans-serif'),
            x=0.5
        ),
        xaxis=dict(
            title='Chronological Age',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        ),
        yaxis=dict(
            title='Predicted Epigenetic Age',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        height=700,
        width=1000
    )

    fig.write_html(OUTPUT_DIR / "05_animated_scatter.html")
    print("  ✓ Saved animated scatter")
    return fig


def create_waterfall_error(metrics_df):
    """
    Waterfall chart showing error breakdown.
    """
    print("Creating waterfall error chart...")

    df = metrics_df[metrics_df['model'] != 'AltumAge'].sort_values('mae')

    # Calculate incremental differences
    base_mae = df['mae'].iloc[0]

    fig = go.Figure(go.Waterfall(
        name="Model Comparison",
        orientation="v",
        x=df['model'].tolist(),
        textposition="outside",
        text=[f"{v:.2f}" for v in df['mae']],
        y=df['mae'].tolist(),
        connector={"line": {"color": NEON['cyan'], "width": 2, "dash": "dot"}},
        decreasing={"marker": {"color": NEON['green'], "line": {"color": NEON['green'], "width": 2}}},
        increasing={"marker": {"color": NEON['red'], "line": {"color": NEON['red'], "width": 2}}},
        totals={"marker": {"color": NEON['purple'], "line": {"color": NEON['purple'], "width": 2}}}
    ))

    fig.update_layout(
        title=dict(
            text='📊 Model Error Comparison (MAE)',
            font=dict(size=28, color=NEON['yellow'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        xaxis=dict(title='Model', color='white', gridcolor='#222'),
        yaxis=dict(title='MAE (years)', color='white', gridcolor='#222'),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white', size=14),
        height=500,
        width=900,
        showlegend=False
    )

    fig.write_html(OUTPUT_DIR / "06_waterfall_error.html")
    print("  ✓ Saved waterfall error chart")
    return fig


def create_polar_features(coef_df):
    """
    Polar/rose chart for top features.
    """
    print("Creating polar feature chart...")

    df = coef_df.head(20).copy()
    df['theta'] = np.linspace(0, 360, len(df), endpoint=False)
    df['color'] = df['coef'].apply(lambda x: NEON['green'] if x > 0 else NEON['red'])

    fig = go.Figure()

    # Add bars in polar coordinates
    fig.add_trace(go.Barpolar(
        r=df['abs_coef'],
        theta=df['theta'],
        width=15,
        marker=dict(
            color=df['color'],
            line=dict(color='white', width=1),
            opacity=0.8
        ),
        text=df['feature'],
        hovertemplate='<b>%{text}</b><br>Coefficient: %{r:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='🧬 Top 20 CpG Biomarkers',
            font=dict(size=28, color=NEON['cyan'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        polar=dict(
            bgcolor='#0a0a0a',
            radialaxis=dict(
                visible=True,
                color='white',
                gridcolor='#333'
            ),
            angularaxis=dict(
                visible=False
            )
        ),
        paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=700,
        width=800,
        annotations=[
            dict(x=0.5, y=-0.1, xref='paper', yref='paper',
                 text=f'<span style="color:{NEON["green"]}">● Aging markers</span>  '
                      f'<span style="color:{NEON["red"]}">● Youth markers</span>',
                 showarrow=False, font=dict(size=14))
        ]
    )

    fig.write_html(OUTPUT_DIR / "07_polar_features.html")
    print("  ✓ Saved polar feature chart")
    return fig


def create_treemap_optimization(opt_df):
    """
    Treemap showing optimization results.
    """
    print("Creating optimization treemap...")

    df = opt_df.copy()
    df['config'] = df['strategy'] + '_' + df['n_features'].astype(str)

    # Keep best result per config per model
    df = df.loc[df.groupby(['config', 'model_name'])['mae_test'].idxmin()]

    fig = px.treemap(
        df,
        path=['strategy', 'n_features', 'model_name'],
        values='n_trials',
        color='mae_test',
        color_continuous_scale='RdYlGn_r',
        title='🗺️ Hyperparameter Optimization Map'
    )

    fig.update_traces(
        textfont=dict(size=14, color='white'),
        marker=dict(line=dict(color='#0a0a0a', width=2))
    )

    fig.update_layout(
        title=dict(
            font=dict(size=28, color=NEON['magenta'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=700,
        width=1100,
        coloraxis_colorbar=dict(
            title=dict(text='MAE', font=dict(color='white')),
            tickfont=dict(color='white')
        )
    )

    fig.write_html(OUTPUT_DIR / "08_optimization_treemap.html")
    print("  ✓ Saved optimization treemap")
    return fig


def create_violin_ensemble(pred_df):
    """
    Beautiful violin plots for all models.
    """
    print("Creating violin ensemble...")

    df = pred_df.copy()
    df['error'] = df['age_pred'] - df['age']

    # Exclude AltumAge for cleaner viz
    df = df[df['model'] != 'AltumAge']

    colors = [NEON['cyan'], NEON['magenta'], NEON['green'],
              NEON['orange'], NEON['purple']]

    fig = go.Figure()

    for i, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]['error']

        fig.add_trace(go.Violin(
            y=model_data,
            name=model,
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[i % len(colors)],
            opacity=0.7,
            line_color='white',
            points='outliers',
            pointpos=0,
            jitter=0.05
        ))

    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color=NEON['yellow'], line_width=2,
                  annotation_text='Perfect prediction', annotation_position='right')

    fig.update_layout(
        title=dict(
            text='🎻 Prediction Error Distribution by Model',
            font=dict(size=28, color=NEON['cyan'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        xaxis=dict(title='Model', color='white', gridcolor='#222'),
        yaxis=dict(title='Prediction Error (years)', color='white', gridcolor='#222'),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=600,
        width=1000,
        showlegend=False
    )

    fig.write_html(OUTPUT_DIR / "09_violin_ensemble.html")
    print("  ✓ Saved violin ensemble")
    return fig


def create_hexbin_density(pred_df):
    """
    Hexbin density plot for predictions.
    """
    print("Creating hexbin density plot...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()

    fig = go.Figure()

    # Create 2D histogram (hexbin-like)
    fig.add_trace(go.Histogram2dContour(
        x=df['age'],
        y=df['age_pred'],
        colorscale='Turbo',
        contours=dict(
            showlabels=True,
            labelfont=dict(color='white')
        ),
        line=dict(width=2, color='white'),
        ncontours=15,
        colorbar=dict(
            title=dict(text='Density', font=dict(color='white')),
            tickfont=dict(color='white')
        )
    ))

    # Add scatter on top
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df['age_pred'],
        mode='markers',
        marker=dict(
            size=4,
            color='white',
            opacity=0.3
        ),
        hoverinfo='skip'
    ))

    # Perfect line
    fig.add_trace(go.Scatter(
        x=[df['age'].min(), df['age'].max()],
        y=[df['age'].min(), df['age'].max()],
        mode='lines',
        line=dict(color=NEON['red'], width=3, dash='dash'),
        name='Perfect'
    ))

    fig.update_layout(
        title=dict(
            text='🔥 Prediction Density Map',
            font=dict(size=28, color=NEON['orange'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        xaxis=dict(title='Chronological Age', color='white', gridcolor='#222'),
        yaxis=dict(title='Predicted Age', color='white', gridcolor='#222'),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=700,
        width=800,
        showlegend=False
    )

    fig.write_html(OUTPUT_DIR / "10_hexbin_density.html")
    print("  ✓ Saved hexbin density plot")
    return fig


def create_ultimate_dashboard(data):
    """
    The ultimate multi-panel dashboard.
    """
    print("Creating ultimate dashboard...")

    metrics = data['metrics']
    pred = data['predictions']

    best = metrics.loc[metrics['mae'].idxmin()]
    pred_en = pred[(pred['model'] == 'ElasticNet')]

    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'bar', 'colspan': 2}, None, {'type': 'pie'}],
            [{'type': 'scatter', 'colspan': 2}, None, {'type': 'histogram'}]
        ],
        subplot_titles=['', '', '', 'Model Performance', 'Ethnicity Distribution',
                       'Predictions vs Actual', 'Error Distribution'],
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    # Row 1: KPIs
    kpis = [
        (best['mae'], 'MAE', NEON['cyan'], 'years'),
        (best['r2'] * 100, 'R² Score', NEON['green'], '%'),
        (best['correlation'] * 100, 'Correlation', NEON['magenta'], '%')
    ]

    for col, (val, title, color, suffix) in enumerate(kpis, 1):
        fig.add_trace(go.Indicator(
            mode='number',
            value=val,
            number={'suffix': suffix, 'font': {'size': 48, 'color': color}},
            title={'text': title, 'font': {'size': 16, 'color': 'white'}}
        ), row=1, col=col)

    # Row 2: Bar chart
    df = metrics[metrics['model'] != 'AltumAge'].sort_values('mae')
    colors = [NEON['cyan'], NEON['magenta'], NEON['green'], NEON['orange'], NEON['purple']]
    fig.add_trace(go.Bar(
        x=df['model'],
        y=df['mae'],
        marker_color=colors[:len(df)],
        text=df['mae'].round(2),
        textposition='outside',
        textfont=dict(color='white')
    ), row=2, col=1)

    # Pie chart
    eth_counts = pred_en['ethnicity'].value_counts()
    fig.add_trace(go.Pie(
        labels=eth_counts.index,
        values=eth_counts.values,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textfont=dict(color='white'),
        hole=0.4
    ), row=2, col=3)

    # Row 3: Scatter
    fig.add_trace(go.Scatter(
        x=pred_en['age'],
        y=pred_en['age_pred'],
        mode='markers',
        marker=dict(color=NEON['cyan'], size=5, opacity=0.6),
        showlegend=False
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=[pred_en['age'].min(), pred_en['age'].max()],
        y=[pred_en['age'].min(), pred_en['age'].max()],
        mode='lines',
        line=dict(color=NEON['red'], dash='dash', width=2),
        showlegend=False
    ), row=3, col=1)

    # Histogram
    errors = pred_en['age_pred'] - pred_en['age']
    fig.add_trace(go.Histogram(
        x=errors,
        marker_color=NEON['magenta'],
        opacity=0.7,
        nbinsx=30
    ), row=3, col=3)

    fig.update_layout(
        title=dict(
            text='🚀 Age Prediction Ultimate Dashboard',
            font=dict(size=32, color=NEON['yellow'], family='Orbitron, sans-serif'),
            x=0.5
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white', family='Rajdhani, sans-serif'),
        height=1000,
        width=1400,
        showlegend=False
    )

    # Update axes colors
    fig.update_xaxes(gridcolor='#222', color='white')
    fig.update_yaxes(gridcolor='#222', color='white')

    fig.write_html(OUTPUT_DIR / "00_ultimate_dashboard.html")
    print("  ✓ Saved ultimate dashboard")
    return fig


def main():
    """Generate all epic charts."""
    print("=" * 70)
    print("  🚀 EPIC CHARTS - Next-Level Visualizations")
    print("=" * 70)

    data = load_data()

    if not data:
        print("Error: No data found!")
        return

    print(f"\n📊 Loaded: {len(data['metrics'])} models, {len(data['predictions'])} predictions")
    print(f"   {len(data['optimization'])} optimization trials, {len(data['coefficients'])} features\n")

    print("-" * 70)

    # Generate all charts
    create_ultimate_dashboard(data)
    create_3d_prediction_landscape(data['predictions'])
    create_neon_gauge_dashboard(data['metrics'])
    create_animated_model_race(data['metrics'])
    create_sunburst_analysis(data['predictions'])
    create_animated_scatter(data['predictions'])
    create_waterfall_error(data['metrics'])
    create_polar_features(data['coefficients'])
    create_treemap_optimization(data['optimization'])
    create_violin_ensemble(data['predictions'])
    create_hexbin_density(data['predictions'])

    print("-" * 70)
    print(f"\n✅ All charts saved to: {OUTPUT_DIR}")
    print("\n📁 Generated files:")
    for f in sorted(OUTPUT_DIR.glob("*.html")):
        print(f"   • {f.name}")

    print("\n" + "=" * 70)
    print("  🎉 Done! Open any HTML file in your browser to explore.")
    print("=" * 70)


if __name__ == "__main__":
    main()
