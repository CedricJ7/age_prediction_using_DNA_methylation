#!/usr/bin/env python3
"""
Special Charts: Chord Diagram + Stream Graph
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "special_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
          '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']


def load_data():
    pred = pd.read_csv(RESULTS_DIR / "annot_predictions.csv")
    return pred


def create_chord_diagram(pred_df):
    """
    Chord diagram: Model <-> Ethnicity performance relationships
    """
    print("Creating Chord Diagram...")

    df = pred_df[pred_df['model'] != 'AltumAge'].copy()
    df['abs_error'] = np.abs(df['age_pred'] - df['age'])

    models = df['model'].unique().tolist()
    ethnicities = df['ethnicity'].unique().tolist()

    # All nodes: models first, then ethnicities
    nodes = models + ethnicities
    n_models = len(models)
    n_eth = len(ethnicities)
    n_total = len(nodes)

    # Create matrix: inverse of MAE (higher = better performance = thicker connection)
    matrix = np.zeros((n_total, n_total))

    for i, model in enumerate(models):
        for j, eth in enumerate(ethnicities):
            subset = df[(df['model'] == model) & (df['ethnicity'] == eth)]
            if len(subset) > 0:
                mae = subset['abs_error'].mean()
                # Inverse: good performance = high value
                value = max(0, 10 - mae) * len(subset) / 50
                matrix[i, n_models + j] = value
                matrix[n_models + j, i] = value

    # Build chord diagram with Plotly
    # Calculate node positions on circle
    node_angles = np.linspace(0, 2 * np.pi, n_total, endpoint=False)

    # Gap between model and ethnicity groups
    gap = 0.3
    model_angles = np.linspace(0, np.pi - gap, n_models, endpoint=True)
    eth_angles = np.linspace(np.pi + gap, 2 * np.pi - gap, n_eth, endpoint=True)
    node_angles = np.concatenate([model_angles, eth_angles])

    radius = 1.0
    node_x = radius * np.cos(node_angles)
    node_y = radius * np.sin(node_angles)

    fig = go.Figure()

    # Draw chords (connections)
    for i in range(n_models):
        for j in range(n_eth):
            value = matrix[i, n_models + j]
            if value > 0:
                # Bezier curve from model to ethnicity
                x0, y0 = node_x[i], node_y[i]
                x1, y1 = node_x[n_models + j], node_y[n_models + j]

                # Control points for bezier (through center with offset)
                cx, cy = 0, 0

                # Create bezier path
                t = np.linspace(0, 1, 50)
                bx = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
                by = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1

                opacity = min(0.7, value / 5)
                width = max(1, value * 1.5)

                fig.add_trace(go.Scatter(
                    x=bx, y=by,
                    mode='lines',
                    line=dict(
                        color=COLORS[i % len(COLORS)],
                        width=width
                    ),
                    opacity=opacity,
                    hoverinfo='text',
                    hovertext=f"{models[i]} → {ethnicities[j]}<br>Performance score: {value:.1f}",
                    showlegend=False
                ))

    # Draw nodes (outer arc segments)
    for i, (name, x, y, angle) in enumerate(zip(nodes, node_x, node_y, node_angles)):
        is_model = i < n_models
        color = COLORS[i % len(COLORS)] if is_model else COLORS[(i - n_models + 5) % len(COLORS)]
        size = 25 if is_model else 20

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=size, color=color, line=dict(width=2, color='white')),
            text=[name],
            textposition='top center' if y > 0 else 'bottom center',
            textfont=dict(size=11, color='#333'),
            hoverinfo='text',
            hovertext=f"<b>{name}</b><br>{'Model' if is_model else 'Ethnicity'}",
            showlegend=False
        ))

    # Add outer ring arcs for groups
    theta_model = np.linspace(model_angles[0] - 0.1, model_angles[-1] + 0.1, 50)
    theta_eth = np.linspace(eth_angles[0] - 0.1, eth_angles[-1] + 0.1, 50)

    r_outer = 1.15
    fig.add_trace(go.Scatter(
        x=r_outer * np.cos(theta_model),
        y=r_outer * np.sin(theta_model),
        mode='lines',
        line=dict(color='#4e79a7', width=8),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=r_outer * np.cos(theta_eth),
        y=r_outer * np.sin(theta_eth),
        mode='lines',
        line=dict(color='#e15759', width=8),
        hoverinfo='skip',
        showlegend=False
    ))

    # Labels for groups
    fig.add_annotation(x=0, y=1.35, text="<b>MODELS</b>", showarrow=False,
                       font=dict(size=14, color='#4e79a7'))
    fig.add_annotation(x=0, y=-1.35, text="<b>ETHNICITY</b>", showarrow=False,
                       font=dict(size=14, color='#e15759'))

    fig.update_layout(
        title=dict(
            text='Model-Ethnicity Performance Relationships',
            x=0.5, font=dict(size=20)
        ),
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor='x'),
        paper_bgcolor='#fafafa',
        plot_bgcolor='#fafafa',
        height=700,
        width=750,
        margin=dict(t=80, b=40, l=40, r=40)
    )

    fig.write_html(OUTPUT_DIR / "chord_diagram.html")
    print("  ✓ Saved chord_diagram.html")
    return fig


def create_stream_graph(pred_df):
    """
    Stream graph: Error distribution flowing across age
    """
    print("Creating Stream Graph...")

    df = pred_df[pred_df['model'] != 'AltumAge'].copy()
    df['error'] = df['age_pred'] - df['age']

    # Create age bins
    age_bins = np.arange(20, 95, 5)
    df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=age_bins[:-1])
    df = df.dropna(subset=['age_bin'])
    df['age_bin'] = df['age_bin'].astype(float)

    models = ['ElasticNet', 'Ridge', 'Lasso', 'XGBoost', 'RandomForest']

    # Aggregate: count of predictions per age bin per model
    # Split by positive/negative error for stream effect
    stream_data = []

    for model in models:
        model_df = df[df['model'] == model]

        for age in age_bins[:-1]:
            bin_data = model_df[model_df['age_bin'] == age]

            # Positive errors (predicted older)
            pos_errors = bin_data[bin_data['error'] > 0]['error']
            neg_errors = bin_data[bin_data['error'] < 0]['error']

            stream_data.append({
                'age': age,
                'model': model,
                'pos_count': len(pos_errors),
                'neg_count': len(neg_errors),
                'pos_mean': pos_errors.mean() if len(pos_errors) > 0 else 0,
                'neg_mean': neg_errors.mean() if len(neg_errors) > 0 else 0,
                'total': len(bin_data),
                'mae': bin_data['error'].abs().mean() if len(bin_data) > 0 else 0
            })

    stream_df = pd.DataFrame(stream_data)

    fig = go.Figure()

    # Create stacked streams - one for each model
    # Using stackgroup for stream effect

    colors_pos = ['rgba(78,121,167,0.8)', 'rgba(242,142,43,0.8)', 'rgba(225,87,89,0.8)',
                  'rgba(118,183,178,0.8)', 'rgba(89,161,79,0.8)']
    colors_neg = ['rgba(78,121,167,0.5)', 'rgba(242,142,43,0.5)', 'rgba(225,87,89,0.5)',
                  'rgba(118,183,178,0.5)', 'rgba(89,161,79,0.5)']

    # Positive errors (above zero line)
    for i, model in enumerate(models):
        model_data = stream_df[stream_df['model'] == model].sort_values('age')

        # Smooth the data
        y_vals = model_data['mae'].values * model_data['total'].values / 10

        fig.add_trace(go.Scatter(
            x=model_data['age'],
            y=y_vals,
            name=model,
            mode='lines',
            line=dict(width=0.5, color=COLORS[i]),
            stackgroup='pos',
            groupnorm='',
            fillcolor=colors_pos[i],
            hovertemplate=f'<b>{model}</b><br>Age: %{{x}}<br>Samples: %{{text}}<extra></extra>',
            text=model_data['total']
        ))

    # Mirror for negative (below zero) - creates the stream effect
    for i, model in enumerate(reversed(models)):
        model_data = stream_df[stream_df['model'] == model].sort_values('age')

        y_vals = -model_data['mae'].values * model_data['total'].values / 10

        fig.add_trace(go.Scatter(
            x=model_data['age'],
            y=y_vals,
            name=model,
            mode='lines',
            line=dict(width=0.5, color=COLORS[len(models)-1-i]),
            stackgroup='neg',
            groupnorm='',
            fillcolor=colors_neg[len(models)-1-i],
            showlegend=False,
            hoverinfo='skip'
        ))

    # Center line
    fig.add_hline(y=0, line_color='white', line_width=2)

    # Age markers
    for age in [30, 50, 70, 90]:
        fig.add_vline(x=age, line_dash='dot', line_color='rgba(0,0,0,0.2)', line_width=1)

    fig.update_layout(
        title=dict(
            text='Prediction Volume & Error Across Age Spectrum',
            x=0.5, font=dict(size=20)
        ),
        xaxis=dict(
            title='Age (years)',
            gridcolor='rgba(0,0,0,0.1)',
            range=[20, 90]
        ),
        yaxis=dict(
            title='Weighted Error × Sample Volume',
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white', family='Arial'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=500,
        width=1000,
        margin=dict(t=100, b=60)
    )

    fig.write_html(OUTPUT_DIR / "stream_graph.html")
    print("  ✓ Saved stream_graph.html")
    return fig


def main():
    print("=" * 50)
    print("  Special Charts: Chord + Stream")
    print("=" * 50 + "\n")

    pred_df = load_data()
    print(f"Loaded {len(pred_df)} predictions\n")

    create_chord_diagram(pred_df)
    create_stream_graph(pred_df)

    print(f"\n✓ Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
