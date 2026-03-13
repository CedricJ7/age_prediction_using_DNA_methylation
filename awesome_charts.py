#!/usr/bin/env python3
"""
Awesome Charts for Age Prediction Project
=========================================
A collection of beautiful, insightful visualizations for the epigenetic clock analysis.

Charts included:
1. Model Performance Radar Chart
2. Hyperparameter Optimization Heatmap
3. Demographic Bias Analysis
4. Top CpG Features (Positive vs Negative)
5. Age-Stratified Accuracy Analysis
6. Overfitting Profile Comparison
7. Prediction Confidence Bands
8. Age Acceleration by Demographics
9. Model Agreement Analysis
10. Optimization Trajectory (Parallel Coordinates)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "awesome_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palettes
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#ec4899',    # Pink
    'success': '#10b981',      # Emerald
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'info': '#3b82f6',         # Blue
    'purple': '#8b5cf6',
    'teal': '#14b8a6',
}

MODEL_COLORS = {
    'ElasticNet': '#6366f1',
    'Lasso': '#ec4899',
    'Ridge': '#10b981',
    'XGBoost': '#f59e0b',
    'RandomForest': '#3b82f6',
    'AltumAge': '#8b5cf6',
    'GradientBoosting': '#14b8a6',
    'LightGBM': '#ef4444',
}

# Dark theme template
DARK_TEMPLATE = {
    'layout': {
        'paper_bgcolor': '#1a1a2e',
        'plot_bgcolor': '#16213e',
        'font': {'color': '#e4e4e7', 'family': 'Inter, sans-serif'},
        'title': {'font': {'size': 20, 'color': '#f4f4f5'}},
        'xaxis': {'gridcolor': '#334155', 'zerolinecolor': '#475569'},
        'yaxis': {'gridcolor': '#334155', 'zerolinecolor': '#475569'},
    }
}


def load_data():
    """Load all required datasets."""
    print("Loading data...")

    data = {}

    # Model metrics
    metrics_path = RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        data['metrics'] = pd.read_csv(metrics_path)
        print(f"  Loaded metrics: {len(data['metrics'])} models")

    # Predictions with annotations
    annot_path = RESULTS_DIR / "annot_predictions.csv"
    if annot_path.exists():
        data['predictions'] = pd.read_csv(annot_path)
        print(f"  Loaded predictions: {len(data['predictions'])} rows")

    # Optimization results
    opt_path = RESULTS_DIR / "optimization_complete" / "results_intermediate.csv"
    if opt_path.exists():
        data['optimization'] = pd.read_csv(opt_path)
        print(f"  Loaded optimization: {len(data['optimization'])} trials")

    # Feature coefficients
    for model in ['elasticnet', 'ridge', 'lasso']:
        coef_path = RESULTS_DIR / f"coefficients_{model}.csv"
        if coef_path.exists():
            data[f'coef_{model}'] = pd.read_csv(coef_path)
            print(f"  Loaded {model} coefficients: {len(data[f'coef_{model}'])} features")

    return data


def create_radar_chart(metrics_df):
    """
    Chart 1: Model Performance Radar Chart
    Compare models across multiple metrics in a radar/spider chart.
    """
    print("Creating radar chart...")

    # Normalize metrics to 0-1 scale for comparison
    metrics_to_plot = ['mae', 'rmse', 'r2', 'correlation', 'overfitting_ratio']
    display_names = ['MAE (lower=better)', 'RMSE (lower=better)', 'R² (higher=better)',
                     'Correlation (higher=better)', 'Overfitting (lower=better)']

    # Exclude AltumAge for cleaner visualization
    df = metrics_df[metrics_df['model'] != 'AltumAge'].copy()

    # Normalize: for MAE, RMSE, overfitting - invert so higher is better
    normalized = pd.DataFrame()
    normalized['model'] = df['model']

    for metric in metrics_to_plot:
        values = df[metric].values
        if metric in ['mae', 'rmse', 'overfitting_ratio']:
            # Invert: lower is better
            normalized[metric] = 1 - (values - values.min()) / (values.max() - values.min() + 0.001)
        else:
            # Higher is better
            normalized[metric] = (values - values.min()) / (values.max() - values.min() + 0.001)

    fig = go.Figure()

    for _, row in normalized.iterrows():
        model = row['model']
        values = [row[m] for m in metrics_to_plot]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=display_names + [display_names[0]],
            fill='toself',
            name=model,
            line_color=MODEL_COLORS.get(model, '#888'),
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#334155'),
            angularaxis=dict(gridcolor='#334155'),
            bgcolor='#16213e'
        ),
        title={
            'text': '🎯 Model Performance Comparison',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        height=600,
        width=800
    )

    fig.write_html(OUTPUT_DIR / "01_radar_model_comparison.html")
    print("  Saved radar chart")
    return fig


def create_optimization_heatmap(opt_df):
    """
    Chart 2: Hyperparameter Optimization Heatmap
    Shows MAE across different strategies and models.
    """
    print("Creating optimization heatmap...")

    # Create pivot table: strategy+features vs model
    opt_df['config'] = opt_df['strategy'] + '_' + opt_df['n_features'].astype(str)

    pivot = opt_df.pivot_table(
        values='mae_test',
        index='config',
        columns='model_name',
        aggfunc='mean'
    )

    # Sort by average MAE
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',  # Red=bad, Green=good
        text=np.round(pivot.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='Model: %{x}<br>Config: %{y}<br>MAE: %{z:.2f} years<extra></extra>',
        colorbar=dict(title='MAE (years)', tickfont=dict(color='#e4e4e7'))
    ))

    fig.update_layout(
        title={
            'text': '🔥 Optimization Results Heatmap',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        xaxis_title='Model',
        yaxis_title='Feature Strategy',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        height=700,
        width=1000
    )

    fig.write_html(OUTPUT_DIR / "02_optimization_heatmap.html")
    print("  Saved optimization heatmap")
    return fig


def create_demographic_bias_analysis(pred_df):
    """
    Chart 3: Demographic Bias Analysis
    Shows prediction errors stratified by gender, ethnicity, and age groups.
    """
    print("Creating demographic bias analysis...")

    # Calculate prediction error
    pred_df = pred_df.copy()
    pred_df['error'] = pred_df['age_pred'] - pred_df['age']
    pred_df['abs_error'] = np.abs(pred_df['error'])

    # Age groups
    pred_df['age_group'] = pd.cut(pred_df['age'],
                                   bins=[0, 30, 45, 60, 75, 100],
                                   labels=['<30', '30-45', '45-60', '60-75', '>75'])

    # Filter to best model (ElasticNet)
    df_en = pred_df[pred_df['model'] == 'ElasticNet']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'MAE by Ethnicity',
            'MAE by Gender',
            'MAE by Age Group',
            'Age Acceleration by Ethnicity & Gender'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'box'}]]
    )

    # 1. MAE by Ethnicity
    eth_mae = df_en.groupby('ethnicity')['abs_error'].mean().sort_values()
    fig.add_trace(
        go.Bar(
            x=eth_mae.index,
            y=eth_mae.values,
            marker_color=[COLORS['primary'], COLORS['secondary'], COLORS['success'],
                         COLORS['warning'], COLORS['info']][:len(eth_mae)],
            text=np.round(eth_mae.values, 2),
            textposition='outside',
            hovertemplate='%{x}: %{y:.2f} years<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. MAE by Gender
    df_en['gender'] = df_en['female'].map({True: 'Female', False: 'Male'})
    gender_mae = df_en.groupby('gender')['abs_error'].mean()
    fig.add_trace(
        go.Bar(
            x=gender_mae.index,
            y=gender_mae.values,
            marker_color=[COLORS['secondary'], COLORS['primary']],
            text=np.round(gender_mae.values, 2),
            textposition='outside',
            hovertemplate='%{x}: %{y:.2f} years<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. MAE by Age Group
    age_mae = df_en.groupby('age_group')['abs_error'].mean()
    fig.add_trace(
        go.Bar(
            x=[str(x) for x in age_mae.index],
            y=age_mae.values,
            marker_color=px.colors.sequential.Viridis[:len(age_mae)],
            text=np.round(age_mae.values, 2),
            textposition='outside',
            hovertemplate='Age %{x}: %{y:.2f} years<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Age Acceleration Box Plot
    for i, eth in enumerate(df_en['ethnicity'].unique()):
        eth_data = df_en[df_en['ethnicity'] == eth]
        fig.add_trace(
            go.Box(
                y=eth_data['error'],
                name=eth,
                marker_color=list(COLORS.values())[i % len(COLORS)],
                boxmean=True
            ),
            row=2, col=2
        )

    fig.update_layout(
        title={
            'text': '📊 Demographic Bias Analysis (ElasticNet)',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        showlegend=False,
        height=800,
        width=1100
    )

    # Update axes
    for i in range(1, 5):
        fig.update_xaxes(gridcolor='#334155', row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor='#334155', title_text='MAE (years)' if i < 4 else 'Age Acceleration',
                        row=(i-1)//2+1, col=(i-1)%2+1)

    fig.write_html(OUTPUT_DIR / "03_demographic_bias.html")
    print("  Saved demographic bias analysis")
    return fig


def create_feature_importance_chart(coef_df, model_name='ElasticNet'):
    """
    Chart 4: Top CpG Features (Positive vs Negative Effects)
    Diverging bar chart showing top features.
    """
    print(f"Creating feature importance chart for {model_name}...")

    # Get top 20 positive and negative
    df = coef_df.head(40).copy()
    df['color'] = df['coef'].apply(lambda x: COLORS['success'] if x > 0 else COLORS['danger'])
    df['direction'] = df['coef'].apply(lambda x: 'Positive (aging)' if x > 0 else 'Negative (youth)')

    # Sort by coefficient value
    df = df.sort_values('coef')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['feature'],
        x=df['coef'],
        orientation='h',
        marker_color=df['color'],
        text=df['coef'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>'
    ))

    # Add center line
    fig.add_vline(x=0, line_width=2, line_color='#f4f4f5', line_dash='dash')

    fig.update_layout(
        title={
            'text': f'🧬 Top CpG Biomarkers ({model_name})',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        xaxis_title='Coefficient (Impact on Age Prediction)',
        yaxis_title='CpG Site',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif', size=10),
        height=900,
        width=800,
        xaxis=dict(gridcolor='#334155', zerolinecolor='#f4f4f5'),
        yaxis=dict(gridcolor='#334155'),
        annotations=[
            dict(x=0.3, y=1.02, xref='paper', yref='paper',
                 text='<b style="color:#10b981">→ Aging markers</b>',
                 showarrow=False, font=dict(size=12, color='#10b981')),
            dict(x=-0.1, y=1.02, xref='paper', yref='paper',
                 text='<b style="color:#ef4444">← Youth markers</b>',
                 showarrow=False, font=dict(size=12, color='#ef4444'))
        ]
    )

    fig.write_html(OUTPUT_DIR / f"04_feature_importance_{model_name.lower()}.html")
    print(f"  Saved feature importance chart")
    return fig


def create_age_stratified_scatter(pred_df):
    """
    Chart 5: Age-Stratified Accuracy Analysis
    Faceted scatter plots showing accuracy across age ranges.
    """
    print("Creating age-stratified scatter plot...")

    # Filter to ElasticNet and test set
    df = pred_df[(pred_df['model'] == 'ElasticNet') & (pred_df['split'] == 'test')].copy()

    # Create age groups
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 35, 55, 75, 100],
                             labels=['Young (18-35)', 'Middle (35-55)',
                                    'Senior (55-75)', 'Elderly (75+)'])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[str(g) for g in df['age_group'].unique()],
        shared_xaxes=False,
        shared_yaxes=False
    )

    colors = [COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger']]

    for i, (group, group_df) in enumerate(df.groupby('age_group')):
        row, col = i // 2 + 1, i % 2 + 1

        # Calculate group stats
        mae = np.abs(group_df['age_pred'] - group_df['age']).mean()
        r2 = np.corrcoef(group_df['age'], group_df['age_pred'])[0, 1] ** 2

        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=group_df['age'],
                y=group_df['age_pred'],
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name=str(group),
                hovertemplate='Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>'
            ),
            row=row, col=col
        )

        # Perfect prediction line
        min_age, max_age = group_df['age'].min(), group_df['age'].max()
        fig.add_trace(
            go.Scatter(
                x=[min_age, max_age],
                y=[min_age, max_age],
                mode='lines',
                line=dict(color='#f4f4f5', dash='dash', width=2),
                showlegend=False
            ),
            row=row, col=col
        )

        # Add annotation with stats (use subplot axis refs)
        xref = 'x' if i == 0 else f'x{i+1}'
        yref = 'y' if i == 0 else f'y{i+1}'
        fig.add_annotation(
            x=min_age + 0.05 * (max_age - min_age),
            y=max_age - 0.05 * (max_age - min_age),
            xref=xref, yref=yref,
            text=f'MAE: {mae:.2f}y | R²: {r2:.3f}',
            showarrow=False,
            font=dict(size=11, color='#f4f4f5'),
            bgcolor='rgba(0,0,0,0.5)',
            borderpad=4
        )

    fig.update_layout(
        title={
            'text': '🎂 Age-Stratified Prediction Accuracy',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        showlegend=False,
        height=700,
        width=900
    )

    for i in range(1, 5):
        fig.update_xaxes(title_text='Chronological Age', gridcolor='#334155', row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(title_text='Predicted Age', gridcolor='#334155', row=(i-1)//2+1, col=(i-1)%2+1)

    fig.write_html(OUTPUT_DIR / "05_age_stratified_scatter.html")
    print("  Saved age-stratified scatter plot")
    return fig


def create_overfitting_analysis(metrics_df):
    """
    Chart 6: Overfitting Profile Comparison
    Compares train vs test error with overfitting ratio.
    """
    print("Creating overfitting analysis...")

    df = metrics_df[metrics_df['model'] != 'AltumAge'].copy()
    df = df.sort_values('overfitting_ratio')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Train vs Test MAE', 'Overfitting Ratio'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Train vs Test MAE (grouped bar)
    fig.add_trace(
        go.Bar(
            name='Train MAE',
            x=df['model'],
            y=df['mae_train'],
            marker_color=COLORS['success'],
            text=df['mae_train'].round(2),
            textposition='outside'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            name='Test MAE',
            x=df['model'],
            y=df['mae'],
            marker_color=COLORS['danger'],
            text=df['mae'].round(2),
            textposition='outside'
        ),
        row=1, col=1
    )

    # Overfitting ratio
    colors = ['#10b981' if r < 2 else '#f59e0b' if r < 4 else '#ef4444'
              for r in df['overfitting_ratio']]

    fig.add_trace(
        go.Bar(
            x=df['model'],
            y=df['overfitting_ratio'],
            marker_color=colors,
            text=df['overfitting_ratio'].round(2),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )

    # Add threshold lines
    fig.add_hline(y=2, line_dash='dash', line_color='#f59e0b', row=1, col=2,
                  annotation_text='Acceptable', annotation_position='right')
    fig.add_hline(y=4, line_dash='dash', line_color='#ef4444', row=1, col=2,
                  annotation_text='High', annotation_position='right')

    fig.update_layout(
        title={
            'text': '⚠️ Overfitting Analysis',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        barmode='group',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.25),
        height=500,
        width=1000
    )

    fig.update_xaxes(gridcolor='#334155')
    fig.update_yaxes(gridcolor='#334155', title_text='MAE (years)', row=1, col=1)
    fig.update_yaxes(gridcolor='#334155', title_text='Ratio (Test/Train)', row=1, col=2)

    fig.write_html(OUTPUT_DIR / "06_overfitting_analysis.html")
    print("  Saved overfitting analysis")
    return fig


def create_prediction_bands(pred_df):
    """
    Chart 7: Prediction Confidence Bands
    Shows predictions with error bands across the age range.
    """
    print("Creating prediction confidence bands...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df = df.sort_values('age')
    df['error'] = df['age_pred'] - df['age']

    # Calculate rolling statistics
    window = 20
    df['rolling_mean'] = df['error'].rolling(window, center=True, min_periods=5).mean()
    df['rolling_std'] = df['error'].rolling(window, center=True, min_periods=5).std()

    fig = go.Figure()

    # Confidence band (±2 SD)
    fig.add_trace(go.Scatter(
        x=list(df['age']) + list(df['age'][::-1]),
        y=list(df['rolling_mean'] + 2*df['rolling_std']) + list((df['rolling_mean'] - 2*df['rolling_std'])[::-1]),
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% CI',
        hoverinfo='skip'
    ))

    # Confidence band (±1 SD)
    fig.add_trace(go.Scatter(
        x=list(df['age']) + list(df['age'][::-1]),
        y=list(df['rolling_mean'] + df['rolling_std']) + list((df['rolling_mean'] - df['rolling_std'])[::-1]),
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.4)',
        line=dict(color='rgba(0,0,0,0)'),
        name='68% CI',
        hoverinfo='skip'
    ))

    # Mean prediction error
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df['rolling_mean'],
        mode='lines',
        line=dict(color=COLORS['primary'], width=3),
        name='Mean Error'
    ))

    # Individual points
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df['error'],
        mode='markers',
        marker=dict(color=COLORS['secondary'], size=5, opacity=0.5),
        name='Individual Errors',
        hovertemplate='Age: %{x:.1f}<br>Error: %{y:.2f} years<extra></extra>'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color='#f4f4f5', line_width=2)

    fig.update_layout(
        title={
            'text': '📈 Prediction Error with Confidence Bands',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        xaxis_title='Chronological Age (years)',
        yaxis_title='Prediction Error (years)',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=500,
        width=900,
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155')
    )

    fig.write_html(OUTPUT_DIR / "07_prediction_bands.html")
    print("  Saved prediction confidence bands")
    return fig


def create_age_acceleration_demographics(pred_df):
    """
    Chart 8: Age Acceleration by Demographics
    Ridge plot / violin showing epigenetic age acceleration distribution.
    """
    print("Creating age acceleration demographics chart...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['age_acceleration'] = df['age_pred'] - df['age']
    df['gender'] = df['female'].map({True: 'Female', False: 'Male'})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('By Gender', 'By Ethnicity')
    )

    # By Gender
    for i, gender in enumerate(['Female', 'Male']):
        gender_data = df[df['gender'] == gender]['age_acceleration']
        fig.add_trace(
            go.Violin(
                y=gender_data,
                name=gender,
                box_visible=True,
                meanline_visible=True,
                fillcolor=COLORS['secondary'] if gender == 'Female' else COLORS['primary'],
                line_color='white',
                opacity=0.7
            ),
            row=1, col=1
        )

    # By Ethnicity
    ethnicities = df['ethnicity'].value_counts().index[:5]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'],
              COLORS['warning'], COLORS['purple']]

    for i, eth in enumerate(ethnicities):
        eth_data = df[df['ethnicity'] == eth]['age_acceleration']
        fig.add_trace(
            go.Violin(
                y=eth_data,
                name=eth,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                line_color='white',
                opacity=0.7
            ),
            row=1, col=2
        )

    fig.add_hline(y=0, line_dash='dash', line_color='#f4f4f5', line_width=1)

    fig.update_layout(
        title={
            'text': '⏰ Epigenetic Age Acceleration Distribution',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        showlegend=False,
        height=500,
        width=1000
    )

    fig.update_yaxes(title_text='Age Acceleration (years)', gridcolor='#334155')
    fig.update_xaxes(gridcolor='#334155')

    fig.write_html(OUTPUT_DIR / "08_age_acceleration_demographics.html")
    print("  Saved age acceleration demographics chart")
    return fig


def create_model_agreement(pred_df):
    """
    Chart 9: Model Agreement Analysis
    Shows how much different models agree/disagree on predictions.
    """
    print("Creating model agreement analysis...")

    # Pivot to get predictions per sample per model
    pivot = pred_df.pivot_table(
        values='age_pred',
        index='Sample_Name',
        columns='model',
        aggfunc='first'
    )

    # Calculate standard deviation across models for each sample
    pivot['std'] = pivot.std(axis=1)
    pivot['mean'] = pivot.mean(axis=1)

    # Get actual age
    age_map = pred_df.groupby('Sample_Name')['age'].first()
    pivot['actual_age'] = pivot.index.map(age_map)

    pivot = pivot.sort_values('actual_age')

    fig = go.Figure()

    # Error bars showing model disagreement
    fig.add_trace(go.Scatter(
        x=pivot['actual_age'],
        y=pivot['mean'],
        mode='markers',
        marker=dict(
            color=pivot['std'],
            colorscale='Viridis',
            size=8,
            colorbar=dict(title='Model Disagreement (SD)', tickfont=dict(color='#e4e4e7')),
            line=dict(width=1, color='white')
        ),
        error_y=dict(
            type='data',
            array=pivot['std'],
            visible=True,
            color='rgba(255,255,255,0.3)'
        ),
        hovertemplate='Age: %{x:.1f}<br>Mean Pred: %{y:.1f}<br>SD: %{marker.color:.2f}<extra></extra>'
    ))

    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[pivot['actual_age'].min(), pivot['actual_age'].max()],
        y=[pivot['actual_age'].min(), pivot['actual_age'].max()],
        mode='lines',
        line=dict(color='#ef4444', dash='dash', width=2),
        name='Perfect Agreement'
    ))

    fig.update_layout(
        title={
            'text': '🤝 Model Agreement Analysis',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        xaxis_title='Chronological Age (years)',
        yaxis_title='Mean Predicted Age (years)',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=600,
        width=900,
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155')
    )

    fig.write_html(OUTPUT_DIR / "09_model_agreement.html")
    print("  Saved model agreement analysis")
    return fig


def create_parallel_coordinates(opt_df):
    """
    Chart 10: Optimization Trajectory (Parallel Coordinates)
    Visualize hyperparameter optimization exploration.
    """
    print("Creating parallel coordinates chart...")

    df = opt_df.copy()

    # Create numeric encoding for categorical variables
    df['strategy_num'] = df['strategy'].map({'PCA': 0, 'TopN': 1})
    df['model_num'] = pd.factorize(df['model_name'])[0]

    # Normalize features for better visualization
    df['n_features_norm'] = (df['n_features'] - df['n_features'].min()) / (df['n_features'].max() - df['n_features'].min())

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df['mae_test'],
                colorscale='RdYlGn_r',
                showscale=True,
                cmin=df['mae_test'].min(),
                cmax=df['mae_test'].quantile(0.9),
                colorbar=dict(title='MAE', tickfont=dict(color='#e4e4e7'))
            ),
            dimensions=[
                dict(
                    range=[0, 1],
                    tickvals=[0, 1],
                    ticktext=['PCA', 'TopN'],
                    label='Strategy',
                    values=df['strategy_num']
                ),
                dict(
                    range=[df['n_features'].min(), df['n_features'].max()],
                    label='Features',
                    values=df['n_features']
                ),
                dict(
                    range=[0, df['model_num'].max()],
                    tickvals=list(range(df['model_num'].max() + 1)),
                    ticktext=df.groupby('model_num')['model_name'].first().tolist(),
                    label='Model',
                    values=df['model_num']
                ),
                dict(
                    range=[df['r2_test'].min(), df['r2_test'].max()],
                    label='R² Test',
                    values=df['r2_test']
                ),
                dict(
                    range=[df['mae_test'].min(), df['mae_test'].max()],
                    label='MAE Test',
                    values=df['mae_test']
                ),
                dict(
                    range=[0, min(df['overfitting_ratio'].max(), 20)],
                    label='Overfitting',
                    values=df['overfitting_ratio'].clip(upper=20)
                ),
            ]
        )
    )

    fig.update_layout(
        title={
            'text': '🔍 Hyperparameter Optimization Explorer',
            'x': 0.5,
            'font': {'size': 24, 'color': '#f4f4f5'}
        },
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif', size=11),
        height=600,
        width=1100
    )

    fig.write_html(OUTPUT_DIR / "10_parallel_coordinates.html")
    print("  Saved parallel coordinates chart")
    return fig


def create_summary_dashboard(data):
    """
    Bonus: Summary Dashboard
    A single page with key metrics and mini charts.
    """
    print("Creating summary dashboard...")

    metrics = data['metrics']
    best_model = metrics.loc[metrics['mae'].idxmin()]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            f"Best Model: {best_model['model']}",
            'Model Comparison (MAE)',
            'R² Scores',
            'Predictions vs Actual',
            'Error Distribution',
            'Feature Count vs MAE'
        ),
        specs=[
            [{'type': 'indicator'}, {'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    # 1. KPI Indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=best_model['mae'],
            number={'suffix': ' years', 'font': {'size': 40, 'color': '#10b981'}},
            title={'text': 'MAE', 'font': {'size': 16, 'color': '#e4e4e7'}},
            delta={'reference': 5, 'relative': False, 'valueformat': '.2f'}
        ),
        row=1, col=1
    )

    # 2. MAE Bar Chart
    df = metrics[metrics['model'] != 'AltumAge'].sort_values('mae')
    fig.add_trace(
        go.Bar(
            x=df['model'],
            y=df['mae'],
            marker_color=[MODEL_COLORS.get(m, '#888') for m in df['model']],
            text=df['mae'].round(2),
            textposition='outside'
        ),
        row=1, col=2
    )

    # 3. R² Bar Chart
    fig.add_trace(
        go.Bar(
            x=df['model'],
            y=df['r2'],
            marker_color=[MODEL_COLORS.get(m, '#888') for m in df['model']],
            text=df['r2'].round(3),
            textposition='outside'
        ),
        row=1, col=3
    )

    # 4. Scatter plot (if predictions available)
    if 'predictions' in data:
        pred = data['predictions']
        pred_en = pred[(pred['model'] == 'ElasticNet') & (pred['split'] == 'test')]
        fig.add_trace(
            go.Scatter(
                x=pred_en['age'],
                y=pred_en['age_pred'],
                mode='markers',
                marker=dict(color=COLORS['primary'], size=6, opacity=0.7),
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[pred_en['age'].min(), pred_en['age'].max()],
                y=[pred_en['age'].min(), pred_en['age'].max()],
                mode='lines',
                line=dict(color='#ef4444', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        # 5. Error histogram
        errors = pred_en['age_pred'] - pred_en['age']
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=20,
                marker_color=COLORS['secondary'],
                opacity=0.7
            ),
            row=2, col=2
        )

    # 6. Feature count vs MAE (if optimization available)
    if 'optimization' in data:
        opt = data['optimization']
        opt_linear = opt[opt['model_name'].isin(['Ridge', 'Lasso', 'ElasticNet'])]
        fig.add_trace(
            go.Scatter(
                x=opt_linear['n_features'],
                y=opt_linear['mae_test'],
                mode='markers',
                marker=dict(
                    color=[MODEL_COLORS.get(m, '#888') for m in opt_linear['model_name']],
                    size=8, opacity=0.7
                ),
                text=opt_linear['model_name'],
                hovertemplate='%{text}<br>Features: %{x}<br>MAE: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=3
        )

    fig.update_layout(
        title={
            'text': '📊 Age Prediction Summary Dashboard',
            'x': 0.5,
            'font': {'size': 28, 'color': '#f4f4f5'}
        },
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e4e4e7', family='Inter, sans-serif'),
        showlegend=False,
        height=800,
        width=1400
    )

    # Update all axes
    for i in range(1, 7):
        fig.update_xaxes(gridcolor='#334155')
        fig.update_yaxes(gridcolor='#334155')

    fig.write_html(OUTPUT_DIR / "00_summary_dashboard.html")
    print("  Saved summary dashboard")
    return fig


def main():
    """Generate all awesome charts."""
    print("=" * 60)
    print("🎨 Generating Awesome Charts for Age Prediction Project")
    print("=" * 60)

    # Load all data
    data = load_data()

    if not data:
        print("Error: No data found!")
        return

    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60 + "\n")

    # Generate all charts
    charts = []

    if 'metrics' in data:
        charts.append(create_radar_chart(data['metrics']))
        charts.append(create_overfitting_analysis(data['metrics']))

    if 'optimization' in data:
        charts.append(create_optimization_heatmap(data['optimization']))
        charts.append(create_parallel_coordinates(data['optimization']))

    if 'predictions' in data:
        charts.append(create_demographic_bias_analysis(data['predictions']))
        charts.append(create_age_stratified_scatter(data['predictions']))
        charts.append(create_prediction_bands(data['predictions']))
        charts.append(create_age_acceleration_demographics(data['predictions']))
        charts.append(create_model_agreement(data['predictions']))

    if 'coef_elasticnet' in data:
        charts.append(create_feature_importance_chart(data['coef_elasticnet'], 'ElasticNet'))

    # Summary dashboard
    charts.append(create_summary_dashboard(data))

    print("\n" + "=" * 60)
    print(f"✅ Generated {len(charts)} awesome charts!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")

    return charts


if __name__ == "__main__":
    main()
