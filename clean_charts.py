#!/usr/bin/env python3
"""
Clean, Professional Charts for Age Prediction
==============================================
Publication-quality visualizations with elegant, minimal design.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "clean_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Professional color palette
COLORS = px.colors.qualitative.Set2
SEQUENTIAL = px.colors.sequential.Blues

# Clean theme
THEME = {
    'bg': '#ffffff',
    'paper': '#fafafa',
    'text': '#2d3436',
    'grid': '#e0e0e0',
    'accent': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
}


def load_data():
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


def chart_1_model_comparison(metrics_df):
    """Simple, clear model comparison."""
    print("1. Model comparison...")

    df = metrics_df[metrics_df['model'] != 'AltumAge'].sort_values('mae')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mean Absolute Error (years)', 'R² Score'),
        horizontal_spacing=0.12
    )

    # MAE bars
    fig.add_trace(go.Bar(
        x=df['model'], y=df['mae'],
        marker_color=COLORS[:len(df)],
        text=df['mae'].round(2),
        textposition='outside',
        textfont=dict(size=12),
        showlegend=False
    ), row=1, col=1)

    # R² bars
    fig.add_trace(go.Bar(
        x=df['model'], y=df['r2'],
        marker_color=COLORS[:len(df)],
        text=df['r2'].round(3),
        textposition='outside',
        textfont=dict(size=12),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text='Model Performance Comparison', x=0.5, font=dict(size=20)),
        height=450, width=900,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', size=12, color=THEME['text']),
        margin=dict(t=80, b=60)
    )
    fig.update_xaxes(tickangle=0, gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'])

    fig.write_html(OUTPUT_DIR / "01_model_comparison.html")
    return fig


def chart_2_predictions_scatter(pred_df):
    """Clean scatter plot of predictions vs actual."""
    print("2. Predictions scatter...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['error'] = np.abs(df['age_pred'] - df['age'])

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df['age'], y=df['age_pred'],
        mode='markers',
        marker=dict(
            size=7,
            color=df['error'],
            colorscale='RdYlBu_r',
            colorbar=dict(title='Error (y)', thickness=15),
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        hovertemplate='Age: %{x:.1f}<br>Predicted: %{y:.1f}<br>Error: %{marker.color:.1f}y<extra></extra>'
    ))

    # Perfect line
    min_age, max_age = df['age'].min(), df['age'].max()
    fig.add_trace(go.Scatter(
        x=[min_age, max_age], y=[min_age, max_age],
        mode='lines',
        line=dict(color=THEME['danger'], width=2, dash='dash'),
        name='Perfect prediction'
    ))

    # Stats annotation
    mae = df['error'].mean()
    r2 = np.corrcoef(df['age'], df['age_pred'])[0,1]**2
    fig.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=f'MAE = {mae:.2f} years<br>R² = {r2:.3f}',
        showarrow=False, font=dict(size=13),
        bgcolor='rgba(255,255,255,0.8)', bordercolor=THEME['grid'],
        align='left'
    )

    fig.update_layout(
        title=dict(text='Predicted vs Chronological Age (ElasticNet)', x=0.5, font=dict(size=18)),
        xaxis_title='Chronological Age (years)',
        yaxis_title='Predicted Age (years)',
        height=550, width=650,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text']),
        showlegend=False
    )
    fig.update_xaxes(gridcolor=THEME['grid'], zeroline=False)
    fig.update_yaxes(gridcolor=THEME['grid'], zeroline=False, scaleanchor='x')

    fig.write_html(OUTPUT_DIR / "02_predictions_scatter.html")
    return fig


def chart_3_error_by_age(pred_df):
    """Error distribution across age groups."""
    print("3. Error by age group...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['error'] = df['age_pred'] - df['age']
    df['abs_error'] = np.abs(df['error'])
    df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,75,100],
                              labels=['<30', '30-45', '45-60', '60-75', '75+'])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('MAE by Age Group', 'Error Distribution'),
                        horizontal_spacing=0.1)

    # Bar chart
    agg = df.groupby('age_group', observed=True)['abs_error'].agg(['mean', 'std']).reset_index()
    fig.add_trace(go.Bar(
        x=agg['age_group'].astype(str), y=agg['mean'],
        error_y=dict(type='data', array=agg['std'], visible=True),
        marker_color=SEQUENTIAL[4],
        showlegend=False
    ), row=1, col=1)

    # Box plot
    for i, group in enumerate(df['age_group'].cat.categories):
        group_data = df[df['age_group'] == group]['error']
        fig.add_trace(go.Box(
            y=group_data, name=str(group),
            marker_color=COLORS[i % len(COLORS)],
            boxmean='sd',
            showlegend=False
        ), row=1, col=2)

    fig.add_hline(y=0, line_dash='dash', line_color=THEME['danger'], row=1, col=2)

    fig.update_layout(
        title=dict(text='Prediction Accuracy Across Age Groups', x=0.5, font=dict(size=18)),
        height=450, width=900,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text'])
    )
    fig.update_xaxes(gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'], title_text='MAE (years)', row=1, col=1)
    fig.update_yaxes(gridcolor=THEME['grid'], title_text='Prediction Error (years)', row=1, col=2)

    fig.write_html(OUTPUT_DIR / "03_error_by_age.html")
    return fig


def chart_4_demographics(pred_df):
    """Clean demographic analysis."""
    print("4. Demographics analysis...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['abs_error'] = np.abs(df['age_pred'] - df['age'])
    df['gender'] = df['female'].map({True: 'Female', False: 'Male'})

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('MAE by Gender', 'MAE by Ethnicity'),
                        horizontal_spacing=0.12)

    # Gender
    gender_mae = df.groupby('gender')['abs_error'].mean().sort_values()
    fig.add_trace(go.Bar(
        x=gender_mae.index, y=gender_mae.values,
        marker_color=[COLORS[0], COLORS[1]],
        text=gender_mae.round(2),
        textposition='outside',
        showlegend=False
    ), row=1, col=1)

    # Ethnicity
    eth_mae = df.groupby('ethnicity')['abs_error'].mean().sort_values()
    fig.add_trace(go.Bar(
        x=eth_mae.index, y=eth_mae.values,
        marker_color=COLORS[:len(eth_mae)],
        text=eth_mae.round(2),
        textposition='outside',
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text='Prediction Accuracy by Demographics', x=0.5, font=dict(size=18)),
        height=400, width=900,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text'])
    )
    fig.update_xaxes(gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'], title_text='MAE (years)')

    fig.write_html(OUTPUT_DIR / "04_demographics.html")
    return fig


def chart_5_top_features(coef_df):
    """Top biomarkers horizontal bar chart."""
    print("5. Top features...")

    df = coef_df.head(25).copy()
    df = df.sort_values('coef')
    df['color'] = df['coef'].apply(lambda x: THEME['success'] if x > 0 else THEME['danger'])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['feature'], x=df['coef'],
        orientation='h',
        marker_color=df['color'],
        hovertemplate='%{y}<br>Coefficient: %{x:.4f}<extra></extra>'
    ))

    fig.add_vline(x=0, line_color=THEME['text'], line_width=1)

    fig.update_layout(
        title=dict(text='Top 25 CpG Biomarkers (ElasticNet)', x=0.5, font=dict(size=18)),
        xaxis_title='Coefficient',
        height=650, width=700,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', size=10, color=THEME['text']),
        annotations=[
            dict(x=0.15, y=1.02, xref='paper', yref='paper',
                 text='<b>→ Aging</b>', showarrow=False,
                 font=dict(color=THEME['success'], size=11)),
            dict(x=-0.05, y=1.02, xref='paper', yref='paper',
                 text='<b>← Youth</b>', showarrow=False,
                 font=dict(color=THEME['danger'], size=11))
        ]
    )
    fig.update_xaxes(gridcolor=THEME['grid'], zeroline=False)
    fig.update_yaxes(gridcolor=THEME['grid'])

    fig.write_html(OUTPUT_DIR / "05_top_features.html")
    return fig


def chart_6_optimization(opt_df):
    """Optimization results summary."""
    print("6. Optimization results...")

    # Best per model per strategy
    df = opt_df.copy()
    best = df.loc[df.groupby(['strategy', 'model_name'])['mae_test'].idxmin()]

    fig = px.scatter(
        best, x='n_features', y='mae_test',
        color='model_name', symbol='strategy',
        size='r2_test', size_max=15,
        hover_data=['strategy', 'n_features', 'mae_test', 'r2_test'],
        color_discrete_sequence=COLORS
    )

    fig.update_layout(
        title=dict(text='Hyperparameter Optimization Results', x=0.5, font=dict(size=18)),
        xaxis_title='Number of Features',
        yaxis_title='Test MAE (years)',
        legend_title='Model',
        height=500, width=850,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text'])
    )
    fig.update_xaxes(gridcolor=THEME['grid'], type='log')
    fig.update_yaxes(gridcolor=THEME['grid'])

    fig.write_html(OUTPUT_DIR / "06_optimization.html")
    return fig


def chart_7_residuals(pred_df):
    """Residual analysis."""
    print("7. Residuals analysis...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['residual'] = df['age_pred'] - df['age']
    df = df.sort_values('age')

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Residuals vs Age', 'Residual Distribution'))

    # Residuals vs age
    fig.add_trace(go.Scatter(
        x=df['age'], y=df['residual'],
        mode='markers',
        marker=dict(size=5, color=THEME['accent'], opacity=0.5),
        showlegend=False
    ), row=1, col=1)

    # Trend line
    z = np.polyfit(df['age'], df['residual'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=[df['age'].min(), df['age'].max()],
        y=[p(df['age'].min()), p(df['age'].max())],
        mode='lines',
        line=dict(color=THEME['danger'], width=2),
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash='dash', line_color=THEME['text'], row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(
        x=df['residual'],
        nbinsx=30,
        marker_color=THEME['accent'],
        opacity=0.7,
        showlegend=False
    ), row=1, col=2)

    fig.add_vline(x=0, line_dash='dash', line_color=THEME['danger'], row=1, col=2)

    # Stats
    mean_res = df['residual'].mean()
    std_res = df['residual'].std()
    fig.add_annotation(
        x=0.98, y=0.95, xref='x2 domain', yref='y2 domain',
        text=f'Mean: {mean_res:.2f}<br>SD: {std_res:.2f}',
        showarrow=False, font=dict(size=11),
        bgcolor='white', align='right'
    )

    fig.update_layout(
        title=dict(text='Residual Analysis', x=0.5, font=dict(size=18)),
        height=400, width=900,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text'])
    )
    fig.update_xaxes(gridcolor=THEME['grid'], title_text='Age (years)', row=1, col=1)
    fig.update_xaxes(gridcolor=THEME['grid'], title_text='Residual (years)', row=1, col=2)
    fig.update_yaxes(gridcolor=THEME['grid'], title_text='Residual (years)', row=1, col=1)
    fig.update_yaxes(gridcolor=THEME['grid'], title_text='Count', row=1, col=2)

    fig.write_html(OUTPUT_DIR / "07_residuals.html")
    return fig


def chart_8_bland_altman(pred_df):
    """Bland-Altman plot."""
    print("8. Bland-Altman plot...")

    df = pred_df[pred_df['model'] == 'ElasticNet'].copy()
    df['mean'] = (df['age'] + df['age_pred']) / 2
    df['diff'] = df['age_pred'] - df['age']

    mean_diff = df['diff'].mean()
    std_diff = df['diff'].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['mean'], y=df['diff'],
        mode='markers',
        marker=dict(size=6, color=THEME['accent'], opacity=0.6),
        hovertemplate='Mean: %{x:.1f}<br>Difference: %{y:.2f}<extra></extra>'
    ))

    # Mean line
    fig.add_hline(y=mean_diff, line_color=THEME['success'], line_width=2,
                  annotation_text=f'Mean: {mean_diff:.2f}', annotation_position='right')

    # ±1.96 SD
    fig.add_hline(y=mean_diff + 1.96*std_diff, line_dash='dash', line_color=THEME['warning'],
                  annotation_text=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}', annotation_position='right')
    fig.add_hline(y=mean_diff - 1.96*std_diff, line_dash='dash', line_color=THEME['warning'],
                  annotation_text=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}', annotation_position='right')

    fig.update_layout(
        title=dict(text='Bland-Altman Plot', x=0.5, font=dict(size=18)),
        xaxis_title='Mean of Actual and Predicted Age',
        yaxis_title='Difference (Predicted - Actual)',
        height=500, width=700,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text'])
    )
    fig.update_xaxes(gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'])

    fig.write_html(OUTPUT_DIR / "08_bland_altman.html")
    return fig


def chart_9_model_violin(pred_df):
    """Violin plot comparing all models."""
    print("9. Model violin plot...")

    df = pred_df[pred_df['model'] != 'AltumAge'].copy()
    df['error'] = df['age_pred'] - df['age']

    # Order by median error
    order = df.groupby('model')['error'].apply(lambda x: np.abs(x).median()).sort_values().index

    fig = go.Figure()

    for i, model in enumerate(order):
        model_data = df[df['model'] == model]['error']
        fig.add_trace(go.Violin(
            y=model_data,
            name=model,
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLORS[i % len(COLORS)],
            opacity=0.7,
            line_color=THEME['text']
        ))

    fig.add_hline(y=0, line_dash='dash', line_color=THEME['danger'])

    fig.update_layout(
        title=dict(text='Prediction Error Distribution by Model', x=0.5, font=dict(size=18)),
        yaxis_title='Prediction Error (years)',
        height=500, width=800,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text']),
        showlegend=False
    )
    fig.update_xaxes(gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'])

    fig.write_html(OUTPUT_DIR / "09_model_violin.html")
    return fig


def chart_10_summary(data):
    """Summary dashboard."""
    print("10. Summary dashboard...")

    metrics = data['metrics']
    pred = data['predictions']
    best = metrics.loc[metrics['mae'].idxmin()]
    pred_en = pred[pred['model'] == 'ElasticNet']

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'histogram'}]],
        subplot_titles=['', '', '', 'Predictions', 'Model Comparison', 'Error Distribution'],
        vertical_spacing=0.15, horizontal_spacing=0.08
    )

    # KPIs
    kpis = [
        (best['mae'], 'MAE', 'years', THEME['accent']),
        (best['r2'], 'R²', '', THEME['success']),
        (best['correlation'], 'Correlation', '', THEME['warning'])
    ]

    for col, (val, title, suffix, color) in enumerate(kpis, 1):
        fig.add_trace(go.Indicator(
            mode='number',
            value=val,
            number={'suffix': f' {suffix}' if suffix else '', 'font': {'size': 40, 'color': color}},
            title={'text': title, 'font': {'size': 16}}
        ), row=1, col=col)

    # Scatter
    fig.add_trace(go.Scatter(
        x=pred_en['age'], y=pred_en['age_pred'],
        mode='markers',
        marker=dict(size=4, color=THEME['accent'], opacity=0.5),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[pred_en['age'].min(), pred_en['age'].max()],
        y=[pred_en['age'].min(), pred_en['age'].max()],
        mode='lines', line=dict(color=THEME['danger'], dash='dash'),
        showlegend=False
    ), row=2, col=1)

    # Bar
    df_bar = metrics[metrics['model'] != 'AltumAge'].sort_values('mae')
    fig.add_trace(go.Bar(
        x=df_bar['model'], y=df_bar['mae'],
        marker_color=COLORS[:len(df_bar)],
        showlegend=False
    ), row=2, col=2)

    # Histogram
    errors = pred_en['age_pred'] - pred_en['age']
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=25,
        marker_color=THEME['accent'], opacity=0.7,
        showlegend=False
    ), row=2, col=3)

    fig.update_layout(
        title=dict(text=f'Age Prediction Summary ({best["model"]})', x=0.5, font=dict(size=22)),
        height=650, width=1100,
        paper_bgcolor=THEME['paper'],
        plot_bgcolor=THEME['bg'],
        font=dict(family='Arial', color=THEME['text'])
    )
    fig.update_xaxes(gridcolor=THEME['grid'])
    fig.update_yaxes(gridcolor=THEME['grid'])

    fig.write_html(OUTPUT_DIR / "00_summary.html")
    return fig


def main():
    print("=" * 50)
    print("  Clean Charts Generator")
    print("=" * 50)

    data = load_data()
    print(f"\nLoaded: {len(data['metrics'])} models, {len(data['predictions'])} predictions\n")

    chart_10_summary(data)
    chart_1_model_comparison(data['metrics'])
    chart_2_predictions_scatter(data['predictions'])
    chart_3_error_by_age(data['predictions'])
    chart_4_demographics(data['predictions'])
    chart_5_top_features(data['coefficients'])
    chart_6_optimization(data['optimization'])
    chart_7_residuals(data['predictions'])
    chart_8_bland_altman(data['predictions'])
    chart_9_model_violin(data['predictions'])

    print("\n" + "=" * 50)
    print(f"Done! Files saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
