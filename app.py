from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from scipy import stats
from sklearn.linear_model import LinearRegression

# Import des visualisations révolutionnaires
from revolutionary_viz import (
    create_revolutionary_dashboard,
    create_biological_clock_viz,
    create_age_acceleration_wave,
    create_dna_strand_viz,
    create_wtf_data_art,
)


RESULTS_DIR = Path("results")

# Enhanced model colors with better contrast
MODEL_COLORS = {
    "ElasticNet": "#00ffc8",
    "Lasso": "#22d3ee",
    "Ridge": "#60a5fa",
    "RandomForest": "#a78bfa",
    "XGBoost": "#f472b6",
    "AltumAge": "#fbbf24",
}


def load_results():
    metrics_path = RESULTS_DIR / "metrics.csv"
    preds_path = RESULTS_DIR / "predictions.csv"
    annot_path = RESULTS_DIR / "annot_predictions.csv"
    if not metrics_path.exists() or not preds_path.exists():
        return None, None, None
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(preds_path)
    annot = pd.read_csv(annot_path) if annot_path.exists() else None
    return metrics, preds, annot


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=['assets/style-minimal.css']
)
app.title = "DNAm Age Prediction"

metrics_data, preds_data, annot_data = load_results()

model_options = []
default_model = None
if metrics_data is not None:
    model_options = [{"label": m, "value": m} for m in metrics_data["model"].unique()]
    default_model = model_options[0]["value"]


app.layout = html.Div(
    className="app-shell",
    children=[
        dcc.Download(id="download-csv"),
        
        # Top Bar
        html.Header(
            className="topbar",
            children=[
                html.Div(className="topbar-content", children=[
                    html.Div(className="logo-section", children=[
                        html.Span("DNA Methylation Age Prediction", className="logo-text"),
                    ]),
                    html.Button("Export Report", id="btn-export", className="btn primary"),
                ]),
            ],
        ),
        
        # Content
        html.Div(
            className="content-shell",
            children=[
                # Sidebar
                html.Aside(
                    className="sidebar",
                    children=[
                        html.Div(
                            className="filter-card",
                            children=[
                                html.Div("Sélection du modèle", className="control-label"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="model-dropdown",
                                        options=model_options,
                                        value=default_model,
                                        clearable=False,
                                        disabled=metrics_data is None,
                                    ),
                                    role="listbox",
                                    **{"aria-label": "Sélectionner le modèle de prédiction"},
                                ),
                                html.Hr(className="sidebar-divider"),
                                html.Div(className="metrics-legend", children=[
                                    html.Div(className="legend-item", children=[
                                        html.Span("Corrélation", className="legend-label"),
                                        html.Span("Force de la relation linéaire (Pearson, -1 à 1)", className="legend-desc"),
                                    ]),
                                    html.Div(className="legend-item", children=[
                                        html.Span("Écart moyen", className="legend-label"),
                                        html.Span("Biais moyen du modèle (années)", className="legend-desc"),
                                    ]),
                                    html.Div(className="legend-item", children=[
                                        html.Span("MAE", className="legend-label"),
                                        html.Span("Erreur absolue moyenne (années)", className="legend-desc"),
                                    ]),
                                    html.Div(className="legend-item", children=[
                                        html.Span("R²", className="legend-label"),
                                        html.Span("Variance expliquée (0 à 1)", className="legend-desc"),
                                    ]),
                                ]),
                            ],
                        ),
                    ],
                ),
                
                # Main
                html.Main(
                    className="main",
                    children=[
                        # Hero
                        html.Div(
                            className="hero",
                            children=[
                                html.H1("Epigenetic Clock Benchmark", className="hero-title"),
                                html.P("Compare age prediction models based on DNA methylation data.", className="hero-subtitle"),
                            ],
                        ),
                        
                        # Tabs
                        dcc.Tabs(
                            id="tabs",
                            value="tab-compare",
                            className="tabs",
                            children=[
                                # Comparaison des modèles
                                dcc.Tab(
                                    label="Comparaison des modèles",
                                    value="tab-compare",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        # Métriques cohorte
                                        html.Div(className="section-title", children="Performance Metrics"),
                                        html.Div(className="kpi-row", children=[
                                            html.Div(className="kpi-card", children=[
                                                html.Div("Correlation", className="kpi-label"),
                                                html.Div(id="kpi-corr", className="kpi-value"),
                                            ]),
                                            html.Div(className="kpi-card", children=[
                                                html.Div("Mean Bias", className="kpi-label"),
                                                html.Div(id="kpi-mean-diff", className="kpi-value"),
                                            ]),
                                            html.Div(className="kpi-card", children=[
                                                html.Div("MAE", className="kpi-label"),
                                                html.Div(id="kpi-mae", className="kpi-value"),
                                            ]),
                                            html.Div(className="kpi-card", children=[
                                                html.Div("R²", className="kpi-label"),
                                                html.Div(id="kpi-r2", className="kpi-value"),
                                            ]),
                                        ]),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-mae"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Graphique MAE par modèle"}),
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-r2"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Graphique R² par modèle"}),
                                        ]),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-scatter-all"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Nuage de points tous modèles"}),
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-scatter-single"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Régression modèle sélectionné"}),
                                        ]),
                                        
                                        # Métriques individuelles
                                        html.Div(className="section-title", children="Métriques Individuelles"),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-delta-age"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Delta Age vs Âge chronologique"}),
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-age-accel"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Distribution Age Acceleration"}),
                                        ]),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-box"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Box plot des erreurs"}),
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-hist"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Histogramme Delta Age"}),
                                        ]),
                                        
                                        # Analyses stratifiées
                                        html.Div(className="section-title", children="Analyses Stratifiées"),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-nonlin"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Analyse non-linéarité"}),
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-gender"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Analyse par genre"}),
                                        ]),
                                        html.Div(className="grid grid-single", children=[
                                            html.Div(dcc.Loading(
                                                dcc.Graph(id="chart-batch"),
                                                type="circle", color="var(--primary)"
                                            ), className="card", role="img", **{"aria-label": "Variabilité par lot"}),
                                        ]),
                                    ],
                                ),
                                
                                # Échantillons
                                dcc.Tab(
                                    label="Échantillons",
                                    value="tab-samples",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(
                                            className="card table-card",
                                            children=[
                                                html.H3("Données des échantillons"),
                                                html.Div(className="table-controls", children=[
                                                    # Search input
                                                    html.Div(className="search-group", children=[
                                                        html.Label("Rechercher:", htmlFor="search-sample"),
                                                        dcc.Input(
                                                            id="search-sample",
                                                            type="text",
                                                            placeholder="ID échantillon...",
                                                            className="search-input",
                                                            debounce=True,
                                                        ),
                                                    ]),
                                                    # Age filter
                                                    html.Div(className="filter-group", children=[
                                                        html.Label("Tranche d'âge:", htmlFor="filter-age-range"),
                                                        dcc.Dropdown(
                                                            id="filter-age-range",
                                                            options=[
                                                                {"label": "Tous les âges", "value": "all"},
                                                                {"label": "< 30 ans", "value": "young"},
                                                                {"label": "30-60 ans", "value": "middle"},
                                                                {"label": "> 60 ans", "value": "old"},
                                                            ],
                                                            value="all",
                                                            clearable=False,
                                                            className="filter-dropdown",
                                                        ),
                                                    ]),
                                                    # Split filter
                                                    html.Div(className="filter-group", children=[
                                                        html.Label("Ensemble:"),
                                                        dcc.RadioItems(
                                                            id="split-filter",
                                                            options=[
                                                                {"label": "Tous", "value": "all"},
                                                                {"label": "Test", "value": "test"},
                                                                {"label": "Train", "value": "non_test"},
                                                            ],
                                                            value="all",
                                                            inline=True,
                                                            className="radio-filter",
                                                        ),
                                                    ]),
                                                ]),
                                                # Export button
                                                html.Div(className="export-controls", children=[
                                                    html.Button(
                                                        "Exporter CSV",
                                                        id="btn-export-csv",
                                                        className="btn secondary",
                                                    ),
                                                ]),
                                                dcc.Download(id="download-samples-csv"),
                                                html.Div(id="samples-count", className="samples-count"),
                                                dcc.Loading(
                                                    html.Div(id="samples-table-container"),
                                                    type="circle",
                                                    color="var(--primary)",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                
                                # Contexte
                                dcc.Tab(
                                    label="Contexte",
                                    value="tab-contexte",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(className="education-grid", children=[
                                            html.Div(className="card edu-card", children=[
                                                html.H3("🧬 Méthylation de l'ADN"),
                                                html.P("Modification épigénétique où un groupe méthyle (CH₃) est ajouté à la cytosine des sites CpG. Ce processus régule l'expression des gènes sans modifier la séquence d'ADN."),
                                                html.P("Ces modifications sont réversibles et influencées par l'environnement, l'alimentation, le stress et le vieillissement."),
                                            ]),
                                            html.Div(className="card edu-card", children=[
                                                html.H3("⏰ Pourquoi prédire l'âge ?"),
                                                html.P("L'horloge épigénétique mesure l'âge biologique vs l'âge chronologique. Un écart révèle l'accélération ou la décélération du vieillissement."),
                                                html.P("Applications : diagnostic précoce, traitements anti-âge, études longévité, médecine personnalisée."),
                                            ]),
                                            html.Div(className="card edu-card", children=[
                                                html.H3("📊 Accélération épigénétique"),
                                                html.P("EAA = Âge prédit - Âge chronologique. Une EAA positive indique un vieillissement accéléré, associé aux maladies."),
                                                html.P("Facteurs : tabac, obésité, stress. Protection : exercice, alimentation saine, sommeil."),
                                            ]),
                                        ]),
                                    ],
                                ),
                                
                                # Références
                                dcc.Tab(
                                    label="Références",
                                    value="tab-references",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(className="references-grid", children=[
                                            html.Div(className="card ref-card", children=[
                                                html.H4("Horvath (2013)"),
                                                html.P("DNA methylation age of human tissues"),
                                                html.A("Genome Biology", href="https://doi.org/10.1186/gb-2013-14-10-r115", target="_blank", className="ref-link"),
                                            ]),
                                            html.Div(className="card ref-card", children=[
                                                html.H4("Hannum (2013)"),
                                                html.P("Genome-wide methylation profiles"),
                                                html.A("Molecular Cell", href="https://doi.org/10.1016/j.molcel.2012.10.016", target="_blank", className="ref-link"),
                                            ]),
                                            html.Div(className="card ref-card", children=[
                                                html.H4("Levine (2018)"),
                                                html.P("PhenoAge biomarker"),
                                                html.A("Aging", href="https://doi.org/10.18632/aging.101414", target="_blank", className="ref-link"),
                                            ]),
                                            html.Div(className="card ref-card", children=[
                                                html.H4("Lu (2019)"),
                                                html.P("GrimAge predictor"),
                                                html.A("Aging", href="https://doi.org/10.18632/aging.101684", target="_blank", className="ref-link"),
                                            ]),
                                            html.Div(className="card ref-card", children=[
                                                html.H4("DeepMAge (2021)"),
                                                html.P("Deep learning clock"),
                                                html.A("Aging and Disease", href="https://www.aginganddisease.org/EN/10.14336/AD.2020.1202", target="_blank", className="ref-link"),
                                            ]),
                                        ]),
                                    ],
                                ),
                                
                                # Revolution
                                dcc.Tab(
                                    label="Revolution",
                                    value="tab-revolution",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(className="revolution-container", children=[
                                            # Sélecteur de visualisation
                                            html.Div(className="viz-selector", children=[
                                                html.Div(className="viz-selector-header", children=[
                                                    html.Span("🚀", className="viz-icon"),
                                                    html.Span("Visualisation", className="viz-title"),
                                                ]),
                                                dcc.RadioItems(
                                                    id="viz-type-selector",
                                                    options=[
                                                        {"label": "🎯 Dashboard Futuriste", "value": "dashboard"},
                                                        {"label": "⏱️ Horloge Biologique", "value": "clock"},
                                                        {"label": "🌊 Vagues d'Accélération", "value": "waves"},
                                                        {"label": "🧬 Double Hélice 3D", "value": "dna"},
                                                        {"label": "✨ Chronos Fragmenté", "value": "wtf"},
                                                    ],
                                                    value="dashboard",
                                                    className="viz-radio",
                                                ),
                                            ]),
                                            # Container du graphique
                                            html.Div(className="revolution-graph-container", children=[
                                                dcc.Graph(
                                                    id="revolution-graph",
                                                    config={
                                                        'displayModeBar': True,
                                                        'toImageButtonOptions': {
                                                            'format': 'png',
                                                            'filename': 'epigenetic_revolution',
                                                            'height': 800,
                                                            'width': 1400,
                                                            'scale': 2
                                                        },
                                                    },
                                                    style={'height': '75vh'},
                                                ),
                                            ]),
                                        ]),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# Enhanced chart layout with better styling
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.3)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#cbd5e1", size=13),
    title_font=dict(size=16, color="#f1f5f9", family="Inter"),
    margin=dict(l=60, r=40, t=60, b=50),
    xaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.1)",
        gridwidth=1,
        tickfont=dict(color="#94a3b8", size=12),
        showline=True,
        linewidth=1,
        linecolor="rgba(148, 163, 184, 0.2)",
        mirror=False
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.1)",
        gridwidth=1,
        tickfont=dict(color="#94a3b8", size=12),
        showline=True,
        linewidth=1,
        linecolor="rgba(148, 163, 184, 0.2)",
        mirror=False
    ),
    hoverlabel=dict(
        bgcolor="rgba(15, 23, 42, 0.95)",
        bordercolor="rgba(0, 255, 200, 0.5)",
        font=dict(color="#f1f5f9", size=13, family="Inter")
    ),
    hovermode='closest',
)


@app.callback(
    Output("kpi-corr", "children"),
    Output("kpi-mean-diff", "children"),
    Output("kpi-mae", "children"),
    Output("kpi-r2", "children"),
    Output("chart-mae", "figure"),
    Output("chart-r2", "figure"),
    Output("chart-scatter-all", "figure"),
    Output("chart-scatter-single", "figure"),
    Output("chart-delta-age", "figure"),
    Output("chart-age-accel", "figure"),
    Output("chart-box", "figure"),
    Output("chart-hist", "figure"),
    Output("chart-nonlin", "figure"),
    Output("chart-gender", "figure"),
    Output("chart-batch", "figure"),
    Input("model-dropdown", "value"),
)
def update_charts(model_name):
    empty = go.Figure().update_layout(**CHART_LAYOUT, annotations=[
        dict(text="Aucune donnée", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))
    ])
    
    if metrics_data is None or model_name is None:
        return "--", "--", "--", "--", empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty

    row = metrics_data[metrics_data["model"] == model_name].iloc[0]
    preds_model = preds_data[preds_data["model"] == model_name].copy()
    color = MODEL_COLORS.get(model_name, "#00d4aa")
    
    # === MÉTRIQUES COHORTE ===
    y_true = preds_model["y_true"].values
    y_pred = preds_model["y_pred"].values
    
    # Corrélation
    correlation, _ = stats.pearsonr(y_true, y_pred)
    
    # Écart moyen (biais)
    mean_diff = np.mean(y_pred - y_true)
    

    # === MÉTRIQUES INDIVIDUELLES ===
    # Delta Age = Âge prédit - Âge chronologique
    preds_model["delta_age"] = preds_model["y_pred"] - preds_model["y_true"]
    
    # Age Acceleration = résidus de la régression (âge prédit ~ âge chrono)
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred)
    y_pred_expected = lr.predict(y_true.reshape(-1, 1))
    preds_model["age_acceleration"] = y_pred - y_pred_expected

    # === GRAPHIQUES ===
    
    # MAE par modèle
    fig_mae = px.bar(metrics_data.sort_values("mae"), x="model", y="mae", title="MAE par modèle",
                     color="model", color_discrete_map=MODEL_COLORS)
    fig_mae.update_layout(**CHART_LAYOUT, showlegend=False)
    fig_mae.update_traces(marker_line_width=0)

    # R² par modèle
    fig_r2 = px.bar(metrics_data.sort_values("r2", ascending=False), x="model", y="r2", title="R² par modèle",
                    color="model", color_discrete_map=MODEL_COLORS)
    fig_r2.update_layout(**CHART_LAYOUT, showlegend=False)
    fig_r2.update_traces(marker_line_width=0)

    # Scatter ALL models
    fig_scatter_all = go.Figure()
    for m in metrics_data["model"].unique():
        preds_m = preds_data[preds_data["model"] == m]
        fig_scatter_all.add_trace(go.Scatter(
            x=preds_m["y_true"], y=preds_m["y_pred"],
            mode="markers", name=m,
            marker=dict(size=8, color=MODEL_COLORS.get(m, "#fff"), opacity=0.7),
        ))
    if len(preds_data) > 0:
        min_val, max_val = preds_data["y_true"].min(), preds_data["y_true"].max()
        fig_scatter_all.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Idéal",
            line=dict(dash="dash", color="rgba(255,255,255,0.4)", width=2),
        ))
    fig_scatter_all.update_layout(**CHART_LAYOUT, title="Tous les modèles",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02))

    # Scatter SINGLE model avec régression
    fig_scatter_single = px.scatter(preds_model, x="y_true", y="y_pred", title=f"Régression — {model_name}",
                                     trendline="ols")
    fig_scatter_single.update_traces(marker=dict(size=10, color=color, opacity=0.8))
    fig_scatter_single.update_layout(**CHART_LAYOUT)
    if len(fig_scatter_single.data) > 1:
        fig_scatter_single.data[1].line.color = "rgba(255,255,255,0.6)"

    # Delta Age vs Âge chronologique
    fig_delta = px.scatter(preds_model, x="y_true", y="delta_age", 
                           title=f"Delta Age (ΔAge = Prédit - Chrono) — {model_name}",
                           labels={"y_true": "Âge chronologique", "delta_age": "Delta Age (années)"})
    fig_delta.update_traces(marker=dict(size=10, color=color, opacity=0.8))
    fig_delta.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    # Ajouter ligne de tendance
    z = np.polyfit(y_true, preds_model["delta_age"].values, 1)
    p = np.poly1d(z)
    fig_delta.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], 
                                    y=[p(y_true.min()), p(y_true.max())],
                                    mode="lines", name="Tendance",
                                    line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dot")))
    fig_delta.update_layout(**CHART_LAYOUT, showlegend=False)

    # Age Acceleration distribution
    fig_accel = px.histogram(preds_model, x="age_acceleration", nbins=25,
                             title=f"Age Acceleration (résidus régression) — {model_name}",
                             labels={"age_acceleration": "Age Acceleration (années)"})
    fig_accel.update_traces(marker_color=color, opacity=0.8)
    fig_accel.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    # Ajouter statistiques
    mean_accel = preds_model["age_acceleration"].mean()
    std_accel = preds_model["age_acceleration"].std()
    fig_accel.add_annotation(x=0.98, y=0.95, xref="paper", yref="paper",
                             text=f"μ = {mean_accel:.2f}<br>σ = {std_accel:.2f}",
                             showarrow=False, font=dict(size=11, color="#94a3b8"),
                             bgcolor="rgba(0,0,0,0.5)", borderpad=6)
    fig_accel.update_layout(**CHART_LAYOUT)

    # Box plot erreurs
    preds_err = preds_data.copy()
    preds_err["error"] = preds_err["y_pred"] - preds_err["y_true"]
    fig_box = px.box(preds_err, x="model", y="error", title="Distribution des erreurs (tous modèles)",
                     color="model", color_discrete_map=MODEL_COLORS)
    fig_box.update_layout(**CHART_LAYOUT, showlegend=False)

    # Histogram erreurs modèle sélectionné
    fig_hist = px.histogram(preds_model, x="delta_age", nbins=25, 
                            title=f"Distribution Delta Age — {model_name}",
                            labels={"delta_age": "Delta Age (années)"})
    fig_hist.update_traces(marker_color=color, opacity=0.8)
    fig_hist.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    fig_hist.update_layout(**CHART_LAYOUT)

    # === ANALYSES STRATIFIÉES ===
    
    # Non-linéarité selon l'âge (erreur résiduelle vs âge avec LOWESS)
    fig_nonlin = go.Figure()
    
    # Scatter des erreurs vs âge
    fig_nonlin.add_trace(go.Scatter(
        x=y_true, y=preds_model["delta_age"].values,
        mode="markers", name="Échantillons",
        marker=dict(size=8, color=color, opacity=0.6),
    ))
    
    # Ajouter ligne de tendance polynomiale (degré 2) pour visualiser non-linéarité
    if len(y_true) > 10:
        z = np.polyfit(y_true, preds_model["delta_age"].values, 2)
        p = np.poly1d(z)
        x_line = np.linspace(y_true.min(), y_true.max(), 100)
        fig_nonlin.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode="lines", name="Tendance (poly²)",
            line=dict(color="#f0ad4e", width=3),
        ))
    
    fig_nonlin.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig_nonlin.update_layout(
        **CHART_LAYOUT,
        title=f"Non-linéarité: Erreur vs Âge — {model_name}",
        xaxis_title="Âge chronologique",
        yaxis_title="Delta Age (erreur)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    # Différence selon le genre (si annot_data disponible)
    fig_gender = go.Figure()
    
    if annot_data is not None:
        annot_model = annot_data[annot_data["model"] == model_name].copy()
        if "female" in annot_model.columns and len(annot_model) > 0:
            annot_model["delta_age"] = annot_model["age_pred"] - annot_model["age"]
            # Handle string "True"/"False" or boolean
            annot_model["Genre"] = annot_model["female"].apply(
                lambda x: "Femme" if str(x).lower() == "true" else ("Homme" if str(x).lower() == "false" else None)
            )
            # Filtrer les valeurs inconnues
            annot_gender = annot_model[annot_model["Genre"].notna()].copy()
            
            if len(annot_gender) > 0:
                fig_gender = px.box(
                    annot_gender, x="Genre", y="delta_age",
                    title=f"Erreur par Genre — {model_name}",
                    labels={"delta_age": "Delta Age (années)", "Genre": ""},
                    color="Genre",
                    color_discrete_map={"Femme": "#e879f9", "Homme": "#60a5fa"},
                )
                fig_gender.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_gender.update_layout(**CHART_LAYOUT, showlegend=False)
                
                # Ajouter statistiques
                for genre in ["Femme", "Homme"]:
                    subset = annot_gender[annot_gender["Genre"] == genre]["delta_age"]
                    if len(subset) > 0:
                        mean_g = subset.mean()
                        fig_gender.add_annotation(
                            x=genre, y=mean_g,
                            text=f"μ={mean_g:.2f}",
                            showarrow=False,
                            yshift=15,
                            font=dict(size=11, color="#94a3b8")
                        )
            else:
                fig_gender.update_layout(**CHART_LAYOUT, annotations=[
                    dict(text="Données genre non disponibles", x=0.5, y=0.5, 
                         xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))
                ])
        else:
            fig_gender.update_layout(**CHART_LAYOUT, annotations=[
                dict(text="Colonne genre non disponible", x=0.5, y=0.5, 
                     xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))
            ])
    else:
        fig_gender.update_layout(**CHART_LAYOUT, annotations=[
            dict(text="Données annotation non disponibles", x=0.5, y=0.5, 
                 xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))
        ])

    # Variabilité technique par lot/batch (chip ID)
    fig_batch = go.Figure()
    
    if annot_data is not None and "Sample_description" in annot_data.columns:
        annot_model = annot_data[annot_data["model"] == model_name].copy()
        if len(annot_model) > 0:
            annot_model["delta_age"] = annot_model["age_pred"] - annot_model["age"]
            # Extraire le chip ID (première partie avant "_R")
            annot_model["chip_id"] = annot_model["Sample_description"].str.split("_R").str[0]
            
            # Compter les échantillons par chip et filtrer ceux avec >= 3 échantillons
            chip_counts = annot_model["chip_id"].value_counts()
            valid_chips = chip_counts[chip_counts >= 3].index.tolist()
            
            if len(valid_chips) >= 2:
                annot_filtered = annot_model[annot_model["chip_id"].isin(valid_chips)]
                
                # Calculer statistiques par chip
                chip_stats = annot_filtered.groupby("chip_id")["delta_age"].agg(["mean", "std", "count"]).reset_index()
                chip_stats = chip_stats.sort_values("mean")
                
                fig_batch = px.box(
                    annot_filtered, x="chip_id", y="delta_age",
                    title=f"Variabilité par Lot (Chip) — {model_name}",
                    labels={"chip_id": "Chip ID", "delta_age": "Delta Age (années)"},
                )
                fig_batch.update_traces(marker_color=color)
                fig_batch.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_batch.update_layout(**CHART_LAYOUT)
                fig_batch.update_xaxes(tickangle=45)
                
                # Ajouter annotation avec variabilité inter-batch
                inter_batch_std = chip_stats["mean"].std()
                fig_batch.add_annotation(
                    x=0.98, y=0.95, xref="paper", yref="paper",
                    text=f"Var. inter-lot: σ={inter_batch_std:.2f}",
                    showarrow=False, font=dict(size=11, color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0.5)", borderpad=6
                )
            else:
                fig_batch.update_layout(**CHART_LAYOUT, annotations=[
                    dict(text="Pas assez de lots (chips) pour l'analyse", x=0.5, y=0.5,
                         xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))
                ])
        else:
            fig_batch.update_layout(**CHART_LAYOUT, annotations=[
                dict(text="Aucune donnée pour ce modèle", x=0.5, y=0.5,
                     xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))
            ])
    else:
        fig_batch.update_layout(**CHART_LAYOUT, annotations=[
            dict(text="Données de lot non disponibles", x=0.5, y=0.5,
                 xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))
        ])

    return (
        f"{correlation:.3f}",
        f"{mean_diff:+.2f}",
        f"{row['mae']:.2f}",
        f"{row['r2']:.3f}",
        fig_mae, fig_r2, fig_scatter_all, fig_scatter_single, fig_delta, fig_accel, fig_box, fig_hist,
        fig_nonlin, fig_gender, fig_batch
    )


def clean_ethnicity(eth):
    """Regroupe les catégories d'ethnicité rares en 'Inconnu'."""
    if pd.isna(eth):
        return "Inconnu"
    eth_str = str(eth).strip()
    if eth_str.lower() in ["unavailable", "declined", "other", ""]:
        return "Inconnu"
    return eth_str


@app.callback(
    Output("samples-table-container", "children"),
    Output("samples-count", "children"),
    Input("model-dropdown", "value"),
    Input("split-filter", "value"),
    Input("search-sample", "value"),
    Input("filter-age-range", "value"),
)
def update_samples_table(model_name, split_filter, search_term, age_range):
    if annot_data is None or model_name is None:
        return html.P("Aucune donnée disponible", className="no-data"), ""
    
    df = annot_data[annot_data["model"] == model_name].copy()
    
    # Filtrer par split
    if split_filter and split_filter != "all":
        df = df[df["split"] == split_filter]
    
    # Filtrer par recherche
    if search_term and len(search_term) > 0:
        search_lower = search_term.lower()
        if "Sample_description" in df.columns:
            df = df[df["Sample_description"].str.lower().str.contains(search_lower, na=False)]
    
    # Filtrer par tranche d'âge
    if age_range and age_range != "all" and "age" in df.columns:
        if age_range == "young":
            df = df[df["age"] < 30]
        elif age_range == "middle":
            df = df[(df["age"] >= 30) & (df["age"] <= 60)]
        elif age_range == "old":
            df = df[df["age"] > 60]
    
    total_count = len(df)
    
    # Add Delta Age
    if "age" in df.columns and "age_pred" in df.columns:
        df["delta_age"] = (df["age_pred"] - df["age"]).round(2)
    
    # Transform sex column (handles string "True"/"False" or boolean)
    if "female" in df.columns:
        df["sexe"] = df["female"].apply(
            lambda x: "Femme" if str(x).lower() == "true" else ("Homme" if str(x).lower() == "false" else "?")
        )
    
    # Clean ethnicity
    if "ethnicity" in df.columns:
        df["ethnicity"] = df["ethnicity"].apply(clean_ethnicity)
    
    # Select columns to display
    cols_map = {
        "Sample_description": "Échantillon",
        "Sample_Name": "Nom",
        "sexe": "Sexe", 
        "age": "Âge chrono",
        "age_pred": "Âge prédit",
        "delta_age": "Delta Age",
        "ethnicity": "Ethnicité",
        "split": "Ensemble",
    }
    
    # Si Sample_description n'est pas une colonne mais l'index
    if "Sample_description" not in df.columns and df.index.name == "Sample_description":
        df = df.reset_index()
    
    cols_to_show = [c for c in cols_map.keys() if c in df.columns]
    df_display = df[cols_to_show].copy()
    df_display.columns = [cols_map[c] for c in cols_to_show]
    
    # Round numeric
    for col in df_display.select_dtypes(include=[np.number]).columns:
        df_display[col] = df_display[col].round(2)
    
    # Trier par âge chronologique
    if "Âge chrono" in df_display.columns:
        df_display = df_display.sort_values("Âge chrono")
    
    count_text = f"{total_count} échantillons affichés"
    
    table = html.Div(
        className="table-wrapper",
        children=[
            html.Table(
                className="data-table",
                children=[
                    html.Thead(html.Tr([html.Th(col) for col in df_display.columns])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                str(row[col]) if pd.notna(row[col]) else "—",
                                className=("cell-positive" if col == "Delta Age" and pd.notna(row[col]) and row[col] > 0 
                                          else ("cell-negative" if col == "Delta Age" and pd.notna(row[col]) and row[col] < 0 else ""))
                            )
                            for col in df_display.columns
                        ])
                        for _, row in df_display.iterrows()
                    ]),
                ],
            ),
        ],
    )
    
    return table, count_text


@app.callback(
    Output("download-samples-csv", "data"),
    Input("btn-export-csv", "n_clicks"),
    Input("model-dropdown", "value"),
    Input("split-filter", "value"),
    Input("search-sample", "value"),
    Input("filter-age-range", "value"),
    prevent_initial_call=True,
)
def export_samples_csv(n_clicks, model_name, split_filter, search_term, age_range):
    """Export filtered samples to CSV."""
    from dash import ctx
    if ctx.triggered_id != "btn-export-csv" or not n_clicks:
        return None
    if annot_data is None or model_name is None:
        return None
    
    df = annot_data[annot_data["model"] == model_name].copy()
    
    # Apply same filters as table
    if split_filter and split_filter != "all":
        df = df[df["split"] == split_filter]
    
    if search_term and len(search_term) > 0:
        search_lower = search_term.lower()
        if "Sample_description" in df.columns:
            df = df[df["Sample_description"].str.lower().str.contains(search_lower, na=False)]
    
    if age_range and age_range != "all" and "age" in df.columns:
        if age_range == "young":
            df = df[df["age"] < 30]
        elif age_range == "middle":
            df = df[(df["age"] >= 30) & (df["age"] <= 60)]
        elif age_range == "old":
            df = df[df["age"] > 60]
    
    # Add Delta Age
    if "age" in df.columns and "age_pred" in df.columns:
        df["delta_age"] = (df["age_pred"] - df["age"]).round(2)
    
    # Select columns for export
    export_cols = ["Sample_description", "age", "age_pred", "delta_age", "split"]
    export_cols = [c for c in export_cols if c in df.columns]
    df_export = df[export_cols]
    
    return dcc.send_data_frame(df_export.to_csv, f"samples_{model_name}.csv", index=False)


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True,
)
def export_report(n_clicks):
    """Generate and download comprehensive PhD-level PDF report."""
    from dash import ctx
    if ctx.triggered_id != "btn-export" or not n_clicks:
        return None
    if metrics_data is None:
        return None

    try:
        # Import the comprehensive report generator
        from generate_comprehensive_report import generate_comprehensive_report

        # Generate the PDF report
        pdf_path = generate_comprehensive_report()

        # Return the PDF file for download
        return dcc.send_file(pdf_path)

    except Exception as e:
        print(f"Error generating comprehensive report: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# PUBLICATION QUALITY FIGURE
# =============================================================================

PUBLICATION_LAYOUT = dict(
    font=dict(family="Arial, Helvetica, sans-serif", size=14, color="#1a1a1a"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=70, r=40, t=50, b=70),
    xaxis=dict(
        showgrid=False,
        showline=True,
        linewidth=1.5,
        linecolor="#1a1a1a",
        tickfont=dict(size=12, color="#1a1a1a"),
        title_font=dict(size=14, color="#1a1a1a"),
        ticks="outside",
        tickwidth=1.5,
        ticklen=6,
        tickcolor="#1a1a1a",
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=False,
        showline=True,
        linewidth=1.5,
        linecolor="#1a1a1a",
        tickfont=dict(size=12, color="#1a1a1a"),
        title_font=dict(size=14, color="#1a1a1a"),
        ticks="outside",
        tickwidth=1.5,
        ticklen=6,
        tickcolor="#1a1a1a",
        zeroline=False,
    ),
    hoverlabel=dict(bgcolor="white", bordercolor="#1a1a1a", font=dict(color="#1a1a1a", size=11)),
)


@app.callback(
    Output("pub-scatter", "figure"),
    Output("pub-stats-content", "children"),
    Input("model-dropdown", "value"),
)
def update_publication_figure(model_name):
    """Génère une figure de qualité publication."""
    
    if metrics_data is None or model_name is None:
        empty = go.Figure()
        empty.update_layout(**PUBLICATION_LAYOUT)
        empty.add_annotation(text="Aucune donnée", x=0.5, y=0.5, xref="paper", yref="paper", 
                            showarrow=False, font=dict(size=16, color="#666"))
        return empty, html.P("Données non disponibles")
    
    preds_model = preds_data[preds_data["model"] == model_name].copy()
    y_true = preds_model["y_true"].values
    y_pred = preds_model["y_pred"].values
    
    # Statistiques
    mae = np.mean(np.abs(y_true - y_pred))
    mad = np.median(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    corr, p_val = stats.pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Régression
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred)
    x_line = np.linspace(y_true.min() - 2, y_true.max() + 2, 100)
    y_line = lr.predict(x_line.reshape(-1, 1))
    
    # Limites
    min_val = min(y_true.min(), y_pred.min()) - 5
    max_val = max(y_true.max(), y_pred.max()) + 5
    
    # Figure
    fig = go.Figure()
    
    # Scatter points - style publication
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(
            size=8,
            color="#0072B2",  # Bleu accessible daltoniens
            opacity=0.6,
            line=dict(width=0),
        ),
        name="Samples",
        hovertemplate="<b>Chronological:</b> %{x:.1f} y<br><b>Predicted:</b> %{y:.1f} y<extra></extra>",
    ))
    
    # Ligne identité (diagonale)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#999999", width=1.5, dash="dash"),
        name="Identity",
        hoverinfo="skip",
    ))
    
    # Ligne de régression
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        line=dict(color="#D55E00", width=2),  # Orange accessible
        name="Regression",
        hoverinfo="skip",
    ))
    
    # Layout publication
    fig.update_layout(
        **PUBLICATION_LAYOUT,
        xaxis_title="Chronological age (years)",
        yaxis_title="Predicted age (years)",
        xaxis_range=[min_val, max_val],
        yaxis_range=[min_val, max_val],
        width=600,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#1a1a1a",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    
    # Annotation statistiques
    stats_annotation = f"r = {corr:.3f}<br>MAE = {mae:.2f} years<br>R² = {r2:.3f}"
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=stats_annotation,
        showarrow=False,
        font=dict(family="Arial", size=12, color="#1a1a1a"),
        align="right",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#1a1a1a",
        borderwidth=1,
        borderpad=8,
    )
    
    # Annotation n
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text=f"n = {len(y_true)}",
        showarrow=False,
        font=dict(family="Arial", size=11, color="#666666"),
        align="left",
    )
    
    # Panneau statistiques
    stats_panel = html.Div(className="pub-stats-grid", children=[
        html.Div(className="pub-stat-item", children=[
            html.Span("Pearson r", className="pub-stat-label"),
            html.Span(f"{corr:.4f}", className="pub-stat-value"),
        ]),
        html.Div(className="pub-stat-item", children=[
            html.Span("MAE", className="pub-stat-label"),
            html.Span(f"{mae:.2f} years", className="pub-stat-value"),
        ]),
        html.Div(className="pub-stat-item", children=[
            html.Span("MAD", className="pub-stat-label"),
            html.Span(f"{mad:.2f} years", className="pub-stat-value"),
        ]),
        html.Div(className="pub-stat-item", children=[
            html.Span("RMSE", className="pub-stat-label"),
            html.Span(f"{rmse:.2f} years", className="pub-stat-value"),
        ]),
        html.Div(className="pub-stat-item", children=[
            html.Span("R²", className="pub-stat-label"),
            html.Span(f"{r2:.4f}", className="pub-stat-value"),
        ]),
        html.Div(className="pub-stat-item", children=[
            html.Span("p-value", className="pub-stat-label"),
            html.Span(f"< 0.001" if p_val < 0.001 else f"{p_val:.4f}", className="pub-stat-value"),
        ]),
    ])
    
    return fig, stats_panel


@app.callback(
    Output("revolution-graph", "figure"),
    Input("viz-type-selector", "value"),
)
def update_revolution_graph(viz_type):
    """Met à jour le graphique révolutionnaire selon la sélection."""
    try:
        if viz_type == "dashboard":
            return create_revolutionary_dashboard()
        elif viz_type == "clock":
            return create_biological_clock_viz()
        elif viz_type == "waves":
            return create_age_acceleration_wave()
        elif viz_type == "dna":
            return create_dna_strand_viz()
        elif viz_type == "wtf":
            return create_wtf_data_art()
        else:
            return create_revolutionary_dashboard()
    except Exception as e:
        # En cas d'erreur, retourner un graphique vide
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='#0a0a0f',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            annotations=[dict(
                text=f"Erreur: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color='#ff6b6b')
            )]
        )
        return fig


if __name__ == "__main__":
    app.run(debug=True)
