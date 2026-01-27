from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from scipy import stats
from sklearn.linear_model import LinearRegression


RESULTS_DIR = Path("results")

MODEL_COLORS = {
    "ElasticNet": "#00d4aa",
    "Lasso": "#00a896",
    "Ridge": "#2e86ab",
    "RandomForest": "#0096c7",
    "XGBoost": "#7b68ee",
    "AltumAge": "#f0ad4e",
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


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "DNAm Age Prediction Benchmark"

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
                html.Div(className="brand", children=[html.Span("DNAme"), html.Span("Clock")]),
                html.Button("Exporter Rapport (Meilleur Modèle)", id="btn-export", className="btn primary"),
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
                                dcc.Dropdown(
                                    id="model-dropdown",
                                    options=model_options,
                                    value=default_model,
                                    clearable=False,
                                    disabled=metrics_data is None,
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
                                html.H1("Horloge Épigénétique"),
                                html.P("Explorez les performances des modèles de prédiction d'âge basés sur la méthylation de l'ADN."),
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
                                        html.Div(className="section-title", children="Métriques Cohorte"),
                                        html.Div(className="kpi-row", children=[
                                            html.Div(className="kpi-card", children=[
                                                html.Div("Corrélation", className="kpi-label"),
                                                html.Div(id="kpi-corr", className="kpi-value"),
                                            ]),
                                            html.Div(className="kpi-card", children=[
                                                html.Div("Écart moyen", className="kpi-label"),
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
                                            html.Div(dcc.Graph(id="chart-mae"), className="card"),
                                            html.Div(dcc.Graph(id="chart-r2"), className="card"),
                                        ]),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Graph(id="chart-scatter-all"), className="card"),
                                            html.Div(dcc.Graph(id="chart-scatter-single"), className="card"),
                                        ]),
                                        
                                        # Métriques individuelles
                                        html.Div(className="section-title", children="Métriques Individuelles"),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Graph(id="chart-delta-age"), className="card"),
                                            html.Div(dcc.Graph(id="chart-age-accel"), className="card"),
                                        ]),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Graph(id="chart-box"), className="card"),
                                            html.Div(dcc.Graph(id="chart-hist"), className="card"),
                                        ]),
                                        
                                        # Analyses stratifiées
                                        html.Div(className="section-title", children="Analyses Stratifiées"),
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Graph(id="chart-nonlin"), className="card"),
                                            html.Div(dcc.Graph(id="chart-gender"), className="card"),
                                        ]),
                                        html.Div(className="grid grid-single", children=[
                                            html.Div(dcc.Graph(id="chart-batch"), className="card"),
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
                                                    html.Label("Filtrer par ensemble:"),
                                                    dcc.RadioItems(
                                                        id="split-filter",
                                                        options=[
                                                            {"label": "Tous", "value": "all"},
                                                            {"label": "Test uniquement", "value": "test"},
                                                            {"label": "Entraînement", "value": "non_test"},
                                                        ],
                                                        value="all",
                                                        inline=True,
                                                        className="radio-filter",
                                                    ),
                                                ]),
                                                html.Div(id="samples-count", className="samples-count"),
                                                html.Div(id="samples-table-container"),
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
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#e6edf3", size=12),
    title_font=dict(size=14, color="#e6edf3"),
    margin=dict(l=50, r=30, t=45, b=45),
    xaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    yaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font=dict(color="#e6edf3")),
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
)
def update_samples_table(model_name, split_filter):
    if annot_data is None or model_name is None:
        return html.P("Aucune donnée disponible", className="no-data"), ""
    
    df = annot_data[annot_data["model"] == model_name].copy()
    
    # Filtrer par split
    if split_filter and split_filter != "all":
        df = df[df["split"] == split_filter]
    
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
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True,
)
def export_report(n_clicks):
    from dash import ctx
    if ctx.triggered_id != "btn-export" or not n_clicks:
        return None
    if metrics_data is None:
        return None
    
    # Utilise automatiquement le meilleur modèle (MAE minimum)
    best_model_name = metrics_data.loc[metrics_data["mae"].idxmin(), "model"]
    model_name = best_model_name
    
    row = metrics_data[metrics_data["model"] == model_name].iloc[0]
    preds_model = preds_data[preds_data["model"] == model_name]
    
    y_true = preds_model["y_true"].values
    y_pred = preds_model["y_pred"].values
    correlation, _ = stats.pearsonr(y_true, y_pred)
    mean_diff = np.mean(y_pred - y_true)
    
    # Age Acceleration calculation
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred)
    y_expected = lr.predict(y_true.reshape(-1, 1))
    age_accel = y_pred - y_expected
    
    # Non-linearity (polynomial fit)
    delta_age = y_pred - y_true
    z = np.polyfit(y_true, delta_age, 2)
    
    # Gender stats if available
    gender_section = ""
    if annot_data is not None and "female" in annot_data.columns:
        annot_model = annot_data[annot_data["model"] == model_name].copy()
        annot_model["delta_age"] = annot_model["age_pred"] - annot_model["age"]
        annot_model["Genre"] = annot_model["female"].apply(
            lambda x: "Femme" if str(x).lower() == "true" else "Homme"
        )
        gender_stats = annot_model.groupby("Genre")["delta_age"].agg(["mean", "std", "count"])
        gender_section = r"""
\subsection{Analyse par Genre}

\begin{table}[htbp]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Genre} & \textbf{Delta Age moyen} & \textbf{Écart-type} & \textbf{N} \\
\hline
"""
        for genre, stats_row in gender_stats.iterrows():
            gender_section += f"{genre} & {stats_row['mean']:.2f} & {stats_row['std']:.2f} & {int(stats_row['count'])} \\\\\n"
        gender_section += r"""\hline
\end{tabular}
\caption{Statistiques du Delta Age par genre}
\end{table}
"""
    
    # Build LaTeX report
    report = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{xcolor}

\geometry{margin=2.5cm}
\definecolor{primary}{RGB}{0,150,136}
\hypersetup{colorlinks=true,linkcolor=primary,urlcolor=primary}

\title{\textbf{Rapport d'Analyse — Horloge Épigénétique}\\
\large Modèle: """ + model_name + r"""}
\author{DNAm Age Prediction Benchmark}
\date{""" + pd.Timestamp.now().strftime("%d %B %Y") + r"""}

\begin{document}
\maketitle

\section{Introduction}

La méthylation de l'ADN est une modification épigénétique consistant en l'ajout d'un groupe 
méthyle (CH\textsubscript{3}) sur les cytosines des dinucléotides CpG. Ces modifications 
évoluent avec l'âge de manière prévisible, permettant de construire des "horloges épigénétiques" 
capables de prédire l'âge biologique d'un individu.

Ce rapport présente les performances du modèle \textbf{""" + model_name + r"""} pour la prédiction 
de l'âge à partir des profils de méthylation.

\section{Métriques de Performance}

\subsection{Métriques au Niveau Cohorte}

\begin{table}[htbp]
\centering
\begin{tabular}{lr}
\hline
\textbf{Métrique} & \textbf{Valeur} \\
\hline
Corrélation (Pearson) & """ + f"{correlation:.4f}" + r""" \\
Écart moyen (biais) & """ + f"{mean_diff:+.2f}" + r""" années \\
MAE (Mean Absolute Error) & """ + f"{row['mae']:.2f}" + r""" années \\
MAD (Median Absolute Deviation) & """ + f"{row['mad']:.2f}" + r""" années \\
R² (Coefficient de détermination) & """ + f"{row['r2']:.4f}" + r""" \\
\hline
\end{tabular}
\caption{Métriques de performance du modèle """ + model_name + r"""}
\end{table}

\subsection{Définitions des Métriques}

\begin{itemize}
    \item \textbf{Corrélation} : Force de la relation linéaire entre âge prédit et âge réel (-1 à 1)
    \item \textbf{Écart moyen} : Biais systématique du modèle (surestimation si positif)
    \item \textbf{MAE} : Erreur absolue moyenne en années
    \item \textbf{MAD} : Médiane des erreurs absolues (robuste aux outliers)
    \item \textbf{R²} : Proportion de variance expliquée (0 à 1)
\end{itemize}

\section{Données d'Entraînement}

\begin{table}[htbp]
\centering
\begin{tabular}{lr}
\hline
\textbf{Paramètre} & \textbf{Valeur} \\
\hline
Échantillons d'entraînement & """ + f"{int(row['n_train'])}" + r""" \\
Échantillons de test & """ + f"{int(row['n_test'])}" + r""" \\
Nombre de features & """ + f"{int(row['n_features'])}" + r""" \\
\hline
\end{tabular}
\caption{Caractéristiques des données}
\end{table}

\section{Analyse Individuelle}

\subsection{Delta Age}

Le Delta Age ($\Delta$Age) représente la différence entre l'âge prédit et l'âge chronologique :
$$\Delta\text{Age} = \text{Âge}_{\text{prédit}} - \text{Âge}_{\text{chronologique}}$$

\begin{table}[htbp]
\centering
\begin{tabular}{lr}
\hline
\textbf{Statistique} & \textbf{Valeur} \\
\hline
Moyenne & """ + f"{np.mean(delta_age):.2f}" + r""" ans \\
Écart-type & """ + f"{np.std(delta_age):.2f}" + r""" ans \\
Minimum & """ + f"{np.min(delta_age):.2f}" + r""" ans \\
Maximum & """ + f"{np.max(delta_age):.2f}" + r""" ans \\
\hline
\end{tabular}
\caption{Distribution du Delta Age}
\end{table}

\subsection{Age Acceleration}

L'accélération de l'âge est le résidu de la régression âge prédit $\sim$ âge chronologique :
$$\text{AgeAccel} = \text{Âge}_{\text{prédit}} - (\alpha + \beta \times \text{Âge}_{\text{chronologique}})$$

\begin{itemize}
    \item Moyenne : """ + f"{np.mean(age_accel):.2f}" + r""" ans
    \item Écart-type : """ + f"{np.std(age_accel):.2f}" + r""" ans
\end{itemize}

\section{Analyses Stratifiées}

\subsection{Non-linéarité selon l'Âge}

Régression polynomiale de degré 2 du Delta Age sur l'âge chronologique :
$$\Delta\text{Age} = """ + f"{z[0]:.4f}" + r""" \times \text{Âge}^2 """ + f"{z[1]:+.4f}" + r""" \times \text{Âge} """ + f"{z[2]:+.2f}" + r"""$$

""" + gender_section + r"""

\section{Comparaison des Modèles}

\begin{table}[htbp]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Modèle} & \textbf{MAE (ans)} & \textbf{R²} \\
\hline
"""
    for _, m in metrics_data.sort_values("mae").iterrows():
        marker = r" $\star$" if m['model'] == model_name else ""
        report += f"{m['model']}{marker} & {m['mae']:.2f} & {m['r2']:.4f} \\\\\n"
    
    report += r"""\hline
\end{tabular}
\caption{Comparaison des performances ($\star$ = modèle analysé)}
\end{table}

\section{Références}

\begin{enumerate}
    \item Horvath, S. (2013). DNA methylation age of human tissues. \textit{Genome Biology}, 14(10), R115.
    \item Hannum, G., et al. (2013). Genome-wide methylation profiles. \textit{Molecular Cell}, 49(2), 359-367.
    \item Levine, M. E., et al. (2018). PhenoAge biomarker. \textit{Aging}, 10(4), 573-591.
\end{enumerate}

\end{document}
"""
    
    return dict(content=report, filename=f"rapport_{model_name}.tex")


if __name__ == "__main__":
    app.run(debug=True)
