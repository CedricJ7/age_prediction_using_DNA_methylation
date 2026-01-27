"""
Benchmark Platform — DNAm Age Prediction Challenge

Application permettant aux utilisateurs de soumettre leurs prédictions d'âge
basées sur la méthylation de l'ADN et de se comparer sur un leaderboard.
"""

from pathlib import Path
from datetime import datetime
import json
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io


# === CONFIGURATION ===
RESULTS_DIR = Path("results")
BENCHMARK_DIR = Path("benchmark_data")
BENCHMARK_DIR.mkdir(exist_ok=True)

GROUND_TRUTH_FILE = BENCHMARK_DIR / "ground_truth.csv"
SUBMISSIONS_FILE = BENCHMARK_DIR / "submissions.json"

MODEL_CATEGORIES = [
    "ElasticNet",
    "Random Forest",
    "XGBoost",
    "Neural Network",
    "Autre",
]


def init_ground_truth():
    """Initialise le fichier ground truth à partir des données existantes."""
    if GROUND_TRUTH_FILE.exists():
        return pd.read_csv(GROUND_TRUTH_FILE)
    
    # Charger depuis les prédictions existantes (set de test uniquement)
    annot_path = RESULTS_DIR / "annot_predictions.csv"
    if annot_path.exists():
        annot = pd.read_csv(annot_path)
        # Garder uniquement les échantillons de test (une seule fois)
        test_samples = annot[annot["split"] == "test"].drop_duplicates(subset=["Sample_Name"])
        ground_truth = test_samples[["Sample_Name", "age"]].copy()
        ground_truth.columns = ["sample_id", "true_age"]
        ground_truth.to_csv(GROUND_TRUTH_FILE, index=False)
        return ground_truth
    
    return pd.DataFrame(columns=["sample_id", "true_age"])


def load_submissions():
    """Charge les soumissions existantes."""
    if SUBMISSIONS_FILE.exists():
        with open(SUBMISSIONS_FILE, "r") as f:
            return json.load(f)
    return []


def save_submissions(submissions):
    """Sauvegarde les soumissions."""
    with open(SUBMISSIONS_FILE, "w") as f:
        json.dump(submissions, f, indent=2, default=str)


def calculate_metrics(y_true, y_pred):
    """Calcule les métriques de performance."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Median Absolute Deviation
    mad = np.median(np.abs(y_true - y_pred))
    
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "correlation": round(corr, 4),
        "mad": round(mad, 3),
    }


def parse_uploaded_csv(contents, filename):
    """Parse le fichier CSV uploadé."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        return None, f"Erreur de lecture du CSV: {str(e)}"
    
    # Vérifier les colonnes requises
    required_cols = {"sample_id", "predicted_age"}
    alt_cols = {"Sample_Name": "sample_id", "age_pred": "predicted_age", "prediction": "predicted_age"}
    
    # Renommer si nécessaire
    for old, new in alt_cols.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    if not required_cols.issubset(df.columns):
        return None, f"Colonnes requises: {required_cols}. Colonnes trouvées: {set(df.columns)}"
    
    return df[["sample_id", "predicted_age"]], None


# === INITIALISATION ===
ground_truth = init_ground_truth()
submissions = load_submissions()


# === APPLICATION DASH ===
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "DNAm Age Benchmark Challenge"


def create_leaderboard_table(submissions_data):
    """Crée le tableau du leaderboard."""
    if not submissions_data:
        return html.P("Aucune soumission pour le moment. Soyez le premier!", className="no-data")
    
    # Trier par MAE (meilleur en premier)
    sorted_subs = sorted(submissions_data, key=lambda x: x["metrics"]["mae"])
    
    rows = []
    for i, sub in enumerate(sorted_subs):
        rank = i + 1
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        
        rows.append(html.Tr([
            html.Td(medal, className="rank-cell"),
            html.Td(sub["username"], className="user-cell"),
            html.Td(sub["model_category"], className="model-cell"),
            html.Td(f"{sub['metrics']['mae']:.2f}", className="metric-cell mae-cell"),
            html.Td(f"{sub['metrics']['r2']:.3f}", className="metric-cell"),
            html.Td(f"{sub['metrics']['correlation']:.3f}", className="metric-cell"),
            html.Td(sub["n_samples"], className="metric-cell"),
            html.Td(sub["date"][:10], className="date-cell"),
        ]))
    
    return html.Table(
        className="leaderboard-table",
        children=[
            html.Thead(html.Tr([
                html.Th("Rang"),
                html.Th("Utilisateur"),
                html.Th("Type de Modèle"),
                html.Th("MAE ↓", className="sortable"),
                html.Th("R²"),
                html.Th("Corr."),
                html.Th("N"),
                html.Th("Date"),
            ])),
            html.Tbody(rows),
        ],
    )


def create_comparison_chart(submissions_data):
    """Crée le graphique de comparaison des modèles."""
    if not submissions_data:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, annotations=[
            dict(text="Soumettez vos prédictions pour voir les comparaisons", 
                 x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, 
                 font=dict(size=14, color="#64748b"))
        ])
        return fig
    
    df = pd.DataFrame([
        {
            "username": s["username"],
            "model_category": s["model_category"],
            "mae": s["metrics"]["mae"],
            "r2": s["metrics"]["r2"],
        }
        for s in submissions_data
    ])
    
    fig = px.scatter(
        df, x="mae", y="r2",
        color="model_category",
        hover_name="username",
        title="Comparaison des Soumissions (MAE vs R²)",
        labels={"mae": "MAE (années)", "r2": "R²", "model_category": "Type de modèle"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    
    fig.update_traces(marker=dict(size=14, line=dict(width=2, color='white')))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    
    return fig


def create_mae_by_category_chart(submissions_data):
    """Crée le graphique MAE par catégorie de modèle."""
    if not submissions_data:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT)
        return fig
    
    df = pd.DataFrame([
        {
            "username": s["username"],
            "model_category": s["model_category"],
            "mae": s["metrics"]["mae"],
        }
        for s in submissions_data
    ])
    
    fig = px.box(
        df, x="model_category", y="mae",
        title="Distribution MAE par Type de Modèle",
        labels={"mae": "MAE (années)", "model_category": ""},
        color="model_category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    
    fig.update_layout(**CHART_LAYOUT, showlegend=False)
    fig.update_xaxes(tickangle=20)
    
    return fig


CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#e6edf3", size=12),
    title_font=dict(size=14, color="#e6edf3"),
    margin=dict(l=50, r=30, t=50, b=60),
    xaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    yaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font=dict(color="#e6edf3")),
)


# === LAYOUT ===
app.layout = html.Div(
    className="app-shell",
    children=[
        # Stores
        dcc.Store(id="submissions-store", data=submissions),
        
        # Header
        html.Header(
            className="topbar",
            children=[
                html.Div(className="brand", children=[
                    html.Span("DNAm"), 
                    html.Span("Challenge")
                ]),
                html.Div(className="header-stats", children=[
                    html.Span(f"{len(ground_truth)} échantillons de test", className="stat-badge"),
                    html.Span(f"{len(submissions)} soumissions", className="stat-badge", id="submissions-count"),
                ]),
            ],
        ),
        
        # Content
        html.Div(
            className="content-shell benchmark-content",
            children=[
                # Sidebar - Soumission
                html.Aside(
                    className="sidebar",
                    children=[
                        html.Div(
                            className="filter-card submission-card",
                            children=[
                                html.H3("Soumettre vos prédictions", className="card-title"),
                                
                                # Username
                                html.Div(className="form-group", children=[
                                    html.Label("Nom d'utilisateur", className="control-label"),
                                    dcc.Input(
                                        id="input-username",
                                        type="text",
                                        placeholder="Votre nom ou pseudo",
                                        className="text-input",
                                    ),
                                ]),
                                
                                # Model category
                                html.Div(className="form-group", children=[
                                    html.Label("Type de modèle", className="control-label"),
                                    dcc.Dropdown(
                                        id="input-model-category",
                                        options=[{"label": c, "value": c} for c in MODEL_CATEGORIES],
                                        placeholder="Sélectionnez...",
                                        clearable=False,
                                    ),
                                ]),
                                
                                # Description
                                html.Div(className="form-group", children=[
                                    html.Label("Description de la méthode", className="control-label"),
                                    dcc.Textarea(
                                        id="input-description",
                                        placeholder="Décrivez brièvement votre approche (preprocessing, features, hyperparamètres...)",
                                        className="text-area",
                                    ),
                                ]),
                                
                                # File upload
                                html.Div(className="form-group", children=[
                                    html.Label("Fichier de prédictions (CSV)", className="control-label"),
                                    dcc.Upload(
                                        id="upload-predictions",
                                        children=html.Div([
                                            html.Span("📁 Glissez-déposez ou "),
                                            html.A("parcourir", className="upload-link"),
                                        ]),
                                        className="upload-zone",
                                        multiple=False,
                                    ),
                                    html.Div(id="upload-feedback", className="upload-feedback"),
                                ]),
                                
                                # Format info
                                html.Div(className="format-info", children=[
                                    html.Strong("Format requis:"),
                                    html.Code("sample_id,predicted_age", className="code-block"),
                                    html.P("Les sample_id doivent correspondre aux échantillons de test."),
                                ]),
                                
                                # Submit button
                                html.Button(
                                    "Soumettre",
                                    id="btn-submit",
                                    className="btn primary submit-btn",
                                    disabled=True,
                                ),
                                
                                html.Div(id="submission-result", className="submission-result"),
                            ],
                        ),
                        
                        # Règles
                        html.Div(
                            className="filter-card rules-card",
                            children=[
                                html.H4("Règles du Challenge"),
                                html.Ul([
                                    html.Li("Utilisez uniquement les données de méthylation fournies"),
                                    html.Li("Décrivez votre méthode de manière reproductible"),
                                    html.Li("Pas de data leakage (n'utilisez pas les âges du test set)"),
                                    html.Li("Une soumission par utilisateur et par méthode"),
                                ]),
                            ],
                        ),
                    ],
                ),
                
                # Main - Leaderboard
                html.Main(
                    className="main",
                    children=[
                        # Hero
                        html.Div(
                            className="hero",
                            children=[
                                html.H1("DNAm Age Prediction Challenge"),
                                html.P("Comparez vos modèles de prédiction d'âge épigénétique avec la communauté."),
                            ],
                        ),
                        
                        # KPIs
                        html.Div(className="kpi-row", children=[
                            html.Div(className="kpi-card", children=[
                                html.Div("Meilleur MAE", className="kpi-label"),
                                html.Div(id="kpi-best-mae", className="kpi-value"),
                            ]),
                            html.Div(className="kpi-card", children=[
                                html.Div("Soumissions", className="kpi-label"),
                                html.Div(id="kpi-total-subs", className="kpi-value"),
                            ]),
                            html.Div(className="kpi-card", children=[
                                html.Div("Participants", className="kpi-label"),
                                html.Div(id="kpi-participants", className="kpi-value"),
                            ]),
                            html.Div(className="kpi-card", children=[
                                html.Div("Échantillons Test", className="kpi-label"),
                                html.Div(f"{len(ground_truth)}", className="kpi-value"),
                            ]),
                        ]),
                        
                        # Tabs
                        dcc.Tabs(
                            id="tabs",
                            value="tab-leaderboard",
                            className="tabs",
                            children=[
                                # Leaderboard
                                dcc.Tab(
                                    label="Leaderboard",
                                    value="tab-leaderboard",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(className="card table-card", children=[
                                            html.H3("Classement"),
                                            html.Div(id="leaderboard-container"),
                                        ]),
                                    ],
                                ),
                                
                                # Comparaisons
                                dcc.Tab(
                                    label="Comparaisons",
                                    value="tab-compare",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(className="grid", children=[
                                            html.Div(dcc.Graph(id="chart-comparison"), className="card"),
                                            html.Div(dcc.Graph(id="chart-mae-category"), className="card"),
                                        ]),
                                    ],
                                ),
                                
                                # Détails des soumissions
                                dcc.Tab(
                                    label="Détails",
                                    value="tab-details",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(className="card", children=[
                                            html.H3("Détails des Soumissions"),
                                            html.Div(id="details-container"),
                                        ]),
                                    ],
                                ),
                                
                                # Comment participer
                                dcc.Tab(
                                    label="Comment Participer",
                                    value="tab-howto",
                                    className="tab",
                                    selected_className="tab-selected",
                                    children=[
                                        html.Div(className="education-grid howto-grid", children=[
                                            html.Div(className="card edu-card", children=[
                                                html.H3("1. Préparez vos données"),
                                                html.P("Utilisez les données de méthylation ADN fournies. Le fichier contient les valeurs beta de ~5000 CpG sites pour chaque échantillon."),
                                                html.P("Split train/test: 80/20. Vous avez accès aux âges du train set uniquement."),
                                            ]),
                                            html.Div(className="card edu-card", children=[
                                                html.H3("2. Entraînez votre modèle"),
                                                html.P("Choisissez votre approche: régression linéaire, forêts aléatoires, XGBoost, réseaux de neurones..."),
                                                html.P("Optimisez vos hyperparamètres par validation croisée sur le train set."),
                                            ]),
                                            html.Div(className="card edu-card", children=[
                                                html.H3("3. Soumettez vos prédictions"),
                                                html.P("Exportez un CSV avec les colonnes: sample_id, predicted_age"),
                                                html.P("Remplissez le formulaire avec une description reproductible de votre méthode."),
                                            ]),
                                        ]),
                                        
                                        html.Div(className="card code-example", children=[
                                            html.H4("Exemple de code Python avec GridSearchCV"),
                                            html.Pre("""
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# Charger les données
X_train = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv", index_col=0)["age"]
X_test = pd.read_csv("data/X_test.csv", index_col=0)

# Définir la grille d'hyperparamètres
param_grid = {
    "alpha": [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95]
}

# GridSearchCV pour optimiser les hyperparamètres
model = GridSearchCV(
    ElasticNet(max_iter=5000, random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
model.fit(X_train, y_train)

print(f"Meilleurs paramètres: {model.best_params_}")

# Prédire avec le meilleur modèle
predictions = model.predict(X_test)

# Exporter
output = pd.DataFrame({
    "sample_id": X_test.index,
    "predicted_age": predictions
})
output.to_csv("my_predictions.csv", index=False)
""", className="code-block-large"),
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


# === CALLBACKS ===

@app.callback(
    Output("upload-feedback", "children"),
    Output("btn-submit", "disabled"),
    Input("upload-predictions", "contents"),
    State("upload-predictions", "filename"),
    State("input-username", "value"),
    State("input-model-category", "value"),
)
def validate_upload(contents, filename, username, model_cat):
    """Valide le fichier uploadé."""
    if not contents:
        return "", True
    
    df, error = parse_uploaded_csv(contents, filename)
    
    if error:
        return html.Div(f"❌ {error}", className="error-msg"), True
    
    # Vérifier que les sample_id correspondent
    uploaded_ids = set(df["sample_id"].astype(str))
    expected_ids = set(ground_truth["sample_id"].astype(str))
    
    matched = uploaded_ids & expected_ids
    missing = expected_ids - uploaded_ids
    extra = uploaded_ids - expected_ids
    
    if len(matched) == 0:
        return html.Div("❌ Aucun sample_id ne correspond aux données de test!", className="error-msg"), True
    
    feedback = [
        html.Div(f"✅ {filename}", className="success-msg"),
        html.Div(f"📊 {len(matched)}/{len(expected_ids)} échantillons trouvés", className="info-msg"),
    ]
    
    if missing:
        feedback.append(html.Div(f"⚠️ {len(missing)} échantillons manquants", className="warning-msg"))
    
    # Vérifier les autres champs
    can_submit = bool(username and model_cat and len(matched) > 0)
    
    if not username:
        feedback.append(html.Div("⚠️ Entrez votre nom d'utilisateur", className="warning-msg"))
    if not model_cat:
        feedback.append(html.Div("⚠️ Sélectionnez le type de modèle", className="warning-msg"))
    
    return html.Div(feedback), not can_submit


@app.callback(
    Output("submissions-store", "data"),
    Output("submission-result", "children"),
    Output("upload-predictions", "contents"),
    Input("btn-submit", "n_clicks"),
    State("upload-predictions", "contents"),
    State("upload-predictions", "filename"),
    State("input-username", "value"),
    State("input-model-category", "value"),
    State("input-description", "value"),
    State("submissions-store", "data"),
    prevent_initial_call=True,
)
def submit_predictions(n_clicks, contents, filename, username, model_cat, description, current_subs):
    """Soumet les prédictions et calcule les métriques."""
    if not n_clicks or not contents:
        raise PreventUpdate
    
    df, error = parse_uploaded_csv(contents, filename)
    if error:
        return current_subs, html.Div(f"❌ {error}", className="error-msg"), contents
    
    # Merger avec ground truth
    merged = pd.merge(
        df.astype({"sample_id": str}),
        ground_truth.astype({"sample_id": str}),
        on="sample_id",
        how="inner"
    )
    
    if len(merged) == 0:
        return current_subs, html.Div("❌ Aucun échantillon correspondant!", className="error-msg"), contents
    
    # Calculer les métriques
    metrics = calculate_metrics(merged["true_age"].values, merged["predicted_age"].values)
    
    # Créer la soumission
    submission_id = hashlib.md5(f"{username}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    
    new_submission = {
        "id": submission_id,
        "username": username,
        "model_category": model_cat,
        "description": description or "Non fournie",
        "metrics": metrics,
        "n_samples": len(merged),
        "date": datetime.now().isoformat(),
    }
    
    # Ajouter à la liste
    updated_subs = current_subs + [new_submission]
    
    # Sauvegarder
    save_submissions(updated_subs)
    
    result = html.Div([
        html.Div("✅ Soumission réussie!", className="success-msg"),
        html.Div(f"MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.3f}", className="metrics-summary"),
    ])
    
    return updated_subs, result, None


@app.callback(
    Output("leaderboard-container", "children"),
    Output("chart-comparison", "figure"),
    Output("chart-mae-category", "figure"),
    Output("details-container", "children"),
    Output("kpi-best-mae", "children"),
    Output("kpi-total-subs", "children"),
    Output("kpi-participants", "children"),
    Output("submissions-count", "children"),
    Input("submissions-store", "data"),
)
def update_displays(subs_data):
    """Met à jour tous les affichages."""
    # Leaderboard
    leaderboard = create_leaderboard_table(subs_data)
    
    # Charts
    fig_comparison = create_comparison_chart(subs_data)
    fig_mae_cat = create_mae_by_category_chart(subs_data)
    
    # Détails
    if subs_data:
        details = []
        for sub in sorted(subs_data, key=lambda x: x["metrics"]["mae"]):
            details.append(html.Div(className="detail-card", children=[
                html.Div(className="detail-header", children=[
                    html.Strong(sub["username"]),
                    html.Span(f"MAE: {sub['metrics']['mae']:.2f}", className="detail-mae"),
                ]),
                html.Div(f"Modèle: {sub['model_category']}", className="detail-model"),
                html.Div(f"Description: {sub['description']}", className="detail-desc"),
                html.Div(f"Date: {sub['date'][:10]} | N={sub['n_samples']}", className="detail-meta"),
            ]))
        details_content = html.Div(details, className="details-list")
    else:
        details_content = html.P("Aucune soumission", className="no-data")
    
    # KPIs
    if subs_data:
        best_mae = min(s["metrics"]["mae"] for s in subs_data)
        total_subs = len(subs_data)
        participants = len(set(s["username"] for s in subs_data))
    else:
        best_mae = "--"
        total_subs = 0
        participants = 0
    
    best_mae_str = f"{best_mae:.2f}" if isinstance(best_mae, float) else best_mae
    subs_count_str = f"{total_subs} soumissions"
    
    return (
        leaderboard, 
        fig_comparison, 
        fig_mae_cat, 
        details_content,
        best_mae_str,
        str(total_subs),
        str(participants),
        subs_count_str,
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DNAm Age Prediction Challenge - Benchmark Platform")
    print("="*60)
    print(f"\n  Ground truth: {len(ground_truth)} échantillons de test")
    print(f"  Soumissions existantes: {len(submissions)}")
    print("\n  Démarrage du serveur sur http://localhost:8051")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8051)
