from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output


RESULTS_DIR = Path("results")


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


app = Dash(__name__)
app.title = "DNAm Age Prediction Benchmark"


def layout_from_data(metrics: pd.DataFrame, preds: pd.DataFrame):
    model_options = [{"label": m, "value": m} for m in metrics["model"].unique()]
    return html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="topbar",
                children=[
                    html.Div(
                        className="brand",
                        children=[
                            html.Img(src=app.get_asset_url("uca.webp"), className="logo"),
                            html.Span("DNAme"),
                            html.Span("Clock"),
                        ],
                    ),
                    html.Div(
                        className="topbar-actions",
                        children=[
                            html.Button("Exporter résultats", className="btn primary"),
                            html.Button("Paramètres", className="btn ghost"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="content-shell",
                children=[
                    html.Aside(
                        className="sidebar",
                        children=[
                            html.H3("Navigation"),
                            html.Ul(
                                className="menu",
                                children=[
                                    html.Li("Dashboard"),
                                    html.Li("Modèles"),
                                    html.Li("Interprétabilité"),
                                    html.Li("Rapports"),
                                ],
                            ),
                            html.Div(
                                className="filter-card",
                                children=[
                                    html.Div("Sélection modèle", className="control-label"),
                                    dcc.Dropdown(
                                        id="model-dropdown",
                                        options=model_options,
                                        value=model_options[0]["value"],
                                        clearable=False,
                                        className="dropdown",
                                    ),
                                    html.Div(id="metrics-summary", className="metrics-summary"),
                                ],
                            ),
                        ],
                    ),
                    html.Main(
                        className="main",
                        children=[
                            html.Div(
                                className="hero",
                                children=[
                                    html.H1("Horloge épigénétique — Benchmark des modèles"),
                                    html.P(
                                        "Interface interactive et épurée pour comparer MAE, MAD et R²."
                                    ),
                                ],
                            ),
                            dcc.Tabs(
                                id="page-tabs",
                                value="tab-dashboard",
                                className="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Dashboard",
                                        value="tab-dashboard",
                                        className="tab",
                                        selected_className="tab-selected",
                                        children=[
                                            html.Div(
                                                className="kpi-row",
                                                children=[
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[
                                                            html.Div("MAE", className="kpi-label"),
                                                            html.Div(id="kpi-mae", className="kpi-value"),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[
                                                            html.Div("MAD", className="kpi-label"),
                                                            html.Div(id="kpi-mad", className="kpi-value"),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[
                                                            html.Div("R²", className="kpi-label"),
                                                            html.Div(id="kpi-r2", className="kpi-value"),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[
                                                            html.Div("Test", className="kpi-label"),
                                                            html.Div(id="kpi-n", className="kpi-value"),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="grid",
                                                children=[
                                                    html.Div(dcc.Graph(id="metrics-mae"), className="card"),
                                                    html.Div(dcc.Graph(id="metrics-r2"), className="card"),
                                                    html.Div(dcc.Graph(id="metrics-mad"), className="card"),
                                                    html.Div(dcc.Graph(id="error-box"), className="card"),
                                                ],
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Prédictions",
                                        value="tab-pred",
                                        className="tab",
                                        selected_className="tab-selected",
                                        children=[
                                            html.Div(
                                                className="grid",
                                                children=[
                                                    html.Div(dcc.Graph(id="pred-scatter"), className="card"),
                                                    html.Div(dcc.Graph(id="error-hist"), className="card"),
                                                ],
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Comparaison",
                                        value="tab-compare",
                                        className="tab",
                                        selected_className="tab-selected",
                                        children=[
                                            html.Div(
                                                className="card table-card",
                                                children=[
                                                    html.H3("Tableau comparatif des performances"),
                                                    dash_table.DataTable(
                                                        id="metrics-table",
                                                        data=metrics.to_dict("records"),
                                                        columns=[{"name": c, "id": c} for c in metrics.columns],
                                                        sort_action="native",
                                                        page_size=10,
                                                        style_cell={
                                                            "backgroundColor": "rgba(0,0,0,0)",
                                                            "color": "#EDEDED",
                                                        },
                                                        style_header={
                                                            "backgroundColor": "rgba(255,255,255,0.08)",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Échantillons",
                                        value="tab-samples",
                                        className="tab",
                                        selected_className="tab-selected",
                                        children=[
                                            html.Div(
                                                className="card table-card",
                                                children=[
                                                    html.H3("Annot + prédictions (meilleur modèle)"),
                                                    dash_table.DataTable(
                                                        id="annot-table",
                                                        data=(annot_data.to_dict("records") if annot_data is not None else []),
                                                        columns=(
                                                            [{"name": c, "id": c} for c in annot_data.columns]
                                                            if annot_data is not None
                                                            else []
                                                        ),
                                                        sort_action="native",
                                                        page_size=12,
                                                        style_cell={
                                                            "backgroundColor": "rgba(0,0,0,0)",
                                                            "color": "#EDEDED",
                                                        },
                                                        style_header={
                                                            "backgroundColor": "rgba(255,255,255,0.08)",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                ],
                                            )
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


def layout_missing_results():
    empty_fig = go.Figure().update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="topbar",
                children=[
                    html.Div(
                        className="brand",
                        children=[
                            html.Img(src=app.get_asset_url("uca.webp"), className="logo"),
                            html.Span("DNAme"),
                            html.Span("Clock"),
                        ],
                    ),
                    html.Div(
                        className="topbar-actions",
                        children=[
                            html.Button("Exporter résultats", className="btn primary"),
                            html.Button("Paramètres", className="btn ghost"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="content-shell",
                children=[
                    html.Aside(
                        className="sidebar",
                        children=[
                            html.H3("Navigation"),
                            html.Ul(
                                className="menu",
                                children=[
                                    html.Li("Dashboard"),
                                    html.Li("Modèles"),
                                    html.Li("Interprétabilité"),
                                    html.Li("Rapports"),
                                ],
                            ),
                            html.Div(
                                className="filter-card",
                                children=[
                                    html.Div(
                                        id="status-message",
                                        children="Exécutez: python train_models.py",
                                    ),
                                    dcc.Dropdown(
                                        id="model-dropdown",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        disabled=True,
                                        className="dropdown",
                                    ),
                                    html.Div(id="metrics-summary", className="metrics-summary"),
                                ],
                            ),
                        ],
                    ),
                    html.Main(
                        className="main",
                        children=[
                            html.Div(
                                className="hero",
                                children=[
                                    html.H1("Horloge épigénétique — Benchmark des modèles"),
                                    html.P("Résultats manquants. Lancez l'entraînement pour générer les fichiers."),
                                ],
                            ),
                            dcc.Tabs(
                                id="page-tabs",
                                value="tab-dashboard",
                                className="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Dashboard",
                                        value="tab-dashboard",
                                        className="tab",
                                        selected_className="tab-selected",
                                        children=[
                                            html.Div(
                                                className="kpi-row",
                                                children=[
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[html.Div("MAE", className="kpi-label")],
                                                    ),
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[html.Div("MAD", className="kpi-label")],
                                                    ),
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[html.Div("R²", className="kpi-label")],
                                                    ),
                                                    html.Div(
                                                        className="kpi-card",
                                                        children=[html.Div("Test", className="kpi-label")],
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="grid",
                                                children=[
                                                    html.Div(dcc.Graph(id="metrics-mae", figure=empty_fig), className="card"),
                                                    html.Div(dcc.Graph(id="metrics-r2", figure=empty_fig), className="card"),
                                                    html.Div(dcc.Graph(id="metrics-mad", figure=empty_fig), className="card"),
                                                    html.Div(dcc.Graph(id="error-box", figure=empty_fig), className="card"),
                                                ],
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Prédictions",
                                        value="tab-pred",
                                        className="tab",
                                        selected_className="tab-selected",
                                        children=[
                                            html.Div(
                                                className="grid",
                                                children=[
                                                    html.Div(dcc.Graph(id="pred-scatter", figure=empty_fig), className="card"),
                                                    html.Div(dcc.Graph(id="error-hist", figure=empty_fig), className="card"),
                                                ],
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Échantillons",
                                        value="tab-samples",
                                        className="tab",
                                        selected_className="tab-selected",
                                        children=[
                                            html.Div(
                                                className="card table-card",
                                                children=[
                                                    html.H3("Annot + prédictions (meilleur modèle)"),
                                                    dash_table.DataTable(
                                                        id="annot-table",
                                                        data=[],
                                                        columns=[],
                                                        page_size=12,
                                                        style_cell={
                                                            "backgroundColor": "rgba(0,0,0,0)",
                                                            "color": "#EDEDED",
                                                        },
                                                        style_header={
                                                            "backgroundColor": "rgba(255,255,255,0.08)",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                ],
                                            )
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


metrics_data, preds_data, annot_data = load_results()

if metrics_data is None or preds_data is None:
    app.layout = layout_missing_results()
else:
    app.layout = layout_from_data(metrics_data, preds_data)


@app.callback(
    Output("metrics-summary", "children"),
    Output("kpi-mae", "children"),
    Output("kpi-mad", "children"),
    Output("kpi-r2", "children"),
    Output("kpi-n", "children"),
    Output("metrics-mae", "figure"),
    Output("metrics-r2", "figure"),
    Output("metrics-mad", "figure"),
    Output("error-box", "figure"),
    Output("pred-scatter", "figure"),
    Output("error-hist", "figure"),
    Input("model-dropdown", "value"),
)
def update_model_view(model_name):
    if metrics_data is None or preds_data is None or model_name is None:
        empty_fig = go.Figure().update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        return (
            "Aucun résultat chargé.",
            "--",
            "--",
            "--",
            "--",
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
        )

    metrics = metrics_data
    preds = preds_data[preds_data["model"] == model_name]
    row = metrics.loc[metrics["model"] == model_name].iloc[0]

    test_mae = row.get("mae", float("nan"))
    test_mad = row.get("mad", float("nan"))
    test_r2 = row.get("r2", float("nan"))

    if preds["y_true"].notna().any() and preds["y_pred"].notna().any():
        if pd.isna(test_mae):
            test_mae = (preds["y_true"] - preds["y_pred"]).abs().mean()
        if pd.isna(test_mad):
            test_mad = (preds["y_true"] - preds["y_pred"]).abs().median()
        if pd.isna(test_r2):
            y_true = preds["y_true"].to_numpy()
            y_pred = preds["y_pred"].to_numpy()
            denom = ((y_true - y_true.mean()) ** 2).sum()
            test_r2 = 1 - ((y_true - y_pred) ** 2).sum() / denom if denom != 0 else 0.0

    summary = f"Test MAE: {test_mae:.2f} | Test MAD: {test_mad:.2f} | Test R2: {test_r2:.3f}"
    kpi_mae = f"{test_mae:.2f}"
    kpi_mad = f"{test_mad:.2f}"
    kpi_r2 = f"{test_r2:.3f}"
    kpi_n = f"{int(row['n_train'])} / {int(row['n_test'])}"

    mae_fig = px.bar(
        metrics.sort_values("mae"),
        x="model",
        y="mae",
        title="MAE par modèle",
        labels={"model": "Modèle", "mae": "MAE"},
    )
    mae_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    r2_fig = px.bar(
        metrics.sort_values("r2", ascending=False),
        x="model",
        y="r2",
        title="R² par modèle",
        labels={"model": "Modèle", "r2": "R²"},
    )
    r2_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    if "mad" in metrics.columns:
        mad_fig = px.bar(
            metrics.sort_values("mad"),
            x="model",
            y="mad",
            title="MAD par modèle",
            labels={"model": "Modèle", "mad": "MAD"},
        )
    else:
        mad_fig = go.Figure()
    mad_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    error_box = preds_data.copy()
    error_box["error"] = error_box["y_pred"] - error_box["y_true"]
    error_box_fig = px.box(
        error_box,
        x="model",
        y="error",
        title="Distribution des erreurs par modèle",
        labels={"model": "Modèle", "error": "Erreur (prédit - réel)"},
    )
    error_box_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    scatter_fig = px.scatter(
        preds,
        x="y_true",
        y="y_pred",
        title=f"Prédictions vs Âge réel — {model_name}",
        labels={"y_true": "Âge réel", "y_pred": "Âge prédit"},
    )
    scatter_fig.add_shape(
        type="line",
        x0=preds["y_true"].min(),
        x1=preds["y_true"].max(),
        y0=preds["y_true"].min(),
        y1=preds["y_true"].max(),
        line={"dash": "dash", "color": "gray"},
    )
    scatter_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    error_hist = preds.copy()
    error_hist["error"] = error_hist["y_pred"] - error_hist["y_true"]
    hist_fig = px.histogram(
        error_hist,
        x="error",
        nbins=30,
        title=f"Distribution des erreurs — {model_name}",
        labels={"error": "Erreur (prédit - réel)"},
    )
    hist_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    return (
        summary,
        kpi_mae,
        kpi_mad,
        kpi_r2,
        kpi_n,
        mae_fig,
        r2_fig,
        mad_fig,
        error_box_fig,
        scatter_fig,
        hist_fig,
    )


if __name__ == "__main__":
    app.run(debug=True)
