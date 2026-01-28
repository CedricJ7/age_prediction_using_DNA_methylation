"""Layout components for the dashboard."""

from dash import dcc, html


def create_topbar() -> html.Div:
    """
    Create the top navigation bar.

    Returns:
        Dash HTML component
    """
    return html.Div(
        className="topbar",
        children=[
            html.Div(className="topbar-content", children=[
                html.Div(className="logo-section", children=[
                    html.Span("🧬", className="logo-icon"),
                    html.Span("DNAm Age Predictor", className="logo-text"),
                ]),
                html.Button(
                    "📥 Exporter LaTeX",
                    id="btn-export",
                    className="btn primary",
                    **{"aria-label": "Export report to LaTeX format"}
                ),
                dcc.Download(id="download-csv"),
            ]),
        ],
        role="banner"
    )


def create_hero_section() -> html.Div:
    """
    Create the hero section with title and description.

    Returns:
        Dash HTML component
    """
    return html.Div(
        className="hero",
        children=[
            html.H1(
                "Prédiction d'Âge par Méthylation de l'ADN",
                className="hero-title"
            ),
            html.P(
                "Tableau de bord interactif pour l'évaluation de modèles d'horloges épigénétiques",
                className="hero-subtitle"
            ),
        ],
        role="region",
        **{"aria-label": "Dashboard header"}
    )


def create_sidebar(model_options: list, default_model: str, metrics_data) -> html.Div:
    """
    Create the sidebar with model selector and metrics legend.

    Args:
        model_options: List of model options for dropdown
        default_model: Default selected model
        metrics_data: Metrics data for legend

    Returns:
        Dash HTML component
    """
    return html.Div(
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
                        html.Div(className="legend-title", children="📊 Métriques"),
                        html.Div(className="legend-items", children=[
                            _create_legend_item("📈", "Corrélation", "Force de la relation linéaire (0-1)"),
                            _create_legend_item("📏", "MAE", "Erreur absolue moyenne en années"),
                            _create_legend_item("🎯", "R²", "Coefficient de détermination (0-1)"),
                            _create_legend_item("⚖️", "Écart", "Biais systématique du modèle"),
                        ]),
                    ]),
                ],
            ),
        ],
        role="complementary",
        **{"aria-label": "Model selection and metrics legend"}
    )


def _create_legend_item(icon: str, label: str, description: str) -> html.Div:
    """Create a legend item."""
    return html.Div(
        className="legend-item",
        children=[
            html.Span(icon, className="legend-icon"),
            html.Div(className="legend-content", children=[
                html.Span(label, className="legend-label"),
                html.Span(description, className="legend-desc"),
            ]),
        ]
    )
