"""KPI card components for the dashboard."""

from dash import html


def create_kpi_card(label: str, value: str, icon: str = "📊") -> html.Div:
    """
    Create a KPI card component.

    Args:
        label: The metric label
        value: The metric value
        icon: Optional emoji icon

    Returns:
        Dash HTML component
    """
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(className="kpi-icon", children=icon),
            html.Div(className="kpi-content", children=[
                html.Div(className="kpi-label", children=label),
                html.Div(className="kpi-value", children=value),
            ]),
        ],
        role="article",
        **{"aria-label": f"{label}: {value}"}
    )


def create_kpi_grid(metrics: dict) -> html.Div:
    """
    Create a grid of KPI cards.

    Args:
        metrics: Dictionary with keys: correlation, mae, r2, delta_age

    Returns:
        Dash HTML component with KPI grid
    """
    cards = []

    if "correlation" in metrics:
        cards.append(create_kpi_card(
            "Corrélation",
            f"{metrics['correlation']:.3f}",
            "📈"
        ))

    if "mae" in metrics:
        cards.append(create_kpi_card(
            "MAE",
            f"{metrics['mae']:.2f} ans",
            "📏"
        ))

    if "r2" in metrics:
        cards.append(create_kpi_card(
            "R²",
            f"{metrics['r2']:.3f}",
            "🎯"
        ))

    if "delta_age" in metrics:
        cards.append(create_kpi_card(
            "Écart moyen",
            f"{metrics['delta_age']:.2f} ans",
            "⚖️"
        ))

    return html.Div(
        className="kpi-grid",
        children=cards,
        role="region",
        **{"aria-label": "Key performance indicators"}
    )
