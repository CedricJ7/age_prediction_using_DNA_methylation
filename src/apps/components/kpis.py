"""
KPI card components for the Dash application.
"""

from dash import html
import numpy as np
from scipy import stats

from ...utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_kpi_card(label: str, value: str, trend: str = None) -> html.Div:
    """
    Create a single KPI card component.
    
    Args:
        label: KPI label
        value: KPI value (formatted string)
        trend: Optional trend indicator
        
    Returns:
        Dash HTML Div component
    """
    children = [
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
    ]
    
    if trend:
        trend_class = "kpi-trend-up" if trend.startswith("+") else "kpi-trend-down"
        children.append(html.Div(trend, className=f"kpi-trend {trend_class}"))
    
    return html.Div(className="kpi-card", children=children)


def create_kpi_row(metrics: dict) -> html.Div:
    """
    Create a row of KPI cards.
    
    Args:
        metrics: Dictionary with metric names and values
        
    Returns:
        Dash HTML Div with KPI cards
    """
    kpi_configs = [
        ("Corrélation", "correlation", lambda x: f"{x:.3f}"),
        ("Écart moyen", "mean_diff", lambda x: f"{x:+.2f}"),
        ("MAE", "mae", lambda x: f"{x:.2f}"),
        ("R²", "r2", lambda x: f"{x:.3f}"),
    ]
    
    cards = []
    for label, key, formatter in kpi_configs:
        value = metrics.get(key, None)
        if value is not None:
            cards.append(create_kpi_card(label, formatter(value)))
        else:
            cards.append(create_kpi_card(label, "--"))
    
    return html.Div(className="kpi-row", children=cards)


def compute_cohort_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute cohort-level metrics for KPIs.
    
    Args:
        y_true: True age values
        y_pred: Predicted age values
        
    Returns:
        Dictionary with computed metrics
    """
    try:
        correlation, _ = stats.pearsonr(y_true, y_pred)
        mean_diff = np.mean(y_pred - y_true)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        
        return {
            "correlation": correlation,
            "mean_diff": mean_diff,
            "mae": mae,
            "r2": r2,
        }
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return {}


def format_kpi_value(value, metric_type: str) -> str:
    """
    Format a KPI value based on its type.
    
    Args:
        value: Raw metric value
        metric_type: Type of metric (correlation, mae, r2, etc.)
        
    Returns:
        Formatted string
    """
    if value is None:
        return "--"
    
    formatters = {
        "correlation": lambda x: f"{x:.3f}",
        "mean_diff": lambda x: f"{x:+.2f}",
        "mae": lambda x: f"{x:.2f}",
        "mad": lambda x: f"{x:.2f}",
        "r2": lambda x: f"{x:.3f}",
        "rmse": lambda x: f"{x:.2f}",
        "count": lambda x: f"{int(x):,}",
    }
    
    formatter = formatters.get(metric_type, lambda x: f"{x:.2f}")
    return formatter(value)
