"""
Main callbacks for the Dash application with proper error handling.
"""

from typing import Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from ..components.charts import (
    create_empty_figure,
    create_error_figure,
    create_mae_bar_chart,
    create_r2_bar_chart,
    create_scatter_all_models,
    create_scatter_single_model,
    create_delta_age_chart,
    MODEL_COLORS,
    CHART_LAYOUT,
)
from ..components.kpis import compute_cohort_metrics, format_kpi_value
from ..components.tables import (
    create_data_table,
    prepare_samples_dataframe,
    SAMPLES_COLUMNS_MAP,
)
from ...utils.logging_config import setup_logger

logger = setup_logger(__name__)


def safe_callback(func):
    """
    Decorator for safe callback execution with error handling.
    
    Catches exceptions and returns appropriate error states.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Callback error in {func.__name__}: {e}")
            # Return error figures/states based on expected output count
            return None
    return wrapper


@safe_callback
def update_main_charts(
    model_name: str,
    metrics_data: pd.DataFrame,
    preds_data: pd.DataFrame,
    annot_data: pd.DataFrame = None,
) -> Tuple[Any, ...]:
    """
    Update all main dashboard charts.
    
    Args:
        model_name: Selected model name
        metrics_data: DataFrame with model metrics
        preds_data: DataFrame with predictions
        annot_data: Optional annotations DataFrame
        
    Returns:
        Tuple of (kpi_corr, kpi_mean_diff, kpi_mae, kpi_r2, 
                  fig_mae, fig_r2, fig_scatter_all, fig_scatter_single,
                  fig_delta, fig_accel, fig_box, fig_hist,
                  fig_nonlin, fig_gender, fig_batch)
    """
    empty = create_empty_figure("Aucune donnée")
    empty_kpis = ("--", "--", "--", "--")
    empty_figs = tuple([empty] * 11)
    
    # Validate inputs
    if metrics_data is None or model_name is None:
        logger.warning("No data or model selected")
        return empty_kpis + empty_figs
    
    if model_name not in metrics_data["model"].values:
        logger.warning(f"Model {model_name} not found in metrics")
        return empty_kpis + empty_figs
    
    # Get model data
    row = metrics_data[metrics_data["model"] == model_name].iloc[0]
    preds_model = preds_data[preds_data["model"] == model_name].copy()
    color = MODEL_COLORS.get(model_name, "#00d4aa")
    
    # Compute cohort metrics
    y_true = preds_model["y_true"].values
    y_pred = preds_model["y_pred"].values
    
    metrics = compute_cohort_metrics(y_true, y_pred)
    
    # Format KPIs
    kpi_corr = format_kpi_value(metrics.get("correlation"), "correlation")
    kpi_mean_diff = format_kpi_value(metrics.get("mean_diff"), "mean_diff")
    kpi_mae = format_kpi_value(row.get("mae"), "mae")
    kpi_r2 = format_kpi_value(row.get("r2"), "r2")
    
    # Create charts
    fig_mae = create_mae_bar_chart(metrics_data, model_name)
    fig_r2 = create_r2_bar_chart(metrics_data, model_name)
    fig_scatter_all = create_scatter_all_models(preds_data, metrics_data)
    fig_scatter_single = create_scatter_single_model(preds_model, model_name, color)
    fig_delta = create_delta_age_chart(preds_model, model_name, color)
    
    # Age acceleration histogram
    fig_accel = create_age_acceleration_chart(preds_model, model_name, color)
    
    # Error box plot
    fig_box = create_error_boxplot(preds_data)
    
    # Delta age histogram
    fig_hist = create_delta_histogram(preds_model, model_name, color)
    
    # Stratified analyses
    fig_nonlin = create_nonlinearity_chart(preds_model, model_name, color)
    fig_gender = create_gender_chart(annot_data, model_name)
    fig_batch = create_batch_chart(annot_data, model_name, color)
    
    return (
        kpi_corr, kpi_mean_diff, kpi_mae, kpi_r2,
        fig_mae, fig_r2, fig_scatter_all, fig_scatter_single,
        fig_delta, fig_accel, fig_box, fig_hist,
        fig_nonlin, fig_gender, fig_batch
    )


def create_age_acceleration_chart(preds_model: pd.DataFrame, model_name: str, color: str):
    """Create age acceleration histogram."""
    import plotly.express as px
    import plotly.graph_objects as go
    
    try:
        y_true = preds_model["y_true"].values
        y_pred = preds_model["y_pred"].values
        
        # Calculate age acceleration (residuals from regression)
        lr = LinearRegression()
        lr.fit(y_true.reshape(-1, 1), y_pred)
        y_expected = lr.predict(y_true.reshape(-1, 1))
        age_accel = y_pred - y_expected
        
        fig = px.histogram(
            x=age_accel,
            nbins=25,
            title=f"Age Acceleration — {model_name}",
            labels={"x": "Age Acceleration (années)"},
        )
        fig.update_traces(marker_color=color, opacity=0.8)
        fig.add_vline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        
        # Add stats annotation
        mean_accel = np.mean(age_accel)
        std_accel = np.std(age_accel)
        fig.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=f"μ = {mean_accel:.2f}<br>σ = {std_accel:.2f}",
            showarrow=False,
            font=dict(size=11, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=6,
        )
        fig.update_layout(**CHART_LAYOUT)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating age acceleration chart: {e}")
        return create_error_figure("Could not create chart")


def create_error_boxplot(preds_data: pd.DataFrame):
    """Create error distribution box plot for all models."""
    import plotly.express as px
    
    try:
        preds_err = preds_data.copy()
        preds_err["error"] = preds_err["y_pred"] - preds_err["y_true"]
        
        fig = px.box(
            preds_err,
            x="model",
            y="error",
            title="Distribution des erreurs",
            color="model",
            color_discrete_map=MODEL_COLORS,
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating box plot: {e}")
        return create_error_figure("Could not create chart")


def create_delta_histogram(preds_model: pd.DataFrame, model_name: str, color: str):
    """Create delta age histogram."""
    import plotly.express as px
    
    try:
        delta_age = preds_model["y_pred"] - preds_model["y_true"]
        
        fig = px.histogram(
            x=delta_age,
            nbins=25,
            title=f"Distribution Delta Age — {model_name}",
            labels={"x": "Delta Age (années)"},
        )
        fig.update_traces(marker_color=color, opacity=0.8)
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig.update_layout(**CHART_LAYOUT)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating histogram: {e}")
        return create_error_figure("Could not create chart")


def create_nonlinearity_chart(preds_model: pd.DataFrame, model_name: str, color: str):
    """Create nonlinearity analysis chart."""
    import plotly.graph_objects as go
    
    try:
        y_true = preds_model["y_true"].values
        delta_age = preds_model["y_pred"].values - y_true
        
        fig = go.Figure()
        
        # Scatter of errors
        fig.add_trace(go.Scatter(
            x=y_true, y=delta_age,
            mode="markers", name="Échantillons",
            marker=dict(size=8, color=color, opacity=0.6),
        ))
        
        # Polynomial trend
        if len(y_true) > 10:
            z = np.polyfit(y_true, delta_age, 2)
            p = np.poly1d(z)
            x_line = np.linspace(y_true.min(), y_true.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines", name="Tendance (poly²)",
                line=dict(color="#f0ad4e", width=3),
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            **CHART_LAYOUT,
            title=f"Non-linéarité — {model_name}",
            xaxis_title="Âge chronologique",
            yaxis_title="Delta Age",
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating nonlinearity chart: {e}")
        return create_error_figure("Could not create chart")


def create_gender_chart(annot_data: pd.DataFrame, model_name: str):
    """Create gender stratified analysis chart."""
    import plotly.express as px
    import plotly.graph_objects as go
    
    try:
        if annot_data is None or "female" not in annot_data.columns:
            return create_empty_figure("Données genre non disponibles")
        
        annot_model = annot_data[annot_data["model"] == model_name].copy()
        if len(annot_model) == 0:
            return create_empty_figure("Aucune donnée pour ce modèle")
        
        annot_model["delta_age"] = annot_model["age_pred"] - annot_model["age"]
        annot_model["Genre"] = annot_model["female"].apply(
            lambda x: "Femme" if str(x).lower() == "true" 
            else ("Homme" if str(x).lower() == "false" else None)
        )
        annot_gender = annot_model[annot_model["Genre"].notna()]
        
        if len(annot_gender) == 0:
            return create_empty_figure("Pas de données genre valides")
        
        fig = px.box(
            annot_gender, x="Genre", y="delta_age",
            title=f"Erreur par Genre — {model_name}",
            color="Genre",
            color_discrete_map={"Femme": "#e879f9", "Homme": "#60a5fa"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating gender chart: {e}")
        return create_error_figure("Could not create chart")


def create_batch_chart(annot_data: pd.DataFrame, model_name: str, color: str):
    """Create batch/chip variability chart."""
    import plotly.express as px
    
    try:
        if annot_data is None or "Sample_description" not in annot_data.columns:
            return create_empty_figure("Données de lot non disponibles")
        
        annot_model = annot_data[annot_data["model"] == model_name].copy()
        if len(annot_model) == 0:
            return create_empty_figure("Aucune donnée")
        
        annot_model["delta_age"] = annot_model["age_pred"] - annot_model["age"]
        annot_model["chip_id"] = annot_model["Sample_description"].str.split("_R").str[0]
        
        # Filter chips with enough samples
        chip_counts = annot_model["chip_id"].value_counts()
        valid_chips = chip_counts[chip_counts >= 3].index.tolist()
        
        if len(valid_chips) < 2:
            return create_empty_figure("Pas assez de lots pour l'analyse")
        
        annot_filtered = annot_model[annot_model["chip_id"].isin(valid_chips)]
        
        fig = px.box(
            annot_filtered, x="chip_id", y="delta_age",
            title=f"Variabilité par Lot — {model_name}",
        )
        fig.update_traces(marker_color=color)
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(**CHART_LAYOUT)
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating batch chart: {e}")
        return create_error_figure("Could not create chart")


@safe_callback
def update_samples_table(
    model_name: str,
    split_filter: str,
    annot_data: pd.DataFrame,
) -> Tuple[Any, str]:
    """
    Update samples table display.
    
    Args:
        model_name: Selected model name
        split_filter: Split filter value
        annot_data: Annotations DataFrame
        
    Returns:
        Tuple of (table_component, count_text)
    """
    from dash import html
    
    if annot_data is None or model_name is None:
        return html.P("Aucune donnée disponible", className="no-data"), ""
    
    df = prepare_samples_dataframe(annot_data, model_name, split_filter)
    
    if df.empty:
        return html.P("Aucune donnée disponible", className="no-data"), ""
    
    count_text = f"{len(df)} échantillons affichés"
    table = create_data_table(df, SAMPLES_COLUMNS_MAP)
    
    return table, count_text
