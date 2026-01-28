"""
Reusable chart components for the Dash application.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression

from ...utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Color palette for models
MODEL_COLORS = {
    "ElasticNet": "#00d4aa",
    "Lasso": "#00a896",
    "Ridge": "#2e86ab",
    "RandomForest": "#0096c7",
    "XGBoost": "#7b68ee",
    "AltumAge": "#f0ad4e",
}

# Base chart layout (dark theme)
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

# Publication-quality layout (white theme)
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
)


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """
    Create an empty figure with a centered message.
    
    Args:
        message: Message to display
        
    Returns:
        Empty Plotly figure with message
    """
    fig = go.Figure()
    fig.update_layout(
        **CHART_LAYOUT,
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="#64748b"),
            )
        ],
    )
    return fig


def create_error_figure(error_message: str) -> go.Figure:
    """
    Create a figure showing an error state.
    
    Args:
        error_message: Error message to display
        
    Returns:
        Plotly figure with error styling
    """
    fig = go.Figure()
    fig.update_layout(
        **CHART_LAYOUT,
        annotations=[
            dict(
                text=f"⚠️ {error_message}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="#ff6b6b"),
            )
        ],
    )
    return fig


def create_mae_bar_chart(metrics_df, highlight_model: str = None) -> go.Figure:
    """
    Create MAE comparison bar chart.
    
    Args:
        metrics_df: DataFrame with model metrics
        highlight_model: Model to highlight
        
    Returns:
        Plotly bar chart figure
    """
    try:
        sorted_df = metrics_df.sort_values("mae")
        
        fig = px.bar(
            sorted_df,
            x="model",
            y="mae",
            title="MAE par modèle",
            color="model",
            color_discrete_map=MODEL_COLORS,
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        fig.update_traces(marker_line_width=0)
        
        logger.debug("Created MAE bar chart")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating MAE chart: {e}")
        return create_error_figure("Could not create MAE chart")


def create_r2_bar_chart(metrics_df, highlight_model: str = None) -> go.Figure:
    """
    Create R² comparison bar chart.
    
    Args:
        metrics_df: DataFrame with model metrics
        highlight_model: Model to highlight
        
    Returns:
        Plotly bar chart figure
    """
    try:
        sorted_df = metrics_df.sort_values("r2", ascending=False)
        
        fig = px.bar(
            sorted_df,
            x="model",
            y="r2",
            title="R² par modèle",
            color="model",
            color_discrete_map=MODEL_COLORS,
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        fig.update_traces(marker_line_width=0)
        
        logger.debug("Created R² bar chart")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating R² chart: {e}")
        return create_error_figure("Could not create R² chart")


def create_scatter_all_models(preds_df, metrics_df) -> go.Figure:
    """
    Create scatter plot comparing all models.
    
    Args:
        preds_df: DataFrame with predictions
        metrics_df: DataFrame with model metrics
        
    Returns:
        Plotly scatter figure
    """
    try:
        fig = go.Figure()
        
        for model in metrics_df["model"].unique():
            preds_m = preds_df[preds_df["model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=preds_m["y_true"],
                    y=preds_m["y_pred"],
                    mode="markers",
                    name=model,
                    marker=dict(
                        size=8,
                        color=MODEL_COLORS.get(model, "#fff"),
                        opacity=0.7,
                    ),
                )
            )
        
        # Add identity line
        if len(preds_df) > 0:
            min_val, max_val = preds_df["y_true"].min(), preds_df["y_true"].max()
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Idéal",
                    line=dict(dash="dash", color="rgba(255,255,255,0.4)", width=2),
                )
            )
        
        fig.update_layout(
            **CHART_LAYOUT,
            title="Tous les modèles",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        
        logger.debug("Created all-models scatter plot")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating scatter plot: {e}")
        return create_error_figure("Could not create scatter plot")


def create_scatter_single_model(preds_model, model_name: str, color: str) -> go.Figure:
    """
    Create scatter plot for a single model with regression line.
    
    Args:
        preds_model: DataFrame with predictions for single model
        model_name: Name of the model
        color: Color for the markers
        
    Returns:
        Plotly scatter figure with regression
    """
    try:
        fig = px.scatter(
            preds_model,
            x="y_true",
            y="y_pred",
            title=f"Régression — {model_name}",
            trendline="ols",
        )
        fig.update_traces(marker=dict(size=10, color=color, opacity=0.8))
        fig.update_layout(**CHART_LAYOUT)
        
        # Style the trendline
        if len(fig.data) > 1:
            fig.data[1].line.color = "rgba(255,255,255,0.6)"
        
        logger.debug(f"Created scatter plot for {model_name}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating single model scatter: {e}")
        return create_error_figure(f"Could not create scatter for {model_name}")


def create_delta_age_chart(preds_model, model_name: str, color: str) -> go.Figure:
    """
    Create Delta Age vs Chronological Age chart.
    
    Args:
        preds_model: DataFrame with predictions
        model_name: Name of the model
        color: Color for markers
        
    Returns:
        Plotly scatter figure
    """
    try:
        y_true = preds_model["y_true"].values
        delta_age = preds_model["y_pred"].values - y_true
        
        fig = px.scatter(
            x=y_true,
            y=delta_age,
            title=f"Delta Age (ΔAge = Prédit - Chrono) — {model_name}",
            labels={"x": "Âge chronologique", "y": "Delta Age (années)"},
        )
        fig.update_traces(marker=dict(size=10, color=color, opacity=0.8))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        
        # Add trend line
        z = np.polyfit(y_true, delta_age, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=[y_true.min(), y_true.max()],
                y=[p(y_true.min()), p(y_true.max())],
                mode="lines",
                name="Tendance",
                line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dot"),
            )
        )
        
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating delta age chart: {e}")
        return create_error_figure("Could not create Delta Age chart")


def create_publication_scatter(y_true, y_pred, model_name: str) -> go.Figure:
    """
    Create publication-quality scatter plot.
    
    Args:
        y_true: True ages
        y_pred: Predicted ages
        model_name: Name of the model
        
    Returns:
        Publication-quality Plotly figure
    """
    try:
        # Statistics
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        corr, _ = stats.pearsonr(y_true, y_pred)
        
        # Regression line
        lr = LinearRegression()
        lr.fit(y_true.reshape(-1, 1), y_pred)
        x_line = np.linspace(y_true.min() - 2, y_true.max() + 2, 100)
        y_line = lr.predict(x_line.reshape(-1, 1))
        
        # Limits
        min_val = min(y_true.min(), y_pred.min()) - 5
        max_val = max(y_true.max(), y_pred.max()) + 5
        
        fig = go.Figure()
        
        # Data points
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                marker=dict(size=8, color="#0072B2", opacity=0.6),
                name="Samples",
            )
        )
        
        # Identity line
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#999999", width=1.5, dash="dash"),
                name="Identity",
            )
        )
        
        # Regression line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="#D55E00", width=2),
                name="Regression",
            )
        )
        
        fig.update_layout(
            **PUBLICATION_LAYOUT,
            xaxis_title="Chronological age (years)",
            yaxis_title="Predicted age (years)",
            xaxis_range=[min_val, max_val],
            yaxis_range=[min_val, max_val],
            width=600,
            height=600,
        )
        
        # Statistics annotation
        fig.add_annotation(
            x=0.98,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"r = {corr:.3f}<br>MAE = {mae:.2f} years<br>R² = {r2:.3f}",
            showarrow=False,
            font=dict(size=12),
            align="right",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#1a1a1a",
            borderwidth=1,
            borderpad=8,
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating publication scatter: {e}")
        return create_error_figure("Could not create publication figure")
