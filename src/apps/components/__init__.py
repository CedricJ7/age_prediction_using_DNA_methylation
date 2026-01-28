"""
Reusable Dash components for the DNA methylation age prediction app.
"""

from .charts import (
    create_empty_figure,
    create_error_figure,
    create_mae_bar_chart,
    create_r2_bar_chart,
    create_scatter_all_models,
    create_scatter_single_model,
    create_delta_age_chart,
    create_publication_scatter,
    MODEL_COLORS,
    CHART_LAYOUT,
    PUBLICATION_LAYOUT,
)

from .kpis import (
    create_kpi_card,
    create_kpi_row,
    compute_cohort_metrics,
    format_kpi_value,
)

from .tables import (
    create_data_table,
    prepare_samples_dataframe,
    clean_ethnicity,
    SAMPLES_COLUMNS_MAP,
)

__all__ = [
    # Charts
    "create_empty_figure",
    "create_error_figure",
    "create_mae_bar_chart",
    "create_r2_bar_chart",
    "create_scatter_all_models",
    "create_scatter_single_model",
    "create_delta_age_chart",
    "create_publication_scatter",
    "MODEL_COLORS",
    "CHART_LAYOUT",
    "PUBLICATION_LAYOUT",
    # KPIs
    "create_kpi_card",
    "create_kpi_row",
    "compute_cohort_metrics",
    "format_kpi_value",
    # Tables
    "create_data_table",
    "prepare_samples_dataframe",
    "clean_ethnicity",
    "SAMPLES_COLUMNS_MAP",
]
