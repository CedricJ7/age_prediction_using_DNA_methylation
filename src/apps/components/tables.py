"""
Table components for the Dash application.
"""

from dash import html
import pandas as pd
import numpy as np

from ...utils.logging_config import setup_logger

logger = setup_logger(__name__)


def clean_ethnicity(eth) -> str:
    """
    Clean and standardize ethnicity values.
    
    Args:
        eth: Raw ethnicity value
        
    Returns:
        Cleaned ethnicity string
    """
    if pd.isna(eth):
        return "Inconnu"
    eth_str = str(eth).strip()
    if eth_str.lower() in ["unavailable", "declined", "other", ""]:
        return "Inconnu"
    return eth_str


def create_data_table(
    df: pd.DataFrame,
    columns_map: dict = None,
    max_rows: int = 100,
) -> html.Div:
    """
    Create a styled data table component.
    
    Args:
        df: DataFrame to display
        columns_map: Dictionary mapping column names to display names
        max_rows: Maximum rows to display
        
    Returns:
        Dash HTML table component
    """
    try:
        if df is None or df.empty:
            return html.P("Aucune donnée disponible", className="no-data")
        
        # Apply column mapping
        if columns_map:
            cols_to_show = [c for c in columns_map.keys() if c in df.columns]
            df_display = df[cols_to_show].copy()
            df_display.columns = [columns_map[c] for c in cols_to_show]
        else:
            df_display = df.copy()
        
        # Round numeric columns
        for col in df_display.select_dtypes(include=[np.number]).columns:
            df_display[col] = df_display[col].round(2)
        
        # Limit rows
        if len(df_display) > max_rows:
            df_display = df_display.head(max_rows)
            logger.debug(f"Table truncated to {max_rows} rows")
        
        # Create table
        table = html.Table(
            className="data-table",
            role="table",
            children=[
                html.Thead(
                    html.Tr([
                        html.Th(col, scope="col") 
                        for col in df_display.columns
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            str(row[col]) if pd.notna(row[col]) else "—",
                            className=get_cell_class(col, row[col]),
                        )
                        for col in df_display.columns
                    ])
                    for _, row in df_display.iterrows()
                ]),
            ],
        )
        
        return html.Div(className="table-wrapper", children=[table])
        
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        return html.P(f"Erreur: {str(e)}", className="error-message")


def get_cell_class(column: str, value) -> str:
    """
    Determine CSS class for a table cell based on column and value.
    
    Args:
        column: Column name
        value: Cell value
        
    Returns:
        CSS class name
    """
    # Delta Age coloring
    if column in ["Delta Age", "delta_age", "Δ Age"]:
        if pd.notna(value):
            try:
                val = float(value)
                if val > 0:
                    return "cell-positive"
                elif val < 0:
                    return "cell-negative"
            except (ValueError, TypeError):
                pass
    return ""


def prepare_samples_dataframe(
    annot_df: pd.DataFrame,
    model_name: str,
    split_filter: str = "all",
) -> pd.DataFrame:
    """
    Prepare samples DataFrame for display.
    
    Args:
        annot_df: Annotations DataFrame
        model_name: Selected model name
        split_filter: Split filter ("all", "test", "non_test")
        
    Returns:
        Prepared DataFrame
    """
    try:
        df = annot_df[annot_df["model"] == model_name].copy()
        
        # Apply split filter
        if split_filter and split_filter != "all":
            df = df[df["split"] == split_filter]
        
        # Add Delta Age
        if "age" in df.columns and "age_pred" in df.columns:
            df["delta_age"] = (df["age_pred"] - df["age"]).round(2)
        
        # Transform sex column
        if "female" in df.columns:
            df["sexe"] = df["female"].apply(
                lambda x: "Femme" if str(x).lower() == "true" 
                else ("Homme" if str(x).lower() == "false" else "?")
            )
        
        # Clean ethnicity
        if "ethnicity" in df.columns:
            df["ethnicity"] = df["ethnicity"].apply(clean_ethnicity)
        
        # Sort by age
        if "age" in df.columns:
            df = df.sort_values("age")
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing samples DataFrame: {e}")
        return pd.DataFrame()


# Default column mapping for samples table
SAMPLES_COLUMNS_MAP = {
    "Sample_description": "Échantillon",
    "Sample_Name": "Nom",
    "sexe": "Sexe",
    "age": "Âge chrono",
    "age_pred": "Âge prédit",
    "delta_age": "Delta Age",
    "ethnicity": "Ethnicité",
    "split": "Ensemble",
}
