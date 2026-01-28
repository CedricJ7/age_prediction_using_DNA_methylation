"""
Demographic feature engineering for DNA methylation age prediction.

This module handles encoding of demographic variables (gender, ethnicity)
for use as features in predictive models.
"""

import numpy as np
import pandas as pd

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


def clean_ethnicity(ethnicity: str) -> str:
    """
    Clean and group ethnicity categories.

    Groups 'Unavailable', 'Declined', 'Other', and missing values into 'Inconnu' (Unknown).

    Args:
        ethnicity: Raw ethnicity string

    Returns:
        Cleaned ethnicity category

    Example:
        >>> clean_ethnicity("White")
        'White'
        >>> clean_ethnicity("Unavailable")
        'Inconnu'
        >>> clean_ethnicity(np.nan)
        'Inconnu'
    """
    if pd.isna(ethnicity):
        return "Inconnu"

    eth = str(ethnicity).strip()

    if eth.lower() in ["unavailable", "declined", "other", ""]:
        return "Inconnu"

    return eth


def add_demographic_features(annot: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare demographic features from annotations.

    Processes gender and ethnicity variables:
    - Gender: Binary encoding (1=Female, 0=Male)
    - Ethnicity: One-hot encoding with rare category grouping

    Args:
        annot: DataFrame with annotations including 'female' and/or 'ethnicity' columns

    Returns:
        DataFrame with encoded demographic features

    Example:
        >>> annot = pd.DataFrame({
        ...     'female': [True, False, True],
        ...     'ethnicity': ['White', 'Asian', 'White']
        ... }, index=['S1', 'S2', 'S3'])
        >>> demo = add_demographic_features(annot)
        >>> 'is_female' in demo.columns
        True
        >>> 'eth_White' in demo.columns
        True
    """
    logger.info("Preparing demographic features")

    demo_features = pd.DataFrame(index=annot.index)

    # Gender encoding
    if "female" in annot.columns:
        demo_features["is_female"] = annot["female"].apply(
            lambda x: 1 if str(x).lower() == "true" else (
                0 if str(x).lower() == "false" else np.nan
            )
        )

        valid_count = demo_features["is_female"].notna().sum()
        logger.info(f"Gender: {valid_count} valid values")
    else:
        logger.warning("No 'female' column found in annotations")

    # Ethnicity encoding
    if "ethnicity" in annot.columns:
        # Clean and group ethnicity categories
        ethnicity_clean = annot["ethnicity"].apply(clean_ethnicity)

        # Log distribution
        eth_counts = ethnicity_clean.value_counts().to_dict()
        logger.info(f"Ethnicity distribution (after cleaning): {eth_counts}")

        # One-hot encoding
        ethnicity_dummies = pd.get_dummies(
            ethnicity_clean,
            prefix="eth",
            dummy_na=False
        )

        demo_features = pd.concat([demo_features, ethnicity_dummies], axis=1)

        logger.info(f"Ethnicity features created: {list(ethnicity_dummies.columns)}")
    else:
        logger.warning("No 'ethnicity' column found in annotations")

    logger.info(f"Total demographic features: {len(demo_features.columns)}")

    return demo_features
