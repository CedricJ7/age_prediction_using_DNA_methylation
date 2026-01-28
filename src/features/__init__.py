"""
Feature extraction and engineering modules.
"""

from .selection import (
    filter_cpgs_by_missing_rate,
    select_top_k_cpgs,
)

from .demographic import add_demographic_features

from .stability_selection import (
    stability_selection,
    stability_selection_cv,
    get_stable_features,
)

from .engineering import (
    create_polynomial_features,
    create_demographic_interactions,
    create_age_group_features,
    compute_cpg_region_scores,
    select_features_by_variance,
    normalize_features,
)

__all__ = [
    # Selection
    "filter_cpgs_by_missing_rate",
    "select_top_k_cpgs",
    # Demographic
    "add_demographic_features",
    # Stability selection
    "stability_selection",
    "stability_selection_cv",
    "get_stable_features",
    # Engineering
    "create_polynomial_features",
    "create_demographic_interactions",
    "create_age_group_features",
    "compute_cpg_region_scores",
    "select_features_by_variance",
    "normalize_features",
]
