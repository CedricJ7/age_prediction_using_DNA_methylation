"""
Pytest fixtures for DNA methylation age prediction tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_annotations():
    """Create sample annotation DataFrame for testing."""
    return pd.DataFrame({
        'Sample_name': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'female': [True, False, True, False, True],
        'ethnicity': ['White', 'Asian', 'White', 'Hispanic', 'Unavailable'],
        'age': [25.0, 30.0, 45.0, 60.0, 35.0],
        'Sample_description': ['D1', 'D2', 'D3', 'D4', 'D5']
    })


@pytest.fixture
def sample_methylation_data():
    """Create sample methylation matrix for testing."""
    np.random.seed(42)
    n_cpgs = 100
    n_samples = 5

    # Create methylation data (beta values between 0 and 1)
    data = np.random.beta(2, 5, size=(n_cpgs, n_samples))

    # No missing values for model testing
    # (missing values should be handled by imputation first)

    df = pd.DataFrame(
        data,
        columns=['D1', 'D2', 'D3', 'D4', 'D5'],
        index=[f'cg{i:05d}' for i in range(n_cpgs)]
    )

    return df


@pytest.fixture
def sample_features_and_targets(sample_methylation_data, sample_annotations):
    """Create sample X and y for model testing."""
    X = sample_methylation_data.T  # Transpose to samples x features
    y = sample_annotations.set_index('Sample_description')['age'].values
    return X, y


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory with sample files."""
    data_dir = tmp_path / "Data"
    data_dir.mkdir()

    # Create sample annotation file
    annot = pd.DataFrame({
        'Sample_name': ['S1', 'S2', 'S3'],
        'female': [True, False, True],
        'ethnicity': ['White', 'Asian', 'White'],
        'age': [25.0, 30.0, 45.0],
        'Sample_description': ['D1', 'D2', 'D3']
    })
    annot.to_csv(data_dir / "annot_projet.csv", index=False)

    # Create sample CpG names file
    cpg_names = pd.DataFrame({
        'cpg_names': [f'cg{i:07d}' for i in range(1000)]
    })
    cpg_names.to_csv(data_dir / "cpg_names_projet.csv", index=False)

    # Create sample methylation file
    np.random.seed(42)
    methylation = pd.DataFrame(
        np.random.beta(2, 5, size=(1000, 3)),
        columns=['D1', 'D2', 'D3']
    )
    methylation.to_csv(data_dir / "c_sample.csv", index=False)

    return data_dir


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    from src.utils.config import Config, DataConfig, ModelConfig, OptimizationConfig

    return Config(
        data=DataConfig(
            data_dir=Path("Data"),
            top_k_features=100,
            chunk_size=50,
            test_size=0.2,
        ),
        models=ModelConfig(
            ridge_alpha=100.0,
            xgboost_reg_alpha=1.0,
        ),
        optimization=OptimizationConfig(
            cv_folds=3,  # Reduced for faster testing
            n_iter=10,
            random_state=42,
        )
    )
