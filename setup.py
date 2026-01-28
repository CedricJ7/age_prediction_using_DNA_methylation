"""Setup file for DNA Methylation Age Prediction project."""

from setuptools import setup, find_packages

setup(
    name="dna-methylation-age-prediction",
    version="1.0.0",
    description="Epigenetic clock models for predicting chronological age from DNA methylation",
    author="Master IA Project",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "scipy>=1.11.0",

        # Visualization
        "plotly>=5.17.0",
        "dash>=2.14.0",
        "kaleido>=0.2.1",

        # Report generation
        "fpdf2>=2.7.0",

        # Configuration
        "pyyaml>=6.0",

        # Optimization
        "optuna>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "dna-train=scripts.train:main",
        ],
    },
)
