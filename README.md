# DNA Methylation Age Prediction

[![Tests](https://github.com/YOUR_USERNAME/age_prediction_using_DNA_methylation/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/age_prediction_using_DNA_methylation/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Epigenetic clock models for predicting chronological age from DNA methylation profiles (EPICv2 array, ~900,000 CpG sites).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Or use the new modular script
python scripts/train.py --config config/model_config.yaml

# Run the interactive dashboard
python app.py
# Open http://localhost:8050
```

## Features

- **6 ML Models**: Ridge, Lasso, ElasticNet, RandomForest, XGBoost, MLP (AltumAge)
- **Modular Architecture**: Clean separation of data, features, models, and evaluation
- **Interactive Dashboard**: Dash-based visualization with multiple analysis views
- **Publication-Ready Figures**: Nature-style and revolutionary data art visualizations
- **Comprehensive Reports**: PDF and LaTeX report generation
- **Full Test Suite**: 32 unit tests with pytest

## Project Structure

```
age_prediction_using_DNA_methylation/
├── src/                          # Modular source code
│   ├── data/                     # Data loading and imputation
│   ├── features/                 # Feature selection and engineering
│   ├── models/                   # Model definitions
│   ├── evaluation/               # Metrics and evaluation
│   ├── optimization/             # Bayesian optimization (Optuna)
│   ├── apps/                     # Dash components and callbacks
│   └── utils/                    # Configuration and logging
├── tests/                        # Unit tests (32 tests)
├── config/                       # YAML configuration files
├── scripts/                      # Entry point scripts
├── assets/                       # CSS styles and images
├── results/                      # Output files and models
├── app.py                        # Main Dash application
├── train_models.py               # Legacy training script
└── requirements.txt              # Dependencies
```

## Installation

### Requirements

- Python 3.10+
- 8GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/age_prediction_using_DNA_methylation.git
cd age_prediction_using_DNA_methylation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For development (includes testing and linting tools)
pip install -r requirements-dev.txt
```

## Usage

### Training Models

```bash
# Standard training with default config
python scripts/train.py

# With custom configuration
python scripts/train.py --config config/model_config.yaml

# With stacked ensemble
python scripts/train.py --ensemble

# Legacy script (still supported)
python train_models.py --top-k 10000 --optimize
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Dashboard

```bash
python app.py
# Open http://localhost:8050
```

### Generate Reports

```bash
# PhD-level PDF report
python generate_phd_report.py

# Publication figures (Nature style)
python generate_publication_figure.py

# Revolutionary visualizations
python revolutionary_viz.py
```

## Configuration

Configuration is managed via YAML files in `config/`:

```yaml
# config/model_config.yaml
data:
  data_dir: "Data"
  top_k_features: 10000
  test_size: 0.2

models:
  ridge_alpha: 100.0
  xgboost_reg_alpha: 1.0
  xgboost_reg_lambda: 10.0

optimization:
  cv_folds: 10
  random_state: 42
  use_optuna: false
```

## Models

| Model | Description | Best For |
|-------|-------------|----------|
| **Ridge** | L2 regularized linear regression | Baseline, interpretability |
| **Lasso** | L1 regularized (sparse) | Feature selection |
| **ElasticNet** | L1+L2 combined | Balance of both |
| **RandomForest** | Ensemble of decision trees | Robustness |
| **XGBoost** | Gradient boosting | Best performance |
| **AltumAge (MLP)** | Neural network | Non-linear relationships |

## Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (years) |
| **MAD** | Median Absolute Deviation |
| **R²** | Coefficient of determination |
| **Correlation** | Pearson correlation |
| **Overfitting Ratio** | Test MAE / Train MAE |

## API Reference

### Data Loading

```python
from src.data.data_loader import load_annotations, load_cpg_names

annot = load_annotations(Path("Data"))
cpg_names = load_cpg_names(Path("Data"))
```

### Model Creation

```python
from src.utils.config import ModelConfig
from src.models.linear_models import create_ridge_model

config = ModelConfig(ridge_alpha=100.0)
model = create_ridge_model(config)
model.fit(X_train, y_train)
```

### Evaluation

```python
from src.evaluation.metrics import evaluate_model

metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
print(f"MAE: {metrics['mae']:.2f} years")
```

### Optimization (Optuna)

```python
from src.optimization import optimize_ridge

results = optimize_ridge(X_train, y_train, n_trials=100)
print(f"Best alpha: {results['best_params']['alpha']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## References

1. Horvath, S. (2013). DNA methylation age of human tissues. *Genome Biology*, 14(10), R115.
2. Hannum, G., et al. (2013). Genome-wide methylation profiles. *Molecular Cell*, 49(2), 359-367.
3. Levine, M. E., et al. (2018). PhenoAge biomarker. *Aging*, 10(4), 573-591.
4. Lu, A. T., et al. (2019). GrimAge predictor. *Aging*, 11(2), 303-327.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*DNA Methylation Age Prediction - Epigenetic Clock Benchmark*
