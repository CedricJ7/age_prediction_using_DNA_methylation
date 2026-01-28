"""
Configuration management for the DNA methylation age prediction project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: Path = Path("Data")
    top_k_features: int = 10000
    chunk_size: int = 2000
    test_size: float = 0.2
    use_pca: bool = False
    pca_components: int = 400
    missing_rate_threshold: float = 0.05


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    ridge_alpha: float = 100.0
    elasticnet_alpha: float = 0.1
    elasticnet_l1_ratio: float = 0.5
    lasso_alpha: float = 0.1
    xgboost_n_estimators: int = 400
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 6
    xgboost_reg_alpha: float = 1.0
    xgboost_reg_lambda: float = 10.0
    xgboost_early_stopping_rounds: int = 20
    rf_n_estimators: int = 300
    rf_max_depth: int = 20
    mlp_hidden_layers: tuple = (128, 64, 32)
    mlp_alpha: float = 0.001
    # DeepMAge parameters
    deepmage_hidden_size: int = 512
    deepmage_dropout: float = 0.3
    deepmage_learning_rate: float = 0.001
    deepmage_batch_size: int = 32
    deepmage_epochs: int = 100
    deepmage_early_stopping_patience: int = 10
    deepmage_random_state: int = 42


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    cv_folds: int = 10
    n_iter: int = 50
    random_state: int = 42
    scoring: str = "neg_mean_absolute_error"
    use_optuna: bool = False
    n_jobs: int = -1


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output_dir: Path = Path("results")

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance

        Example:
            >>> config = Config.from_yaml(Path("config/model_config.yaml"))
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            data=DataConfig(**data.get('data', {})),
            models=ModelConfig(**data.get('models', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            output_dir=Path(data.get('output_dir', 'results'))
        )

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path where to save the configuration
        """
        data = {
            'data': {
                'data_dir': str(self.data.data_dir),
                'top_k_features': self.data.top_k_features,
                'chunk_size': self.data.chunk_size,
                'test_size': self.data.test_size,
                'use_pca': self.data.use_pca,
                'pca_components': self.data.pca_components,
                'missing_rate_threshold': self.data.missing_rate_threshold,
            },
            'models': {
                'ridge_alpha': self.models.ridge_alpha,
                'elasticnet_alpha': self.models.elasticnet_alpha,
                'elasticnet_l1_ratio': self.models.elasticnet_l1_ratio,
                'lasso_alpha': self.models.lasso_alpha,
                'xgboost_n_estimators': self.models.xgboost_n_estimators,
                'xgboost_learning_rate': self.models.xgboost_learning_rate,
                'xgboost_max_depth': self.models.xgboost_max_depth,
                'xgboost_reg_alpha': self.models.xgboost_reg_alpha,
                'xgboost_reg_lambda': self.models.xgboost_reg_lambda,
                'xgboost_early_stopping_rounds': self.models.xgboost_early_stopping_rounds,
                'rf_n_estimators': self.models.rf_n_estimators,
                'rf_max_depth': self.models.rf_max_depth,
                'mlp_hidden_layers': list(self.models.mlp_hidden_layers),
                'mlp_alpha': self.models.mlp_alpha,
                'deepmage_hidden_size': self.models.deepmage_hidden_size,
                'deepmage_dropout': self.models.deepmage_dropout,
                'deepmage_learning_rate': self.models.deepmage_learning_rate,
                'deepmage_batch_size': self.models.deepmage_batch_size,
                'deepmage_epochs': self.models.deepmage_epochs,
                'deepmage_early_stopping_patience': self.models.deepmage_early_stopping_patience,
                'deepmage_random_state': self.models.deepmage_random_state,
            },
            'optimization': {
                'cv_folds': self.optimization.cv_folds,
                'n_iter': self.optimization.n_iter,
                'random_state': self.optimization.random_state,
                'scoring': self.optimization.scoring,
                'use_optuna': self.optimization.use_optuna,
                'n_jobs': self.optimization.n_jobs,
            },
            'output_dir': str(self.output_dir),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
