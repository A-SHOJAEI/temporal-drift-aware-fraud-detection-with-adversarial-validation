"""Configuration management for drift-aware fraud detection system."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    model_type: str
    params: Dict[str, Any]


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    random_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    n_drift_periods: int = 5
    target_column: str = 'isFraud'
    ensemble_method: str = 'weighted_average'
    calibrate_probabilities: bool = True
    early_stopping_rounds: int = 50
    n_trials_optuna: int = 100


@dataclass
class DataConfig:
    """Data configuration parameters."""
    data_path: Optional[str] = None
    use_synthetic: bool = True
    time_column: str = 'TransactionDT'
    id_column: str = 'TransactionID'
    target_column: str = 'isFraud'
    feature_engineering: bool = True
    handle_missing: bool = True
    scale_features: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    target_metrics: Dict[str, float] = None
    calibration_bins: int = 10
    drift_threshold: float = 0.6
    business_cost_fp: float = 1.0
    business_cost_fn: float = 10.0
    min_precision_at_recall_80: float = 0.5

    def __post_init__(self):
        if self.target_metrics is None:
            self.target_metrics = {
                'auroc': 0.94,
                'auprc': 0.65,
                'drift_detection_recall': 0.85,
                'calibration_ece': 0.03
            }


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_to_file: bool = True
    log_file: str = 'logs/training.log'


@dataclass
class MLflowConfig:
    """MLflow configuration parameters."""
    enabled: bool = True
    experiment_name: str = 'drift_aware_fraud_detection'
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {
                'project': 'temporal-drift-aware-fraud-detection',
                'model_type': 'ensemble',
                'feature_type': 'tabular'
            }


class Config:
    """Main configuration class for the drift-aware fraud detection system."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = Path(config_path) if config_path else None

        # Default configurations
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        self.logging = LoggingConfig()
        self.mlflow = MLflowConfig()
        self.base_models = self._get_default_model_configs()

        # Load configuration from file if provided
        if self.config_path and self.config_path.exists():
            self.load_from_file(self.config_path)

    def _get_default_model_configs(self) -> Dict[str, ModelConfig]:
        """Get default model configurations.

        Returns:
            Dictionary of default model configurations.
        """
        return {
            'lightgbm': ModelConfig(
                model_type='lightgbm',
                params={
                    'objective': 'binary',
                    'metric': 'auc',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'n_estimators': 1000,
                    'verbose': -1,
                    'random_state': self.training.random_seed if hasattr(self, 'training') else 42
                }
            ),
            'xgboost': ModelConfig(
                model_type='xgboost',
                params={
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 1000,
                    'subsample': 0.8,
                    'colsample_bytree': 0.9,
                    'random_state': self.training.random_seed if hasattr(self, 'training') else 42,
                    'verbosity': 0
                }
            ),
            'catboost': ModelConfig(
                model_type='catboost',
                params={
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'random_seed': self.training.random_seed if hasattr(self, 'training') else 42,
                    'verbose': False
                }
            )
        }

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.
        """
        config_path = Path(config_path)
        logger.info(f"Loading configuration from {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            # Update configurations
            if 'training' in config_dict:
                self._update_dataclass(self.training, config_dict['training'])

            if 'data' in config_dict:
                self._update_dataclass(self.data, config_dict['data'])

            if 'evaluation' in config_dict:
                self._update_dataclass(self.evaluation, config_dict['evaluation'])

            if 'logging' in config_dict:
                self._update_dataclass(self.logging, config_dict['logging'])

            if 'mlflow' in config_dict:
                self._update_dataclass(self.mlflow, config_dict['mlflow'])

            # Update model configurations
            if 'base_models' in config_dict:
                for model_name, model_config in config_dict['base_models'].items():
                    if model_name in self.base_models:
                        # Update existing model config
                        if 'params' in model_config:
                            self.base_models[model_name].params.update(model_config['params'])
                    else:
                        # Add new model config
                        self.base_models[model_name] = ModelConfig(
                            model_type=model_config.get('model_type', model_name),
                            params=model_config.get('params', {})
                        )

            # Ensure random seeds are consistent
            self._sync_random_seeds()

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.warning("Using default configuration")

    def _update_dataclass(self, dataclass_instance: Any, update_dict: Dict[str, Any]) -> None:
        """Update dataclass instance with dictionary values.

        Args:
            dataclass_instance: Dataclass instance to update.
            update_dict: Dictionary with new values.
        """
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def _sync_random_seeds(self) -> None:
        """Ensure all random seeds are consistent."""
        for model_config in self.base_models.values():
            if 'random_state' in model_config.params:
                model_config.params['random_state'] = self.training.random_seed
            elif 'random_seed' in model_config.params:
                model_config.params['random_seed'] = self.training.random_seed

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            config_path: Path to save configuration file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            'training': asdict(self.training),
            'data': asdict(self.data),
            'evaluation': asdict(self.evaluation),
            'logging': asdict(self.logging),
            'mlflow': asdict(self.mlflow),
            'base_models': {
                name: asdict(model_config)
                for name, model_config in self.base_models.items()
            }
        }

        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def get_model_config_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations as dictionary for ensemble initialization.

        Returns:
            Dictionary suitable for DriftAwareEnsemble initialization.
        """
        return {
            name: {
                'type': config.model_type,
                'params': config.params
            }
            for name, config in self.base_models.items()
        }

    def setup_logging(self) -> None:
        """Set up logging based on configuration."""
        # Create logs directory if needed
        if self.logging.log_to_file:
            log_file = Path(self.logging.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)

        handlers = [logging.StreamHandler()]

        if self.logging.log_to_file:
            handlers.append(logging.FileHandler(self.logging.log_file))

        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            handlers=handlers,
            force=True
        )

        logger.info("Logging configuration applied")

    def validate(self) -> bool:
        """Validate configuration parameters.

        Returns:
            True if configuration is valid, False otherwise.
        """
        logger.info("Validating configuration")

        try:
            # Validate training configuration
            assert 0 < self.training.train_ratio <= 1.0, "train_ratio must be between 0 and 1"
            assert 0 < self.training.val_ratio <= 1.0, "val_ratio must be between 0 and 1"
            assert 0 < self.training.test_ratio <= 1.0, "test_ratio must be between 0 and 1"
            assert abs(self.training.train_ratio + self.training.val_ratio + self.training.test_ratio - 1.0) < 1e-6, \
                "train_ratio + val_ratio + test_ratio must equal 1"

            assert self.training.n_drift_periods >= 2, "Need at least 2 drift periods"
            assert self.training.random_seed >= 0, "Random seed must be non-negative"

            # Validate data configuration
            if self.data.data_path:
                data_path = Path(self.data.data_path)
                if not data_path.exists():
                    logger.warning(f"Data path does not exist: {data_path}")

            # Validate evaluation configuration
            assert 0 <= self.evaluation.drift_threshold <= 1.0, "drift_threshold must be between 0 and 1"
            assert self.evaluation.calibration_bins > 0, "calibration_bins must be positive"

            # Validate target metrics
            for metric_name, target_value in self.evaluation.target_metrics.items():
                if 'auc' in metric_name.lower() or 'recall' in metric_name.lower():
                    assert 0 <= target_value <= 1.0, f"{metric_name} must be between 0 and 1"

            # Validate model configurations
            assert len(self.base_models) > 0, "Need at least one base model"

            for model_name, model_config in self.base_models.items():
                assert model_config.model_type in ['lightgbm', 'xgboost', 'catboost', 'rf'], \
                    f"Unsupported model type: {model_config.model_type}"

            logger.info("Configuration validation passed")
            return True

        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            'training': asdict(self.training),
            'data': asdict(self.data),
            'evaluation': asdict(self.evaluation),
            'logging': asdict(self.logging),
            'mlflow': asdict(self.mlflow),
            'base_models': {
                name: asdict(model_config)
                for name, model_config in self.base_models.items()
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        try:
            keys = key.split('.')
            value = self
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except:
            return default


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration with optional path.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Loaded configuration.
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            'configs/default.yaml',
            'config.yaml',
            Path.home() / '.fraud_detection' / 'config.yaml'
        ]

        for path in default_paths:
            if Path(path).exists():
                config_path = path
                break

    config = Config(config_path)

    if not config.validate():
        logger.warning("Configuration validation failed. Check parameters.")

    config.setup_logging()
    return config