"""Training pipeline for drift-aware fraud detection with comprehensive monitoring."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
import json
import time
from datetime import datetime

from temporal_drift_aware_fraud_detection_with_adversarial_validation.models.model import (
    DriftAwareEnsemble, AdversarialValidator
)
from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.preprocessing import (
    FeatureEngineer, TemporalSplitter
)
from temporal_drift_aware_fraud_detection_with_adversarial_validation.evaluation.metrics import (
    DriftDetectionMetrics, CalibrationMetrics
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking disabled.")


class DriftAwareTrainer:
    """Advanced trainer for drift-aware fraud detection with comprehensive monitoring.

    This trainer implements a sophisticated training pipeline that:
    1. Handles temporal data splits to simulate realistic deployment
    2. Trains adversarial validators for drift detection
    3. Optimizes ensemble weights based on drift robustness
    4. Provides comprehensive evaluation and monitoring
    """

    def __init__(self,
                 config: Dict[str, Any],
                 save_dir: str = "models",
                 experiment_name: str = "drift_aware_fraud_detection"):
        """Initialize trainer.

        Args:
            config: Configuration dictionary with training parameters.
            save_dir: Directory to save models and artifacts.
            experiment_name: Name for MLflow experiment.
        """
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.experiment_name = experiment_name

        # Initialize components
        self.feature_engineer = FeatureEngineer(random_seed=config.get('random_seed', 42))
        self.temporal_splitter = TemporalSplitter(random_seed=config.get('random_seed', 42))
        self.ensemble = None
        self.adversarial_validator = None

        # Training state
        self.training_history = {}
        self.best_model_path = None
        self.training_data = None
        self.validation_data = None

        # Initialize MLflow
        if MLFLOW_AVAILABLE:
            self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow experiment tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")

    def train(self,
              train_data: pd.DataFrame,
              validation_data: Optional[pd.DataFrame] = None,
              save_best: bool = True) -> Dict[str, Any]:
        """Train the complete drift-aware fraud detection system.

        Args:
            train_data: Training dataset with temporal information.
            validation_data: Optional validation data. If None, split from training data.
            save_best: Whether to save the best model.

        Returns:
            Dictionary with training metrics and model information.
        """
        start_time = time.time()
        logger.info("Starting drift-aware fraud detection training")

        # Start MLflow run
        if MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        try:
            # Prepare data
            train_processed, val_processed, drift_periods = self._prepare_data(train_data, validation_data)

            # Train models
            training_metrics = self._train_models(train_processed, val_processed, drift_periods)

            # Comprehensive evaluation
            evaluation_metrics = self._evaluate_models(val_processed, drift_periods)

            # Combine all metrics
            all_metrics = {**training_metrics, **evaluation_metrics}

            # Log metrics
            self._log_metrics(all_metrics)

            # Save models if requested
            if save_best:
                self._save_models()

            # Record training time
            training_time = time.time() - start_time
            all_metrics['training_time_seconds'] = training_time

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final ensemble AUC: {all_metrics.get('ensemble_auc', 'N/A'):.4f}")

            return all_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.end_run()
                except:
                    pass

    def _prepare_data(self,
                     train_data: pd.DataFrame,
                     validation_data: Optional[pd.DataFrame] = None) -> Tuple[Tuple, Tuple, List]:
        """Prepare and preprocess data for training.

        Args:
            train_data: Raw training data.
            validation_data: Optional raw validation data.

        Returns:
            Tuple of (train_processed, val_processed, drift_periods).
        """
        logger.info("Preparing and preprocessing data")

        # Temporal split if no validation data provided
        if validation_data is None:
            train_split, val_split = self.temporal_splitter.stratified_temporal_split(
                train_data, train_ratio=self.config.get('train_ratio', 0.8)
            )
        else:
            train_split = train_data
            val_split = validation_data

        # Create drift periods for adversarial validation
        drift_periods = self.temporal_splitter.create_drift_periods(
            train_split, n_periods=self.config.get('n_drift_periods', 5)
        )

        # Feature engineering
        # Prepare target and features
        target_col = self.config.get('target_column', 'isFraud')

        # Fit feature engineer on training data
        train_features = self.feature_engineer.fit_transform(train_split)
        val_features = self.feature_engineer.transform(val_split)

        # Separate features and targets
        feature_cols = [col for col in train_features.columns
                       if col not in [target_col, 'TransactionID']]

        X_train = train_features[feature_cols]
        y_train = train_features[target_col]
        X_val = val_features[feature_cols]
        y_val = val_features[target_col]

        # Store processed data
        self.training_data = (X_train, y_train)
        self.validation_data = (X_val, y_val)

        logger.info(f"Data prepared - Train: {X_train.shape}, Validation: {X_val.shape}")
        logger.info(f"Train fraud rate: {y_train.mean():.4f}, Val fraud rate: {y_val.mean():.4f}")

        return (X_train, y_train), (X_val, y_val), drift_periods

    def _train_models(self,
                     train_data: Tuple,
                     val_data: Tuple,
                     drift_periods: List[pd.DataFrame]) -> Dict[str, Any]:
        """Train ensemble models and adversarial validator.

        Args:
            train_data: Tuple of (X_train, y_train).
            val_data: Tuple of (X_val, y_val).
            drift_periods: List of drift periods for adversarial validation.

        Returns:
            Training metrics dictionary.
        """
        logger.info("Training models")

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Configure base models
        base_models_config = self.config.get('base_models', {})
        if not base_models_config:
            base_models_config = self._get_default_model_config()

        # Initialize ensemble
        self.ensemble = DriftAwareEnsemble(
            base_models=base_models_config,
            ensemble_method=self.config.get('ensemble_method', 'weighted_average'),
            calibrate_probabilities=self.config.get('calibrate_probabilities', True),
            random_seed=self.config.get('random_seed', 42)
        )

        # Preprocess drift periods for adversarial validation
        processed_drift_periods = []
        for period in drift_periods:
            period_features = self.feature_engineer.transform(period)
            feature_cols = [col for col in period_features.columns
                          if col not in ['isFraud', 'TransactionID']]
            processed_drift_periods.append(period_features[feature_cols])

        # Train ensemble
        training_metrics = self.ensemble.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            drift_periods=processed_drift_periods
        )

        # Train standalone adversarial validator for drift detection
        if len(processed_drift_periods) > 1:
            logger.info("Training standalone adversarial validator")
            self.adversarial_validator = AdversarialValidator(
                model_type=self.config.get('adversarial_model_type', 'lightgbm'),
                random_seed=self.config.get('random_seed', 42)
            )

            # Use first half of training data as reference, second half as target
            mid_point = len(X_train) // 2
            X_ref = pd.DataFrame(X_train.iloc[:mid_point])
            X_target = pd.DataFrame(X_train.iloc[mid_point:])

            adv_metrics = self.adversarial_validator.fit(X_ref, X_target)
            training_metrics.update({f'adversarial_{k}': v for k, v in adv_metrics.items()})

        return training_metrics

    def _evaluate_models(self,
                        val_data: Tuple,
                        drift_periods: List[pd.DataFrame]) -> Dict[str, Any]:
        """Comprehensive evaluation of trained models.

        Args:
            val_data: Validation data tuple.
            drift_periods: List of drift periods.

        Returns:
            Evaluation metrics dictionary.
        """
        logger.info("Evaluating models")

        X_val, y_val = val_data
        evaluation_metrics = {}

        # Basic ensemble performance
        ensemble_probs = self.ensemble.predict_proba(X_val)
        ensemble_preds = self.ensemble.predict(X_val)

        # Drift detection evaluation
        drift_metrics = DriftDetectionMetrics()
        drift_eval = drift_metrics.evaluate_drift_detection(
            ensemble=self.ensemble,
            drift_periods=[self.feature_engineer.transform(period) for period in drift_periods],
            reference_data=X_val
        )
        evaluation_metrics.update(drift_eval)

        # Calibration evaluation
        calibration_metrics = CalibrationMetrics()
        cal_eval = calibration_metrics.evaluate_calibration(y_val, ensemble_probs)
        evaluation_metrics.update(cal_eval)

        # Model robustness across time periods
        if len(drift_periods) > 1:
            robustness_metrics = self._evaluate_temporal_robustness(drift_periods)
            evaluation_metrics.update(robustness_metrics)

        # Individual model contributions
        model_contributions = self.ensemble.get_model_contributions(X_val)
        for name, contrib in model_contributions.items():
            evaluation_metrics[f'{name}_contribution_mean'] = float(np.mean(contrib))
            evaluation_metrics[f'{name}_contribution_std'] = float(np.std(contrib))

        return evaluation_metrics

    def _evaluate_temporal_robustness(self, drift_periods: List[pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate model robustness across temporal periods.

        Args:
            drift_periods: List of drift periods.

        Returns:
            Robustness metrics dictionary.
        """
        logger.info("Evaluating temporal robustness")

        robustness_metrics = {}
        period_performances = []

        for i, period in enumerate(drift_periods):
            if 'isFraud' not in period.columns:
                continue

            # Preprocess period data
            period_processed = self.feature_engineer.transform(period)
            feature_cols = [col for col in period_processed.columns
                          if col not in ['isFraud', 'TransactionID']]

            X_period = period_processed[feature_cols]
            y_period = period_processed['isFraud']

            if len(y_period.unique()) < 2:  # Skip periods with only one class
                continue

            # Evaluate ensemble on this period
            try:
                period_probs = self.ensemble.predict_proba(X_period)
                from sklearn.metrics import roc_auc_score, average_precision_score

                period_auc = roc_auc_score(y_period, period_probs)
                period_ap = average_precision_score(y_period, period_probs)

                period_performances.append({
                    'period': i,
                    'auc': period_auc,
                    'average_precision': period_ap,
                    'fraud_rate': y_period.mean()
                })

                robustness_metrics[f'period_{i}_auc'] = period_auc
                robustness_metrics[f'period_{i}_ap'] = period_ap

            except Exception as e:
                logger.warning(f"Failed to evaluate period {i}: {e}")

        # Compute robustness statistics
        if period_performances:
            aucs = [p['auc'] for p in period_performances]
            aps = [p['average_precision'] for p in period_performances]

            robustness_metrics.update({
                'temporal_auc_mean': float(np.mean(aucs)),
                'temporal_auc_std': float(np.std(aucs)),
                'temporal_auc_min': float(np.min(aucs)),
                'temporal_auc_max': float(np.max(aucs)),
                'temporal_ap_mean': float(np.mean(aps)),
                'temporal_ap_std': float(np.std(aps)),
                'temporal_stability_score': 1.0 - float(np.std(aucs))  # Higher = more stable
            })

        return robustness_metrics

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to console and MLflow.

        Args:
            metrics: Metrics dictionary to log.
        """
        # Log to console
        logger.info("Training Results:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        mlflow.log_metric(key, float(value))
                    elif isinstance(value, (str, bool)):
                        mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

        # Save metrics to file
        metrics_file = self.save_dir / "training_metrics.json"
        try:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, np.number):
                    serializable_metrics[k] = float(v)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    serializable_metrics[k] = [float(x) if isinstance(x, np.number) else x for x in v]
                else:
                    serializable_metrics[k] = v

            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    def _save_models(self):
        """Save trained models and artifacts."""
        logger.info("Saving models")

        # Create checkpoints subdirectory
        checkpoints_dir = self.save_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        # Save ensemble
        ensemble_path = self.save_dir / "drift_aware_ensemble.pkl"
        self.ensemble.save(str(ensemble_path))
        self.best_model_path = str(ensemble_path)

        # Save checkpoint copy
        checkpoint_path = checkpoints_dir / f"ensemble_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.ensemble.save(str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save adversarial validator
        if self.adversarial_validator is not None:
            adv_validator_path = self.save_dir / "adversarial_validator.pkl"
            with open(adv_validator_path, 'wb') as f:
                import pickle
                pickle.dump(self.adversarial_validator, f)

        # Save feature engineer
        feature_engineer_path = self.save_dir / "feature_engineer.pkl"
        with open(feature_engineer_path, 'wb') as f:
            import pickle
            pickle.dump(self.feature_engineer, f)

        # Save configuration
        config_path = self.save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(str(ensemble_path))
                mlflow.log_artifact(str(config_path))
                if self.adversarial_validator is not None:
                    mlflow.log_artifact(str(adv_validator_path))
                mlflow.log_artifact(str(feature_engineer_path))
            except Exception as e:
                logger.warning(f"Failed to log artifacts to MLflow: {e}")

        logger.info(f"Models saved to {self.save_dir}")

    def _get_default_model_config(self) -> Dict[str, Dict]:
        """Get default model configuration."""
        random_seed = self.config.get('random_seed', 42)

        return {
            'lightgbm': {
                'type': 'lightgbm',
                'params': {
                    'objective': 'binary',
                    'metric': 'auc',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'n_estimators': 1000,
                    'verbose': -1,
                    'random_state': random_seed
                }
            },
            'xgboost': {
                'type': 'xgboost',
                'params': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 1000,
                    'subsample': 0.8,
                    'colsample_bytree': 0.9,
                    'random_state': random_seed,
                    'verbosity': 0
                }
            },
            'catboost': {
                'type': 'catboost',
                'params': {
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'random_seed': random_seed,
                    'verbose': False
                }
            }
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with trained ensemble.

        Args:
            X: Input features.

        Returns:
            Tuple of (predicted_probabilities, predicted_labels).
        """
        if self.ensemble is None:
            raise ValueError("Model not trained. Call train() first.")

        # Preprocess features
        X_processed = self.feature_engineer.transform(X)
        feature_cols = [col for col in X_processed.columns
                       if col not in ['isFraud', 'TransactionID']]
        X_features = X_processed[feature_cols]

        # Make predictions
        probs = self.ensemble.predict_proba(X_features)
        preds = self.ensemble.predict(X_features)

        return probs, preds

    def load_model(self, model_path: str):
        """Load a trained model.

        Args:
            model_path: Path to the saved model directory.
        """
        model_dir = Path(model_path)

        # Load ensemble
        ensemble_path = model_dir / "drift_aware_ensemble.pkl"
        if ensemble_path.exists():
            self.ensemble = DriftAwareEnsemble.load(str(ensemble_path))

        # Load feature engineer
        feature_engineer_path = model_dir / "feature_engineer.pkl"
        if feature_engineer_path.exists():
            with open(feature_engineer_path, 'rb') as f:
                import pickle
                self.feature_engineer = pickle.load(f)

        # Load adversarial validator
        adv_validator_path = model_dir / "adversarial_validator.pkl"
        if adv_validator_path.exists():
            with open(adv_validator_path, 'rb') as f:
                import pickle
                self.adversarial_validator = pickle.load(f)

        logger.info(f"Model loaded from {model_path}")

    def evaluate_drift(self, new_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate drift between new data and reference data.

        Args:
            new_data: New data to evaluate.
            reference_data: Reference training data.

        Returns:
            Drift evaluation results.
        """
        if self.adversarial_validator is None:
            raise ValueError("Adversarial validator not available. Train with drift periods.")

        # Preprocess both datasets
        new_processed = self.feature_engineer.transform(new_data)
        ref_processed = self.feature_engineer.transform(reference_data)

        # Get feature columns
        feature_cols = [col for col in new_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        new_features = new_processed[feature_cols]
        ref_features = ref_processed[feature_cols]

        # Detect drift
        drift_results = self.adversarial_validator.predict_drift(new_features, ref_features)

        return drift_results