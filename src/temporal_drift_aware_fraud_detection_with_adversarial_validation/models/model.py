"""Advanced fraud detection models with adversarial validation and drift awareness."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
from pathlib import Path

# Graceful handling of missing ML dependencies
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    LogisticRegression = None
    roc_auc_score = None
    average_precision_score = None
    CalibratedClassifierCV = None
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check GPU availability for different ML frameworks."""
    gpu_info = {
        'torch_cuda': False,
        'lightgbm_gpu': False,
        'xgboost_gpu': False,
        'catboost_gpu': False
    }

    try:
        import torch
        gpu_info['torch_cuda'] = torch.cuda.is_available()
        if gpu_info['torch_cuda']:
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        # LightGBM GPU support check
        gpu_info['lightgbm_gpu'] = hasattr(lgb, 'create_valid_data')  # Simple check
    except ImportError:
        pass

    try:
        import xgboost as xgb
        # XGBoost GPU support check
        gpu_info['xgboost_gpu'] = True  # XGBoost can auto-detect
    except ImportError:
        pass

    try:
        import catboost as cb
        # CatBoost GPU support check
        gpu_info['catboost_gpu'] = True  # CatBoost can auto-detect
    except ImportError:
        pass

    return gpu_info


class AdversarialValidator:
    """Adversarial validator to detect distribution drift between time periods.

    This model learns to distinguish between different temporal periods in the data,
    helping identify when the production distribution diverges from training data.
    """

    def __init__(self,
                 model_type: str = 'lightgbm',
                 random_seed: int = 42,
                 **model_params):
        """Initialize adversarial validator.

        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'catboost', 'rf').
            random_seed: Random seed for reproducibility.
            **model_params: Additional model parameters.
        """
        self.model_type = model_type
        self.random_seed = random_seed
        self.model_params = model_params
        self.model = None
        self.feature_importance = None
        self.drift_threshold = 0.6  # Threshold for detecting drift

    def _create_model(self):
        """Create the underlying model with GPU optimization if available."""
        gpu_info = check_gpu_availability()

        if self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed. Please install with: pip install lightgbm")

            default_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_seed,
                'device_type': 'gpu' if gpu_info['lightgbm_gpu'] and gpu_info['torch_cuda'] else 'cpu'
            }
            default_params.update(self.model_params)
            self.model = lgb.LGBMClassifier(**default_params)

        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")

            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'random_state': self.random_seed,
                'tree_method': 'gpu_hist' if gpu_info['xgboost_gpu'] and gpu_info['torch_cuda'] else 'auto'
            }
            default_params.update(self.model_params)
            self.model = xgb.XGBClassifier(**default_params)

        elif self.model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost is not installed. Please install with: pip install catboost")

            default_params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': self.random_seed,
                'verbose': False,
                'task_type': 'GPU' if gpu_info['catboost_gpu'] and gpu_info['torch_cuda'] else 'CPU'
            }
            default_params.update(self.model_params)
            self.model = cb.CatBoostClassifier(**default_params)

        elif self.model_type == 'rf':
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")

            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_seed,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def prepare_drift_data(self,
                          source_data: pd.DataFrame,
                          target_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for adversarial validation.

        Args:
            source_data: Source domain data (e.g., training data).
            target_data: Target domain data (e.g., new production data).

        Returns:
            Tuple of (features, labels) where labels indicate domain.
        """
        # Combine data
        source_data = source_data.copy()
        target_data = target_data.copy()

        # Remove target column if present
        feature_cols = [col for col in source_data.columns if col not in ['isFraud', 'TransactionID']]

        source_features = source_data[feature_cols]
        target_features = target_data[feature_cols]

        # Ensure same columns
        common_cols = source_features.columns.intersection(target_features.columns)
        source_features = source_features[common_cols]
        target_features = target_features[common_cols]

        # Create labels (0 for source, 1 for target)
        source_labels = np.zeros(len(source_features))
        target_labels = np.ones(len(target_features))

        # Combine
        X = np.vstack([source_features.values, target_features.values])
        y = np.hstack([source_labels, target_labels])

        return X, y

    def fit(self,
            source_data: pd.DataFrame,
            target_data: pd.DataFrame,
            validation_split: float = 0.2) -> Dict[str, float]:
        """Fit adversarial validator.

        Args:
            source_data: Source domain data.
            target_data: Target domain data.
            validation_split: Proportion of data for validation.

        Returns:
            Training metrics dictionary.
        """
        logger.info(f"Training adversarial validator with {self.model_type}")

        # Prepare data
        X, y = self.prepare_drift_data(source_data, target_data)

        # Create model
        self._create_model()

        # Split data for validation
        n_samples = len(X)
        val_size = int(n_samples * validation_split)
        train_size = n_samples - val_size

        # Shuffle data
        indices = np.random.RandomState(self.random_seed).permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Train model
        if self.model_type in ['lightgbm', 'xgboost', 'catboost']:
            import lightgbm as _lgb
            if self.model_type == 'lightgbm':
                self.model.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)],
                    callbacks=[_lgb.early_stopping(50, verbose=False), _lgb.log_evaluation(period=0)]
                )
            else:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

        # Calculate metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        val_pred = self.model.predict_proba(X_val)[:, 1]

        metrics = {
            'train_auc': roc_auc_score(y_train, train_pred),
            'val_auc': roc_auc_score(y_val, val_pred),
            'drift_score': roc_auc_score(y_val, val_pred)
        }

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        logger.info(f"Adversarial validation AUC: {metrics['drift_score']:.4f}")

        return metrics

    def predict_drift(self, new_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, float]:
        """Predict drift between reference and new data.

        Args:
            new_data: New data to check for drift.
            reference_data: Reference data (training distribution).

        Returns:
            Dictionary with drift metrics.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X, y = self.prepare_drift_data(reference_data, new_data)
        drift_probs = self.model.predict_proba(X)[:, 1]

        # Separate predictions for reference and new data
        n_reference = len(reference_data)
        reference_scores = drift_probs[:n_reference]
        new_scores = drift_probs[n_reference:]

        drift_metrics = {
            'drift_score': np.mean(new_scores),
            'drift_auc': roc_auc_score(y, drift_probs),
            'is_drift_detected': np.mean(new_scores) > self.drift_threshold,
            'reference_mean_score': np.mean(reference_scores),
            'new_mean_score': np.mean(new_scores),
            'score_difference': np.mean(new_scores) - np.mean(reference_scores)
        }

        return drift_metrics

    def get_drift_features(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get top features contributing to drift.

        Args:
            top_k: Number of top features to return.

        Returns:
            List of (feature_index, importance) tuples.
        """
        if self.feature_importance is None:
            return []

        feature_importance_pairs = [(i, importance)
                                  for i, importance in enumerate(self.feature_importance)]
        return sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)[:top_k]


class DriftAwareEnsemble:
    """Ensemble model that dynamically reweights based on drift detection.

    This ensemble combines multiple base models (LightGBM, CatBoost, XGBoost) and
    adjusts their weights based on their robustness to detected drift regimes.
    """

    def __init__(self,
                 base_models: Optional[Dict[str, Any]] = None,
                 ensemble_method: str = 'weighted_average',
                 calibrate_probabilities: bool = True,
                 random_seed: int = 42):
        """Initialize drift-aware ensemble.

        Args:
            base_models: Dictionary of base model configurations.
            ensemble_method: Method for combining predictions.
            calibrate_probabilities: Whether to calibrate output probabilities.
            random_seed: Random seed for reproducibility.
        """
        self.ensemble_method = ensemble_method
        self.calibrate_probabilities = calibrate_probabilities
        self.random_seed = random_seed

        # Initialize base models
        if base_models is None:
            base_models = self._get_default_models()

        self.base_models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.calibrators = {}
        self.adversarial_validator = None

        # Create base models
        for name, config in base_models.items():
            self.base_models[name] = self._create_base_model(name, config)
            self.model_weights[name] = 1.0 / len(base_models)  # Equal initial weights

    def _get_default_models(self) -> Dict[str, Dict]:
        """Get default model configurations with GPU optimization."""
        gpu_info = check_gpu_availability()

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
                    'verbose': -1,
                    'random_state': self.random_seed,
                    'device_type': 'gpu' if gpu_info['lightgbm_gpu'] and gpu_info['torch_cuda'] else 'cpu'
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
                    'random_state': self.random_seed,
                    'tree_method': 'gpu_hist' if gpu_info['xgboost_gpu'] and gpu_info['torch_cuda'] else 'auto'
                }
            },
            'catboost': {
                'type': 'catboost',
                'params': {
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'random_seed': self.random_seed,
                    'verbose': False,
                    'task_type': 'GPU' if gpu_info['catboost_gpu'] and gpu_info['torch_cuda'] else 'CPU'
                }
            }
        }

    def _create_base_model(self, name: str, config: Dict):
        """Create a base model from configuration."""
        model_type = config.get('type') or config.get('model_type') or name
        params = config.get('params', {})

        if model_type == 'lightgbm':
            return lgb.LGBMClassifier(**params)
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif model_type == 'catboost':
            return cb.CatBoostClassifier(**params)
        elif model_type == 'rf':
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            drift_periods: Optional[List[pd.DataFrame]] = None) -> Dict[str, Any]:
        """Fit ensemble models with drift awareness.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            drift_periods: List of data from different time periods for drift analysis.

        Returns:
            Training metrics and model information.
        """
        logger.info(f"Training ensemble with {len(self.base_models)} base models")

        training_metrics = {}

        # Train each base model
        for name, model in list(self.base_models.items()):
            logger.info(f"Training {name}")

            try:
                if name in ['lightgbm', 'xgboost', 'catboost']:
                    import lightgbm as _lgb
                    if name == 'lightgbm':
                        model.fit(
                            X_train, y_train, eval_set=[(X_val, y_val)],
                            callbacks=[_lgb.early_stopping(50, verbose=False), _lgb.log_evaluation(period=0)]
                        )
                    else:
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)

                # Evaluate model
                val_pred = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, val_pred)
                ap = average_precision_score(y_val, val_pred)

                self.model_performance[name] = {
                    'auc': auc,
                    'average_precision': ap
                }

                training_metrics[f'{name}_auc'] = auc
                training_metrics[f'{name}_ap'] = ap

                logger.info(f"{name} - AUC: {auc:.4f}, AP: {ap:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                # Remove failed model
                del self.base_models[name]
                if name in self.model_weights:
                    del self.model_weights[name]

        # Calibrate probabilities if requested
        if self.calibrate_probabilities:
            self._calibrate_models(X_val, y_val)

        # Train adversarial validator if drift periods provided
        if drift_periods and len(drift_periods) > 1:
            self._train_adversarial_validator(drift_periods)

        # Optimize ensemble weights
        self._optimize_weights(X_val, y_val)

        # Calculate ensemble performance
        ensemble_pred = self.predict_proba(X_val)
        training_metrics['ensemble_auc'] = roc_auc_score(y_val, ensemble_pred)
        training_metrics['ensemble_ap'] = average_precision_score(y_val, ensemble_pred)

        logger.info(f"Ensemble - AUC: {training_metrics['ensemble_auc']:.4f}, "
                   f"AP: {training_metrics['ensemble_ap']:.4f}")

        return training_metrics

    def _calibrate_models(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calibrate model probabilities."""
        logger.info("Calibrating model probabilities")

        for name, model in self.base_models.items():
            try:
                calibrator = CalibratedClassifierCV(model, cv=3, method='isotonic')
                calibrator.fit(X_val, y_val)
                self.calibrators[name] = calibrator
            except Exception as e:
                logger.warning(f"Failed to calibrate {name}: {e}")

    def _train_adversarial_validator(self, drift_periods: List[pd.DataFrame]):
        """Train adversarial validator on drift periods."""
        logger.info("Training adversarial validator")

        # Use first period as reference, combine others as target
        reference_period = drift_periods[0]
        target_period = pd.concat(drift_periods[1:], ignore_index=True)

        self.adversarial_validator = AdversarialValidator(random_seed=self.random_seed)
        drift_metrics = self.adversarial_validator.fit(reference_period, target_period)

        logger.info(f"Adversarial validator trained. Drift AUC: {drift_metrics['drift_score']:.4f}")

    def _optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Optimize ensemble weights based on validation performance."""
        # Simple performance-based weighting
        total_performance = sum(perf['auc'] for perf in self.model_performance.values())

        for name in self.base_models.keys():
            if name in self.model_performance:
                self.model_weights[name] = self.model_performance[name]['auc'] / total_performance

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        for name in self.model_weights:
            self.model_weights[name] /= total_weight

        logger.info(f"Optimized ensemble weights: {self.model_weights}")

    def predict_proba(self, X: pd.DataFrame,
                     adaptive_weights: bool = True) -> np.ndarray:
        """Predict probabilities with ensemble.

        Args:
            X: Input features.
            adaptive_weights: Whether to use adaptive weighting based on drift.

        Returns:
            Predicted probabilities.
        """
        if not self.base_models:
            raise ValueError("No trained models available")

        predictions = []
        weights = []

        for name, model in self.base_models.items():
            if name in self.calibrators:
                pred = self.calibrators[name].predict_proba(X)[:, 1]
            else:
                pred = model.predict_proba(X)[:, 1]

            predictions.append(pred)
            weights.append(self.model_weights.get(name, 1.0))

        predictions = np.array(predictions)
        weights = np.array(weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: Input features.
            threshold: Classification threshold.

        Returns:
            Predicted binary labels.
        """
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def detect_drift(self, X: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in new data.

        Args:
            X: New data to check for drift.
            reference_data: Reference training data.

        Returns:
            Drift detection results.
        """
        if self.adversarial_validator is None:
            return {'drift_detected': False, 'message': 'No adversarial validator trained'}

        return self.adversarial_validator.predict_drift(X, reference_data)

    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual model contributions to ensemble prediction.

        Args:
            X: Input features.

        Returns:
            Dictionary mapping model names to their predictions.
        """
        contributions = {}

        for name, model in self.base_models.items():
            if name in self.calibrators:
                pred = self.calibrators[name].predict_proba(X)[:, 1]
            else:
                pred = model.predict_proba(X)[:, 1]

            contributions[name] = pred

        return contributions

    def save(self, filepath: str):
        """Save the ensemble model.

        Args:
            filepath: Path to save the model.
        """
        save_data = {
            'base_models': self.base_models,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'calibrators': self.calibrators,
            'adversarial_validator': self.adversarial_validator,
            'ensemble_method': self.ensemble_method,
            'calibrate_probabilities': self.calibrate_probabilities,
            'random_seed': self.random_seed
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Ensemble model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DriftAwareEnsemble':
        """Load a saved ensemble model.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded DriftAwareEnsemble instance.
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Create new instance
        ensemble = cls(
            ensemble_method=save_data['ensemble_method'],
            calibrate_probabilities=save_data['calibrate_probabilities'],
            random_seed=save_data['random_seed']
        )

        # Restore saved state
        ensemble.base_models = save_data['base_models']
        ensemble.model_weights = save_data['model_weights']
        ensemble.model_performance = save_data['model_performance']
        ensemble.calibrators = save_data['calibrators']
        ensemble.adversarial_validator = save_data['adversarial_validator']

        logger.info(f"Ensemble model loaded from {filepath}")
        return ensemble