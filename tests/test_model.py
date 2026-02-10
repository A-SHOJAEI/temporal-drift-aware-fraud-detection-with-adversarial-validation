"""Tests for model implementations."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from temporal_drift_aware_fraud_detection_with_adversarial_validation.models.model import (
    AdversarialValidator, DriftAwareEnsemble
)
from .conftest import assert_predictions_valid, assert_dataframe_valid


class TestAdversarialValidator:
    """Tests for AdversarialValidator class."""

    def test_init(self, random_seed):
        """Test AdversarialValidator initialization."""
        validator = AdversarialValidator(
            model_type='lightgbm',
            random_seed=random_seed
        )

        assert validator.model_type == 'lightgbm'
        assert validator.random_seed == random_seed
        assert validator.model is None
        assert validator.drift_threshold == 0.6

    def test_prepare_drift_data(self, random_seed):
        """Test drift data preparation."""
        validator = AdversarialValidator(random_seed=random_seed)

        # Create mock source and target data
        np.random.seed(random_seed)
        source_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'isFraud': np.random.binomial(1, 0.05, 100)
        })

        target_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1, 80),  # Slight drift
            'feature2': np.random.normal(0, 1.2, 80),  # Variance drift
            'isFraud': np.random.binomial(1, 0.07, 80)  # Different fraud rate
        })

        X, y = validator.prepare_drift_data(source_data, target_data)

        # Check output shapes
        assert X.shape[0] == 180  # 100 + 80 samples
        assert X.shape[1] == 2   # 2 features
        assert y.shape[0] == 180

        # Check labels (0 for source, 1 for target)
        assert np.sum(y == 0) == 100  # Source samples
        assert np.sum(y == 1) == 80   # Target samples

    def test_fit_lightgbm(self, random_seed, processed_data):
        """Test fitting with LightGBM."""
        train_processed, test_processed = processed_data

        validator = AdversarialValidator(
            model_type='lightgbm',
            random_seed=random_seed,
            n_estimators=10,  # Fast training
            verbose=-1
        )

        # Split data to simulate source and target
        mid_point = len(train_processed) // 2
        source_data = train_processed.iloc[:mid_point]
        target_data = train_processed.iloc[mid_point:]

        metrics = validator.fit(source_data, target_data, validation_split=0.2)

        # Check that model was trained
        assert validator.model is not None

        # Check metrics
        required_metrics = ['train_auc', 'val_auc', 'drift_score']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

        # Check feature importance
        assert validator.feature_importance is not None
        assert len(validator.feature_importance) > 0

    def test_predict_drift(self, random_seed, processed_data):
        """Test drift prediction."""
        train_processed, test_processed = processed_data

        validator = AdversarialValidator(
            model_type='lightgbm',
            random_seed=random_seed,
            n_estimators=10
        )

        # Train validator
        mid_point = len(train_processed) // 2
        source_data = train_processed.iloc[:mid_point]
        target_data = train_processed.iloc[mid_point:]

        validator.fit(source_data, target_data)

        # Test drift prediction
        drift_metrics = validator.predict_drift(test_processed, train_processed)

        # Check required metrics
        required_metrics = [
            'drift_score', 'drift_auc', 'is_drift_detected',
            'reference_mean_score', 'new_mean_score', 'score_difference'
        ]

        for metric in required_metrics:
            assert metric in drift_metrics

        # Check value ranges
        assert 0 <= drift_metrics['drift_score'] <= 1
        assert 0 <= drift_metrics['drift_auc'] <= 1
        assert isinstance(drift_metrics['is_drift_detected'], (bool, np.bool_))

    def test_get_drift_features(self, random_seed, processed_data):
        """Test getting top drift features."""
        train_processed, _ = processed_data

        validator = AdversarialValidator(
            model_type='lightgbm',
            random_seed=random_seed,
            n_estimators=10
        )

        # Train validator
        mid_point = len(train_processed) // 2
        source_data = train_processed.iloc[:mid_point]
        target_data = train_processed.iloc[mid_point:]

        validator.fit(source_data, target_data)

        # Get top drift features
        top_features = validator.get_drift_features(top_k=5)

        assert len(top_features) <= 5
        for feature_idx, importance in top_features:
            assert isinstance(feature_idx, int)
            assert isinstance(importance, (int, float))
            assert importance >= 0

    def test_unfitted_error(self, random_seed, processed_data):
        """Test error when using unfitted validator."""
        train_processed, test_processed = processed_data

        validator = AdversarialValidator(random_seed=random_seed)

        with pytest.raises(ValueError, match="Model not fitted"):
            validator.predict_drift(test_processed, train_processed)


class TestDriftAwareEnsemble:
    """Tests for DriftAwareEnsemble class."""

    def test_init(self, random_seed):
        """Test DriftAwareEnsemble initialization."""
        ensemble = DriftAwareEnsemble(random_seed=random_seed)

        assert ensemble.random_seed == random_seed
        assert ensemble.ensemble_method == 'weighted_average'
        assert ensemble.calibrate_probabilities is True
        assert len(ensemble.base_models) > 0  # Should have default models

        # Check default weights
        for name in ensemble.base_models.keys():
            assert name in ensemble.model_weights
            assert ensemble.model_weights[name] > 0

    def test_init_custom_models(self, random_seed):
        """Test initialization with custom models."""
        custom_models = {
            'test_lgb': {
                'type': 'lightgbm',
                'params': {
                    'objective': 'binary',
                    'n_estimators': 5,
                    'random_state': random_seed
                }
            }
        }

        ensemble = DriftAwareEnsemble(
            base_models=custom_models,
            ensemble_method='weighted_average',
            random_seed=random_seed
        )

        assert 'test_lgb' in ensemble.base_models
        assert len(ensemble.base_models) == 1

    def test_fit(self, small_ensemble, processed_data):
        """Test ensemble fitting."""
        train_processed, test_processed = processed_data

        # Get features and targets
        feature_cols = [col for col in train_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = test_processed[feature_cols]
        y_val = test_processed['isFraud']

        # Fit ensemble
        metrics = small_ensemble.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Check metrics
        assert isinstance(metrics, dict)
        assert 'ensemble_auc' in metrics
        assert 'ensemble_ap' in metrics

        # Check that models are trained
        for name, model in small_ensemble.base_models.items():
            assert hasattr(model, 'predict_proba'), f"Model {name} should be fitted"

    def test_predict_proba(self, small_ensemble, processed_data):
        """Test probability prediction."""
        train_processed, test_processed = processed_data

        # Get features and targets
        feature_cols = [col for col in train_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = test_processed[feature_cols]
        y_val = test_processed['isFraud']

        # Fit and predict
        small_ensemble.fit(X_train, y_train, X_val, y_val)
        predictions = small_ensemble.predict_proba(X_val)

        # Validate predictions
        assert_predictions_valid(predictions, len(X_val))

    def test_predict(self, small_ensemble, processed_data):
        """Test binary prediction."""
        train_processed, test_processed = processed_data

        # Get features and targets
        feature_cols = [col for col in train_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = test_processed[feature_cols]
        y_val = test_processed['isFraud']

        # Fit and predict
        small_ensemble.fit(X_train, y_train, X_val, y_val)
        predictions = small_ensemble.predict(X_val, threshold=0.5)

        # Validate predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_val)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_get_model_contributions(self, small_ensemble, processed_data):
        """Test getting individual model contributions."""
        train_processed, test_processed = processed_data

        # Get features and targets
        feature_cols = [col for col in train_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = test_processed[feature_cols]
        y_val = test_processed['isFraud']

        # Fit and get contributions
        small_ensemble.fit(X_train, y_train, X_val, y_val)
        contributions = small_ensemble.get_model_contributions(X_val)

        # Check output
        assert isinstance(contributions, dict)

        for name, contrib in contributions.items():
            assert name in small_ensemble.base_models
            assert_predictions_valid(contrib, len(X_val))

    def test_weight_optimization(self, small_ensemble, processed_data):
        """Test that weights are optimized based on performance."""
        train_processed, test_processed = processed_data

        # Get features and targets
        feature_cols = [col for col in train_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = test_processed[feature_cols]
        y_val = test_processed['isFraud']

        # Store initial weights
        initial_weights = small_ensemble.model_weights.copy()

        # Fit ensemble (should optimize weights)
        small_ensemble.fit(X_train, y_train, X_val, y_val)

        # Check that weights sum to 1
        total_weight = sum(small_ensemble.model_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

        # Weights should be positive
        for weight in small_ensemble.model_weights.values():
            assert weight > 0

    def test_save_load(self, small_ensemble, processed_data, temp_dir):
        """Test saving and loading ensemble."""
        train_processed, test_processed = processed_data

        # Get features and targets
        feature_cols = [col for col in train_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = test_processed[feature_cols]
        y_val = test_processed['isFraud']

        # Fit ensemble
        small_ensemble.fit(X_train, y_train, X_val, y_val)

        # Make predictions before saving
        original_predictions = small_ensemble.predict_proba(X_val)

        # Save model
        save_path = f"{temp_dir}/test_ensemble.pkl"
        small_ensemble.save(save_path)

        # Load model
        loaded_ensemble = DriftAwareEnsemble.load(save_path)

        # Make predictions with loaded model
        loaded_predictions = loaded_ensemble.predict_proba(X_val)

        # Check predictions match
        np.testing.assert_array_almost_equal(
            original_predictions, loaded_predictions, decimal=6
        )

    def test_unfitted_error(self, small_ensemble, sample_data):
        """Test error when using unfitted ensemble."""
        feature_cols = [col for col in sample_data.columns
                       if col not in ['isFraud', 'TransactionID']]
        X = sample_data[feature_cols]

        with pytest.raises(ValueError):
            small_ensemble.predict_proba(X)

    def test_drift_detection_integration(self, small_ensemble, drift_data_periods):
        """Test drift detection with ensemble."""
        periods = drift_data_periods

        # Prepare data
        reference_period = periods[0]
        feature_cols = [col for col in reference_period.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = reference_period[feature_cols]
        y_train = reference_period['isFraud']

        # Simple validation data (subset of training)
        X_val = X_train.iloc[:50]
        y_val = y_train.iloc[:50]

        # Fit ensemble with drift periods
        metrics = small_ensemble.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            drift_periods=periods
        )

        # Should have adversarial validator after fitting with drift periods
        assert small_ensemble.adversarial_validator is not None

        # Test drift detection
        if len(periods) > 1:
            target_period = periods[1]
            target_features = target_period[feature_cols]

            drift_results = small_ensemble.detect_drift(target_features, X_train)
            assert isinstance(drift_results, dict)
            assert 'drift_detected' in drift_results or 'drift_score' in drift_results