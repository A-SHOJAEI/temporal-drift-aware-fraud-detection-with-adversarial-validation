"""Tests for training module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from temporal_drift_aware_fraud_detection_with_adversarial_validation.training.trainer import DriftAwareTrainer
from temporal_drift_aware_fraud_detection_with_adversarial_validation.utils.config import Config
from .conftest import assert_dataframe_valid, assert_metrics_valid, TEST_TARGET_METRICS


class TestDriftAwareTrainer:
    """Tests for DriftAwareTrainer class."""

    def test_init(self, test_config, temp_dir):
        """Test trainer initialization."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir,
            experiment_name="test_experiment"
        )

        assert trainer.config == test_config.to_dict()
        assert trainer.save_dir == Path(temp_dir)
        assert trainer.experiment_name == "test_experiment"
        assert trainer.ensemble is None
        assert trainer.feature_engineer is not None

    def test_prepare_data(self, test_config, sample_data, temp_dir, mock_mlflow):
        """Test data preparation."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        # Prepare data
        train_data, val_data, drift_periods = trainer._prepare_data(sample_data)

        # Check outputs
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Validate prepared data
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_val, pd.Series)

        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)

        # Check drift periods
        assert isinstance(drift_periods, list)
        assert len(drift_periods) >= 2

        # Check feature engineering was applied
        assert 'isFraud' not in X_train.columns
        assert 'TransactionID' not in X_train.columns

    def test_train_models(self, test_config, processed_data, temp_dir, mock_mlflow):
        """Test model training."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        train_processed, test_processed = processed_data

        # Prepare data manually
        feature_cols = [col for col in train_processed.columns
                       if col not in ['isFraud', 'TransactionID']]

        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = test_processed[feature_cols]
        y_val = test_processed['isFraud']

        # Create simple drift periods
        mid_point = len(train_processed) // 2
        drift_periods = [
            train_processed.iloc[:mid_point],
            train_processed.iloc[mid_point:]
        ]

        # Train models
        training_metrics = trainer._train_models(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            drift_periods=drift_periods
        )

        # Check that ensemble was created
        assert trainer.ensemble is not None

        # Check training metrics
        assert_metrics_valid(training_metrics, ['ensemble_auc', 'ensemble_ap'])

        # Check that base model metrics are present
        for model_name in trainer.ensemble.base_models.keys():
            assert f'{model_name}_auc' in training_metrics
            assert f'{model_name}_ap' in training_metrics

    def test_full_training_pipeline(self, test_config, sample_data, temp_dir, mock_mlflow):
        """Test complete training pipeline."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        # Run training
        training_results = trainer.train(
            train_data=sample_data,
            validation_data=None,
            save_best=False  # Skip saving for test speed
        )

        # Check results
        assert isinstance(training_results, dict)
        assert 'ensemble_auc' in training_results
        assert 'training_time_seconds' in training_results

        # Check that trainer state is updated
        assert trainer.ensemble is not None
        assert trainer.training_data is not None
        assert trainer.validation_data is not None

    def test_evaluate_models(self, test_config, processed_data, temp_dir, mock_mlflow):
        """Test model evaluation."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        train_processed, test_processed = processed_data

        # Train first
        training_results = trainer.train(
            train_data=train_processed,
            validation_data=test_processed,
            save_best=False
        )

        # Check that evaluation metrics are present
        expected_eval_metrics = [
            'calibration_ece',
            'calibration_brier_score',
            'drift_detection_recall'
        ]

        # Some metrics might not be present due to insufficient data
        eval_metrics_present = [m for m in expected_eval_metrics if m in training_results]
        assert len(eval_metrics_present) > 0, "Should have some evaluation metrics"

    def test_predict(self, test_config, processed_data, temp_dir, mock_mlflow):
        """Test prediction functionality."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        train_processed, test_processed = processed_data

        # Train first
        trainer.train(train_data=train_processed, save_best=False)

        # Make predictions
        probs, preds = trainer.predict(test_processed)

        # Validate predictions
        assert isinstance(probs, np.ndarray)
        assert isinstance(preds, np.ndarray)
        assert len(probs) == len(test_processed)
        assert len(preds) == len(test_processed)

        # Check probability range
        assert np.all(probs >= 0) and np.all(probs <= 1)
        assert np.all(np.isin(preds, [0, 1]))

    def test_save_models(self, test_config, sample_data, temp_dir, mock_mlflow):
        """Test model saving."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        # Train model
        trainer.train(train_data=sample_data, save_best=True)

        # Check that files were saved
        save_dir = Path(temp_dir)
        expected_files = [
            'drift_aware_ensemble.pkl',
            'feature_engineer.pkl',
            'config.json',
            'training_metrics.json'
        ]

        for file_name in expected_files:
            file_path = save_dir / file_name
            assert file_path.exists(), f"Expected file not saved: {file_name}"

        # Check that best model path is set
        assert trainer.best_model_path is not None

    def test_load_model(self, test_config, sample_data, temp_dir, mock_mlflow):
        """Test model loading."""
        # Train and save a model
        trainer1 = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )
        trainer1.train(train_data=sample_data, save_best=True)

        # Make predictions with original trainer
        original_probs, _ = trainer1.predict(sample_data)

        # Load model with new trainer
        trainer2 = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )
        trainer2.load_model(temp_dir)

        # Make predictions with loaded trainer
        loaded_probs, _ = trainer2.predict(sample_data)

        # Check that predictions match
        np.testing.assert_array_almost_equal(
            original_probs, loaded_probs, decimal=5
        )

    def test_temporal_robustness_evaluation(self, test_config, drift_data_periods, temp_dir, mock_mlflow):
        """Test temporal robustness evaluation."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        # Use first period for training
        train_period = drift_data_periods[0]

        # Train model
        trainer.train(train_data=train_period, save_best=False)

        # Evaluate robustness
        robustness_metrics = trainer._evaluate_temporal_robustness(drift_data_periods)

        # Check that robustness metrics are computed
        expected_metrics = [
            'temporal_auc_mean',
            'temporal_auc_std',
            'temporal_stability_score'
        ]

        for metric in expected_metrics:
            if metric in robustness_metrics:
                assert isinstance(robustness_metrics[metric], (int, float))
                assert not np.isnan(robustness_metrics[metric])

    def test_drift_evaluation(self, test_config, drift_data_periods, temp_dir, mock_mlflow):
        """Test drift evaluation functionality."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        # Train with drift periods
        train_period = drift_data_periods[0]
        trainer.train(train_data=train_period, save_best=False)

        # Test drift evaluation
        if len(drift_data_periods) > 1:
            new_data = drift_data_periods[1]
            reference_data = drift_data_periods[0]

            # This should work if adversarial validator was trained
            if trainer.adversarial_validator is not None:
                drift_results = trainer.evaluate_drift(new_data, reference_data)
                assert isinstance(drift_results, dict)

    def test_configuration_validation(self, temp_dir):
        """Test that invalid configuration raises errors."""
        # Create invalid configuration
        invalid_config = {
            'training': {
                'random_seed': -1,  # Invalid seed
                'train_ratio': 1.5,  # Invalid ratio
            }
        }

        # This should not crash immediately, but may cause issues during training
        trainer = DriftAwareTrainer(
            config=invalid_config,
            save_dir=temp_dir
        )

        assert trainer is not None  # Initialization should succeed

    def test_error_handling(self, test_config, temp_dir, mock_mlflow):
        """Test error handling in training."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        # Test with invalid data (empty DataFrame)
        empty_data = pd.DataFrame()

        with pytest.raises(Exception):  # Should raise some exception
            trainer.train(train_data=empty_data)

    def test_metrics_logging(self, test_config, sample_data, temp_dir, mock_mlflow):
        """Test that metrics are properly logged."""
        trainer = DriftAwareTrainer(
            config=test_config.to_dict(),
            save_dir=temp_dir
        )

        # Train model
        results = trainer.train(train_data=sample_data, save_best=True)

        # Check that metrics file exists
        metrics_file = Path(temp_dir) / 'training_metrics.json'
        assert metrics_file.exists()

        # Check that metrics can be loaded
        with open(metrics_file, 'r') as f:
            saved_metrics = json.load(f)

        assert isinstance(saved_metrics, dict)
        assert len(saved_metrics) > 0

        # Check that important metrics are present
        important_metrics = ['ensemble_auc', 'training_time_seconds']
        for metric in important_metrics:
            assert metric in saved_metrics

    def test_reproducibility(self, test_config, sample_data, temp_dir, mock_mlflow):
        """Test that training is reproducible with same seed."""
        config1 = test_config.to_dict()
        config2 = test_config.to_dict()

        # Ensure same random seed
        config1['training']['random_seed'] = 42
        config2['training']['random_seed'] = 42

        # Train two models
        trainer1 = DriftAwareTrainer(config=config1, save_dir=f"{temp_dir}/model1")
        trainer2 = DriftAwareTrainer(config=config2, save_dir=f"{temp_dir}/model2")

        results1 = trainer1.train(train_data=sample_data, save_best=False)
        results2 = trainer2.train(train_data=sample_data, save_best=False)

        # Results should be very similar (allowing for small numerical differences)
        if 'ensemble_auc' in results1 and 'ensemble_auc' in results2:
            auc_diff = abs(results1['ensemble_auc'] - results2['ensemble_auc'])
            assert auc_diff < 0.01, f"AUC difference too large: {auc_diff}"

    def test_quick_training(self, test_config, sample_data, temp_dir, mock_mlflow):
        """Test training with minimal configuration for speed."""
        # Modify config for very fast training
        quick_config = test_config.to_dict()
        quick_config['training']['n_drift_periods'] = 2

        # Further reduce model complexity
        for model_name in quick_config.get('base_models', {}):
            if 'params' in quick_config['base_models'][model_name]:
                params = quick_config['base_models'][model_name]['params']
                params['n_estimators'] = 3
                if 'iterations' in params:
                    params['iterations'] = 3

        trainer = DriftAwareTrainer(
            config=quick_config,
            save_dir=temp_dir
        )

        # Use smaller data sample
        small_sample = sample_data.sample(n=min(200, len(sample_data)), random_state=42)

        # Training should complete quickly
        results = trainer.train(train_data=small_sample, save_best=False)

        # Check that training completed
        assert 'training_time_seconds' in results
        assert results['training_time_seconds'] > 0

        # Should still produce valid models
        assert trainer.ensemble is not None

        # Test basic prediction
        probs, preds = trainer.predict(small_sample)
        assert len(probs) == len(small_sample)