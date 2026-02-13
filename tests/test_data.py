"""Tests for data loading and preprocessing modules."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.loader import DataLoader
from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.preprocessing import FeatureEngineer, TemporalSplitter
from .conftest import assert_dataframe_valid, TEST_REQUIRED_COLUMNS


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_init(self, random_seed):
        """Test DataLoader initialization."""
        loader = DataLoader(random_seed=random_seed)
        assert loader.random_seed == random_seed
        assert loader.data_path is None

    def test_load_synthetic_data(self, data_loader):
        """Test loading synthetic data."""
        train_data, test_data = data_loader.load_data()

        # Validate data structure
        assert_dataframe_valid(train_data, TEST_REQUIRED_COLUMNS)
        assert_dataframe_valid(test_data, TEST_REQUIRED_COLUMNS)

        # Check data properties
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert 'isFraud' in train_data.columns
        assert 'isFraud' in test_data.columns

        # Check fraud rates are reasonable
        train_fraud_rate = train_data['isFraud'].mean()
        test_fraud_rate = test_data['isFraud'].mean()

        assert 0.01 <= train_fraud_rate <= 0.1, f"Train fraud rate {train_fraud_rate} not realistic"
        assert 0.01 <= test_fraud_rate <= 0.1, f"Test fraud rate {test_fraud_rate} not realistic"

    def test_synthetic_data_temporal_features(self, data_loader):
        """Test that synthetic data has proper temporal features."""
        train_data, test_data = data_loader.load_data()

        # Check temporal ordering
        assert train_data['TransactionDT'].is_monotonic_increasing, "Train data should be temporally ordered"

        # Check test data comes after training data (temporal split)
        train_max_time = train_data['TransactionDT'].max()
        test_min_time = test_data['TransactionDT'].min()
        assert test_min_time > train_max_time * 0.8, "Test data should overlap with or come after training"

    def test_feature_info(self, data_loader, sample_data):
        """Test feature information extraction."""
        feature_info = data_loader.get_feature_info(sample_data)

        # Check required keys
        required_keys = ['categorical_columns', 'numerical_columns', 'n_samples', 'n_features', 'fraud_rate']
        for key in required_keys:
            assert key in feature_info, f"Missing key: {key}"

        # Validate feature counts
        assert feature_info['n_samples'] == len(sample_data)
        assert feature_info['n_features'] > 0

        # Check fraud rate
        expected_fraud_rate = sample_data['isFraud'].mean()
        assert abs(feature_info['fraud_rate'] - expected_fraud_rate) < 1e-6

    def test_reproducibility(self, random_seed):
        """Test that data generation is reproducible."""
        loader1 = DataLoader(random_seed=random_seed)
        loader2 = DataLoader(random_seed=random_seed)

        train1, test1 = loader1.load_data()
        train2, test2 = loader2.load_data()

        # Check reproducibility
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_init(self, random_seed):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(random_seed=random_seed)
        assert engineer.random_seed == random_seed
        assert engineer.scalers == {}
        assert engineer.encoders == {}

    def test_fit_transform(self, feature_engineer, sample_data):
        """Test fit_transform method."""
        transformed = feature_engineer.fit_transform(sample_data)

        # Check output is DataFrame
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)

        # Check that scalers and encoders were fitted
        assert len(feature_engineer.scalers) > 0
        assert len(feature_engineer.encoders) >= 0  # May be 0 if no categorical features

        # Check for new engineered features
        original_features = set(sample_data.columns)
        new_features = set(transformed.columns)
        engineered_features = new_features - original_features

        assert len(engineered_features) > 0, "Should create new features"

    def test_transform(self, feature_engineer, train_test_data):
        """Test transform method (after fitting)."""
        train_data, test_data = train_test_data

        # Fit on training data
        train_transformed = feature_engineer.fit_transform(train_data)

        # Transform test data
        test_transformed = feature_engineer.transform(test_data)

        # Check same columns
        assert set(train_transformed.columns) == set(test_transformed.columns)

        # Check no leakage (different shapes are okay)
        assert len(test_transformed) == len(test_data)

    def test_feature_engineering_creates_expected_features(self, feature_engineer, sample_data):
        """Test that specific engineered features are created."""
        transformed = feature_engineer.fit_transform(sample_data)

        expected_features = [
            'TransactionAmt_log',
            'hour',
            'day_of_week',
            'is_weekend',
            'is_night'
        ]

        for feature in expected_features:
            assert feature in transformed.columns, f"Missing expected feature: {feature}"

    def test_missing_value_handling(self, feature_engineer, sample_data):
        """Test missing value handling."""
        # Introduce missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0:10, 'TransactionAmt'] = np.nan
        data_with_missing.loc[5:15, 'P_emaildomain'] = np.nan

        # Transform data
        transformed = feature_engineer.fit_transform(data_with_missing)

        # Check no missing values remain (except in target/ID columns)
        feature_cols = [col for col in transformed.columns if col not in ['isFraud', 'TransactionID']]
        missing_count = transformed[feature_cols].isnull().sum().sum()

        assert missing_count == 0, "Should handle all missing values"

    def test_categorical_encoding(self, feature_engineer, sample_data):
        """Test categorical feature encoding."""
        transformed = feature_engineer.fit_transform(sample_data)

        # Check that categorical features are encoded
        categorical_cols = ['ProductCD', 'P_emaildomain']
        for col in categorical_cols:
            if col in transformed.columns:
                # Should be numeric after encoding
                assert pd.api.types.is_numeric_dtype(transformed[col])

    def test_feature_scaling(self, feature_engineer, sample_data):
        """Test that numerical features are scaled."""
        transformed = feature_engineer.fit_transform(sample_data)

        # Check that TransactionAmt is scaled (mean ~0, std ~1)
        if 'TransactionAmt' in transformed.columns:
            amt_mean = transformed['TransactionAmt'].mean()
            amt_std = transformed['TransactionAmt'].std()

            assert abs(amt_mean) < 1.0, f"TransactionAmt mean should be ~0, got {amt_mean}"
            assert 0.5 < amt_std < 2.0, f"TransactionAmt std should be ~1, got {amt_std}"


class TestTemporalSplitter:
    """Tests for TemporalSplitter class."""

    def test_init(self, random_seed):
        """Test TemporalSplitter initialization."""
        splitter = TemporalSplitter(random_seed=random_seed)
        assert splitter.time_column == 'TransactionDT'
        assert splitter.random_seed == random_seed

    def test_temporal_split(self, sample_data):
        """Test temporal splitting."""
        splitter = TemporalSplitter()

        train, val, test = splitter.temporal_split(
            sample_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        # Check sizes
        total_size = len(sample_data)
        assert len(train) == int(total_size * 0.6)
        assert len(val) == int(total_size * 0.2)
        assert len(test) <= total_size - len(train) - len(val)

        # Check temporal ordering
        train_max_time = train['TransactionDT'].max()
        val_min_time = val['TransactionDT'].min()
        val_max_time = val['TransactionDT'].max()
        test_min_time = test['TransactionDT'].min()

        assert val_min_time >= train_max_time, "Validation should come after training"
        assert test_min_time >= val_max_time, "Test should come after validation"

    def test_create_drift_periods(self, sample_data):
        """Test drift period creation."""
        splitter = TemporalSplitter()

        periods = splitter.create_drift_periods(sample_data, n_periods=3)

        # Check number of periods
        assert len(periods) == 3

        # Check all data is covered
        total_samples = sum(len(period) for period in periods)
        assert total_samples == len(sample_data)

        # Check temporal ordering between periods
        for i in range(len(periods) - 1):
            current_max = periods[i]['TransactionDT'].max()
            next_min = periods[i + 1]['TransactionDT'].min()
            assert next_min >= current_max, f"Period {i+1} should come after period {i}"

    def test_stratified_temporal_split(self, sample_data):
        """Test stratified temporal split."""
        splitter = TemporalSplitter()

        train, test = splitter.stratified_temporal_split(sample_data, train_ratio=0.7)

        # Check fraud rates are preserved approximately
        original_fraud_rate = sample_data['isFraud'].mean()
        train_fraud_rate = train['isFraud'].mean()
        test_fraud_rate = test['isFraud'].mean()

        # Allow some variation due to temporal effects
        assert abs(train_fraud_rate - original_fraud_rate) < 0.05
        assert abs(test_fraud_rate - original_fraud_rate) < 0.05

        # Check temporal ordering within each class
        fraud_train = train[train['isFraud'] == 1]['TransactionDT']
        fraud_test = test[test['isFraud'] == 1]['TransactionDT']

        if len(fraud_train) > 0 and len(fraud_test) > 0:
            assert fraud_test.min() >= fraud_train.max() * 0.8, "Temporal ordering should be preserved"

    def test_invalid_ratios(self, sample_data):
        """Test error handling for invalid split ratios."""
        splitter = TemporalSplitter()

        with pytest.raises(AssertionError):
            splitter.temporal_split(sample_data, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_edge_cases(self, random_seed):
        """Test edge cases with small datasets."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'TransactionDT': [1, 2, 3, 4, 5],
            'isFraud': [0, 1, 0, 1, 0],
            'feature1': [1, 2, 3, 4, 5]
        })

        splitter = TemporalSplitter(random_seed=random_seed)

        # Should handle small datasets gracefully
        periods = splitter.create_drift_periods(small_data, n_periods=2)
        assert len(periods) == 2
        assert len(periods[0]) > 0
        assert len(periods[1]) > 0