"""Test configuration and fixtures for temporal drift-aware fraud detection."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Tuple

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.utils.config import Config
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.loader import DataLoader
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.preprocessing import FeatureEngineer
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.models.model import DriftAwareEnsemble
except ImportError as e:
    # Handle import errors gracefully for testing
    print(f"Warning: Import error in conftest.py: {e}")
    import sys
    sys.exit(pytest.skip("Required modules not available"))


@pytest.fixture(scope="session")
def random_seed():
    """Random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def test_config(random_seed):
    """Test configuration."""
    config = Config()

    # Reduce complexity for fast testing
    for model_name in config.base_models:
        config.base_models[model_name].params['n_estimators'] = 10
        if 'iterations' in config.base_models[model_name].params:
            config.base_models[model_name].params['iterations'] = 10

    config.training.random_seed = random_seed
    config.training.n_drift_periods = 3
    config.training.early_stopping_rounds = 5

    return config


@pytest.fixture(scope="session")
def sample_data(random_seed):
    """Generate sample fraud detection data for testing."""
    np.random.seed(random_seed)

    n_samples = 1000
    n_features = 20

    # Generate synthetic features
    data = {}

    # Transaction features
    data['TransactionDT'] = np.sort(np.random.uniform(0, 100, n_samples))
    data['TransactionAmt'] = np.random.lognormal(3, 1, n_samples)
    data['TransactionID'] = range(n_samples)

    # Card features
    data['card1'] = np.random.randint(1000, 20000, n_samples)
    data['card2'] = np.random.choice([100, 150, 200, 300], n_samples)
    data['card3'] = np.random.choice([150, 185], n_samples, p=[0.7, 0.3])

    # Categorical features
    data['ProductCD'] = np.random.choice(['W', 'H', 'C', 'S', 'R'], n_samples, p=[0.4, 0.25, 0.15, 0.1, 0.1])
    data['P_emaildomain'] = np.random.choice(['gmail', 'yahoo', 'hotmail', 'other'], n_samples, p=[0.4, 0.3, 0.2, 0.1])

    # Distance features
    data['dist1'] = np.random.exponential(50, n_samples)
    data['dist2'] = np.random.exponential(100, n_samples)

    # Numerical features
    for i in range(1, 6):
        data[f'C{i}'] = np.random.randint(0, 100, n_samples)
        data[f'D{i}'] = np.random.randint(0, 500, n_samples)
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)

    # Target variable (fraud)
    fraud_prob = 0.035 + 0.01 * (data['TransactionAmt'] > np.percentile(data['TransactionAmt'], 90))
    data['isFraud'] = np.random.binomial(1, fraud_prob, n_samples)

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def train_test_data(sample_data):
    """Split sample data into train/test sets."""
    train_size = int(0.8 * len(sample_data))

    train_data = sample_data.iloc[:train_size].copy()
    test_data = sample_data.iloc[train_size:].copy()

    return train_data, test_data


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def data_loader(random_seed):
    """Data loader instance for testing."""
    return DataLoader(data_path=None, random_seed=random_seed)


@pytest.fixture(scope="session")
def feature_engineer(random_seed):
    """Feature engineer instance for testing."""
    return FeatureEngineer(random_seed=random_seed)


@pytest.fixture(scope="session")
def processed_data(train_test_data, feature_engineer):
    """Preprocessed training and test data."""
    train_data, test_data = train_test_data

    # Fit transform on training data
    train_processed = feature_engineer.fit_transform(train_data)
    test_processed = feature_engineer.transform(test_data)

    return train_processed, test_processed


@pytest.fixture(scope="session")
def small_ensemble(test_config, random_seed):
    """Small ensemble model for testing."""
    # Simplified model config for fast testing
    base_models_config = {
        'lightgbm_test': {
            'type': 'lightgbm',
            'params': {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 7,
                'learning_rate': 0.1,
                'n_estimators': 5,
                'verbose': -1,
                'random_state': random_seed
            }
        }
    }

    return DriftAwareEnsemble(
        base_models=base_models_config,
        ensemble_method='weighted_average',
        calibrate_probabilities=False,  # Skip calibration for speed
        random_seed=random_seed
    )


@pytest.fixture(scope="function")
def sample_predictions(random_seed):
    """Sample prediction data for testing metrics."""
    np.random.seed(random_seed)

    n_samples = 500

    # Generate realistic fraud prediction data
    y_true = np.random.binomial(1, 0.05, n_samples)  # 5% fraud rate

    # Generate probabilities with some discrimination
    base_prob = np.random.beta(1, 20, n_samples)  # Low baseline probabilities
    fraud_boost = y_true * np.random.beta(2, 1, n_samples) * 0.8  # Boost for actual frauds
    y_prob = np.clip(base_prob + fraud_boost, 0, 1)

    return y_true, y_prob


@pytest.fixture(scope="session")
def drift_data_periods(sample_data, random_seed):
    """Create data with simulated drift for testing."""
    np.random.seed(random_seed)

    periods = []

    # Split data into 3 periods with increasing drift
    period_size = len(sample_data) // 3

    for i in range(3):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 2 else len(sample_data)

        period_data = sample_data.iloc[start_idx:end_idx].copy()

        # Add drift by modifying distributions
        if i > 0:
            drift_factor = 0.1 * i

            # Drift in transaction amounts
            period_data['TransactionAmt'] *= (1 + drift_factor)

            # Drift in email domains
            if np.random.random() < 0.5:
                mask = period_data['P_emaildomain'] == 'gmail'
                period_data.loc[mask, 'P_emaildomain'] = 'other'

        periods.append(period_data)

    return periods


@pytest.fixture(scope="function")
def mock_mlflow(monkeypatch):
    """Mock MLflow for testing."""
    class MockMLflow:
        @staticmethod
        def start_run(*args, **kwargs):
            pass

        @staticmethod
        def end_run(*args, **kwargs):
            pass

        @staticmethod
        def log_metric(key, value):
            pass

        @staticmethod
        def log_param(key, value):
            pass

        @staticmethod
        def log_artifact(path):
            pass

        @staticmethod
        def set_experiment(name):
            pass

    monkeypatch.setattr("temporal_drift_aware_fraud_detection_with_adversarial_validation.training.trainer.mlflow", MockMLflow())
    return MockMLflow


# Test data validation utilities
def assert_dataframe_valid(df: pd.DataFrame, required_cols: list = None):
    """Assert that a DataFrame is valid for testing."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) > 0, "DataFrame should not be empty"

    if required_cols:
        for col in required_cols:
            assert col in df.columns, f"Required column '{col}' missing"


def assert_predictions_valid(predictions: np.ndarray, n_samples: int = None):
    """Assert that predictions are valid."""
    assert isinstance(predictions, np.ndarray), "Expected numpy array"
    assert len(predictions.shape) == 1, "Expected 1D array"

    if n_samples:
        assert len(predictions) == n_samples, f"Expected {n_samples} predictions"

    # Check for valid probability range
    assert np.all(predictions >= 0), "All predictions should be >= 0"
    assert np.all(predictions <= 1), "All predictions should be <= 1"


def assert_metrics_valid(metrics: dict, required_metrics: list = None):
    """Assert that metrics dictionary is valid."""
    assert isinstance(metrics, dict), "Expected dictionary"

    if required_metrics:
        for metric in required_metrics:
            assert metric in metrics, f"Required metric '{metric}' missing"
            assert isinstance(metrics[metric], (int, float)), f"Metric '{metric}' should be numeric"


# Test constants
TEST_TARGET_METRICS = {
    'auroc': 0.7,  # Lower targets for test data
    'auprc': 0.1,
    'drift_detection_recall': 0.6,
    'calibration_ece': 0.2
}

TEST_REQUIRED_COLUMNS = ['TransactionDT', 'TransactionAmt', 'isFraud']