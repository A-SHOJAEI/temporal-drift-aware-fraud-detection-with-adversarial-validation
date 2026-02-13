"""Data loading utilities for IEEE-CIS Fraud Detection dataset."""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare IEEE-CIS Fraud Detection dataset for training.

    This class handles loading the IEEE-CIS fraud detection data, which contains
    transaction data with temporal information crucial for drift detection.
    """

    def __init__(self, data_path: Optional[str] = None, random_seed: int = 42):
        """Initialize data loader.

        Args:
            data_path: Path to data directory. If None, generates synthetic data.
            random_seed: Random seed for reproducibility.
        """
        self.data_path = Path(data_path) if data_path else None
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and testing data.

        Returns:
            Tuple of (train_df, test_df) DataFrames.
        """
        if self.data_path and self.data_path.exists():
            return self._load_ieee_cis_data()
        else:
            logger.warning("IEEE-CIS data not found. Generating synthetic data for demonstration.")
            return self._generate_synthetic_data()

    def _load_ieee_cis_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load actual IEEE-CIS dataset.

        Returns:
            Tuple of (train_df, test_df).
        """
        logger.info(f"Loading IEEE-CIS data from {self.data_path}")

        # Load transaction data
        train_transaction = pd.read_csv(self.data_path / "train_transaction.csv")
        test_transaction = pd.read_csv(self.data_path / "test_transaction.csv")

        # Load identity data if available
        try:
            train_identity = pd.read_csv(self.data_path / "train_identity.csv")
            test_identity = pd.read_csv(self.data_path / "test_identity.csv")

            # Merge transaction and identity data
            train_df = train_transaction.merge(train_identity, on='TransactionID', how='left')
            test_df = test_transaction.merge(test_identity, on='TransactionID', how='left')
        except FileNotFoundError:
            logger.warning("Identity data not found. Using transaction data only.")
            train_df = train_transaction
            test_df = test_transaction

        # Add temporal features for drift detection
        train_df = self._add_temporal_features(train_df)
        test_df = self._add_temporal_features(test_df)

        logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")
        return train_df, test_df

    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic fraud detection data with temporal drift.

        Returns:
            Tuple of (train_df, test_df).
        """
        logger.info("Generating synthetic IEEE-CIS-like fraud detection data")

        n_train = 10000
        n_test = 5000

        # Generate temporal features (crucial for drift detection)
        train_timestamps = np.sort(np.random.uniform(0, 100, n_train))
        test_timestamps = np.sort(np.random.uniform(80, 180, n_test))  # Temporal shift

        # Generate transaction features
        train_data = self._generate_transaction_features(n_train, train_timestamps, is_train=True)
        test_data = self._generate_transaction_features(n_test, test_timestamps, is_train=False)

        # Add target variable (fraud indicator)
        train_data['isFraud'] = self._generate_fraud_labels(train_data, train_timestamps)
        test_data['isFraud'] = self._generate_fraud_labels(test_data, test_timestamps)

        logger.info(f"Generated {n_train} training and {n_test} test samples")
        logger.info(f"Fraud rate - Train: {train_data['isFraud'].mean():.3f}, Test: {test_data['isFraud'].mean():.3f}")

        return train_data, test_data

    def _generate_transaction_features(self, n_samples: int, timestamps: np.ndarray, is_train: bool) -> pd.DataFrame:
        """Generate synthetic transaction features with temporal drift.

        Args:
            n_samples: Number of samples to generate.
            timestamps: Timestamp array.
            is_train: Whether this is training data.

        Returns:
            DataFrame with transaction features.
        """
        # Base drift factor - increases over time to simulate distribution shift
        drift_factor = timestamps / 100.0

        # Transaction amount features
        transaction_amt = np.random.lognormal(3, 1, n_samples)
        transaction_amt *= (1 + 0.3 * drift_factor)  # Inflation over time

        # Card features with temporal drift
        card1 = np.random.randint(1000, 20000, n_samples)
        card2 = np.random.choice([100, 150, 200, 300], n_samples)
        card3 = np.random.choice([150, 185], n_samples, p=[0.7, 0.3])

        # Product and merchant features
        product_cd = np.random.choice(['W', 'H', 'C', 'S', 'R'], n_samples, p=[0.4, 0.25, 0.15, 0.1, 0.1])

        # Distance features (geographic drift)
        dist1 = np.random.exponential(50, n_samples)
        dist2 = np.random.exponential(100, n_samples)

        # Email domain features (evolving over time)
        email_domains = ['gmail', 'yahoo', 'hotmail', 'other']
        if is_train:
            domain_probs = [0.4, 0.3, 0.2, 0.1]
        else:
            domain_probs = [0.35, 0.25, 0.15, 0.25]  # Shift towards 'other'

        email_domain = np.random.choice(email_domains, n_samples, p=domain_probs)

        # Device info features
        device_type = np.random.choice(['desktop', 'mobile'], n_samples, p=[0.6, 0.4])
        device_info = np.random.choice(['Windows', 'iOS', 'MacOS', 'Android'], n_samples)

        # Time-based features
        hour = (timestamps % 24).astype(int)
        day_of_week = ((timestamps // 24) % 7).astype(int)

        return pd.DataFrame({
            'TransactionDT': timestamps,
            'TransactionAmt': transaction_amt,
            'card1': card1,
            'card2': card2,
            'card3': card3,
            'ProductCD': product_cd,
            'dist1': dist1,
            'dist2': dist2,
            'P_emaildomain': email_domain,
            'DeviceType': device_type,
            'DeviceInfo': device_info,
            'hour': hour,
            'day_of_week': day_of_week,
            # Additional numerical features
            'C1': np.random.randint(0, 5000, n_samples),
            'C2': np.random.randint(0, 100, n_samples),
            'C3': np.random.randint(0, 50, n_samples),
            'C4': np.random.randint(0, 10, n_samples),
            'D1': np.random.randint(0, 1000, n_samples),
            'D2': np.random.randint(0, 500, n_samples),
            'D3': np.random.randint(0, 200, n_samples),
            'V1': np.random.normal(0, 1, n_samples),
            'V2': np.random.normal(0, 2, n_samples),
            'V3': np.random.normal(1, 0.5, n_samples),
            'V4': np.random.exponential(2, n_samples),
        })

    def _generate_fraud_labels(self, data: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
        """Generate fraud labels with realistic patterns.

        Args:
            data: Feature DataFrame.
            timestamps: Timestamp array.

        Returns:
            Binary fraud labels.
        """
        n_samples = len(data)
        base_fraud_rate = 0.035  # ~3.5% base fraud rate

        # Fraud probability based on features
        fraud_prob = np.full(n_samples, base_fraud_rate)

        # Higher fraud probability for:
        # - Large transactions
        fraud_prob += 0.02 * (data['TransactionAmt'] > data['TransactionAmt'].quantile(0.9))

        # - Night hours
        fraud_prob += 0.01 * ((data['hour'] < 6) | (data['hour'] > 22))

        # - Certain device types
        fraud_prob += 0.015 * (data['DeviceType'] == 'mobile')

        # - High distance transactions
        fraud_prob += 0.01 * (data['dist1'] > data['dist1'].quantile(0.8))

        # - Temporal drift effects (fraud patterns evolve)
        drift_factor = timestamps / 100.0
        fraud_prob += 0.01 * drift_factor  # Slight increase in fraud over time

        # Generate binary labels
        return np.random.binomial(1, np.clip(fraud_prob, 0, 1), n_samples)

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for drift detection.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with additional temporal features.
        """
        df = df.copy()

        # Extract time-based features if not present
        if 'hour' not in df.columns and 'TransactionDT' in df.columns:
            df['hour'] = (df['TransactionDT'] % (24 * 3600)) // 3600
            df['day_of_week'] = (df['TransactionDT'] // (24 * 3600)) % 7

        # Add time period for drift analysis
        if 'TransactionDT' in df.columns:
            df['time_period'] = pd.cut(df['TransactionDT'],
                                     bins=10,
                                     labels=False)

        return df

    def get_feature_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get feature information for preprocessing.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with feature information.
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and ID columns
        if 'isFraud' in numerical_cols:
            numerical_cols.remove('isFraud')
        if 'TransactionID' in numerical_cols:
            numerical_cols.remove('TransactionID')

        return {
            'categorical_columns': categorical_cols,
            'numerical_columns': numerical_cols,
            'n_samples': len(df),
            'n_features': len(categorical_cols) + len(numerical_cols),
            'fraud_rate': df['isFraud'].mean() if 'isFraud' in df.columns else None
        }