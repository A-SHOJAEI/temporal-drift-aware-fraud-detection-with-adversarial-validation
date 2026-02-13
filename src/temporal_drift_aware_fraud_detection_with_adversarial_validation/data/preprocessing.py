"""Data preprocessing and feature engineering for fraud detection."""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for fraud detection with drift awareness."""

    def __init__(self, random_seed: int = 42):
        """Initialize feature engineer.

        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_stats = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit transformers and transform training data.

        Args:
            df: Training DataFrame.

        Returns:
            Transformed DataFrame.
        """
        logger.info("Fitting and transforming training data")

        df = df.copy()

        # Store original feature statistics for drift detection
        self._compute_feature_stats(df)

        # Handle missing values
        df = self._handle_missing_values(df, fit=True)

        # Engineer features
        df = self._engineer_features(df)

        # Encode categorical variables
        df = self._encode_categorical(df, fit=True)

        # Scale numerical features
        df = self._scale_numerical(df, fit=True)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        logger.info("Transforming new data")

        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df, fit=False)

        # Engineer features
        df = self._engineer_features(df)

        # Encode categorical variables
        df = self._encode_categorical(df, fit=False)

        # Scale numerical features
        df = self._scale_numerical(df, fit=False)

        return df

    def _compute_feature_stats(self, df: pd.DataFrame) -> None:
        """Compute feature statistics for drift detection.

        Args:
            df: Input DataFrame.
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        self.feature_stats = {
            'numerical': {
                col: {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                } for col in numerical_cols if col not in ['isFraud', 'TransactionID']
            },
            'categorical': {
                col: df[col].value_counts(normalize=True).to_dict()
                for col in categorical_cols
            }
        }

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values in the dataset.

        Args:
            df: Input DataFrame.
            fit: Whether to fit the imputers.

        Returns:
            DataFrame with missing values handled.
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Remove target and ID columns from imputation
        if 'isFraud' in numerical_cols:
            numerical_cols.remove('isFraud')
        if 'TransactionID' in numerical_cols:
            numerical_cols.remove('TransactionID')

        if fit:
            # Fit numerical imputer
            if numerical_cols:
                self.imputers['numerical'] = SimpleImputer(strategy='median')
                df[numerical_cols] = self.imputers['numerical'].fit_transform(df[numerical_cols])

            # Fit categorical imputer
            if categorical_cols:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = self.imputers['categorical'].fit_transform(df[categorical_cols])
        else:
            # Transform using fitted imputers
            if numerical_cols and 'numerical' in self.imputers:
                df[numerical_cols] = self.imputers['numerical'].transform(df[numerical_cols])

            if categorical_cols and 'categorical' in self.imputers:
                df[categorical_cols] = self.imputers['categorical'].transform(df[categorical_cols])

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer domain-specific features for fraud detection.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with engineered features.
        """
        df = df.copy()

        # Transaction amount features
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
            df['TransactionAmt_zscore'] = (df['TransactionAmt'] - df['TransactionAmt'].mean()) / df['TransactionAmt'].std()

        # Time-based features
        if 'TransactionDT' in df.columns:
            df['hour'] = (df['TransactionDT'] % (24 * 3600)) // 3600
            df['day_of_week'] = (df['TransactionDT'] // (24 * 3600)) % 7
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_night'] = df['hour'].isin(list(range(23, 24)) + list(range(0, 6))).astype(int)

        # Card features interaction
        if all(col in df.columns for col in ['card1', 'card2']):
            df['card1_card2_ratio'] = df['card1'] / (df['card2'] + 1)

        # Distance features
        if all(col in df.columns for col in ['dist1', 'dist2']):
            df['dist_total'] = df['dist1'] + df['dist2']
            df['dist_ratio'] = df['dist1'] / (df['dist2'] + 1)

        # Velocity features (transaction frequency)
        if 'TransactionDT' in df.columns and 'card1' in df.columns:
            # Sort by card and time for velocity calculation
            df_sorted = df.sort_values(['card1', 'TransactionDT'])
            df_sorted['time_diff'] = df_sorted.groupby('card1')['TransactionDT'].diff()
            df_sorted['velocity'] = 1 / (df_sorted['time_diff'] + 1)  # Inverse time as velocity
            df['velocity'] = df_sorted['velocity']

        # C features combinations (if present)
        c_cols = [col for col in df.columns if col.startswith('C') and col[1:].isdigit()]
        if len(c_cols) >= 2:
            df['C_sum'] = df[c_cols].sum(axis=1)
            df['C_mean'] = df[c_cols].mean(axis=1)

        # D features combinations (if present)
        d_cols = [col for col in df.columns if col.startswith('D') and col[1:].isdigit()]
        if len(d_cols) >= 2:
            df['D_sum'] = df[d_cols].sum(axis=1)
            df['D_mean'] = df[d_cols].mean(axis=1)

        # V features (if present) - PCA-like transformation
        v_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        if len(v_cols) >= 3:
            df['V_sum'] = df[v_cols].sum(axis=1)
            df['V_std'] = df[v_cols].std(axis=1)

        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical variables.

        Args:
            df: Input DataFrame.
            fit: Whether to fit the encoders.

        Returns:
            DataFrame with encoded categorical variables.
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if not categorical_cols:
            return df

        if fit:
            for col in categorical_cols:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
        else:
            for col in categorical_cols:
                if col in self.encoders:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str))
                    known_values = set(self.encoders[col].classes_)
                    unknown_values = unique_values - known_values

                    if unknown_values:
                        logger.warning(f"Unknown categories in {col}: {unknown_values}")
                        # Replace unknown categories with the most frequent known category
                        most_frequent = self.encoders[col].classes_[0]
                        df[col] = df[col].astype(str).replace(list(unknown_values), most_frequent)

                    df[col] = self.encoders[col].transform(df[col].astype(str))

        return df

    def _scale_numerical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features.

        Args:
            df: Input DataFrame.
            fit: Whether to fit the scaler.

        Returns:
            DataFrame with scaled numerical features.
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and ID columns from scaling
        excluded_cols = ['isFraud', 'TransactionID']
        numerical_cols = [col for col in numerical_cols if col not in excluded_cols]

        if not numerical_cols:
            return df

        if fit:
            self.scalers['standard'] = StandardScaler()
            df[numerical_cols] = self.scalers['standard'].fit_transform(df[numerical_cols])
        else:
            if 'standard' in self.scalers:
                df[numerical_cols] = self.scalers['standard'].transform(df[numerical_cols])

        return df

    def get_drift_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract features specifically for drift detection.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with drift detection features.
        """
        drift_features = {}

        # Statistical features for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['isFraud', 'TransactionID']:
                drift_features[f'{col}_mean'] = np.array([df[col].mean()])
                drift_features[f'{col}_std'] = np.array([df[col].std()])
                drift_features[f'{col}_skew'] = np.array([df[col].skew()])

        # Distribution features for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            # Use top 5 most frequent categories as features
            for i, (value, freq) in enumerate(value_counts.head(5).items()):
                drift_features[f'{col}_top{i+1}_freq'] = np.array([freq])

        return drift_features


class TemporalSplitter:
    """Split data temporally for realistic fraud detection evaluation."""

    def __init__(self, time_column: str = 'TransactionDT', random_seed: int = 42):
        """Initialize temporal splitter.

        Args:
            time_column: Column name containing temporal information.
            random_seed: Random seed for reproducibility.
        """
        self.time_column = time_column
        self.random_seed = random_seed

    def temporal_split(self, df: pd.DataFrame,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally to simulate real-world deployment.

        Args:
            df: Input DataFrame.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        # Sort by time
        df_sorted = df.sort_values(self.time_column).reset_index(drop=True)

        # Calculate split indices
        n_samples = len(df_sorted)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Split temporally
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        logger.info(f"Temporal split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def create_drift_periods(self, df: pd.DataFrame, n_periods: int = 5) -> List[pd.DataFrame]:
        """Split data into temporal periods for drift analysis.

        Args:
            df: Input DataFrame.
            n_periods: Number of periods to create.

        Returns:
            List of DataFrames, one for each period.
        """
        df_sorted = df.sort_values(self.time_column).reset_index(drop=True)

        period_size = len(df_sorted) // n_periods
        periods = []

        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df_sorted)
            periods.append(df_sorted.iloc[start_idx:end_idx].copy())

        logger.info(f"Created {len(periods)} drift periods with sizes: {[len(p) for p in periods]}")
        return periods

    def stratified_temporal_split(self, df: pd.DataFrame,
                                train_ratio: float = 0.7,
                                target_column: str = 'isFraud') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a stratified temporal split maintaining class balance.

        Args:
            df: Input DataFrame.
            train_ratio: Proportion of data for training.
            target_column: Target column name.

        Returns:
            Tuple of (train_df, test_df).
        """
        df_sorted = df.sort_values(self.time_column).reset_index(drop=True)

        # Split each class temporally
        fraud_df = df_sorted[df_sorted[target_column] == 1]
        normal_df = df_sorted[df_sorted[target_column] == 0]

        # Split fraud transactions
        fraud_split_idx = int(len(fraud_df) * train_ratio)
        fraud_train = fraud_df.iloc[:fraud_split_idx]
        fraud_test = fraud_df.iloc[fraud_split_idx:]

        # Split normal transactions
        normal_split_idx = int(len(normal_df) * train_ratio)
        normal_train = normal_df.iloc[:normal_split_idx]
        normal_test = normal_df.iloc[normal_split_idx:]

        # Combine and sort by time
        train_df = pd.concat([fraud_train, normal_train]).sort_values(self.time_column)
        test_df = pd.concat([fraud_test, normal_test]).sort_values(self.time_column)

        logger.info(f"Stratified temporal split - Train: {len(train_df)}, Test: {len(test_df)}")
        logger.info(f"Train fraud rate: {train_df[target_column].mean():.3f}, "
                   f"Test fraud rate: {test_df[target_column].mean():.3f}")

        return train_df, test_df