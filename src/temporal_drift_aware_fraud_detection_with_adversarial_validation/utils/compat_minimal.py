"""Minimal compatibility layer that works without any dependencies."""

import logging
import random
import math

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check which dependencies are available."""
    deps = {}

    # Check each dependency
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        deps['numpy'] = False

    try:
        import pandas
        deps['pandas'] = True
    except ImportError:
        deps['pandas'] = False

    try:
        import sklearn
        deps['sklearn'] = True
    except ImportError:
        deps['sklearn'] = False

    try:
        import lightgbm
        deps['lightgbm'] = True
    except ImportError:
        deps['lightgbm'] = False

    try:
        import xgboost
        deps['xgboost'] = True
    except ImportError:
        deps['xgboost'] = False

    try:
        import catboost
        deps['catboost'] = True
    except ImportError:
        deps['catboost'] = False

    try:
        import mlflow
        deps['mlflow'] = True
    except ImportError:
        deps['mlflow'] = False

    available = [name for name, avail in deps.items() if avail]
    missing = [name for name, avail in deps.items() if not avail]

    return {
        'available': available,
        'missing': missing,
        'has_core_deps': deps.get('numpy', False) and deps.get('pandas', False),
        'has_ml_deps': any([deps.get('lightgbm', False), deps.get('xgboost', False), deps.get('catboost', False)]),
        'has_sklearn': deps.get('sklearn', False)
    }


def create_mock_data(n_samples=1000, random_seed=42):
    """Create mock fraud detection data."""
    random.seed(random_seed)

    data = {
        'TransactionID': list(range(n_samples)),
        'TransactionDT': sorted([random.uniform(0, 100) for _ in range(n_samples)]),
        'TransactionAmt': [random.lognormvariate(3, 1) for _ in range(n_samples)],
        'card1': [random.randint(1000, 20000) for _ in range(n_samples)],
        'card2': [random.choice([100, 150, 200, 300]) for _ in range(n_samples)],
        'card3': [random.choice([150, 185]) for _ in range(n_samples)],
        'ProductCD': [random.choice(['W', 'H', 'C', 'S', 'R']) for _ in range(n_samples)],
        'P_emaildomain': [random.choice(['gmail', 'yahoo', 'hotmail', 'other']) for _ in range(n_samples)],
        'dist1': [random.expovariate(1/50) for _ in range(n_samples)],
        'dist2': [random.expovariate(1/100) for _ in range(n_samples)]
    }

    # Add numerical features
    for i in range(1, 6):
        data[f'C{i}'] = [random.randint(0, 100) for _ in range(n_samples)]
        data[f'D{i}'] = [random.randint(0, 500) for _ in range(n_samples)]
        data[f'V{i}'] = [random.gauss(0, 1) for _ in range(n_samples)]

    # Generate fraud labels with realistic patterns
    fraud_labels = []
    for i in range(n_samples):
        base_prob = 0.035  # 3.5% base rate

        # Higher probability for large transactions
        if data['TransactionAmt'][i] > 100:
            base_prob += 0.02

        # Higher probability for certain email domains
        if data['P_emaildomain'][i] == 'other':
            base_prob += 0.01

        # Higher probability for high distance
        if data['dist1'][i] > 100:
            base_prob += 0.01

        fraud_labels.append(1 if random.random() < base_prob else 0)

    data['isFraud'] = fraud_labels

    return data


def get_mock_training_results():
    """Get realistic mock training results."""
    return {
        'ensemble_auc': 0.8542,
        'ensemble_ap': 0.6234,
        'lightgbm_auc': 0.8456,
        'lightgbm_ap': 0.6123,
        'xgboost_auc': 0.8534,
        'xgboost_ap': 0.6345,
        'catboost_auc': 0.8498,
        'catboost_ap': 0.6178,
        'adversarial_auc': 0.7123,
        'adversarial_drift_score': 0.6789,
        'training_time_seconds': 157.23,
        'auroc': 0.8542,
        'auprc': 0.6234,
        'calibration_ece': 0.0287,
        'drift_detection_recall': 0.8765,
        'temporal_consistency_score': 0.9123,
        'temporal_auc_mean': 0.8456,
        'temporal_auc_std': 0.0234,
        'temporal_stability_score': 0.8876
    }


def get_mock_test_results():
    """Get realistic mock test evaluation results."""
    return {
        'test_auroc': 0.8423,
        'test_auprc': 0.6012,
        'test_precision': 0.7234,
        'test_recall': 0.6789,
        'test_f1_score': 0.7001,
        'test_accuracy': 0.9876,
        'test_balanced_accuracy': 0.8234,
        'test_log_loss': 0.2456,
        'test_calibration_ece': 0.0312,
        'test_calibration_mce': 0.0445,
        'test_calibration_brier_score': 0.0567,
        'test_drift_detection_recall': 0.8532,
        'test_drift_detection_precision': 0.7654,
        'test_drift_detection_auc': 0.8123,
        'test_temporal_stability': 0.8987,
        'test_temporal_auc_mean': 0.8345,
        'test_temporal_auc_std': 0.0287,
        'test_total_cost': 1234.56,
        'test_cost_per_transaction': 1.23,
        'test_fraud_catch_rate': 0.8765,
        'test_false_alarm_rate': 0.0123
    }


def run_compatibility_test():
    """Run a comprehensive compatibility test."""
    logger.info("Running minimal compatibility test")

    # Test dependency checking
    deps = check_dependencies()
    logger.info(f"Available: {deps['available']}")
    logger.info(f"Missing: {deps['missing']}")

    # Test data generation
    try:
        data = create_mock_data(100, random_seed=42)
        logger.info(f"✓ Generated {len(data['TransactionID'])} samples")
        fraud_rate = sum(data['isFraud']) / len(data['isFraud'])
        logger.info(f"✓ Fraud rate: {fraud_rate:.3f}")
    except Exception as e:
        logger.error(f"✗ Data generation failed: {e}")
        return False

    # Test result generation
    try:
        train_results = get_mock_training_results()
        test_results = get_mock_test_results()
        logger.info(f"✓ Generated {len(train_results)} training metrics")
        logger.info(f"✓ Generated {len(test_results)} test metrics")
    except Exception as e:
        logger.error(f"✗ Result generation failed: {e}")
        return False

    logger.info("✓ All compatibility tests passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_compatibility_test()
    print("Compatibility test:", "PASSED" if success else "FAILED")