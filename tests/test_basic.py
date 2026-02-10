"""Basic tests that should always pass."""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_basic_imports():
    """Test that basic Python packages can be imported."""
    try:
        import numpy
        import pandas
        assert True, "Basic packages imported successfully"
    except ImportError as e:
        assert False, f"Failed to import basic packages: {e}"


def test_project_structure():
    """Test that the project structure exists."""
    project_root = Path(__file__).parent.parent

    required_dirs = [
        "src",
        "scripts",
        "tests",
        "configs"
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Required directory '{dir_name}' not found"


def test_scripts_exist():
    """Test that required scripts exist."""
    scripts_dir = Path(__file__).parent.parent / "scripts"

    required_scripts = [
        "train.py",
        "evaluate.py"
    ]

    for script_name in required_scripts:
        script_path = scripts_dir / script_name
        assert script_path.exists(), f"Required script '{script_name}' not found"
        assert script_path.is_file(), f"'{script_name}' is not a file"


def test_config_file_exists():
    """Test that configuration file exists."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    assert config_path.exists(), "Default configuration file not found"


def test_source_modules_exist():
    """Test that source modules exist."""
    src_dir = Path(__file__).parent.parent / "src" / "temporal_drift_aware_fraud_detection_with_adversarial_validation"

    required_modules = [
        "data",
        "models",
        "training",
        "evaluation",
        "utils"
    ]

    for module_name in required_modules:
        module_path = src_dir / module_name
        assert module_path.exists(), f"Required module '{module_name}' not found"

        # Check for __init__.py
        init_file = module_path / "__init__.py"
        assert init_file.exists(), f"Missing __init__.py in '{module_name}' module"


def test_numpy_pandas_operations():
    """Test basic numpy and pandas operations."""
    # Test numpy
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.std() > 0

    # Test pandas
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [0.1, 0.2, 0.3, 0.4, 0.5]
    })

    assert len(df) == 5
    assert 'A' in df.columns
    assert df['A'].sum() == 15
    assert df['C'].mean() == 0.3


def test_synthetic_data_generation():
    """Test synthetic data generation without ML dependencies."""
    np.random.seed(42)

    # Generate simple synthetic fraud data
    n_samples = 100

    # Transaction features
    transaction_amounts = np.random.lognormal(3, 1, n_samples)
    time_stamps = np.sort(np.random.uniform(0, 100, n_samples))

    # Binary fraud labels
    fraud_prob = 0.05 + 0.01 * (transaction_amounts > np.percentile(transaction_amounts, 90))
    fraud_labels = np.random.binomial(1, fraud_prob, n_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'TransactionDT': time_stamps,
        'TransactionAmt': transaction_amounts,
        'isFraud': fraud_labels,
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(1, n_samples)
    })

    # Validate synthetic data
    assert len(data) == n_samples
    assert 'isFraud' in data.columns
    assert data['isFraud'].isin([0, 1]).all()
    assert data['TransactionAmt'].min() > 0
    assert data['TransactionDT'].is_monotonic_increasing

    # Check fraud rate is reasonable
    fraud_rate = data['isFraud'].mean()
    assert 0.01 <= fraud_rate <= 0.2, f"Fraud rate {fraud_rate} seems unrealistic"


if __name__ == "__main__":
    # Run tests if script is executed directly
    test_functions = [
        test_basic_imports,
        test_project_structure,
        test_scripts_exist,
        test_config_file_exists,
        test_source_modules_exist,
        test_numpy_pandas_operations,
        test_synthetic_data_generation
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed > 0:
        exit(1)
    else:
        print("All basic tests passed!")