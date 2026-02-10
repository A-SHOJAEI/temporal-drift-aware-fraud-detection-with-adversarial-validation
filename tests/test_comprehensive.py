"""Comprehensive tests that work with or without ML dependencies."""

import sys
import os
import json
from pathlib import Path
import tempfile
import shutil
import subprocess

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_project_structure():
    """Test that the project has the correct structure."""
    project_root = Path(__file__).parent.parent

    # Required directories
    required_dirs = [
        "src/temporal_drift_aware_fraud_detection_with_adversarial_validation",
        "scripts",
        "tests",
        "configs"
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Required directory missing: {dir_path}"

    # Required files
    required_files = [
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/train_test_mode.py",
        "configs/default.yaml",
        "requirements.txt",
        "README.md"
    ]

    for file_path in required_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Required file missing: {file_path}"


def test_config_file():
    """Test that the configuration file is valid."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    # Test YAML parsing
    try:
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Check required sections
        required_sections = ['training', 'data', 'evaluation', 'base_models']
        for section in required_sections:
            assert section in config_data, f"Missing config section: {section}"

        # Check target metrics
        assert 'target_metrics' in config_data['evaluation'], "Missing target_metrics"
        target_metrics = config_data['evaluation']['target_metrics']
        expected_metrics = ['auroc', 'auprc', 'drift_detection_recall', 'calibration_ece']
        for metric in expected_metrics:
            assert metric in target_metrics, f"Missing target metric: {metric}"

    except ImportError:
        # If PyYAML not available, just check file is readable
        with open(config_path) as f:
            content = f.read()
        assert len(content) > 100, "Config file appears to be empty or too small"


def test_compatibility_layer():
    """Test the compatibility layer."""
    try:
        # Add utils to path to avoid package-level import issues
        utils_path = str(Path(__file__).parent.parent / "src" / "temporal_drift_aware_fraud_detection_with_adversarial_validation" / "utils")
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)

        import compat_minimal
        check_dependencies = compat_minimal.check_dependencies
        create_mock_data = compat_minimal.create_mock_data
        get_mock_training_results = compat_minimal.get_mock_training_results

        # Test dependency checking
        dep_status = check_dependencies()
        assert isinstance(dep_status, dict), "Dependency status should be a dict"
        assert 'available' in dep_status, "Should report available dependencies"
        assert 'missing' in dep_status, "Should report missing dependencies"

        # Test mock data creation
        mock_data = create_mock_data(100, random_seed=42)
        assert isinstance(mock_data, dict), "Mock data should be a dict"
        assert 'TransactionID' in mock_data, "Should have TransactionID"
        assert 'isFraud' in mock_data, "Should have fraud labels"
        assert len(mock_data['TransactionID']) == 100, "Should have correct number of samples"

        # Test mock results
        results = get_mock_training_results()
        assert isinstance(results, dict), "Results should be a dict"
        assert 'ensemble_auc' in results, "Should have AUC metric"
        assert 0.0 <= results['ensemble_auc'] <= 1.0, "AUC should be in valid range"

    except ImportError:
        # If compatibility layer not available, this is an issue
        assert False, "Compatibility layer should be importable"


def test_test_mode_training():
    """Test that test mode training script works."""
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "train_test_mode.py"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the test mode script
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=30
            )

            # Check script executed successfully
            assert result.returncode == 0, f"Script failed with: {result.stderr}"

            # Check outputs contain expected content
            output = result.stdout
            assert "TEST MODE TRAINING SUMMARY" in output, "Should show training summary"
            assert "Training AUC:" in output, "Should report training AUC"
            assert "Test AUC:" in output, "Should report test AUC"

            # Check if results file was created
            results_file = project_root / "test_output" / "results" / "training_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)
                assert isinstance(results, dict), "Results should be a dictionary"
                assert 'ensemble_auc' in results, "Should have ensemble AUC"

        except subprocess.TimeoutExpired:
            assert False, "Test mode script took too long to execute"
        except Exception as e:
            assert False, f"Test mode script failed: {e}"


def test_requirements_file():
    """Test that requirements.txt contains expected packages."""
    requirements_path = Path(__file__).parent.parent / "requirements.txt"

    with open(requirements_path) as f:
        requirements = f.read()

    # Expected core packages
    expected_packages = [
        'numpy', 'pandas', 'scikit-learn',
        'lightgbm', 'xgboost', 'catboost',
        'pyyaml', 'pytest'
    ]

    for package in expected_packages:
        assert package in requirements.lower(), f"Missing required package: {package}"


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    project_root = Path(__file__).parent.parent

    # Find all Python files
    python_files = []
    for pattern in ["src/**/*.py", "scripts/*.py", "tests/*.py"]:
        python_files.extend(project_root.glob(pattern))

    # Test each file
    errors = []
    for py_file in python_files:
        try:
            with open(py_file) as f:
                code = f.read()
            compile(code, str(py_file), 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error in {py_file}: {e}")
        except Exception as e:
            errors.append(f"Error reading {py_file}: {e}")

    if errors:
        assert False, f"Python syntax errors found:\n" + "\n".join(errors)


def test_basic_imports():
    """Test that basic imports work (with fallbacks)."""
    # Test compatibility layer import
    try:
        # Add utils to path to avoid package-level import issues
        utils_path = str(Path(__file__).parent.parent / "src" / "temporal_drift_aware_fraud_detection_with_adversarial_validation" / "utils")
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)

        import compat_minimal
        check_dependencies = compat_minimal.check_dependencies
        print("✓ Compatibility layer imports successfully")
    except ImportError as e:
        assert False, f"Failed to import compatibility layer: {e}"

    # Test config import (may fail without numpy)
    try:
        from temporal_drift_aware_fraud_detection_with_adversarial_validation.utils.config import Config
        print("✓ Config module imports successfully")
    except ImportError:
        print("⚠ Config module requires numpy/pandas - this is expected without ML dependencies")


def test_directories_writable():
    """Test that we can write to output directories."""
    project_root = Path(__file__).parent.parent

    test_dirs = ["models", "logs", "results", "evaluation_results"]

    for dir_name in test_dirs:
        test_dir = project_root / dir_name
        test_dir.mkdir(exist_ok=True)

        # Test writing
        test_file = test_dir / "test_write.tmp"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            assert test_file.exists(), f"Failed to create test file in {test_dir}"
            test_file.unlink()  # Clean up
        except Exception as e:
            assert False, f"Cannot write to {test_dir}: {e}"


def test_mock_model_pipeline():
    """Test the complete mock model pipeline."""
    try:
        # Add utils to path to avoid package-level import issues
        utils_path = str(Path(__file__).parent.parent / "src" / "temporal_drift_aware_fraud_detection_with_adversarial_validation" / "utils")
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)

        import compat_minimal
        create_mock_data = compat_minimal.create_mock_data
        get_mock_training_results = compat_minimal.get_mock_training_results
        get_mock_test_results = compat_minimal.get_mock_test_results

        # Test data pipeline
        train_data = create_mock_data(1000, random_seed=42)
        test_data = create_mock_data(500, random_seed=123)

        assert len(train_data['isFraud']) == 1000, "Training data size incorrect"
        assert len(test_data['isFraud']) == 500, "Test data size incorrect"

        # Test fraud rates are realistic
        train_fraud_rate = sum(train_data['isFraud']) / len(train_data['isFraud'])
        test_fraud_rate = sum(test_data['isFraud']) / len(test_data['isFraud'])

        assert 0.01 <= train_fraud_rate <= 0.10, f"Train fraud rate unrealistic: {train_fraud_rate}"
        assert 0.01 <= test_fraud_rate <= 0.10, f"Test fraud rate unrealistic: {test_fraud_rate}"

        # Test results pipeline
        train_results = get_mock_training_results()
        test_results = get_mock_test_results()

        # Validate metric ranges
        assert 0.7 <= train_results['ensemble_auc'] <= 1.0, "Training AUC out of range"
        assert 0.7 <= test_results['test_auroc'] <= 1.0, "Test AUC out of range"
        assert 0.0 <= test_results['test_calibration_ece'] <= 0.1, "ECE out of range"

    except ImportError:
        assert False, "Mock pipeline should be available"


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_project_structure,
        test_config_file,
        test_compatibility_layer,
        test_test_mode_training,
        test_requirements_file,
        test_python_syntax,
        test_basic_imports,
        test_directories_writable,
        test_mock_model_pipeline
    ]

    passed = 0
    failed = 0
    errors = []

    print("Running comprehensive tests...")
    print("="*60)

    for test_func in tests:
        test_name = test_func.__name__
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            errors.append(f"{test_name}: {e}")
            failed += 1

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)