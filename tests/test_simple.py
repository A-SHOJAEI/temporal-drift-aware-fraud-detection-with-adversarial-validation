"""Simple tests that work with standard library only."""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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


def test_python_syntax():
    """Test that Python files have valid syntax."""
    project_root = Path(__file__).parent.parent

    # Test main scripts
    scripts = [
        project_root / "scripts" / "train.py",
        project_root / "scripts" / "evaluate.py"
    ]

    for script_path in scripts:
        if script_path.exists():
            try:
                compile(open(script_path).read(), str(script_path), 'exec')
            except SyntaxError as e:
                assert False, f"Syntax error in {script_path}: {e}"


def test_config_file_format():
    """Test that config file is valid YAML."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    if config_path.exists():
        try:
            # Try to parse YAML using available modules
            import yaml
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
            assert isinstance(config_data, dict), "Config should be a dictionary"

            # Check for required sections
            required_sections = ['training', 'data', 'evaluation']
            for section in required_sections:
                assert section in config_data, f"Missing config section: {section}"

        except ImportError:
            # If PyYAML not available, just check it's readable
            with open(config_path) as f:
                content = f.read()
            assert len(content) > 0, "Config file is empty"


def test_directories_writable():
    """Test that we can create files in necessary directories."""
    project_root = Path(__file__).parent.parent

    test_dirs = [
        project_root / "models",
        project_root / "logs",
        project_root / "results"
    ]

    for dir_path in test_dirs:
        dir_path.mkdir(exist_ok=True)
        assert dir_path.exists(), f"Could not create directory: {dir_path}"

        # Test writing a file
        test_file = dir_path / "test_write.tmp"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()  # Clean up
        except Exception as e:
            assert False, f"Cannot write to {dir_path}: {e}"


if __name__ == "__main__":
    # Run tests if script is executed directly
    test_functions = [
        test_project_structure,
        test_scripts_exist,
        test_config_file_exists,
        test_source_modules_exist,
        test_python_syntax,
        test_config_file_format,
        test_directories_writable
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
        print("All simple tests passed!")