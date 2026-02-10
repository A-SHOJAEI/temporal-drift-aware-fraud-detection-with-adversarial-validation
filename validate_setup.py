#!/usr/bin/env python3
"""Setup validation script for temporal drift-aware fraud detection system."""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version} is too old. Requires Python 3.8+")
        return False
    else:
        print(f"✅ Python {sys.version} is compatible")
        return True

def check_dependencies():
    """Check required dependencies."""
    print("\n🔍 Checking dependencies...")

    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'catboost': 'catboost',
        'yaml': 'PyYAML',
        'scipy': 'scipy'
    }

    optional_packages = {
        'torch': 'torch (for GPU support)',
        'mlflow': 'mlflow (for experiment tracking)'
    }

    missing_required = []
    missing_optional = []

    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - REQUIRED")
            missing_required.append(package)

    for module, package in optional_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - OPTIONAL")
            missing_optional.append(package)

    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + ' '.join(missing_required))
        return False

    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + ' '.join(missing_optional))

    return True

def check_project_structure():
    """Check project structure."""
    print("\n🔍 Checking project structure...")

    project_root = Path(__file__).parent

    required_dirs = [
        "src/temporal_drift_aware_fraud_detection_with_adversarial_validation",
        "scripts",
        "tests",
        "configs"
    ]

    required_files = [
        "scripts/train.py",
        "scripts/evaluate.py",
        "configs/default.yaml",
        "src/temporal_drift_aware_fraud_detection_with_adversarial_validation/__init__.py"
    ]

    all_good = True

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_good = False

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False

    return all_good

def check_gpu_support():
    """Check GPU support."""
    print("\n🔍 Checking GPU support...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot detect GPU")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n🔧 Creating necessary directories...")

    project_root = Path(__file__).parent

    dirs_to_create = [
        "models",
        "models/checkpoints",
        "logs",
        "results",
        "data"
    ]

    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {dir_name}/")
        except Exception as e:
            print(f"❌ Failed to create {dir_name}/: {e}")

def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🔍 Running basic functionality tests...")

    project_root = Path(__file__).parent
    test_script = project_root / "tests" / "test_simple.py"

    if test_script.exists():
        try:
            import subprocess
            result = subprocess.run([sys.executable, str(test_script)],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Basic tests passed")
                return True
            else:
                print(f"❌ Basic tests failed:\n{result.stdout}\n{result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Failed to run tests: {e}")
            return False
    else:
        print("⚠️  Test script not found")
        return False

def main():
    """Main validation function."""
    print("=" * 70)
    print("TEMPORAL DRIFT-AWARE FRAUD DETECTION - SETUP VALIDATION")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("GPU Support", check_gpu_support),
    ]

    results = []

    for name, check_func in checks:
        result = check_func()
        results.append((name, result))

    # Create directories regardless of other checks
    create_directories()

    # Run tests if basic requirements are met
    if results[1][1] and results[2][1]:  # Dependencies and structure OK
        test_result = run_basic_tests()
        results.append(("Basic Tests", test_result))

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = 0
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20}: {status}")
        if result:
            passed += 1

    print(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 Setup validation completed successfully!")
        print("You can now run: python scripts/train.py --quick-test")
    else:
        print("\n⚠️  Setup validation found issues. Please fix them before training.")

        if not results[1][1]:  # Dependencies failed
            print("\n📋 TO FIX DEPENDENCY ISSUES:")
            print("pip install numpy pandas scikit-learn lightgbm xgboost catboost PyYAML scipy")
            print("pip install torch mlflow  # optional but recommended")

        if not results[2][1]:  # Structure failed
            print("\n📋 TO FIX STRUCTURE ISSUES:")
            print("Ensure you're running this script from the project root directory")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)