#!/usr/bin/env python3
"""Training script for temporal drift-aware fraud detection system.

This script implements a complete training pipeline that:
1. Loads and preprocesses IEEE-CIS fraud detection data (or synthetic data)
2. Trains an ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost)
3. Implements adversarial validation for drift detection
4. Evaluates model performance with comprehensive metrics
5. Saves trained models and artifacts

Usage:
    python scripts/train.py [--config CONFIG_PATH] [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Dependency checking and graceful handling
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install required packages:")
    print("pip install numpy pandas scikit-learn lightgbm xgboost catboost PyYAML scipy")
    sys.exit(1)

try:
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.utils.config import load_config
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.loader import DataLoader
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.training.trainer import DriftAwareTrainer
    from temporal_drift_aware_fraud_detection_with_adversarial_validation.evaluation.metrics import ComprehensiveEvaluator
except ImportError as e:
    print(f"Error: Project modules not found: {e}")
    print("Please ensure the project is properly installed or run from project root.")
    sys.exit(1)

# GPU availability checking
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ GPU not available, falling back to CPU")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ PyTorch not available, cannot detect GPU")

warnings.filterwarnings('ignore')


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Set up logging configuration.

    Args:
        log_level: Logging level.
        log_file: Optional log file path.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train temporal drift-aware fraud detection system"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to IEEE-CIS data directory (optional)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of IEEE-CIS data"
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced data size"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override with command line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    if args.synthetic:
        config.data.use_synthetic = True

    # Adjust for quick test
    if args.quick_test:
        print("Running in quick test mode - reducing data size and iterations")
        # Reduce model complexity for quick testing
        for model_name in config.base_models:
            config.base_models[model_name].params['n_estimators'] = 100
            if 'iterations' in config.base_models[model_name].params:
                config.base_models[model_name].params['iterations'] = 100
        config.training.early_stopping_rounds = 10
        config.training.n_drift_periods = 3

    # Set up logging
    setup_logging(
        log_level="DEBUG" if args.verbose else config.logging.level,
        log_file=config.logging.log_file if config.logging.log_to_file else None
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting temporal drift-aware fraud detection training")
    logger.info(f"GPU available: {GPU_AVAILABLE}")
    logger.info(f"Configuration: {config.to_dict()}")

    # Create necessary directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")
    logger.info(f"Models directory: {models_dir}")

    try:
        # Load and prepare data
        logger.info("Loading and preparing data")
        data_loader = DataLoader(
            data_path=config.data.data_path,
            random_seed=config.training.random_seed
        )

        train_data, test_data = data_loader.load_data()

        # Log data statistics
        feature_info = data_loader.get_feature_info(train_data)
        logger.info(f"Data loaded - Train: {len(train_data)}, Test: {len(test_data)}")
        logger.info(f"Features: {feature_info}")

        if args.quick_test:
            # Sample smaller dataset for quick testing
            sample_size = min(5000, len(train_data))
            train_data = train_data.sample(n=sample_size, random_state=config.training.random_seed)
            test_data = test_data.sample(n=min(2000, len(test_data)), random_state=config.training.random_seed)
            logger.info(f"Quick test mode - sampled Train: {len(train_data)}, Test: {len(test_data)}")

        # Initialize trainer
        logger.info("Initializing drift-aware trainer")
        trainer = DriftAwareTrainer(
            config=config.to_dict(),
            save_dir=str(models_dir),  # Save to models subdirectory
            experiment_name="drift_aware_fraud_detection"
        )

        # Train models
        logger.info("Starting model training")
        training_results = trainer.train(
            train_data=train_data,
            validation_data=None,  # Will split from training data
            save_best=True
        )

        # Log training results
        logger.info("Training completed successfully")
        logger.info("Key Training Results:")
        for key, value in training_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")

        # Evaluate on test data
        logger.info("Evaluating on test data")
        test_predictions, test_labels = trainer.predict(test_data)

        # Comprehensive evaluation
        evaluator = ComprehensiveEvaluator()

        # Preprocess test data through the feature engineer to encode categoricals
        test_processed = trainer.feature_engineer.transform(test_data)
        feature_cols = [col for col in test_processed.columns
                       if col not in ['isFraud', 'TransactionID']]
        test_features = test_processed[feature_cols]
        test_targets = test_processed['isFraud']

        test_results = evaluator.full_evaluation(
            model=trainer.ensemble,
            X_test=test_features,
            y_test=test_targets,
            drift_periods=None,  # Could add test periods here
            reference_data=None
        )

        # Log test results
        logger.info("Test Evaluation Results:")
        for key, value in test_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")

        # Check if target metrics are met
        target_metrics = config.evaluation.target_metrics
        met_targets = {}

        for metric_name, target_value in target_metrics.items():
            actual_value = test_results.get(metric_name, 0)
            met_target = actual_value >= target_value
            met_targets[metric_name] = met_target

            status = "✓" if met_target else "✗"
            logger.info(f"  {status} {metric_name}: {actual_value:.4f} (target: {target_value:.4f})")

        # Summary
        total_targets = len(target_metrics)
        met_count = sum(met_targets.values())
        logger.info(f"Met {met_count}/{total_targets} target metrics")

        if met_count == total_targets:
            logger.info("🎉 All target metrics achieved!")
        else:
            logger.warning(f"⚠️  {total_targets - met_count} target metrics not met")

        # Save final results
        results_dir = Path(args.output_dir) / "results"
        results_dir.mkdir(exist_ok=True)

        # Combine all results
        final_results = {
            **training_results,
            **{f"test_{k}": v for k, v in test_results.items()},
            "target_metrics_met": met_targets,
            "config": config.to_dict()
        }

        # Save as JSON
        import json
        results_file = results_dir / "training_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for k, v in final_results.items():
                if isinstance(v, np.number):
                    serializable_results[k] = float(v)
                elif isinstance(v, dict):
                    serializable_results[k] = {
                        kk: float(vv) if isinstance(vv, np.number) else vv
                        for kk, vv in v.items()
                    }
                else:
                    serializable_results[k] = v

            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Print final summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        def _fmt(val, fmt=".4f"):
            try:
                return f"{float(val):{fmt}}"
            except (ValueError, TypeError):
                return str(val)
        print(f"  Model trained and saved to: {args.output_dir}")
        print(f"  Training AUC: {_fmt(training_results.get('ensemble_auc', 'N/A'))}")
        print(f"  Test AUC: {_fmt(test_results.get('auroc', 'N/A'))}")
        print(f"  Test AP: {_fmt(test_results.get('auprc', 'N/A'))}")
        print(f"  Calibration ECE: {_fmt(test_results.get('calibration_ece', 'N/A'))}")
        print(f"  Results saved to: {results_file}")

        if met_count == total_targets:
            print("🎉 ALL TARGET METRICS ACHIEVED!")
        else:
            print(f"⚠️  {total_targets - met_count}/{total_targets} target metrics not met")

        print("="*80)

        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)