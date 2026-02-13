#!/usr/bin/env python3
"""Evaluation script for temporal drift-aware fraud detection system.

This script loads a trained model and performs comprehensive evaluation including:
1. Standard classification metrics
2. Drift detection performance
3. Calibration quality assessment
4. Temporal stability analysis
5. Business impact metrics

Usage:
    python scripts/evaluate.py [--model-path MODEL_PATH] [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from temporal_drift_aware_fraud_detection_with_adversarial_validation.utils.config import load_config
from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.loader import DataLoader
from temporal_drift_aware_fraud_detection_with_adversarial_validation.training.trainer import DriftAwareTrainer
from temporal_drift_aware_fraud_detection_with_adversarial_validation.evaluation.metrics import (
    ComprehensiveEvaluator, DriftDetectionMetrics, CalibrationMetrics
)

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
        description="Evaluate temporal drift-aware fraud detection system"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models",
        help="Path to trained model directory"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to evaluation data (IEEE-CIS format)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for evaluation"
    )

    parser.add_argument(
        "--drift-analysis",
        action="store_true",
        help="Perform detailed drift detection analysis"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_trained_model(model_path: str, config_path: str) -> DriftAwareTrainer:
    """Load a trained model.

    Args:
        model_path: Path to the trained model directory.
        config_path: Path to configuration file.

    Returns:
        Loaded DriftAwareTrainer instance.
    """
    config = load_config(config_path)

    trainer = DriftAwareTrainer(
        config=config.to_dict(),
        save_dir=model_path
    )

    trainer.load_model(model_path)
    return trainer


def evaluate_standard_metrics(trainer: DriftAwareTrainer,
                             test_data: pd.DataFrame,
                             logger: logging.Logger) -> dict:
    """Evaluate standard classification metrics.

    Args:
        trainer: Trained model.
        test_data: Test dataset.
        logger: Logger instance.

    Returns:
        Dictionary with standard metrics.
    """
    logger.info("Evaluating standard classification metrics")

    # Make predictions
    test_probs, test_preds = trainer.predict(test_data)

    # Get feature columns
    feature_cols = [col for col in test_data.columns
                   if col not in ['isFraud', 'TransactionID']]
    test_features = test_data[feature_cols]
    test_targets = test_data['isFraud']

    # Use comprehensive evaluator
    evaluator = ComprehensiveEvaluator()
    results = evaluator.full_evaluation(
        model=trainer.ensemble,
        X_test=test_features,
        y_test=test_targets
    )

    return results


def evaluate_drift_detection(trainer: DriftAwareTrainer,
                            test_data: pd.DataFrame,
                            config,
                            logger: logging.Logger) -> dict:
    """Evaluate drift detection capabilities.

    Args:
        trainer: Trained model.
        test_data: Test dataset.
        config: Configuration object.
        logger: Logger instance.

    Returns:
        Dictionary with drift detection metrics.
    """
    logger.info("Evaluating drift detection capabilities")

    drift_metrics = {}

    try:
        # Create temporal periods for drift analysis
        from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.preprocessing import TemporalSplitter

        temporal_splitter = TemporalSplitter(
            time_column=config.data.time_column,
            random_seed=config.training.random_seed
        )

        # Create drift periods
        drift_periods = temporal_splitter.create_drift_periods(
            test_data, n_periods=config.training.n_drift_periods
        )

        # Evaluate drift detection
        drift_evaluator = DriftDetectionMetrics(random_seed=config.training.random_seed)

        # Use first period as reference
        reference_period = drift_periods[0]

        drift_results = drift_evaluator.evaluate_drift_detection(
            ensemble=trainer.ensemble,
            drift_periods=drift_periods,
            reference_data=reference_period
        )

        drift_metrics.update(drift_results)

        # Evaluate temporal stability
        predictions_by_period = {}
        true_labels_by_period = {}

        for i, period in enumerate(drift_periods):
            if 'isFraud' not in period.columns:
                continue

            feature_cols = [col for col in period.columns
                           if col not in ['isFraud', 'TransactionID']]

            period_features = period[feature_cols]
            period_targets = period['isFraud']

            if len(period_targets.unique()) > 1:  # Need both classes
                try:
                    period_probs = trainer.ensemble.predict_proba(period_features)
                    predictions_by_period[f'period_{i}'] = period_probs
                    true_labels_by_period[f'period_{i}'] = period_targets.values
                except Exception as e:
                    logger.warning(f"Failed to evaluate period {i}: {e}")

        if predictions_by_period:
            stability_results = drift_evaluator.temporal_stability_score(
                predictions_by_period, true_labels_by_period
            )
            drift_metrics.update(stability_results)

    except Exception as e:
        logger.error(f"Failed to evaluate drift detection: {e}")
        drift_metrics['drift_evaluation_error'] = str(e)

    return drift_metrics


def evaluate_calibration(trainer: DriftAwareTrainer,
                        test_data: pd.DataFrame,
                        config,
                        logger: logging.Logger) -> dict:
    """Evaluate model calibration.

    Args:
        trainer: Trained model.
        test_data: Test dataset.
        config: Configuration object.
        logger: Logger instance.

    Returns:
        Dictionary with calibration metrics.
    """
    logger.info("Evaluating model calibration")

    calibration_metrics = {}

    try:
        # Make predictions
        test_probs, _ = trainer.predict(test_data)
        test_targets = test_data['isFraud']

        # Evaluate calibration
        calibration_evaluator = CalibrationMetrics(
            n_bins=config.evaluation.calibration_bins
        )

        calibration_results = calibration_evaluator.evaluate_calibration(
            y_true=test_targets.values,
            y_prob=test_probs
        )

        calibration_metrics.update(calibration_results)

    except Exception as e:
        logger.error(f"Failed to evaluate calibration: {e}")
        calibration_metrics['calibration_evaluation_error'] = str(e)

    return calibration_metrics


def generate_evaluation_report(results: dict, output_dir: str, logger: logging.Logger):
    """Generate a comprehensive evaluation report.

    Args:
        results: Evaluation results dictionary.
        output_dir: Output directory for the report.
        logger: Logger instance.
    """
    logger.info("Generating evaluation report")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSON
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, np.number):
                serializable_results[k] = float(v)
            elif isinstance(v, dict):
                serializable_results[k] = {
                    kk: float(vv) if isinstance(vv, np.number) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, (list, tuple)):
                serializable_results[k] = [
                    float(item) if isinstance(item, np.number) else item
                    for item in v
                ]
            else:
                serializable_results[k] = v

        json.dump(serializable_results, f, indent=2)

    logger.info(f"Detailed results saved to {results_file}")

    # Generate summary report
    summary_file = output_path / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("TEMPORAL DRIFT-AWARE FRAUD DETECTION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Standard Metrics
        f.write("CLASSIFICATION PERFORMANCE\n")
        f.write("-"*30 + "\n")
        f.write(f"AUC-ROC: {results.get('auroc', 'N/A'):.4f}\n")
        f.write(f"AUC-PR:  {results.get('auprc', 'N/A'):.4f}\n")
        f.write(f"Precision: {results.get('precision', 'N/A'):.4f}\n")
        f.write(f"Recall: {results.get('recall', 'N/A'):.4f}\n")
        f.write(f"F1-Score: {results.get('f1_score', 'N/A'):.4f}\n\n")

        # Drift Detection
        if 'drift_detection_recall' in results:
            f.write("DRIFT DETECTION PERFORMANCE\n")
            f.write("-"*30 + "\n")
            f.write(f"Drift Detection Recall: {results.get('drift_detection_recall', 'N/A'):.4f}\n")
            f.write(f"Drift Detection Precision: {results.get('drift_detection_precision', 'N/A'):.4f}\n")
            f.write(f"Drift Detection AUC: {results.get('drift_detection_auc', 'N/A'):.4f}\n")
            f.write(f"Temporal Stability: {results.get('temporal_consistency_score', 'N/A'):.4f}\n\n")

        # Calibration
        if 'calibration_ece' in results:
            f.write("CALIBRATION QUALITY\n")
            f.write("-"*30 + "\n")
            f.write(f"Expected Calibration Error: {results.get('calibration_ece', 'N/A'):.4f}\n")
            f.write(f"Maximum Calibration Error: {results.get('calibration_mce', 'N/A'):.4f}\n")
            f.write(f"Brier Score: {results.get('calibration_brier_score', 'N/A'):.4f}\n\n")

        # Business Metrics
        if 'total_cost' in results:
            f.write("BUSINESS IMPACT\n")
            f.write("-"*30 + "\n")
            f.write(f"Total Cost: {results.get('total_cost', 'N/A'):.2f}\n")
            f.write(f"Cost per Transaction: {results.get('cost_per_transaction', 'N/A'):.4f}\n")
            f.write(f"Fraud Catch Rate: {results.get('fraud_catch_rate', 'N/A'):.4f}\n")
            f.write(f"False Alarm Rate: {results.get('false_alarm_rate', 'N/A'):.4f}\n\n")

    logger.info(f"Summary report saved to {summary_file}")


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()

    # Set up logging
    setup_logging(
        log_level="DEBUG" if args.verbose else "INFO",
        log_file=f"{args.output_dir}/evaluation.log"
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation")

    try:
        # Load configuration
        config = load_config(args.config)

        # Override data path if provided
        if args.data_path:
            config.data.data_path = args.data_path
        if args.synthetic:
            config.data.use_synthetic = True

        # Load trained model
        logger.info(f"Loading trained model from: {args.model_path}")
        trainer = load_trained_model(args.model_path, args.config)

        # Load evaluation data
        logger.info("Loading evaluation data")
        data_loader = DataLoader(
            data_path=config.data.data_path,
            random_seed=config.training.random_seed
        )

        # For evaluation, we can use test data or generate new synthetic data
        if config.data.use_synthetic:
            # Generate fresh synthetic data for evaluation
            _, test_data = data_loader.load_data()
        else:
            # Load IEEE-CIS test data
            _, test_data = data_loader.load_data()

        logger.info(f"Evaluation data loaded: {len(test_data)} samples")

        # Perform evaluations
        all_results = {}

        # Standard classification metrics
        standard_results = evaluate_standard_metrics(trainer, test_data, logger)
        all_results.update(standard_results)

        # Drift detection evaluation
        if args.drift_analysis:
            drift_results = evaluate_drift_detection(trainer, test_data, config, logger)
            all_results.update(drift_results)

        # Calibration evaluation
        calibration_results = evaluate_calibration(trainer, test_data, config, logger)
        all_results.update(calibration_results)

        # Generate comprehensive report
        generate_evaluation_report(all_results, args.output_dir, logger)

        # Print summary to console
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"✓ AUC-ROC: {all_results.get('auroc', 'N/A'):.4f}")
        print(f"✓ AUC-PR:  {all_results.get('auprc', 'N/A'):.4f}")
        print(f"✓ Precision: {all_results.get('precision', 'N/A'):.4f}")
        print(f"✓ Recall: {all_results.get('recall', 'N/A'):.4f}")

        if args.drift_analysis:
            print(f"✓ Drift Detection Recall: {all_results.get('drift_detection_recall', 'N/A'):.4f}")
            print(f"✓ Temporal Stability: {all_results.get('temporal_consistency_score', 'N/A'):.4f}")

        print(f"✓ Calibration ECE: {all_results.get('calibration_ece', 'N/A'):.4f}")
        print(f"✓ Results saved to: {args.output_dir}")

        # Check target metrics
        target_metrics = config.evaluation.target_metrics
        met_count = 0
        total_count = 0

        for metric_name, target_value in target_metrics.items():
            actual_value = all_results.get(metric_name, 0)
            met_target = actual_value >= target_value
            total_count += 1

            if met_target:
                met_count += 1

            status = "✓" if met_target else "✗"
            print(f"{status} {metric_name}: {actual_value:.4f} (target: {target_value:.4f})")

        print(f"\n✓ Met {met_count}/{total_count} target metrics")
        print("="*80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)