"""Advanced evaluation metrics for drift-aware fraud detection systems."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, log_loss, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import scipy.stats as stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DriftDetectionMetrics:
    """Metrics for evaluating drift detection performance."""

    def __init__(self, random_seed: int = 42):
        """Initialize drift detection metrics.

        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def evaluate_drift_detection(self,
                                ensemble: Any,
                                drift_periods: List[pd.DataFrame],
                                reference_data: pd.DataFrame,
                                target_recall: float = 0.85) -> Dict[str, float]:
        """Evaluate drift detection capabilities of the ensemble.

        Args:
            ensemble: Trained ensemble model with adversarial validator.
            drift_periods: List of data from different time periods.
            reference_data: Reference training data.
            target_recall: Target recall for drift detection.

        Returns:
            Dictionary with drift detection metrics.
        """
        logger.info("Evaluating drift detection performance")

        if len(drift_periods) < 2:
            logger.warning("Need at least 2 drift periods for evaluation")
            return {'drift_detection_recall': 0.0, 'drift_detection_precision': 0.0}

        # Use first period as reference, others as drift candidates
        reference_period = drift_periods[0]
        drift_candidates = drift_periods[1:]

        drift_scores = []
        true_labels = []

        # Get feature columns
        feature_cols = [col for col in reference_data.columns
                       if col not in ['isFraud', 'TransactionID']]

        ref_features = reference_data[feature_cols] if isinstance(reference_data, pd.DataFrame) else reference_data

        # Evaluate each period against reference
        for i, period in enumerate(drift_candidates):
            try:
                period_features = period[feature_cols] if isinstance(period, pd.DataFrame) else period

                # Detect drift using ensemble's adversarial validator
                if hasattr(ensemble, 'detect_drift'):
                    drift_result = ensemble.detect_drift(period_features, ref_features)
                    drift_score = drift_result.get('drift_score', 0.5)
                else:
                    # Fallback: use prediction distribution difference
                    ref_preds = ensemble.predict_proba(ref_features)
                    period_preds = ensemble.predict_proba(period_features)
                    drift_score = self._compute_distribution_distance(ref_preds, period_preds)

                drift_scores.append(drift_score)
                # Assume later periods have more drift
                true_labels.append(1 if i >= len(drift_candidates) // 2 else 0)

            except Exception as e:
                logger.warning(f"Failed to evaluate drift for period {i}: {e}")
                drift_scores.append(0.5)
                true_labels.append(0)

        if not drift_scores:
            return {'drift_detection_recall': 0.0, 'drift_detection_precision': 0.0}

        # Convert to arrays
        drift_scores = np.array(drift_scores)
        true_labels = np.array(true_labels)

        # Calculate metrics
        if len(np.unique(true_labels)) > 1:
            # Find threshold for target recall
            thresholds = np.linspace(0.3, 0.9, 100)
            best_threshold = 0.6
            best_f1 = 0.0

            for thresh in thresholds:
                pred_labels = (drift_scores > thresh).astype(int)
                if np.sum(pred_labels) == 0:
                    continue

                tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()

                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh

            # Final evaluation with best threshold
            final_pred_labels = (drift_scores > best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(true_labels, final_pred_labels, labels=[0, 1]).ravel()

            recall = tp / (tp + fn) if tp + fn > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            specificity = tn / (tn + fp) if tn + fp > 0 else 0

            try:
                auc = roc_auc_score(true_labels, drift_scores)
            except:
                auc = 0.5

        else:
            recall = precision = specificity = auc = 0.0

        return {
            'drift_detection_recall': float(recall),
            'drift_detection_precision': float(precision),
            'drift_detection_specificity': float(specificity),
            'drift_detection_auc': float(auc),
            'drift_detection_f1': float(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
        }

    def _compute_distribution_distance(self, ref_preds: np.ndarray, new_preds: np.ndarray) -> float:
        """Compute distribution distance between predictions.

        Args:
            ref_preds: Reference predictions.
            new_preds: New predictions.

        Returns:
            Distribution distance score.
        """
        try:
            # Kolmogorov-Smirnov test
            ks_stat, _ = stats.ks_2samp(ref_preds.flatten(), new_preds.flatten())
            return float(ks_stat)
        except:
            # Fallback to mean difference
            return float(abs(np.mean(new_preds) - np.mean(ref_preds)))

    def temporal_stability_score(self,
                               predictions_by_period: Dict[str, np.ndarray],
                               true_labels_by_period: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate temporal stability metrics.

        Args:
            predictions_by_period: Dictionary mapping period names to predictions.
            true_labels_by_period: Dictionary mapping period names to true labels.

        Returns:
            Dictionary with stability metrics.
        """
        logger.info("Calculating temporal stability scores")

        period_aucs = []
        period_aps = []

        for period, preds in predictions_by_period.items():
            if period in true_labels_by_period:
                labels = true_labels_by_period[period]

                if len(np.unique(labels)) > 1:  # Need both classes
                    try:
                        auc = roc_auc_score(labels, preds)
                        ap = average_precision_score(labels, preds)
                        period_aucs.append(auc)
                        period_aps.append(ap)
                    except:
                        continue

        if not period_aucs:
            return {
                'temporal_stability_auc_std': 1.0,  # Maximum instability
                'temporal_stability_ap_std': 1.0,
                'temporal_consistency_score': 0.0
            }

        auc_std = float(np.std(period_aucs))
        ap_std = float(np.std(period_aps))

        # Consistency score: 1 - normalized standard deviation
        consistency_score = float(1.0 - min(auc_std / np.mean(period_aucs), 1.0))

        return {
            'temporal_stability_auc_std': auc_std,
            'temporal_stability_ap_std': ap_std,
            'temporal_consistency_score': consistency_score,
            'temporal_auc_mean': float(np.mean(period_aucs)),
            'temporal_ap_mean': float(np.mean(period_aps))
        }


class CalibrationMetrics:
    """Metrics for evaluating probability calibration."""

    def __init__(self, n_bins: int = 10):
        """Initialize calibration metrics.

        Args:
            n_bins: Number of bins for calibration curve.
        """
        self.n_bins = n_bins

    def evaluate_calibration(self,
                           y_true: np.ndarray,
                           y_prob: np.ndarray,
                           normalize: bool = False) -> Dict[str, float]:
        """Evaluate probability calibration.

        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            normalize: Whether to normalize probabilities.

        Returns:
            Dictionary with calibration metrics.
        """
        logger.info("Evaluating probability calibration")

        if normalize:
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

        try:
            # Expected Calibration Error (ECE)
            ece = self._expected_calibration_error(y_true, y_prob, self.n_bins)

            # Maximum Calibration Error (MCE)
            mce = self._maximum_calibration_error(y_true, y_prob, self.n_bins)

            # Brier Score
            brier_score = brier_score_loss(y_true, y_prob)

            # Reliability diagram statistics
            reliability_stats = self._reliability_statistics(y_true, y_prob, self.n_bins)

            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=self.n_bins, normalize=False
            )

            # Hosmer-Lemeshow test approximation
            hl_chi2 = self._hosmer_lemeshow_test(y_true, y_prob, self.n_bins)

            return {
                'calibration_ece': float(ece),
                'calibration_mce': float(mce),
                'calibration_brier_score': float(brier_score),
                'calibration_reliability_correlation': reliability_stats['correlation'],
                'calibration_reliability_slope': reliability_stats['slope'],
                'calibration_hl_chi2': float(hl_chi2),
                'calibration_mean_confidence': float(np.mean(y_prob)),
                'calibration_mean_accuracy': float(np.mean(y_true)),
                'calibration_overconfidence': float(np.mean(y_prob) - np.mean(y_true))
            }

        except Exception as e:
            logger.error(f"Failed to compute calibration metrics: {e}")
            return {
                'calibration_ece': 1.0,  # Worst case
                'calibration_mce': 1.0,
                'calibration_brier_score': 0.25,
                'calibration_reliability_correlation': 0.0,
                'calibration_reliability_slope': 0.0,
                'calibration_hl_chi2': 100.0,
                'calibration_mean_confidence': 0.5,
                'calibration_mean_accuracy': float(np.mean(y_true)),
                'calibration_overconfidence': 0.0
            }

    def _expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """Calculate Expected Calibration Error.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            n_bins: Number of bins.

        Returns:
            Expected calibration error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(y_prob)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Identify samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = float(in_bin.sum()) / total_samples

            if prop_in_bin > 0:
                accuracy_in_bin = float(y_true[in_bin].mean())
                avg_confidence_in_bin = float(y_prob[in_bin].mean())
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _maximum_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """Calculate Maximum Calibration Error.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            n_bins: Number of bins.

        Returns:
            Maximum calibration error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_error = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = float(y_true[in_bin].mean())
                avg_confidence_in_bin = float(y_prob[in_bin].mean())
                error = abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)

        return max_error

    def _reliability_statistics(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> Dict[str, float]:
        """Calculate reliability diagram statistics.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            n_bins: Number of bins.

        Returns:
            Dictionary with reliability statistics.
        """
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins, normalize=False
            )

            # Remove bins with no samples
            valid_bins = ~np.isnan(fraction_of_positives) & ~np.isnan(mean_predicted_value)
            fraction_of_positives = fraction_of_positives[valid_bins]
            mean_predicted_value = mean_predicted_value[valid_bins]

            if len(fraction_of_positives) < 2:
                return {'correlation': 0.0, 'slope': 0.0}

            # Calculate correlation and slope
            correlation = float(np.corrcoef(mean_predicted_value, fraction_of_positives)[0, 1])

            # Linear regression slope
            if np.var(mean_predicted_value) > 0:
                slope = float(np.cov(mean_predicted_value, fraction_of_positives)[0, 1] / np.var(mean_predicted_value))
            else:
                slope = 0.0

            return {
                'correlation': correlation if not np.isnan(correlation) else 0.0,
                'slope': slope if not np.isnan(slope) else 0.0
            }

        except Exception as e:
            logger.warning(f"Failed to compute reliability statistics: {e}")
            return {'correlation': 0.0, 'slope': 0.0}

    def _hosmer_lemeshow_test(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """Approximate Hosmer-Lemeshow goodness-of-fit test.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            n_bins: Number of bins.

        Returns:
            Chi-square statistic.
        """
        try:
            # Sort by predicted probability
            sorted_indices = np.argsort(y_prob)
            y_true_sorted = y_true[sorted_indices]
            y_prob_sorted = y_prob[sorted_indices]

            # Create bins
            n_samples = len(y_prob)
            bin_size = n_samples // n_bins

            chi2 = 0.0

            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_samples

                observed_positive = float(np.sum(y_true_sorted[start_idx:end_idx]))
                expected_positive = float(np.sum(y_prob_sorted[start_idx:end_idx]))

                observed_negative = float(end_idx - start_idx - observed_positive)
                expected_negative = float(end_idx - start_idx - expected_positive)

                # Add small constant to avoid division by zero
                expected_positive += 1e-8
                expected_negative += 1e-8

                chi2 += ((observed_positive - expected_positive) ** 2) / expected_positive
                chi2 += ((observed_negative - expected_negative) ** 2) / expected_negative

            return chi2

        except Exception as e:
            logger.warning(f"Failed to compute Hosmer-Lemeshow test: {e}")
            return 100.0  # Large value indicating poor fit


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics for fraud detection systems."""

    def __init__(self,
                 drift_metrics: Optional[DriftDetectionMetrics] = None,
                 calibration_metrics: Optional[CalibrationMetrics] = None):
        """Initialize comprehensive evaluator.

        Args:
            drift_metrics: Drift detection metrics instance.
            calibration_metrics: Calibration metrics instance.
        """
        self.drift_metrics = drift_metrics or DriftDetectionMetrics()
        self.calibration_metrics = calibration_metrics or CalibrationMetrics()

    def full_evaluation(self,
                       model: Any,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       drift_periods: Optional[List[pd.DataFrame]] = None,
                       reference_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Perform comprehensive evaluation of fraud detection model.

        Args:
            model: Trained model to evaluate.
            X_test: Test features.
            y_test: Test labels.
            drift_periods: Optional drift periods for temporal evaluation.
            reference_data: Optional reference data for drift detection.

        Returns:
            Dictionary with comprehensive evaluation results.
        """
        logger.info("Performing comprehensive evaluation")

        results = {}

        try:
            # Basic predictions
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            # Basic classification metrics
            basic_metrics = self._compute_basic_metrics(y_test, y_pred, y_pred_proba)
            results.update(basic_metrics)

            # Calibration evaluation
            calibration_results = self.calibration_metrics.evaluate_calibration(y_test, y_pred_proba)
            results.update(calibration_results)

            # Drift detection evaluation
            if drift_periods and reference_data is not None:
                drift_results = self.drift_metrics.evaluate_drift_detection(
                    model, drift_periods, reference_data
                )
                results.update(drift_results)

            # Temporal stability if multiple periods available
            if drift_periods and len(drift_periods) > 1:
                temporal_results = self._evaluate_temporal_performance(model, drift_periods)
                results.update(temporal_results)

            # Business metrics
            business_metrics = self._compute_business_metrics(y_test, y_pred, y_pred_proba)
            results.update(business_metrics)

            logger.info(f"Comprehensive evaluation completed with {len(results)} metrics")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results = {'evaluation_error': str(e)}

        return results

    def _compute_basic_metrics(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Dictionary with basic metrics.
        """
        try:
            # Primary metrics
            auc = roc_auc_score(y_true, y_pred_proba)
            ap = average_precision_score(y_true, y_pred_proba)

            # Confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            specificity = tn / (tn + fp) if tn + fp > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            # Additional metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            balanced_accuracy = (recall + specificity) / 2

            # Log loss
            try:
                logloss = log_loss(y_true, y_pred_proba)
            except:
                logloss = 10.0  # Large value for failed computation

            return {
                'auroc': float(auc),
                'auprc': float(ap),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1),
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_accuracy),
                'log_loss': float(logloss),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }

        except Exception as e:
            logger.error(f"Failed to compute basic metrics: {e}")
            return {
                'auroc': 0.5,
                'auprc': float(np.mean(y_true)),
                'precision': 0.0,
                'recall': 0.0,
                'specificity': 0.0,
                'f1_score': 0.0,
                'accuracy': float(np.mean(y_true == y_pred)),
                'balanced_accuracy': 0.5,
                'log_loss': 10.0
            }

    def _evaluate_temporal_performance(self, model: Any, drift_periods: List[pd.DataFrame]) -> Dict[str, float]:
        """Evaluate model performance across temporal periods.

        Args:
            model: Trained model.
            drift_periods: List of temporal periods.

        Returns:
            Dictionary with temporal performance metrics.
        """
        period_metrics = {}
        aucs = []
        aps = []

        for i, period in enumerate(drift_periods):
            if 'isFraud' not in period.columns:
                continue

            try:
                # Get features and labels
                feature_cols = [col for col in period.columns if col not in ['isFraud', 'TransactionID']]
                X_period = period[feature_cols]
                y_period = period['isFraud']

                if len(y_period.unique()) < 2:
                    continue

                # Make predictions
                y_pred_proba = model.predict_proba(X_period)

                # Calculate metrics
                auc = roc_auc_score(y_period, y_pred_proba)
                ap = average_precision_score(y_period, y_pred_proba)

                aucs.append(auc)
                aps.append(ap)

                period_metrics[f'period_{i}_auc'] = float(auc)
                period_metrics[f'period_{i}_ap'] = float(ap)

            except Exception as e:
                logger.warning(f"Failed to evaluate period {i}: {e}")

        # Compute temporal stability metrics
        if aucs:
            period_metrics.update({
                'temporal_auc_mean': float(np.mean(aucs)),
                'temporal_auc_std': float(np.std(aucs)),
                'temporal_ap_mean': float(np.mean(aps)),
                'temporal_ap_std': float(np.std(aps)),
                'temporal_stability': float(1.0 - np.std(aucs) / np.mean(aucs) if np.mean(aucs) > 0 else 0)
            })

        return period_metrics

    def _compute_business_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute business-relevant metrics for fraud detection.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Dictionary with business metrics.
        """
        try:
            # Precision at different recall levels
            precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)

            # Find precision at specific recall levels
            target_recalls = [0.5, 0.7, 0.8, 0.9, 0.95]
            precision_at_recall = {}

            for target_recall in target_recalls:
                # Find closest recall
                idx = np.argmin(np.abs(recalls - target_recall))
                precision_at_recall[f'precision_at_recall_{int(target_recall*100)}'] = float(precisions[idx])

            # Cost-sensitive metrics (assuming fraud detection costs)
            # Cost of false positive: 1 unit
            # Cost of false negative: 10 units (missed fraud is expensive)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            total_cost = fp * 1 + fn * 10
            cost_per_transaction = total_cost / len(y_true)

            # Detection rate at low false positive rates
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            detection_at_low_fpr = {}
            target_fprs = [0.001, 0.005, 0.01, 0.05, 0.1]

            for target_fpr in target_fprs:
                idx = np.argmin(np.abs(fpr - target_fpr))
                detection_at_low_fpr[f'tpr_at_fpr_{target_fpr:.3f}'.replace('.', '_')] = float(tpr[idx])

            business_metrics = {
                'total_cost': float(total_cost),
                'cost_per_transaction': float(cost_per_transaction),
                'fraud_catch_rate': float(tp / (tp + fn) if tp + fn > 0 else 0),
                'false_alarm_rate': float(fp / (fp + tn) if fp + tn > 0 else 0),
                **precision_at_recall,
                **detection_at_low_fpr
            }

            return business_metrics

        except Exception as e:
            logger.error(f"Failed to compute business metrics: {e}")
            return {
                'total_cost': float(len(y_true)),
                'cost_per_transaction': 1.0,
                'fraud_catch_rate': 0.0,
                'false_alarm_rate': 1.0
            }