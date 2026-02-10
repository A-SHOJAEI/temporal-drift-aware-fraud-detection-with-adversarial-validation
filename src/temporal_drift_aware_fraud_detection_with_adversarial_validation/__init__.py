"""Temporal Drift-Aware Fraud Detection with Adversarial Validation.

A production-ready fraud detection system that explicitly models temporal distribution
shift using adversarial validators and dynamic ensemble reweighting.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from temporal_drift_aware_fraud_detection_with_adversarial_validation.models.model import (
    AdversarialValidator,
    DriftAwareEnsemble,
)
from temporal_drift_aware_fraud_detection_with_adversarial_validation.training.trainer import DriftAwareTrainer
from temporal_drift_aware_fraud_detection_with_adversarial_validation.data.loader import DataLoader
from temporal_drift_aware_fraud_detection_with_adversarial_validation.evaluation.metrics import (
    DriftDetectionMetrics,
    CalibrationMetrics,
)

__all__ = [
    "AdversarialValidator",
    "DriftAwareEnsemble",
    "DriftAwareTrainer",
    "DataLoader",
    "DriftDetectionMetrics",
    "CalibrationMetrics",
]