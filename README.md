# Temporal Drift-Aware Fraud Detection with Adversarial Validation

A production-ready fraud detection system that explicitly models temporal distribution shift by training adversarial validators to detect when the production distribution diverges from training data, then dynamically reweights ensemble members (LightGBM, CatBoost, XGBoost) based on their robustness to detected drift regimes.

## Overview

Unlike standard fraud models that degrade silently over time, this pipeline quantifies prediction reliability per-transaction and triggers automated retraining signals when ensemble disagreement exceeds calibrated thresholds. The system addresses the critical challenge of temporal drift in fraud detection where data distributions evolve continuously due to changing attack patterns, seasonal effects, and behavioral shifts.

## Key Features

- **Adversarial Validation**: Dedicated models to detect distribution drift between training and production data
- **Dynamic Ensemble Reweighting**: Automatically adjusts model weights based on drift robustness
- **Temporal Awareness**: Explicit modeling of time-based distribution shifts
- **Production-Ready Architecture**: Comprehensive monitoring, calibration, and retraining triggers
- **Multi-Framework Ensemble**: Combines LightGBM, XGBoost, and CatBoost for robust predictions

## Quick Start

### Installation

```bash
pip install temporal-drift-aware-fraud-detection-with-adversarial-validation
```

### Basic Usage

```python
from temporal_drift_aware_fraud_detection_with_adversarial_validation import DriftAwareTrainer
from temporal_drift_aware_fraud_detection_with_adversarial_validation.utils.config import load_config

# Load configuration
config = load_config('configs/default.yaml')

# Initialize trainer
trainer = DriftAwareTrainer(config=config.to_dict())

# Train the system (uses synthetic IEEE-CIS-like data if no data path provided)
results = trainer.train(train_data=your_data, save_best=True)

# Make predictions with drift monitoring
probs, preds = trainer.predict(new_data)

# Evaluate drift
drift_results = trainer.evaluate_drift(new_data, reference_data)
```

### Command Line Interface

Train the system:
```bash
python scripts/train.py --config configs/default.yaml --output-dir models/
```

Evaluate trained models:
```bash
python scripts/evaluate.py --model-path models/ --output-dir evaluation_results/
```

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| AUC-ROC | 0.94 | Run `python scripts/train.py` to reproduce |
| AUC-PR | 0.65 | Run `python scripts/train.py` to reproduce |
| Drift Detection Recall | 0.85 | Run `python scripts/train.py` to reproduce |
| Calibration ECE | 0.03 | Run `python scripts/train.py` to reproduce |

## Architecture

The system consists of several key components:

1. **Data Processing Pipeline**: Handles IEEE-CIS fraud data with temporal feature engineering
2. **Adversarial Validators**: Detect distribution drift using domain classification
3. **Ensemble Framework**: Combines multiple gradient boosting models with dynamic weighting
4. **Drift Monitoring**: Real-time assessment of prediction reliability
5. **Calibration System**: Ensures probabilistic outputs are well-calibrated
6. **Retraining Triggers**: Automated signals when model performance degrades

## Technical Implementation

- **Ensemble Models**: LightGBM, XGBoost, CatBoost with hyperparameter optimization
- **Drift Detection**: Binary classification between time periods using feature distributions
- **Temporal Splitting**: Realistic evaluation using time-based train/validation/test splits
- **Calibration**: Isotonic regression for probability calibration
- **Monitoring**: Comprehensive metrics tracking including business impact measures

## Configuration

The system is highly configurable via YAML files. Key parameters include:

- Model hyperparameters for each ensemble member
- Drift detection thresholds and sensitivity
- Temporal splitting ratios and period definitions
- Calibration and evaluation settings
- MLflow experiment tracking configuration

## Testing

### With Full Dependencies

Run the comprehensive test suite (requires ML dependencies):

```bash
pytest tests/ -v
```

For quick training test:
```bash
python scripts/train.py --quick-test
```

### Without Dependencies (Test Mode)

For testing the project structure and basic functionality without ML dependencies:

```bash
# Run comprehensive tests that work without ML libraries
python tests/test_comprehensive.py

# Run test mode training (generates mock results)
python scripts/train_test_mode.py
```

The test mode demonstrates the complete training pipeline with realistic mock data and results, suitable for:
- CI/CD environments without ML dependencies
- Project structure validation
- Basic functionality testing

## Project Structure

```
temporal-drift-aware-fraud-detection-with-adversarial-validation/
├── src/temporal_drift_aware_fraud_detection_with_adversarial_validation/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model implementations
│   ├── training/       # Training pipeline
│   ├── evaluation/     # Metrics and evaluation
│   └── utils/          # Configuration and utilities
├── scripts/            # Training and evaluation scripts
├── tests/              # Comprehensive test suite
├── configs/            # Configuration files
└── notebooks/          # Jupyter notebooks for exploration
```

## Requirements

- Python 3.8+
- LightGBM, XGBoost, CatBoost
- scikit-learn, pandas, numpy
- Optuna for hyperparameter optimization
- MLflow for experiment tracking

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.