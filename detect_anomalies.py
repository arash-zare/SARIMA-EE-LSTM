# detect_anomalies.py
"""
This module detects anomalies in time-series data using SARIMA-EE-LSTM and fuzzy logic.
It evaluates anomaly risks for each feature based on predictions and dynamic bounds.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Dict
import logging
from preprocessing import preprocess_for_forecast, load_scaler
from model import load_model
from config import MODEL_PATH, SEQ_LEN, FEATURES, INPUT_DIM, DEVICE, SCALER_PATH, THRESHOLDS, FUZZY_THRESHOLDS
from fuzzy import evaluate_fuzzy_anomaly

def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def detect_anomalies(actual: np.ndarray, predicted: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict:
    """Detect anomalies based on actual and predicted values."""
    setup_logger()
    results = {}
    try:
        if actual.shape != predicted.shape:
            logging.error(f"[❌] Shape mismatch: actual {actual.shape}, predicted {predicted.shape}")
            raise ValueError("Actual and predicted arrays must have the same shape")

        for i, feature in enumerate(FEATURES):
            actual_feature = actual[:, i] if actual.ndim > 1 else actual
            predicted_feature = predicted[:, i] if predicted.ndim > 1 else predicted
            mse = np.mean((actual_feature - predicted_feature) ** 2)
            residual = np.abs(actual_feature - predicted_feature).mean()
            threshold = THRESHOLDS.get(feature, 1.0)
            fuzzy_threshold = FUZZY_THRESHOLDS.get(feature, 0.8)
            fuzzy_risk = evaluate_fuzzy_anomaly(predicted_feature.mean(), actual_feature.mean(), 
                                              predicted_feature.mean() + threshold, 
                                              predicted_feature.mean() - threshold)
            status = "Anomaly" if mse > threshold and mse > 0 else "Normal"
            results[feature] = {
                "fuzzy_risk": fuzzy_risk,
                "status": status,
                "mse": mse,
                "predicted_mean": predicted_feature.mean(),
                "actual_mean": actual_feature.mean()
            }
            logging.info(f"[✅] {feature}: {status} (fuzzy risk = {fuzzy_risk:.2f})")
            logging.debug(f"[DEBUG] {feature}: Anomaly={status}, MSE={mse:.6f}, Risk={fuzzy_risk:.2f}, Predicted={predicted_feature.mean():.2f}, Actual={actual_feature.mean():.2f}")

        # Evaluate against labels if provided
        if labels is not None:
            predicted_anomalies = [1 if results[feature]["status"] == "Anomaly" else 0 for feature in FEATURES]
            accuracy = np.mean([1 if pred == label else 0 for pred, label in zip(predicted_anomalies, labels)])
            logging.info(f"[✅] Anomaly detection accuracy: {accuracy:.2f}")

        return results

    except Exception as e:
        logging.error(f"[❌] Error in detect_anomalies: {str(e)}")
        return {feature: {"fuzzy_risk": 0.0, "status": "Normal", "mse": 0.0} for feature in FEATURES}