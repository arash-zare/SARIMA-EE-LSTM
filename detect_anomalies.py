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

def detect_anomalies(actual_values, predicted_values, model, scaler, sarima_forecasters, thresholds=None):
    """
    Detect anomalies in time-series data using SARIMA-EE-LSTM and fuzzy logic.
    
    Args:
        actual_values (np.ndarray): Actual values
        predicted_values (np.ndarray): Predicted values
        model: Trained LSTM model
        scaler: Fitted scaler
        sarima_forecasters (list): List of SARIMA forecasters
        thresholds (dict): Optional thresholds for each feature
        
    Returns:
        tuple: (anomalies, metrics)
    """
    try:
        if actual_values.shape != predicted_values.shape:
            raise ValueError(f"Shape mismatch: actual {actual_values.shape} vs predicted {predicted_values.shape}")
        
        # Calculate MSE for each feature
        mse_per_feature = np.mean((actual_values - predicted_values) ** 2, axis=0)
        
        # Initialize results
        anomalies = {}
        metrics = {}
        
        # Check each feature
        for i, feature in enumerate(FEATURES):
            mse = mse_per_feature[i]
            threshold = thresholds.get(feature, THRESHOLDS.get(feature, 1.0))
            
            # Calculate fuzzy risk
            fuzzy_risk = evaluate_fuzzy_anomaly(
                predicted_values[0, i],
                actual_values[0, i],
                predicted_values[0, i] + 2 * np.std(actual_values[:, i]),
                predicted_values[0, i] - 2 * np.std(actual_values[:, i])
            )
            
            # Store results
            anomalies[feature] = mse > threshold
            metrics[feature] = {
                'mse': mse,
                'fuzzy_risk': fuzzy_risk,
                'threshold': threshold
            }
            
            # Log results
            status = "Anomaly" if anomalies[feature] else "Normal"
            logging.info(f"{feature}: {status} (MSE: {mse:.4f}, Fuzzy Risk: {fuzzy_risk:.4f})")
        
        return anomalies, metrics
        
    except Exception as e:
        logging.error(f"[❌] Error in detect_anomalies: {str(e)}")
        return None, None