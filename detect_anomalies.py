# detect_anomalies.py
"""
This module detects anomalies in time-series data using SARIMA-EE-LSTM and fuzzy logic.
It evaluates anomaly risks for each feature based on predictions and dynamic bounds.
"""

import torch
import numpy as np
import torch.nn.functional as F
from preprocessing import preprocess_for_forecast, load_scaler
from model import load_model
from config import MODEL_PATH, SEQ_LEN, FEATURES, INPUT_DIM, DEVICE, SCALER_PATH
from fuzzy import evaluate_fuzzy_anomaly

def detect_anomaly(raw_data_batch, model=None):
    """
    Detect anomalies for all features using SARIMA-EE-LSTM and fuzzy logic.

    Args:
        raw_data_batch (np.ndarray): Input data of shape (n_samples, n_features)
        model (tuple, optional): Tuple of (model, scaler, sarima_forecasters). If None, loaded from MODEL_PATH.

    Returns:
        dict: Results for each feature with keys:
              - is_anomaly (bool): True if anomaly detected
              - mse (float): Mean squared error
              - risk_score (float): Fuzzy anomaly risk (0 to 1)
              - predicted (float): Predicted value
              - actual (float): Actual value
    """
    try:
        # Convert input to numpy array
        raw_data_batch = np.asarray(raw_data_batch, dtype=np.float32)
        if raw_data_batch.ndim == 1:
            raw_data_batch = raw_data_batch.reshape(1, -1)

        # Validate input dimensions
        if raw_data_batch.shape[1] != INPUT_DIM:
            raise ValueError(f"[❌] Invalid number of features: expected {INPUT_DIM}, got {raw_data_batch.shape[1]}")
        if raw_data_batch.shape[0] < SEQ_LEN:
            raise ValueError(f"[❌] Not enough samples: expected at least {SEQ_LEN}, got {raw_data_batch.shape[0]}")

        # Load model and scaler if not provided
        if model is None:
            model_tuple = load_model(MODEL_PATH, SCALER_PATH)
            if model_tuple is None:
                raise ValueError("[❌] Failed to load model")
            model, scaler, sarima_forecasters = model_tuple
        else:
            model, scaler, sarima_forecasters = model

        if scaler is None:
            raise ValueError("[❌] Failed to load scaler")

        # Normalize the entire batch at once
        data_normalized = scaler.transform(raw_data_batch)

        # Initialize results
        results = {}
        model.eval()

        for idx, feature in enumerate(FEATURES):
            try:
                # Create a full feature array for preprocessing
                feature_data = np.zeros((raw_data_batch.shape[0], INPUT_DIM))
                feature_data[:, idx] = data_normalized[:, idx]

                # Preprocess for forecasting
                input_tensor = preprocess_for_forecast(feature_data, scaler, seq_len=SEQ_LEN)
                input_tensor = input_tensor.to(DEVICE)

                # Predict
                with torch.no_grad():
                    predicted = model(input_tensor)

                # Convert to numpy and handle the prediction shape
                predicted = predicted.cpu().numpy()
                if predicted.ndim > 1:
                    predicted = predicted.squeeze()
                if predicted.size > 1:
                    predicted = predicted[idx]  # Get the prediction for this feature
                
                actual = raw_data_batch[-1, idx]

                # Create dummy arrays for denormalization
                predicted_dummy = np.zeros((1, INPUT_DIM))
                predicted_dummy[0, idx] = predicted
                predicted_denorm = scaler.inverse_transform(predicted_dummy)[0, idx]

                actual_dummy = np.zeros((1, INPUT_DIM))
                actual_dummy[0, idx] = actual
                actual_denorm = scaler.inverse_transform(actual_dummy)[0, idx]

                # Calculate MSE
                mse = float(F.mse_loss(
                    torch.tensor(predicted_denorm, dtype=torch.float32),
                    torch.tensor(actual_denorm, dtype=torch.float32)
                ).item())

                # Calculate dynamic bounds (mean ± 2*std)
                feature_history = raw_data_batch[:, idx]
                mean_val = np.mean(feature_history)
                std_val = np.std(feature_history) if len(feature_history) > 1 else 1.0
                upper_bound = mean_val + 2 * std_val
                lower_bound = mean_val - 2 * std_val

                # Evaluate fuzzy anomaly risk
                risk_score = evaluate_fuzzy_anomaly(
                    y_pred=predicted_denorm,
                    y_true=actual_denorm,
                    upper=upper_bound,
                    lower=lower_bound
                )

                # Determine anomaly based on fuzzy risk
                is_anomaly = risk_score > 0.5

                # Store results
                results[feature] = {
                    "is_anomaly": is_anomaly,
                    "mse": mse,
                    "risk_score": risk_score,
                    "predicted": float(predicted_denorm),
                    "actual": float(actual_denorm)
                }

                print(f"[DEBUG] {feature}: Anomaly={is_anomaly}, MSE={mse:.6f}, Risk={risk_score:.2f}, Predicted={predicted_denorm:.2f}, Actual={actual_denorm:.2f}")

            except Exception as e:
                print(f"[❌] Error processing feature {feature}: {str(e)}")
                results[feature] = {
                    "is_anomaly": False,
                    "mse": 0.0,
                    "risk_score": 0.0,
                    "predicted": 0.0,
                    "actual": 0.0
                }

        return results

    except Exception as e:
        print(f"[❌] Error in anomaly detection: {str(e)}")
        # Return default results for all features
        return {
            feature: {
                "is_anomaly": False,
                "mse": 0.0,
                "risk_score": 0.0,
                "predicted": 0.0,
                "actual": 0.0
            } for feature in FEATURES
        }