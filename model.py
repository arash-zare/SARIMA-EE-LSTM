# model.py
"""
SARIMA-EE-LSTM model for time-series forecasting and anomaly detection.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from preprocessing import preprocess_for_training, preprocess_for_forecast, fit_scaler, save_scaler, load_scaler, inverse_transform_data
from config import INPUT_DIM, SEQ_LEN, DEVICE, MODEL_PATH, SCALER_PATH, FEATURES, THRESHOLDS, SARIMA_CONFIGS

class EELSTM(nn.Module):
    """
    Ensemble Encoder LSTM for time-series forecasting.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(EELSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_dim).
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class SARIMAForecaster:
    """SARIMA model for time series forecasting."""
    
    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)):
        """
        Initialize SARIMA model.
        
        Args:
            order (tuple): (p, d, q) parameters
            seasonal_order (tuple): (P, D, Q, s) parameters
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.maxiter = 300
        self.method = 'nm'
        self.disp = False
        self.trend = 'c'
        self.enforce_stationarity = False
        self.enforce_invertibility = False
        self.is_fitted = False
    
    def fit(self, data):
        """
        Fit SARIMA model to data.
        
        Args:
            data (np.ndarray): Time series data
        """
        try:
            if len(data) < 10:  # Minimum data length check
                logging.warning(f"[⚠️] Insufficient data for SARIMA fitting: {len(data)} points")
                return None
                
            # Add small constant to handle zero values
            data = data + 1e-6
            
            # Create a pandas Series with a proper index
            import pandas as pd
            data_series = pd.Series(data, index=pd.date_range(start='2024-01-01', periods=len(data), freq='s'))
            
            # Initialize model with improved parameters
            self.model = SARIMAX(
                data_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            
            # Fit with improved optimization settings
            self.model = self.model.fit(
                maxiter=self.maxiter,
                method=self.method,
                disp=self.disp,
                optim_score='approx',
                optim_complex_step=False,
                optim_hessian='approx',
                low_memory=True,
                start_params=None,
                bounds=None
            )
            
            self.is_fitted = True
            return self.model
            
        except Exception as e:
            logging.error(f"[❌] Error fitting SARIMA model: {str(e)}")
            self.is_fitted = False
            return None
    
    def forecast(self, data):
        """
        Generate forecast for the next step.
        
        Args:
            data (np.ndarray): Input data for forecasting
            
        Returns:
            np.ndarray: Forecast values
        """
        try:
            if not self.is_fitted:
                # Try to fit the model if not already fitted
                if self.fit(data) is None:
                    raise ValueError("Model not fitted")
            
            # Create a pandas Series with a proper index
            import pandas as pd
            data_series = pd.Series(data, index=pd.date_range(start='2024-01-01', periods=len(data), freq='s'))
            
            # Get forecast for the next step
            forecast = self.model.get_forecast(steps=1)
            mean_forecast = forecast.predicted_mean.values
            
            # Ensure forecast is 1D array
            if mean_forecast.ndim > 1:
                mean_forecast = mean_forecast.flatten()
            
            if len(mean_forecast) == 0:
                raise ValueError("Empty SARIMA prediction")
                
            return mean_forecast
            
        except Exception as e:
            logging.error(f"[❌] Error generating SARIMA forecast: {str(e)}")
            return None

def detect_anomalies(actual, predicted, thresholds=None):
    """
    Detect anomalies based on prediction errors and thresholds.
    
    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        thresholds (dict): Thresholds for each feature
        
    Returns:
        dict: Anomaly detection results for each feature
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    
    results = {}
    try:
        # Ensure shapes match by taking the first prediction
        if predicted.shape[0] > actual.shape[0]:
            predicted = predicted[0:actual.shape[0]]
        elif predicted.shape[0] < actual.shape[0]:
            actual = actual[0:predicted.shape[0]]
        
        # Ensure both arrays are 2D
        if actual.ndim == 1:
            actual = actual.reshape(1, -1)
        if predicted.ndim == 1:
            predicted = predicted.reshape(1, -1)
        
        for i, feature in enumerate(FEATURES):
            actual_feature = actual[:, i]
            predicted_feature = predicted[:, i]
            
            # Calculate error metrics
            mse = np.mean((actual_feature - predicted_feature) ** 2)
            mae = np.mean(np.abs(actual_feature - predicted_feature))
            
            # Get threshold for this feature
            threshold = thresholds.get(feature, 1.0)
            
            # Calculate fuzzy risk score
            try:
                # Calculate normalized residual
                residual = abs(actual_feature[0] - predicted_feature[0])
                
                # Handle extreme values
                if residual > 1000:  # Cap extreme residuals
                    residual = 1000
                
                # Normalize residual
                residual = min(residual / threshold, 1.0)
                
                # Calculate bounds
                std = np.std(actual_feature)
                if std == 0:  # Handle zero standard deviation
                    std = 1.0
                
                upper_bound = predicted_feature[0] + 2 * std
                lower_bound = predicted_feature[0] - 2 * std
                
                # Calculate differences from bounds
                upper_diff = max(0.0, actual_feature[0] - upper_bound)  # Positive if actual > upper
                lower_diff = max(0.0, lower_bound - actual_feature[0])  # Positive if actual < lower
                
                # Cap extreme differences
                upper_diff = min(upper_diff, 1000)
                lower_diff = min(lower_diff, 1000)
                
                # Normalize differences
                upper_diff = min(upper_diff / threshold, 1.0)
                lower_diff = min(lower_diff / threshold, 1.0)
                
                # Create fuzzy logic instance
                fuzzy_logic = FuzzyLogic()
                
                # Calculate risk score using fuzzy logic
                risk_score = fuzzy_logic.evaluate_risk(residual, upper_diff, lower_diff)
                
                # Log fuzzy inputs for debugging
                logging.debug(f"[DEBUG] Fuzzy inputs: residual={residual:.2f}, upper_diff={upper_diff:.2f}, lower_diff={lower_diff:.2f}")
                logging.debug(f"[DEBUG] Fuzzy output: risk_score={risk_score:.2f}")
                
            except Exception as e:
                logging.error(f"[❌] Error in fuzzy evaluation: {str(e)}")
                # Use MSE-based risk score as fallback
                if mse > threshold * 2:
                    risk_score = 0.8  # High risk
                elif mse > threshold:
                    risk_score = 0.6  # Medium risk
                else:
                    risk_score = 0.2  # Low risk
                logging.debug(f"[DEBUG] Fallback: Using MSE-based risk score = {risk_score:.2f}")
            
            # Determine if anomaly based on both MSE and fuzzy risk
            is_anomaly = mse > threshold or risk_score > 0.7
            
            results[feature] = {
                "mse": float(mse),
                "mae": float(mae),
                "is_anomaly": bool(is_anomaly),
                "threshold": float(threshold),
                "actual_mean": float(actual_feature.mean()),
                "predicted_mean": float(predicted_feature.mean()),
                "fuzzy_risk": float(risk_score)
            }
            
            # Log the results
            status = "Anomaly" if is_anomaly else "Normal"
            logging.info(f"[✅] {feature}: {status} (MSE: {mse:.4f}, Fuzzy Risk: {risk_score:.4f})")
            
            # Log risk level in Persian with adjusted thresholds
            if risk_score > 0.7:  # High risk threshold
                logging.warning(f"[🚨] {feature}: خطر بالا (fuzzy risk = {risk_score:.2f})")
            elif risk_score > 0.4:  # Medium risk threshold
                logging.warning(f"[⚠️] {feature}: خطر متوسط (fuzzy risk = {risk_score:.2f})")
            else:
                logging.info(f"[✅] {feature}: وضعیت نرمال (fuzzy risk = {risk_score:.2f})")
        
        return results
        
    except Exception as e:
        logging.error(f"[❌] Error in detect_anomalies: {str(e)}")
        return {}

def load_model(model_path, scaler_path):
    """
    Load trained model, scaler, and SARIMA forecasters.
    
    Args:
        model_path (str): Path to model file
        scaler_path (str): Path to scaler file
        
    Returns:
        tuple: (model, scaler, sarima_forecasters)
    """
    try:
        # Load LSTM model
        model = EELSTM(input_dim=INPUT_DIM).to(DEVICE)
        
        # Load the saved state dictionary
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle both old and new model formats
        if isinstance(checkpoint, dict) and 'lstm_state_dict' in checkpoint:
            # New format with nested state dict
            model.load_state_dict(checkpoint['lstm_state_dict'])
            sarima_configs = checkpoint.get('sarima_configs', SARIMA_CONFIGS)
            fuzzy_params = checkpoint.get('fuzzy_params', {})
        else:
            # Old format with direct state dict
            model.load_state_dict(checkpoint)
            sarima_configs = SARIMA_CONFIGS
            fuzzy_params = {}
        
        model.eval()
        
        # Load scaler
        scaler = load_scaler(scaler_path)
        
        # Initialize SARIMA forecasters
        sarima_forecasters = {}
        for feature in FEATURES:
            config = sarima_configs.get(feature, {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)})
            forecaster = SARIMAForecaster(
                order=config["order"],
                seasonal_order=config["seasonal_order"]
            )
            sarima_forecasters[feature] = forecaster
        
        logging.info(f"[✔️] Loaded model from {model_path}")
        return model, scaler, sarima_forecasters
        
    except Exception as e:
        logging.error(f"[❌] Error loading model: {str(e)}")
        return None, None, None

def forecast(model, data, scaler, sarima_forecasters, steps=1):
    """
    Generate forecast using the model and SARIMA forecasters.
    
    Args:
        model (EELSTM): Trained model
        data (np.ndarray): Input data
        scaler: Fitted scaler
        sarima_forecasters (list): List of SARIMA forecasters
        steps (int): Number of steps to forecast
        
    Returns:
        tuple: (predictions, upper_bounds, lower_bounds)
    """
    try:
        model.eval()
        with torch.no_grad():
            # Prepare input sequence
            input_seq = preprocess_for_forecast(data, scaler, SEQ_LEN)
            input_seq = input_seq.to(DEVICE)
            
            # Generate LSTM forecast
            lstm_forecast = model(input_seq).cpu().numpy()
            lstm_forecast = inverse_transform_data(lstm_forecast, scaler)
            
            # Initialize arrays for combined forecast
            combined_forecast = np.zeros((1, len(FEATURES)))
            upper_bounds = np.zeros((1, len(FEATURES)))
            lower_bounds = np.zeros((1, len(FEATURES)))
            
            # Get SARIMA forecasts for each feature
            for i, feature in enumerate(FEATURES):
                try:
                    # Ensure we have enough data for SARIMA
                    if len(data[:, i]) < 10:
                        logging.warning(f"[⚠️] Insufficient data for SARIMA on {feature}, using LSTM forecast")
                        combined_forecast[0, i] = lstm_forecast[0, i]
                    else:
                        # Try to get SARIMA forecast
                        forecaster = sarima_forecasters[feature]
                        sarima_forecast = forecaster.forecast(data[:, i])
                        
                        if sarima_forecast is not None and len(sarima_forecast) > 0:
                            # Combine LSTM and SARIMA forecasts with weights
                            lstm_weight = 0.7  # Give more weight to LSTM
                            sarima_weight = 0.3
                            combined_forecast[0, i] = (lstm_weight * lstm_forecast[0, i] + 
                                                     sarima_weight * sarima_forecast[0])
                        else:
                            logging.warning(f"[⚠️] SARIMA forecast failed for {feature}, using LSTM forecast")
                            combined_forecast[0, i] = lstm_forecast[0, i]
                    
                    # Calculate confidence intervals
                    std = np.std(data[:, i])
                    upper_bounds[0, i] = combined_forecast[0, i] + 2 * std
                    lower_bounds[0, i] = combined_forecast[0, i] - 2 * std
                    
                except Exception as e:
                    logging.warning(f"[⚠️] Error in SARIMA forecast for {feature}: {str(e)}")
                    # Use LSTM forecast as fallback
                    combined_forecast[0, i] = lstm_forecast[0, i]
                    std = np.std(data[:, i])
                    upper_bounds[0, i] = lstm_forecast[0, i] + 2 * std
                    lower_bounds[0, i] = lstm_forecast[0, i] - 2 * std
            
            return combined_forecast, upper_bounds, lower_bounds
            
    except Exception as e:
        logging.error(f"[❌] Error in forecast: {str(e)}")
        return None, None, None 