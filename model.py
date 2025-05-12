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
from config import INPUT_DIM, SEQ_LEN, DEVICE, MODEL_PATH, SCALER_PATH, SARIMA_CONFIGS, FEATURES

def setup_logger():
    """
    Set up logging configuration.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

class SARIMAForecaster:
    """
    SARIMA forecaster for individual time-series features.
    """
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.is_fitted = False

    def fit(self, data):
        """
        Fit SARIMA model on the data.
        
        Args:
            data (np.ndarray): Time-series data (n_samples,).
        """
        try:
            self.model = SARIMAX(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model = self.model.fit(disp=False)
            self.is_fitted = True
            logging.info("[✔️] SARIMA model fitted successfully")
        except Exception as e:
            logging.error(f"[❌] Error fitting SARIMA model: {str(e)}")
            self.is_fitted = False

    def forecast(self, steps):
        """
        Forecast future values.
        
        Args:
            steps (int): Number of steps to forecast.
        
        Returns:
            np.ndarray: Forecasted values.
        """
        if not self.is_fitted or self.model is None:
            logging.warning("[⚠️] SARIMA model not fitted, returning zeros")
            return np.zeros(steps)
        try:
            forecast = self.model.forecast(steps=steps)
            return forecast
        except Exception as e:
            logging.error(f"[❌] Error in SARIMA forecast: {str(e)}")
            return np.zeros(steps)

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

def train_model(data, seq_len, epochs, batch_size, model_path, scaler_path, resume_training=False, sarima_configs=None):
    """
    Train SARIMA-EE-LSTM model.
    
    Args:
        data (np.ndarray): Training data of shape (n_samples, n_features).
        seq_len (int): Sequence length.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        model_path (str): Path to save the model.
        scaler_path (str): Path to save the scaler.
        resume_training (bool): Resume training from existing model.
        sarima_configs (dict): SARIMA configurations for each feature.
    
    Returns:
        tuple: (EELSTM model, list of SARIMA forecasters).
    """
    setup_logger()
    try:
        # Validate SARIMA configurations
        if sarima_configs is None or not sarima_configs:
            logging.error("[❌] SARIMA_CONFIGS not provided")
            raise ValueError("SARIMA_CONFIGS is required")

        # Fit scaler
        scaler = fit_scaler(data)
        save_scaler(scaler, scaler_path)

        # Preprocess data
        X, y = preprocess_for_training(data, scaler, seq_len)
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Initialize model
        model = EELSTM(input_dim=INPUT_DIM).to(DEVICE)
        if resume_training and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            logging.info(f"[✔️] Loaded existing model from {model_path}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train SARIMA forecasters
        sarima_forecasters = []
        for i, feature in enumerate(FEATURES):
            config = sarima_configs.get(feature, {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)})
            forecaster = SARIMAForecaster(
                order=config["order"],
                seasonal_order=config["seasonal_order"]
            )
            forecaster.fit(data[:, i])
            sarima_forecasters.append(forecaster)
            logging.info(f"[✔️] Fitted SARIMA for {feature}")

        # Train LSTM
        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            logging.info(f"[DEBUG] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Save model
        torch.save(model.state_dict(), model_path)
        logging.info(f"[✔️] Model saved to {model_path}")

        return model, sarima_forecasters

    except Exception as e:
        logging.error(f"[❌] Error in train_model: {str(e)}")
        return None, []

def load_model(model_path, scaler_path):
    """
    Load trained SARIMA-EE-LSTM model and scaler.
    
    Args:
        model_path (str): Path to the model.
        scaler_path (str): Path to the scaler.
    
    Returns:
        tuple: (EELSTM model, scaler, list of SARIMA forecasters).
    """
    try:
        model = EELSTM(input_dim=INPUT_DIM).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        scaler = load_scaler(scaler_path)
        
        # Load SARIMA forecasters (re-fit on load)
        sarima_forecasters = []
        for feature in FEATURES:
            config = SARIMA_CONFIGS.get(feature, {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)})
            forecaster = SARIMAForecaster(
                order=config["order"],
                seasonal_order=config["seasonal_order"]
            )
            sarima_forecasters.append(forecaster)
            logging.info(f"[✔️] Initialized SARIMA for {feature} (fit required)")

        logging.info(f"[✔️] Loaded model from {model_path}")
        return model, scaler, sarima_forecasters

    except Exception as e:
        logging.error(f"[❌] Error loading model: {str(e)}")
        return None, None, []

def forecast(model, data, scaler, sarima_forecasters, steps):
    """
    Forecast future values using SARIMA-EE-LSTM.
    
    Args:
        model (EELSTM): Trained model.
        data (np.ndarray): Input data of shape (n_samples, n_features).
        scaler: Fitted scaler.
        sarima_forecasters (list): List of SARIMA forecasters.
        steps (int): Number of steps to forecast.
    
    Returns:
        np.ndarray: Forecasted values.
    """
    try:
        model.eval()
        with torch.no_grad():
            input_seq = preprocess_for_forecast(data, scaler, SEQ_LEN)
            input_seq = input_seq.to(DEVICE)
            lstm_forecast = model(input_seq).cpu().numpy()
            lstm_forecast = inverse_transform_data(lstm_forecast, scaler)

        sarima_forecasts = []
        for i, forecaster in enumerate(sarima_forecasters):
            sarima_forecast = forecaster.forecast(steps)
            sarima_forecasts.append(sarima_forecast)

        sarima_forecasts = np.array(sarima_forecasts).T
        combined_forecast = 0.5 * lstm_forecast + 0.5 * sarima_forecasts[:1]
        return combined_forecast

    except Exception as e:
        logging.error(f"[❌] Error in forecast: {str(e)}")
        return np.zeros((steps, INPUT_DIM))