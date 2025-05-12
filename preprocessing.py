# preprocessing.py
"""
Preprocessing module for SARIMA-EE-LSTM model.
Handles data normalization, sequence creation, and scaler management.
"""

import os
import numpy as np
import torch
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import SEQ_LEN

def setup_logger():
    """
    Set up logging configuration if not already set.
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

def save_scaler(scaler, path):
    """
    Save the fitted scaler to a file.
    
    Args:
        scaler: Fitted MinMaxScaler object.
        path (str): Path to save the scaler.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(scaler, path)
        logging.info(f"[✔️] Scaler saved to {path}")
    except Exception as e:
        logging.error(f"[❌] Error saving scaler to {path}: {str(e)}")
        raise

def load_scaler(path):
    """
    Load the scaler from a file.
    
    Args:
        path (str): Path to the scaler file.
    
    Returns:
        MinMaxScaler: Loaded scaler object or None if failed.
    """
    try:
        if not os.path.exists(path):
            logging.error(f"[❌] Scaler file {path} does not exist")
            return None
        scaler = joblib.load(path)
        logging.info(f"[✔️] Scaler loaded from {path}")
        return scaler
    except Exception as e:
        logging.error(f"[❌] Error loading scaler from {path}: {str(e)}")
        return None

def fit_scaler(data):
    """
    Fit a new scaler on the data.
    
    Args:
        data (np.ndarray): Data to fit the scaler, shape (n_samples, n_features).
    
    Returns:
        MinMaxScaler: Fitted scaler object.
    """
    setup_logger()
    try:
        data = np.array(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        
        # Handle NaN and inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning("[⚠️] Data contains NaN or inf. Replacing with median/mean.")
            for i in range(data.shape[1]):
                col = data[:, i]
                mask = np.isnan(col) | np.isinf(col)
                if i in [0, 1, 3, 4, 5, 7]:  # Rate metrics (positive)
                    fill_value = np.median(col[~mask]) if np.sum(~mask) > 0 else 0.0
                    data[:, i][mask] = fill_value
                    data[:, i] = np.maximum(data[:, i], 0)
                else:
                    fill_value = np.mean(col[~mask]) if np.sum(~mask) > 0 else 0.0
                    data[:, i][mask] = fill_value

        scaler = MinMaxScaler()
        scaler.fit(data)
        logging.info("[✔️] Scaler fitted on data with shape: {}".format(data.shape))
        return scaler
    except Exception as e:
        logging.error(f"[❌] Error fitting scaler: {str(e)}")
        raise

def transform_data(data, scaler):
    """
    Transform data using the provided scaler.
    
    Args:
        data (np.ndarray): Data to transform, shape (n_samples, n_features).
        scaler: Fitted MinMaxScaler object.
    
    Returns:
        np.ndarray: Transformed data.
    """
    setup_logger()
    try:
        if scaler is None:
            logging.error("[❌] No scaler provided for transformation")
            raise ValueError("Scaler is None")
        
        data = np.array(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        
        # Handle NaN and inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning("[⚠️] Data contains NaN or inf. Replacing with median/mean.")
            for i in range(data.shape[1]):
                col = data[:, i]
                mask = np.isnan(col) | np.isinf(col)
                if i in [0, 1, 3, 4, 5, 7]:  # Rate metrics
                    fill_value = np.median(col[~mask]) if np.sum(~mask) > 0 else 0.0
                    data[:, i][mask] = fill_value
                    data[:, i] = np.maximum(data[:, i], 0)
                else:
                    fill_value = np.mean(col[~mask]) if np.sum(~mask) > 0 else 0.0
                    data[:, i][mask] = fill_value

        transformed = scaler.transform(data)
        logging.debug(f"[DEBUG] Transformed data shape: {transformed.shape}")
        return transformed
    except Exception as e:
        logging.error(f"[❌] Error transforming data: {str(e)}")
        raise

def inverse_transform_data(data, scaler):
    """
    Inverse transform data back to original scale.
    
    Args:
        data (np.ndarray): Data to inverse transform, shape (n_samples, n_features).
        scaler: Fitted MinMaxScaler object.
    
    Returns:
        np.ndarray: Inverse transformed data.
    """
    setup_logger()
    try:
        if scaler is None:
            logging.error("[❌] No scaler provided for inverse transformation")
            raise ValueError("Scaler is None")
        
        data = np.array(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        
        inverse = scaler.inverse_transform(data)
        logging.debug(f"[DEBUG] Inverse transformed data shape: {inverse.shape}")
        return inverse
    except Exception as e:
        logging.error(f"[❌] Error inverse transforming data: {str(e)}")
        raise

def preprocess_data(data, scaler, seq_len=SEQ_LEN):
    """
    Preprocess input data for SARIMA-EE-LSTM (for training).
    
    Args:
        data (np.ndarray): Input data, shape (n_samples, n_features).
        scaler: Fitted MinMaxScaler object.
        seq_len (int): Sequence length.
    
    Returns:
        torch.Tensor: Sequences of shape (n_sequences, seq_len, n_features).
    """
    setup_logger()
    try:
        data = transform_data(data, scaler)
        data = np.array(data, dtype=np.float32)
        
        if len(data) < seq_len:
            raise ValueError(f"Data length ({len(data)}) is less than seq_len ({seq_len})")
        
        # Use stride_tricks for efficient sequence creation
        shape = (len(data) - seq_len + 1, seq_len, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        sequences = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        sequences = torch.tensor(sequences, dtype=torch.float32)
        
        logging.info(f"[✔️] Preprocessed data into {sequences.shape[0]} sequences")
        return sequences
    except Exception as e:
        logging.error(f"[❌] Error preprocessing data: {str(e)}")
        raise

def preprocess_for_training(data, scaler, seq_len=SEQ_LEN):
    """
    Preprocess data for training (sequence + next-step label).
    
    Args:
        data (np.ndarray): Input data, shape (n_samples, n_features).
        scaler: Fitted MinMaxScaler object.
        seq_len (int): Sequence length.
    
    Returns:
        tuple: (X, y) where X is (n_sequences, seq_len, n_features) and y is (n_sequences, n_features).
    """
    setup_logger()
    try:
        sequences = preprocess_data(data, scaler, seq_len=seq_len)
        if sequences.shape[0] < 2:
            raise ValueError("Not enough sequences for training")
        
        X = sequences[:-1]
        y = sequences[1:, -1, :]  # Predict next timestep
        logging.info(f"[✔️] Prepared training data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"[❌] Error in preprocess_for_training: {str(e)}")
        raise

def preprocess_for_forecast(data, scaler, seq_len=SEQ_LEN, forecast_steps=1):
    """
    Prepare sequences for forecasting.
    
    Args:
        data (np.ndarray): Input data, shape (n_samples, n_features).
        scaler: Fitted MinMaxScaler object.
        seq_len (int): Sequence length.
        forecast_steps (int): Number of forecast steps.
    
    Returns:
        torch.Tensor: Sequence of shape (1, seq_len, n_features).
    """
    setup_logger()
    try:
        data = transform_data(data, scaler)
        data = np.array(data, dtype=np.float32)
        
        if data.ndim == 1:
            data = data[:, None]
        
        if len(data) < seq_len:
            raise ValueError(f"Not enough data for forecasting: have {len(data)}, need {seq_len}")
        
        last_seq = data[-seq_len:]
        tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len, n_features)
        logging.info(f"[✔️] Prepared forecast sequence: shape {tensor.shape}")
        return tensor
    except Exception as e:
        logging.error(f"[❌] Error in preprocess_for_forecast: {str(e)}")
        raise