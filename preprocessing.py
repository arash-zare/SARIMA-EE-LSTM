# preprocessing.py
"""
Preprocessing module for SARIMA-EE-LSTM model.
Handles data normalization, sequence creation, and scaler management.
"""

import os
import numpy as np
import torch
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from config import SEQ_LEN, SCALER_PATH

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
    Fit a StandardScaler to the data.
    
    Args:
        data (np.ndarray): Input data of shape (n_samples, n_features)
    
    Returns:
        StandardScaler: Fitted scaler
    """
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler

def transform_data(data, scaler):
    """
    Transform data using scaler.
    
    Args:
        data (np.ndarray): Input data
        scaler: Fitted scaler
        
    Returns:
        np.ndarray: Transformed data
    """
    try:
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Transform data
        transformed_data = scaler.transform(data)
        
        return transformed_data
        
    except Exception as e:
        logging.error(f"[❌] Error in transform_data: {str(e)}")
        raise

def inverse_transform_data(data, scaler):
    """
    Inverse transform data using scaler.
    
    Args:
        data (np.ndarray): Scaled data
        scaler: Fitted scaler
        
    Returns:
        np.ndarray: Inverse transformed data
    """
    try:
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Inverse transform data
        original_data = scaler.inverse_transform(data)
        
        return original_data
        
    except Exception as e:
        logging.error(f"[❌] Error in inverse_transform_data: {str(e)}")
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

def preprocess_for_training(data, seq_len, batch_size):
    """
    Preprocess data for training with proper train/val/test split.
    
    Args:
        data (np.ndarray): Input data array
        seq_len (int): Sequence length for LSTM
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    try:
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Fit scaler and transform data
        scaler = fit_scaler(data)
        scaled_data = transform_data(data, scaler)
        
        # Save the scaler
        save_scaler(scaler, SCALER_PATH)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i+seq_len])
            y.append(scaled_data[i+seq_len])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Split into train (70%), validation (15%), and test (15%) sets
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        logging.info(f"[✅] Created data loaders with shapes:")
        logging.info(f"Train: {X_train.shape}")
        logging.info(f"Validation: {X_val.shape}")
        logging.info(f"Test: {X_test.shape}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logging.error(f"[❌] Error in preprocess_for_training: {str(e)}")
        raise

def preprocess_for_forecast(data, scaler, seq_len):
    """
    Prepare data for forecasting.
    
    Args:
        data (np.ndarray): Input data
        scaler: Fitted scaler
        seq_len (int): Sequence length
        
    Returns:
        torch.Tensor: Prepared sequence for forecasting
    """
    try:
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale data
        scaled_data = scaler.transform(data)
        
        # Create sequence
        if len(scaled_data) < seq_len:
            # Pad with zeros if sequence is too short
            padding = np.zeros((seq_len - len(scaled_data), scaled_data.shape[1]))
            scaled_data = np.vstack([padding, scaled_data])
        else:
            # Take last seq_len elements
            scaled_data = scaled_data[-seq_len:]
        
        # Reshape for LSTM input (batch_size, seq_len, features)
        sequence = scaled_data.reshape(1, seq_len, -1)
        
        # Convert to tensor
        sequence = torch.FloatTensor(sequence)
        
        logging.info(f"[✔️] Prepared forecast sequence: shape {sequence.shape}")
        return sequence
        
    except Exception as e:
        logging.error(f"[❌] Error in preprocess_for_forecast: {str(e)}")
        raise

def prepare_sequences(data, seq_len):
    """
    Prepare sequences for LSTM input.
    
    Args:
        data (np.ndarray): Input data of shape (n_samples, n_features)
        seq_len (int): Length of each sequence
    
    Returns:
        tuple: (X, y) where:
            - X is the input sequences of shape (n_sequences, seq_len, n_features)
            - y is the target values of shape (n_sequences, n_features)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
        targets.append(data[i+seq_len])
    
    return np.array(sequences), np.array(targets)