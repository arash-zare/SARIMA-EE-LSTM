
# preprocessing.py
import torch
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler  # میتونی MinMaxScaler هم بزاری
from config import SEQ_LEN, INPUT_DIM
import joblib  # برای ذخیره و بارگذاری شیء‌های Python مثل Scaler


# Scaler object globally available
scaler = StandardScaler()

# Track if scaler has been fitted
_scaler_fitted = False



def save_scaler(path):
    """
    Save the fitted scaler to a file.
    """
    global scaler
    joblib.dump(scaler, path)
    print(f"[💾] Scaler saved to {path}.")

def load_scaler(path):
    """
    Load the scaler from a file.
    """
    global scaler, _scaler_fitted
    scaler = joblib.load(path)
    _scaler_fitted = True
    print(f"[📂] Scaler loaded from {path}.")


def fit_scaler(data):
    """
    Fit the scaler on the training data.
    """
    global _scaler_fitted
    data = np.array(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]

    if data.shape[1] != INPUT_DIM:
        raise ValueError(f"[❌] Expected {INPUT_DIM} features, got {data.shape[1]}")

    scaler.fit(data)
    _scaler_fitted = True
    print("[✔️] Scaler fitted on training data.")

def transform_data(data):
    """
    Transform data using fitted scaler.
    """
    if not _scaler_fitted:
        warnings.warn("[⚠️] Scaler has not been fitted. Returning raw data.")
        return data

    data = np.array(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]

    return scaler.transform(data)

def inverse_transform_data(data):
    """
    Inverse transform data back to original scale.
    """
    if not _scaler_fitted:
        warnings.warn("[⚠️] Scaler has not been fitted. Returning raw data.")
        return data

    data = np.array(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]
    return scaler.inverse_transform(data)

def preprocess_data(data, seq_len=SEQ_LEN):
    """
    Preprocess input data for SARIMA-EE-LSTM (for training).

    Returns:
        sequences: torch tensor (n_sequences, seq_len, n_features)
    """
    data = transform_data(data)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    if len(data_tensor) < seq_len:
        raise ValueError(f"[❌] Data length ({len(data_tensor)}) is less than seq_len ({seq_len})")

    sequences = torch.stack([data_tensor[i:i+seq_len] for i in range(len(data_tensor) - seq_len + 1)])

    return sequences

def preprocess_for_training(data, seq_len=SEQ_LEN):
    """
    Preprocess data for training (sequence + next-step label).

    Returns:
        X: (n_sequences, seq_len, n_features)
        y: (n_sequences, n_features)
    """
    sequences = preprocess_data(data, seq_len=seq_len)
    X = sequences[:-1]
    y = sequences[1:, -1, :]  # Predict next timestep
    return X, y

def preprocess_for_forecast(data, seq_len=SEQ_LEN):
    """
    Prepare the latest sequence for forecasting.

    Returns:
        (1, seq_len, n_features)
    """
    data = transform_data(data)
    data = np.array(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]

    if data.shape[1] != INPUT_DIM:
        raise ValueError(f"[❌] Expected {INPUT_DIM} features, got {data.shape[1]}")

    if len(data) < seq_len:
        raise ValueError(f"[❌] Not enough data for forecasting: have {len(data)}, need {seq_len}")

    last_seq = data[-seq_len:]
    tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len, n_features)
    return tensor

