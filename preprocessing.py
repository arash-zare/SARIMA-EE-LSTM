import torch
import numpy as np
import warnings
from config import SEQ_LEN, INPUT_DIM

_mean = None
_std = None

def preprocess_data(data, seq_len=SEQ_LEN, normalize=True):
    """
    Preprocess input data for SARIMA-EE-LSTM (for training).

    Args:
        data: numpy array (n_samples, n_features) or (n_samples,)
        seq_len: sequence length
        normalize: whether to normalize or not

    Returns:
        sequences: torch tensor (n_sequences, seq_len, n_features)
    """
    global _mean, _std

    data = np.array(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]

    if data.shape[1] != INPUT_DIM:
        raise ValueError(f"[❌] Expected {INPUT_DIM} features, got {data.shape[1]}")

    if len(data) < seq_len:
        raise ValueError(f"[❌] Data length ({len(data)}) is less than seq_len ({seq_len})")

    data_tensor = torch.tensor(data, dtype=torch.float32)

    if normalize:
        _mean = data_tensor.mean(dim=0, keepdim=True)
        _std = data_tensor.std(dim=0, keepdim=True) + 1e-8
        data_tensor = (data_tensor - _mean) / _std

    sequences = torch.stack([data_tensor[i:i+seq_len] for i in range(len(data_tensor) - seq_len + 1)])

    return sequences

def preprocess_for_forecast(data, seq_len=SEQ_LEN):
    """
    Prepare a single sequence for forecasting (no batching).

    Args:
        data: numpy array (n_samples, n_features)
        seq_len: sequence length

    Returns:
        torch.Tensor: (1, seq_len, n_features)
    """
    data = np.array(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]

    if data.shape[1] != INPUT_DIM:
        raise ValueError(f"[❌] Expected {INPUT_DIM} features, got {data.shape[1]}")

    if len(data) < seq_len:
        raise ValueError(f"[❌] Not enough data for forecasting: have {len(data)}, need {seq_len}")

    last_seq = data[-seq_len:]  # فقط آخرین seq_len
    tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len, input_dim)

    return tensor

def inverse_preprocess(data):
    """
    Inverse normalization.

    Args:
        data: torch tensor or numpy array

    Returns:
        denormalized_data: numpy array
    """
    global _mean, _std

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if _mean is None or _std is None:
        warnings.warn("[⚠️] Mean/Std not set. Returning data as-is.")
        return data

    mean = _mean.detach().cpu().numpy()
    std = _std.detach().cpu().numpy()

    return data * std + mean

def preprocess_for_training(data, seq_len=SEQ_LEN):
    """
    Preprocess data for training (next-step prediction).

    Args:
        data: numpy array (n_samples, n_features)

    Returns:
        X: (n_sequences, seq_len, n_features)
        y: (n_sequences, n_features)
    """
    sequences = preprocess_data(data, seq_len=seq_len)
    X = sequences[:-1]
    y = sequences[1:, -1, :]  # Predict next timestep
    return X, y
