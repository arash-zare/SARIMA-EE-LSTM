import torch
import numpy as np
import torch.nn.functional as F
from preprocessing import preprocess_for_forecast
from config import THRESHOLDS, FEATURES, SEQ_LEN, INPUT_DIM, DEVICE

def detect_anomaly_per_feature(raw_data_batch, model):
    """
    Detect anomalies for each feature using SARIMA-EE-LSTM.

    Args:
        raw_data_batch (np.ndarray): shape (n_samples, n_features)
        model (torch.nn.Module): the trained SARIMA-EE-LSTM model

    Returns:
        anomalies (dict): feature -> 1 (anomaly) or 0 (normal)
        mse_errors (list): list of MSE per feature
    """
    raw_data_batch = np.asarray(raw_data_batch, dtype=np.float32)

    if raw_data_batch.ndim == 1:
        raw_data_batch = raw_data_batch.reshape(1, -1)

    if raw_data_batch.shape[1] != INPUT_DIM:
        raise ValueError(f"[❌] Invalid number of features: expected {INPUT_DIM}, got {raw_data_batch.shape[1]}")
    
    if raw_data_batch.shape[0] < SEQ_LEN:
        raise ValueError(f"[❌] Not enough samples: expected at least {SEQ_LEN}, got {raw_data_batch.shape[0]}")

    # Preprocessing
    input_tensor = preprocess_for_forecast(raw_data_batch, seq_len=SEQ_LEN)
    input_tensor = input_tensor.to(DEVICE)

    # Predict
    model.eval()
    with torch.no_grad():
        predicted = model(input_tensor)

    # Compare last input vs prediction
    actual_last = torch.tensor(raw_data_batch[-1], dtype=torch.float32, device=DEVICE)
    predicted = predicted.squeeze(0)

    mse_per_feature = F.mse_loss(predicted, actual_last, reduction='none')
    mse_per_feature = mse_per_feature.cpu().numpy()

    anomalies = {}
    for idx, feature in enumerate(FEATURES):
        threshold = THRESHOLDS.get(feature, 1e9)
        anomalies[feature] = int(mse_per_feature[idx] > threshold)

    return anomalies, mse_per_feature.tolist()
