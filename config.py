# config.py
# Configuration settings for the SARIMA-EE-LSTM anomaly detection and forecasting system

import torch

# VictoriaMetrics base URL (without /api/v1/query suffix)
VICTORIA_METRICS_URL = "http://192.168.1.98:8428"

# Node Exporter features to monitor
FEATURES = [
    "node_cpu_seconds_total",
    "node_network_receive_packets_total",
    "node_network_transmit_packets_total",
    "node_nf_conntrack_entries",
    "node_network_receive_bytes_total",
    "node_network_transmit_bytes_total",
]

# Number of features (must match len(FEATURES))
INPUT_DIM = len(FEATURES)

# LSTM parameters
SEQ_LEN = 10  # Length of input sequence (e.g., last 10 seconds of data)
FORECAST_STEPS = 15  # How many future steps to forecast
HIDDEN_DIM = 64  # Hidden layer dimension
NUM_LAYERS = 2  # Number of LSTM layers

# Model save path
MODEL_PATH = "sarima_eelstm_model.pth"

# Fetch interval for new data (in seconds)
FETCH_INTERVAL = 20

# Device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds for anomaly detection based on MSE error
THRESHOLDS = {
    "node_cpu_seconds_total": 1e6,
    "node_network_receive_packets_total": 1e7,
    "node_network_transmit_packets_total": 1e7,
    "node_nf_conntrack_entries": 1e4,
    "node_network_receive_bytes_total": 1e8,
    "node_network_transmit_bytes_total": 1e8,
}

# SARIMA parameters (manual tuning based on data characteristics)
SARIMA_ORDER = (1, 1, 1)  # (p, d, q) order for SARIMA
SARIMA_SEASONAL_ORDER = (0, 0, 0, 0)  # (P, D, Q, s) seasonal order (no seasonality assumed)
