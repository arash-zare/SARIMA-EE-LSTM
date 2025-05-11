# config.py
# Configuration settings for the SARIMA-EE-LSTM anomaly detection and forecasting system

import torch

# VictoriaMetrics base URL (without /api/v1/query suffix)
VICTORIA_METRICS_URL = "http://192.168.1.98:8428"

# Node Exporter features to monitor
# config.py

FEATURE_QUERIES = {
    # 1. سرعت ترافیک دریافتی شبکه (Bytes/sec)
    "network_receive_rate": 'rate(node_network_receive_bytes_total[1m])',

    # 2. سرعت ترافیک ارسالی شبکه (Bytes/sec)
    "network_transmit_rate": 'rate(node_network_transmit_bytes_total[1m])',

    # 3. تعداد اتصال‌های فعال (TCP Connections)
    "active_connections": 'node_nf_conntrack_entries',

    # 4. تعداد درخواست‌های ورودی (Packets/sec)
    "receive_packets_rate": 'rate(node_network_receive_packets_total[1m])',

    # 5. تعداد پکت‌های خراب یا گم شده (Receive Errors)
    "receive_errors_rate": 'rate(node_network_receive_errs_total[1m])',

    # 6. استفاده از CPU (فقط حالت System)
    "cpu_system_usage": 'rate(node_cpu_seconds_total{mode="system"}[1m])',

    # 7. نسبت حافظه آزاد به کل حافظه (Memory Available Ratio)
    "memory_available_ratio": 'node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes',

    # 8. پکت‌های انداخته شده (Dropped Packets)
    "receive_drops_rate": 'rate(node_network_receive_drop_total[1m])',
}


# Feature names (input order to model)
FEATURES = list(FEATURE_QUERIES.keys())

# Input dimensions
INPUT_DIM = len(FEATURES)

# LSTM parameters
SEQ_LEN = 1            # Length of input sequence (timesteps)
FORECAST_STEPS = 2     # How many future steps to forecast
HIDDEN_DIM = 64         # Hidden layer dimension
NUM_LAYERS = 2          # Number of LSTM layers

# Model save/load path
MODEL_PATH = "sarima_eelstm_model.pth"
SCALER_PATH = "sarima_eelstm_scaler.pkl"

# Fetching settings
FETCH_INTERVAL = 30     # Fetch interval (seconds)

# Device setting (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds for anomaly detection based on MSE errors (should be tuned)
# THRESHOLDS = {
#     "network_receive_rate": 1e6,
#     "network_transmit_rate": 1e6,
#     "active_connections": 1e5,
#     "receive_packets_rate": 1e5,
#     "receive_errors_rate": 100,
#     "cpu_system_usage": 0.5,
#     "memory_available_ratio": 0.2,  # If drops below 20%, suspicious
#     "receive_drops_rate": 100,
# }

THRESHOLDS = {
    'network_receive_rate': 7200,
    'network_transmit_rate': 18000,
    'active_connections': 40,
    'receive_packets_rate': 50,
    'receive_errors_rate': 10,
    'cpu_system_usage': 0.01,
    'memory_available_ratio': 0.8,
    'receive_drops_rate': 10
}



# SARIMA parameters (optional fine-tuning)
SARIMA_ORDER = (1, 1, 1)  # (p, d, q)
SARIMA_SEASONAL_ORDER = (0, 0, 0, 0)  # (P, D, Q, s) no seasonality
