# config.py
"""
Configuration settings for the SARIMA-EE-LSTM anomaly detection and forecasting system.
"""

import os
import torch
import re
import logging

# Base directories
MODEL_DIR = "./models/"  # Directory for saving models and scalers
PLOT_DIR = "./saved_plots/"  # Directory for saving plots

# VictoriaMetrics base URL (without /api/v1 suffix)
VICTORIA_METRICS_URL = os.getenv("VICTORIA_METRICS_URL", "http://192.168.1.98:8428")

# Node Exporter features to monitor
FEATURE_QUERIES = {
    # 1. Network receive traffic rate (Bytes/sec)
    "network_receive_rate": 'sum(rate(node_network_receive_bytes_total{device=~"eth.*"}[1m]))',
    # 2. Network transmit traffic rate (Bytes/sec)
    "network_transmit_rate": 'sum(rate(node_network_transmit_bytes_total{device=~"eth.*"}[1m]))',
    # 3. Number of active TCP connections
    "active_connections": 'node_nf_conntrack_entries',
    # 4. Incoming packet rate (Packets/sec)
    "receive_packets_rate": 'sum(rate(node_network_receive_packets_total{device=~"eth.*"}[1m]))',
    # 5. Network receive errors (Errors/sec)
    "receive_errors_rate": 'sum(rate(node_network_receive_errs_total{device=~"eth.*"}[1m]))',
    # 6. CPU usage in system mode (rate)
    "cpu_system_usage": 'sum(rate(node_cpu_seconds_total{mode="system"}[1m]))',
    # 7. Ratio of available memory to total memory
    "memory_available_ratio": 'sum(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)',
    # 8. Dropped packets (Drops/sec)
    "receive_drops_rate": 'sum(rate(node_network_receive_drop_total{device=~"eth.*"}[1m]))',
}

# Feature names (input order to model)
FEATURES = list(FEATURE_QUERIES.keys())

# Input dimensions
INPUT_DIM = len(FEATURES)

# LSTM parameters
SEQ_LEN = 5            # Length of input sequence (timesteps)
FORECAST_STEPS = 5      # Number of future steps to forecast
HIDDEN_DIM = 64         # Hidden layer dimension
NUM_LAYERS = 2          # Number of LSTM layers

# Model save/load paths
MODEL_PATH = os.path.join(MODEL_DIR, "sarima_eelstm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "sarima_eelstm_scaler.pkl")
SAVE_DIR = PLOT_DIR     # Directory for saving plots

# Fetching settings
FETCH_INTERVAL = 10     # Fetch interval (seconds)
MAX_WORKERS = 8         # Max workers for parallel requests

# Device setting (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging level
LOG_LEVEL = logging.INFO

# Enable synthetic data generation if VictoriaMetrics is unavailable
SYNTHETIC_DATA_ENABLED = True

# Thresholds for anomaly detection based on MSE errors (tuned for sample data)
THRESHOLDS = {
    "network_receive_rate": 50000000,      # Bytes/sec, adjust based on real data
    "network_transmit_rate": 200000000,     # Bytes/sec, adjust based on real data
    "active_connections": 100000,          # Connections, adjust based on real data
    "receive_packets_rate": 1000,       # Packets/sec, adjust based on real data
    "receive_errors_rate": 50,          # Errors/sec, adjust based on real data
    "cpu_system_usage": 0.4,            # CPU rate, adjust based on real data
    "memory_available_ratio": 0.7,      # Ratio (below 70% is suspicious)
    "receive_drops_rate": 50,           # Drops/sec, adjust based on real data
}

# Fuzzy thresholds for risk_score (0 to 1)
FUZZY_THRESHOLDS = {
    "network_receive_rate": 0.8,    # High risk if MSE exceeds 80% of threshold
    "network_transmit_rate": 0.8,
    "active_connections": 0.7,
    "receive_packets_rate": 0.8,
    "receive_errors_rate": 0.6,
    "cpu_system_usage": 0.7,
    "memory_available_ratio": 0.6,
    "receive_drops_rate": 0.6,
}

# Default bounds for predictions (percentage of predicted value)
DEFAULT_BOUNDS = {
    "upper": 1.2,  # 120% of predicted value
    "lower": 0.8   # 80% of predicted value
}

# SARIMA parameters (feature-specific, simplified for convergence)
SARIMA_CONFIGS = {
    "network_receive_rate": {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
    "network_transmit_rate": {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
    "active_connections": {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
    "receive_packets_rate": {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
    "receive_errors_rate": {"order": (1, 0, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
    "cpu_system_usage": {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
    "memory_available_ratio": {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
    "receive_drops_rate": {"order": (1, 0, 0), "seasonal_order": (0, 0, 0, 0), "maxiter": 200},
}

# Validate SARIMA_CONFIGS
for feature in FEATURES:
    if feature not in SARIMA_CONFIGS:
        raise ValueError(f"SARIMA_CONFIGS missing configuration for feature: {feature}")

def parse_duration(duration_str):
    """
    Parse duration string like '5m', '1h30m', '7d' into seconds.
    
    Args:
        duration_str (str): Duration string.
    
    Returns:
        int: Duration in seconds.
    """
    try:
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        total_seconds = 0
        pattern = r'(\d+)([smhd])'
        matches = re.findall(pattern, duration_str.lower())

        if not matches:
            raise ValueError(f"Invalid duration format: {duration_str}")

        for number, unit in matches:
            if unit not in units:
                raise ValueError(f"Unknown unit {unit} in duration: {duration_str}")
            total_seconds += int(number) * units[unit]

        return total_seconds
    except Exception as e:
        logging.error(f"[❌] Error parsing duration {duration_str}: {str(e)}")
        raise