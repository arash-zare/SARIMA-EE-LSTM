import numpy as np
import torch
import os

# Dataset configuration
DATASET_PATH = "network_dataset_labeled.csv"  # Update with your dataset path

# Features to monitor (mapped to dataset)
FEATURES = [
    "throughput",              # Maps to network_receive_rate
    "bandwidth",               # Maps to network_transmit_rate
    "Number videos",           # Maps to active_connections
    "Bitrate video",           # Maps to receive_packets_rate
    "packet_loss",             # Maps to receive_errors_rate
    "congestion",              # Maps to receive_drops_rate
    "latency",                 # Additional network metric
    "jitter"                   # Additional network metric
]

# Model parameters
SEQ_LEN = 60
FORECAST_STEPS = 5
INPUT_DIM = len(FEATURES)
HIDDEN_DIM = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# Anomaly detection thresholds (adjusted for dataset)
THRESHOLDS = {
    "throughput": 10.0,          # Adjust based on dataset scale (e.g., Mbps)
    "bandwidth": 10.0,           # Adjust based on dataset scale
    "Number videos": 100,        # Adjust for session count
    "Bitrate video": 1e6,        # Adjust for bitrate scale
    "packet_loss": 1.0,          # Packet loss percentage
    "congestion": 1.0,           # Congestion metric
    "latency": 100.0,            # Latency in ms
    "jitter": 10.0               # Jitter in ms
}

# Fuzzy logic thresholds
FUZZY_THRESHOLDS = {
    feature: 0.8 for feature in FEATURES
}

# Paths
MODEL_DIR = "./models"
SAVE_DIR = "./saved_plots"
MODEL_PATH = os.path.join(MODEL_DIR, "sarima_ee_lstm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "sarima_ee_lstm_scaler.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SARIMA configurations for each feature
SARIMA_CONFIGS = {
    "throughput": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 0, 0)
    },
    "bandwidth": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 0, 0)
    },
    "Number videos": {
        "order": (1, 0, 0),  # Simpler model for discrete data
        "seasonal_order": (0, 0, 0, 0)
    },
    "Bitrate video": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 0, 0)
    },
    "packet_loss": {
        "order": (1, 0, 0),  # Simpler model for percentage data
        "seasonal_order": (0, 0, 0, 0)
    },
    "congestion": {
        "order": (1, 0, 0),  # Simpler model for percentage data
        "seasonal_order": (0, 0, 0, 0)
    },
    "latency": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 0, 0)
    },
    "jitter": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 0, 0)
    }
}