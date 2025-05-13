import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from config import FEATURES, DATASET_PATH

def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def fetch_historical_data(dataset_path: str = DATASET_PATH, start_offset: str = "10m", step: str = "1s", duration: str = "8m") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load historical data from a CSV dataset.

    Args:
        dataset_path (str): Path to the CSV dataset file.
        start_offset (str): Start offset for data fetch (e.g., '10m').
        step (str): Step size for data fetch (e.g., '1s').
        duration (str): Duration for data fetch (e.g., '8m').

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Data array and labels (if available).
    """
    setup_logger()
    try:
        # Load dataset
        logging.info(f"[📂] Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Ensure required columns exist
        required_columns = FEATURES + ['timestamp']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logging.error(f"[❌] Missing columns in dataset: {missing_cols}")
            raise ValueError(f"Dataset must contain columns: {required_columns}")

        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Extract feature data
        data = df[FEATURES].values
        logging.info(f"[✅] Fetched data shape: {data.shape}")

        # Extract labels if available
        labels = df['anomaly'].values if 'anomaly' in df.columns else None
        if labels is not None:
            logging.info(f"[✅] Fetched labels shape: {labels.shape}")

        return data, labels

    except Exception as e:
        logging.error(f"[❌] Error fetching data: {str(e)}")
        raise

def fetch_latest_data(dataset_path: str = DATASET_PATH, n_steps: int = 5) -> np.ndarray:
    """
    Load the latest n_steps data points from the dataset.

    Args:
        dataset_path (str): Path to the CSV dataset file.
        n_steps (int): Number of latest steps to fetch.

    Returns:
        np.ndarray: Latest data array.
    """
    setup_logger()
    try:
        logging.info(f"[📂] Loading latest {n_steps} steps from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Ensure required columns exist
        if not all(col in df.columns for col in FEATURES):
            logging.error(f"[❌] Missing feature columns in dataset")
            raise ValueError(f"Dataset must contain columns: {FEATURES}")

        # Sort by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Get latest n_steps
        latest_data = df[FEATURES].tail(n_steps).values
        logging.info(f"[✅] Fetched latest data shape: {latest_data.shape}")

        return latest_data

    except Exception as e:
        logging.error(f"[❌] Error fetching latest data: {str(e)}")
        raise