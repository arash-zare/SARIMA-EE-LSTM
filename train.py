# train.py
"""
Train SARIMA-EE-LSTM model with historical or synthetic data from VictoriaMetrics.
Supports batch training with multiple configurations.
"""

import os
import numpy as np
import time
import logging
import argparse
from datetime import datetime
from data_fetcher import fetch_historical_data
from model import train_model
from preprocessing import fit_scaler, transform_data
from config import INPUT_DIM, MODEL_PATH, SEQ_LEN, SCALER_PATH, SARIMA_CONFIGS

DATA_CACHE_FILE = "cached_data.npy"

def setup_logger():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def load_or_fetch_data(args):
    """
    Load cached data or fetch historical data from VictoriaMetrics.
    
    Args:
        args: Command-line arguments.
    
    Returns:
        np.ndarray: Data array of shape (n_samples, n_features).
    """
    try:
        if args.clear_cache and os.path.exists(DATA_CACHE_FILE):
            os.remove(DATA_CACHE_FILE)
            logging.info("[✔️] Cleared cached data.")

        if os.path.exists(DATA_CACHE_FILE) and not args.force_refresh:
            logging.info(f"[DEBUG] Loading cached data from {DATA_CACHE_FILE}...")
            data = np.load(DATA_CACHE_FILE)
            logging.info(f"[✔️] Loaded cached data with shape: {data.shape}")
        else:
            logging.info("[DEBUG] Fetching historical data from VictoriaMetrics...")
            data = fetch_historical_data(start_offset=args.start_offset, step=args.step, duration=args.duration)
            logging.info(f"[✔️] Fetched data with shape: {data.shape}")

            if data.shape[0] == 0 or data.shape[1] != INPUT_DIM:
                logging.warning("[⚠️] Invalid data received, using synthetic data...")
                synthetic_samples = max(300, 3 * SEQ_LEN)
                np.random.seed(42)
                # Generate positive data for rate metrics
                data = np.random.uniform(0, 1000, size=(synthetic_samples, INPUT_DIM))
                logging.info(f"[✔️] Using synthetic data with shape: {data.shape}")

            np.save(DATA_CACHE_FILE, data)
            logging.info(f"[✔️] Cached data to {DATA_CACHE_FILE}")

        # Validate data
        if data.shape[0] < SEQ_LEN:
            raise ValueError(f"Data has {data.shape[0]} samples, but SEQ_LEN={SEQ_LEN} is required")
        if data.shape[1] != INPUT_DIM:
            raise ValueError(f"Expected {INPUT_DIM} features, got {data.shape[1]}")

        # Initial normalization check
        try:
            scaler = fit_scaler(data)
            transform_data(data, scaler)
            logging.info("[✔️] Data passed initial normalization check")
        except Exception as e:
            logging.warning(f"[⚠️] Data normalization check failed: {str(e)}. Proceeding with raw data.")

        return data

    except Exception as e:
        logging.error(f"[❌] Error in load_or_fetch_data: {str(e)}")
        raise

def run_training(args, model_path_suffix=""):
    """
    Run training for SARIMA-EE-LSTM model.
    
    Args:
        args: Command-line arguments.
        model_path_suffix (str): Suffix for model and scaler save paths.
    """
    try:
        # Ensure save directories exist
        model_dir = os.path.dirname(MODEL_PATH)
        scaler_dir = os.path.dirname(SCALER_PATH)
        for directory in [model_dir, scaler_dir]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"[✔️] Created directory {directory}")

        # Validate SARIMA_CONFIGS
        if not hasattr(args, 'sarima_configs') or not args.sarima_configs:
            logging.warning("[⚠️] SARIMA_CONFIGS not provided, using defaults from config.py")
            args.sarima_configs = SARIMA_CONFIGS

        data = load_or_fetch_data(args)

        model_save_path = MODEL_PATH.replace(".pt", f"{model_path_suffix}.pt")
        scaler_save_path = SCALER_PATH.replace(".pkl", f"{model_path_suffix}.pkl")

        logging.info(f"[DEBUG] Training SARIMA-EE-LSTM model, saving to {model_save_path}...")
        start_time = time.time()

        model, sarima_forecasters = train_model(
            data=data,
            seq_len=SEQ_LEN,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=model_save_path,
            scaler_path=scaler_save_path,
            resume_training=args.resume_training,
            sarima_configs=args.sarima_configs  # Pass SARIMA configurations
        )

        if model is None or not sarima_forecasters:
            raise ValueError("Training failed: model or SARIMA forecasters not returned")

        elapsed = time.time() - start_time
        logging.info(f"[✔️] Model successfully trained and saved to {model_save_path}")
        logging.info(f"[✔️] Scaler saved to {scaler_save_path}")
        logging.info(f"[✔️] Trained {len(sarima_forecasters)} SARIMA forecasters")
        logging.info(f"[✔️] Training completed in {elapsed:.2f} seconds.")

    except ValueError as ve:
        logging.error(f"[❌] Validation error during training: {ve}")
    except ConnectionError as ce:
        logging.error(f"[❌] Connection error during data fetch: {ce}")
    except Exception as e:
        logging.exception(f"[❌] Unexpected error during training: {e}")
        raise

def batch_train(args):
    """
    Train multiple models with different configurations.
    
    Args:
        args: Command-line arguments.
    """
    configs = [
        {"epochs": 5, "batch_size": 32},
        {"epochs": 10, "batch_size": 32},
        {"epochs": 15, "batch_size": 32},
        {"epochs": 5, "batch_size": 64},
        {"epochs": 10, "batch_size": 64},
        {"epochs": 15, "batch_size": 128}
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx, config in enumerate(configs, 1):
        logging.info("=" * 50)
        logging.info(f"[DEBUG] Starting batch {idx}: {config}")
        logging.info("=" * 50)
        args.epochs = config["epochs"]
        args.batch_size = config["batch_size"]
        suffix = f"_batch{idx}_{timestamp}"
        run_training(args, model_path_suffix=suffix)

if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(description="Train SARIMA-EE-LSTM model with historical or synthetic data.")
    parser.add_argument("--start_offset", type=str, default="10m", help="Start offset for data fetch (e.g., '10m')")
    parser.add_argument("--step", type=str, default="1s", help="Step size for data fetch (e.g., '1s')")
    parser.add_argument("--duration", type=str, default="8m", help="Duration for data fetch (e.g., '8m')")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--force_refresh", action="store_true", help="Force refetch data from VictoriaMetrics")
    parser.add_argument("--batch_mode", action="store_true", help="Train multiple models with different configs")
    parser.add_argument("--clear_cache", action="store_true", help="Clear cached data before training")
    parser.add_argument("--resume_training", action="store_true", help="Resume training from existing model")
    args = parser.parse_args()

    # Add SARIMA_CONFIGS to args
    args.sarima_configs = SARIMA_CONFIGS

    if args.batch_mode:
        batch_train(args)
    else:
        run_training(args)