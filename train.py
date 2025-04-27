import os
import numpy as np
import time
import logging
import argparse
from data_fetcher import fetch_historical_data
from model import train_model
from config import INPUT_DIM, MODEL_PATH, SEQ_LEN

DATA_CACHE_FILE = "cached_data.npy"

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def load_or_fetch_data(args):
    # اگر کش موجود بود، ازش لود کن
    if os.path.exists(DATA_CACHE_FILE) and not args.force_refresh:
        logging.info(f"Loading cached data from {DATA_CACHE_FILE}...")
        data = np.load(DATA_CACHE_FILE)
        logging.info(f"Loaded cached data with shape: {data.shape}")
    else:
        logging.info("Fetching historical data from VictoriaMetrics...")
        data = fetch_historical_data(start_offset=args.start_offset, step=args.step, duration=args.duration)
        logging.info(f"Fetched data with shape: {data.shape}")

        if data.shape[0] == 0:
            logging.warning("No data received from VictoriaMetrics, using synthetic data...")
            synthetic_samples = max(300, 3 * SEQ_LEN)
            np.random.seed(42)
            data = np.random.randn(synthetic_samples, INPUT_DIM) * 100
            logging.info(f"Using synthetic data with shape: {data.shape}")

        # کش کن برای دفعات بعد
        np.save(DATA_CACHE_FILE, data)
        logging.info(f"Cached data to {DATA_CACHE_FILE}")

    return data

def run_training(args, model_path_suffix=""):
    setup_logger()

    try:
        data = load_or_fetch_data(args)

        if data.shape[0] < SEQ_LEN:
            raise ValueError(f"Data has {data.shape[0]} samples, but SEQ_LEN={SEQ_LEN} is required")
        if data.shape[1] != INPUT_DIM:
            raise ValueError(f"Expected {INPUT_DIM} features, got {data.shape[1]}")

        model_save_path = MODEL_PATH.replace(".pt", f"{model_path_suffix}.pt")

        logging.info(f"Training SARIMA-EE-LSTM model, saving to {model_save_path}...")
        start_time = time.time()

        model = train_model(
            data=data,
            seq_len=SEQ_LEN,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=model_save_path
        )

        elapsed = time.time() - start_time
        logging.info(f"✅ Model successfully trained and saved to {model_save_path}")
        logging.info(f"Training completed in {elapsed:.2f} seconds.")

    except ValueError as ve:
        logging.error(f"Validation error during training: {ve}")
    except ConnectionError as ce:
        logging.error(f"Connection error during data fetch: {ce}")
    except Exception as e:
        logging.exception(f"Unexpected error during training: {e}")
        raise

def batch_train(args):
    configs = [
        {"epochs": 5, "batch_size": 32},
        {"epochs": 10, "batch_size": 64},
        {"epochs": 15, "batch_size": 128},
    ]

    for idx, config in enumerate(configs, 1):
        logging.info(f"=== Starting batch {idx}: {config} ===")
        args.epochs = config["epochs"]
        args.batch_size = config["batch_size"]
        run_training(args, model_path_suffix=f"_batch{idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SARIMA-EE-LSTM model with historical or synthetic data.")
    parser.add_argument("--start_offset", type=str, default="10m", help="Start offset for data fetch (e.g., '10m')")
    parser.add_argument("--step", type=str, default="1s", help="Step size for data fetch (e.g., '1s')")
    parser.add_argument("--duration", type=str, default="8m", help="Duration for data fetch (e.g., '8m')")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--force_refresh", action="store_true", help="Force refetch data from VictoriaMetrics instead of cache")
    parser.add_argument("--batch_mode", action="store_true", help="Train multiple models with different configs")
    args = parser.parse_args()

    if args.batch_mode:
        batch_train(args)
    else:
        run_training(args)
