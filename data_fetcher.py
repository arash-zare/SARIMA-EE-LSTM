# data_fetcher.py
"""
Fetch historical and real-time data from VictoriaMetrics.
"""

import requests
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from config import VICTORIA_METRICS_URL, FEATURE_QUERIES, MAX_WORKERS, FEATURES, parse_duration

def setup_logger():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def fetch_single_metric(query, start, end, step):
    """
    Fetch data for a single metric from VictoriaMetrics.
    
    Args:
        query (str): Prometheus query.
        start (int): Start timestamp (Unix).
        end (int): End timestamp (Unix).
        step (str): Step size (e.g., '1s').
    
    Returns:
        tuple: (timestamps, values) or (None, None) if failed.
    """
    try:
        url = f"{VICTORIA_METRICS_URL}/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "success" or not data.get("data", {}).get("result"):
            logging.warning(f"[⚠️] No data returned for query: {query}")
            return None, None

        # Extract timestamps and values
        result = data["data"]["result"]
        if not result:
            logging.warning(f"[⚠️] Empty result for query: {query}")
            return None, None

        # Try to find data for device="eth0" if applicable
        for res in result:
            if "device" in res.get("metric", {}) and res["metric"]["device"] == "eth0" or "device" not in res.get("metric", {}):
                values = res["values"]
                timestamps = np.array([float(v[0]) for v in values])
                values = np.array([float(v[1]) for v in values])
                logging.info(f"[✔️] Fetched {len(values)} samples for query: {query}")
                return timestamps, values

        logging.warning(f"[⚠️] No matching data for device='eth0' in query: {query}")
        return None, None

    except Exception as e:
        logging.error(f"[❌] Error fetching data for query {query}: {str(e)}")
        return None, None

def fetch_historical_data(start_offset="10m", step="1s", duration="8m"):
    """
    Fetch historical data for all features from VictoriaMetrics.
    
    Args:
        start_offset (str): Offset from current time (e.g., '10m').
        step (str): Step size (e.g., '1s').
        duration (str): Duration of data to fetch (e.g., '8m').
    
    Returns:
        np.ndarray: Data array of shape (n_samples, n_features).
    """
    try:
        setup_logger()
        end_time = int(time.time())
        start_time = end_time - parse_duration(start_offset) - parse_duration(duration)
        step_seconds = parse_duration(step)

        results = {}
        timestamps_dict = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_feature = {
                executor.submit(fetch_single_metric, query, start_time, end_time, step): feature
                for feature, query in FEATURE_QUERIES.items()
            }
            for future in future_to_feature:
                feature = future_to_feature[future]
                try:
                    timestamps, values = future.result()
                    if timestamps is not None and values is not None:
                        results[feature] = values
                        timestamps_dict[feature] = timestamps
                        logging.info(f"[✔️] Fetched {feature}: {len(values)} samples")
                    else:
                        results[feature] = np.array([])
                        timestamps_dict[feature] = np.array([])
                        logging.info(f"[✔️] Fetched {feature}: 0 samples")
                except Exception as e:
                    logging.error(f"[❌] Error fetching historical data for {feature}: {str(e)}")
                    results[feature] = np.array([])
                    timestamps_dict[feature] = np.array([])

        # Find common timestamps
        valid_timestamps = [ts for ts in timestamps_dict.values() if ts.size > 0]
        if not valid_timestamps:
            logging.error("[❌] No valid timestamps fetched for any feature")
            return None

        # Use the intersection of timestamps
        common_timestamps = valid_timestamps[0]
        for ts in valid_timestamps[1:]:
            common_timestamps = np.intersect1d(common_timestamps, ts)
        
        if common_timestamps.size == 0:
            logging.warning("[⚠️] No common timestamps found, cannot align data")
            return None

        # Align data to common timestamps
        aligned_data = []
        for feature in FEATURES:
            if results[feature].size == 0:
                logging.warning(f"[⚠️] No data for {feature}, padding with zeros")
                aligned_data.append(np.zeros(len(common_timestamps)))
                continue

            # Interpolate or select values at common timestamps
            feature_ts = timestamps_dict[feature]
            feature_values = results[feature]
            if not np.array_equal(feature_ts, common_timestamps):
                logging.warning(f"[⚠️] Timestamp mismatch for {feature}, aligning to common timestamps")
                # Simple nearest-neighbor interpolation
                aligned_values = np.interp(
                    common_timestamps,
                    feature_ts,
                    feature_values,
                    left=feature_values[0],
                    right=feature_values[-1]
                )
            else:
                aligned_values = feature_values

            aligned_data.append(aligned_values)

        # Stack data and ensure correct shape
        data = np.column_stack(aligned_data)
        if data.shape[1] != len(FEATURES):
            logging.error(f"[❌] Data shape mismatch: got {data.shape[1]} features, expected {len(FEATURES)}")
            return None

        # Ensure data is in the correct shape (n_samples, n_features)
        if data.shape[0] < 1:
            logging.error("[❌] No samples in the data")
            return None

        logging.info(f"[✔️] Final historical data shape: {data.shape}")
        return data

    except Exception as e:
        logging.error(f"[❌] Error in fetch_historical_data: {str(e)}")
        return None