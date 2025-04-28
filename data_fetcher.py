# data_fetcher.py
import requests
import time
import numpy as np
from config import VICTORIA_METRICS_URL, FEATURES

def fetch_latest_data():
    """Fetch latest 1-minute data for all monitored features."""
    values = []
    for feature in FEATURES:
        query = f'{feature}[1m]'
        try:
            print(f"[DEBUG] Sending request for {feature} with query: {query}")
            response = requests.get(
                VICTORIA_METRICS_URL.rstrip("/") + "/api/v1/query",
                params={'query': query}
            )
            response.raise_for_status()
            results = response.json().get('data', {}).get('result', [])

            if results:
                selected_value = select_feature_value(results, feature)
                print(f"[✔️] Fetched {feature}: {selected_value}")
                values.append(selected_value)
            else:
                print(f"[⚠️] No data for {feature}, setting 0.0")
                values.append(0.0)

        except Exception as e:
            print(f"[❌] Error fetching {feature}: {e}")
            values.append(0.0)

    return values

def fetch_historical_data(start_offset="2m", step="60s", duration="2m"):
    """Fetch historical time-series data for all monitored features."""
    end_time = int(time.time())
    duration_seconds = parse_duration(duration)
    start_time = end_time - duration_seconds
    expected_samples = duration_seconds // parse_duration(step)

    feature_data = []

    for feature in FEATURES:
        print(f"[DEBUG] Sending historical request for {feature} ({start_time}-{end_time})")
        try:
            response = requests.get(
                VICTORIA_METRICS_URL.rstrip("/") + "/api/v1/query_range",
                params={
                    "query": feature,
                    "start": start_time,
                    "end": end_time,
                    "step": step
                }
            )
            response.raise_for_status()
            results = response.json().get('data', {}).get('result', [])

            if results:
                feature_values = extract_feature_values(results, feature)
                print(f"[✔️] Fetched {feature}: {len(feature_values)} samples")
                feature_data.append(feature_values)
            else:
                print(f"[⚠️] No historical data for {feature}, filling with zeros")
                feature_data.append([0.0] * expected_samples)

        except Exception as e:
            print(f"[❌] Error fetching historical data for {feature}: {e}")
            feature_data.append([0.0] * expected_samples)

    # Ensure all features are aligned in time
    min_samples = min(len(f) for f in feature_data)
    feature_data = np.array([f[:min_samples] for f in feature_data])

    data = feature_data.T
    print(f"[DEBUG] Final data shape: {data.shape}")
    return data

def select_feature_value(results, feature):
    """Selects a single value for the latest data fetch."""
    if "network" in feature:
        eth0_value = None
        max_value = 0.0
        for r in results:
            device = r.get('metric', {}).get('device', '')
            last_value = float(r['values'][-1][1])

            if device == 'eth0':
                eth0_value = last_value
            if last_value > max_value:
                max_value = last_value

        return eth0_value if eth0_value is not None else max_value
    else:
        return float(results[0]['values'][-1][1])

def extract_feature_values(results, feature):
    """Extracts a list of values from historical data results."""
    if "network" in feature:
        eth0_values = None
        max_values = []

        for r in results:
            device = r.get('metric', {}).get('device', '')
            values = [float(v[1]) for v in r.get('values', []) if v[1] != 'nan']

            if device == 'eth0':
                eth0_values = values
            if values and (not max_values or len(values) > len(max_values)):
                max_values = values

        return eth0_values if eth0_values is not None else max_values
    else:
        return [float(v[1]) for v in results[0]['values'] if v[1] != 'nan']

def parse_duration(duration_str):
    """Parses duration string like '5m', '10s', '7d' into seconds."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    number = ''.join(filter(str.isdigit, duration_str))
    unit = ''.join(filter(str.isalpha, duration_str))

    if not number or unit not in units:
        raise ValueError(f"Invalid duration format: {duration_str}")

    return int(number) * units[unit]
