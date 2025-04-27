# main.py
from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY
import threading
import time
import re
import torch
import numpy as np
from data_fetcher import fetch_latest_data
from detect_anomalies import detect_anomaly_per_feature
from model import SARIMA_EELSTM, forecast
from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQ_LEN, FORECAST_STEPS, DEVICE

app = Flask(__name__)

# --- Helper function: sanitize feature names ---
def sanitize_feature_name(feature):
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature)
    return safe_name.lower()

# --- Remove previous metrics if exist ---
def safe_gauge(name, documentation):
    try:
        metric = Gauge(name, documentation, registry=None)
        REGISTRY.unregister(metric)
    except KeyError:
        pass
    return Gauge(name, documentation)

# --- Create Gauges per feature ---
anomaly_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
    for feature in FEATURES
}

mse_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_MSE", f"MSE error for {feature}")
    for feature in FEATURES
}

forecast_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_forecast_value", f"Forecasted value for {feature}")
    for feature in FEATURES
}

upper_bound_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_upper_bound", f"Upper bound for forecasted {feature}")
    for feature in FEATURES
}

lower_bound_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_lower_bound", f"Lower bound for forecasted {feature}")
    for feature in FEATURES
}

# --- Load trained model ---
model = SARIMA_EELSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded SARIMA-EE-LSTM model from {MODEL_PATH}")

# --- Internal buffer ---
sequence_buffer = []

# --- Monitor function ---
def monitor():
    global sequence_buffer

    while True:
        try:
            # Fetch latest data point
            latest_data = fetch_latest_data()
            if len(latest_data) != len(FEATURES):
                raise ValueError(f"Mismatch between fetched data ({len(latest_data)}) and features ({len(FEATURES)})")

            sequence_buffer.append(latest_data)

            # Keep only needed sequence length
            if len(sequence_buffer) > SEQ_LEN:
                sequence_buffer = sequence_buffer[-SEQ_LEN:]

            if len(sequence_buffer) == SEQ_LEN:
                # Prepare input
                input_sequence = np.array(sequence_buffer, dtype=np.float32)
                
                # Detect anomalies
                # anomalies, mse_per_feature = detect_anomaly_per_feature(input_sequence)
                anomalies, mse_per_feature = detect_anomaly_per_feature(input_sequence, model)
                
                # Forecast future values
                forecast_values, upper_bounds, lower_bounds = forecast(model, input_sequence, forecast_steps=FORECAST_STEPS)

                # Set metrics
                for i, feature in enumerate(FEATURES):
                    mse_value = mse_per_feature[i]
                    mse_gauges[feature].set(mse_value)

                    is_anomaly = anomalies[feature]
                    anomaly_gauges[feature].set(is_anomaly)

                    forecast_gauges[feature].set(forecast_values[0, i])
                    upper_bound_gauges[feature].set(upper_bounds[0, i])
                    lower_bound_gauges[feature].set(lower_bounds[0, i])

                print("✅ Updated metrics:", dict(zip(FEATURES, mse_per_feature)))
                print("✅ Forecasted values:", dict(zip(FEATURES, forecast_values[0])))

        except Exception as e:
            print(f"❌ Error in detection loop: {e}")

        time.sleep(FETCH_INTERVAL)

# --- Flask route ---
@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# --- Main ---
if __name__ == "__main__":
    threading.Thread(target=monitor, daemon=True).start()
    print(f"🚀 Starting VictoriaMetrics exporter server on port 8000...")
    app.run(host="0.0.0.0", port=8000)
