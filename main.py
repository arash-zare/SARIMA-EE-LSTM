# main.py
from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY
import prometheus_client as prom_client
import threading
import time
import re
import torch
import numpy as np
from data_fetcher import fetch_latest_data
from detect_anomalies import detect_anomaly_per_feature
from model import load_model, forecast  # تغییر: استفاده از load_model بجای دستی لود
from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQ_LEN, FORECAST_STEPS, DEVICE

app = Flask(__name__)

# --- Helper function: sanitize feature names ---
def sanitize_feature_name(feature):
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature)
    return safe_name.lower()

# --- Safe Gauge Creation ---
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
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_mse", f"MSE error for {feature}")
    for feature in FEATURES
}

forecast_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_forecast_value", f"Forecasted value for {feature}")
    for feature in FEATURES
}

upper_bound_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_upper_bound", f"Upper bound for {feature}")
    for feature in FEATURES
}

lower_bound_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_lower_bound", f"Lower bound for {feature}")
    for feature in FEATURES
}

# --- Load trained model ---
model = load_model(MODEL_PATH)  # تغییر: استفاده از load_model استاندارد
print(f"✅ Loaded SARIMA-EE-LSTM model and scaler from {MODEL_PATH}")

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

            # Keep only last SEQ_LEN items
            if len(sequence_buffer) > SEQ_LEN:
                sequence_buffer = sequence_buffer[-SEQ_LEN:]

            if len(sequence_buffer) == SEQ_LEN:
                input_sequence = np.array(sequence_buffer, dtype=np.float32)

                # Detect anomalies
                anomalies, mse_per_feature = detect_anomaly_per_feature(input_sequence, model)

                # Forecast future values
                forecast_values, upper_bounds, lower_bounds = forecast(model, input_sequence, forecast_steps=FORECAST_STEPS)

                # Update metrics
                for i, feature in enumerate(FEATURES):
                    mse_gauges[feature].set(mse_per_feature[i])
                    anomaly_gauges[feature].set(anomalies[feature])
                    forecast_gauges[feature].set(forecast_values[0, i])
                    upper_bound_gauges[feature].set(upper_bounds[0, i])
                    lower_bound_gauges[feature].set(lower_bounds[0, i])

                print("✅ Metrics updated:", dict(zip(FEATURES, mse_per_feature)))
                print("✅ Forecast values:", dict(zip(FEATURES, forecast_values[0])))

        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")

        time.sleep(FETCH_INTERVAL)

# --- Clean Prometheus default collectors ---
try:
    REGISTRY.unregister(prom_client.GC_COLLECTOR)
    REGISTRY.unregister(prom_client.PLATFORM_COLLECTOR)
    REGISTRY.unregister(prom_client.PROCESS_COLLECTOR)
except Exception as e:
    print(f"ℹ️ Could not unregister some default collectors: {e}")

# --- Flask route ---
@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# --- Main ---
if __name__ == "__main__":
    threading.Thread(target=monitor, daemon=True).start()
    print(f"🚀 Starting VictoriaMetrics exporter server on port 8000...")
    app.run(host="0.0.0.0", port=8000)
