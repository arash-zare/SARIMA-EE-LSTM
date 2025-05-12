# from flask import Flask, Response
# from prometheus_client import Gauge, generate_latest, REGISTRY
# import prometheus_client as prom_client
# import threading
# import time
# import re
# import torch
# import numpy as np
# from data_fetcher import fetch_latest_data
# from detect_anomalies import detect_anomaly_per_feature
# from model import load_model, forecast, SARIMAForecaster
# from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQ_LEN, FORECAST_STEPS, DEVICE
# from fuzzy import evaluate_fuzzy_anomaly

# app = Flask(__name__)

# def sanitize_feature_name(feature):
#     return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

# def safe_gauge(name, documentation):
#     try:
#         metric = Gauge(name, documentation, registry=None)
#         REGISTRY.unregister(metric)
#     except KeyError:
#         pass
#     return Gauge(name, documentation)

# # Prometheus Metrics
# anomaly_gauges = {
#     feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
#     for feature in FEATURES
# }
# mse_gauges = {
#     feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_mse", f"MSE error for {feature}")
#     for feature in FEATURES
# }
# forecast_gauges = {
#     feature: safe_gauge(f"{sanitize_feature_name(feature)}_forecast_value", f"Forecasted value for {feature}")
#     for feature in FEATURES
# }
# upper_bound_gauges = {
#     feature: safe_gauge(f"{sanitize_feature_name(feature)}_upper_bound", f"Upper bound for {feature}")
#     for feature in FEATURES
# }
# lower_bound_gauges = {
#     feature: safe_gauge(f"{sanitize_feature_name(feature)}_lower_bound", f"Lower bound for {feature}")
#     for feature in FEATURES
# }
# fuzzy_risk_gauges = {
#     feature: safe_gauge(f"{sanitize_feature_name(feature)}_fuzzy_risk", f"Fuzzy anomaly risk for {feature}")
#     for feature in FEATURES
# }

# # Load trained LSTM model
# model = load_model(MODEL_PATH)
# print(f"✅ Loaded SARIMA-EE-LSTM model and scaler from {MODEL_PATH}")

# # Initialize SARIMA model (fit only once)
# sarima_model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), index_col=0)

# # Buffer & SARIMA
# sequence_buffer = []

# def monitor():
#     global sequence_buffer, sarima_model

#     while True:
#         try:
#             latest_data = fetch_latest_data()
#             if len(latest_data) != len(FEATURES):
#                 raise ValueError(f"Mismatch between fetched data ({len(latest_data)}) and features ({len(FEATURES)})")

#             sequence_buffer.append(latest_data)
#             if len(sequence_buffer) > SEQ_LEN:
#                 sequence_buffer = sequence_buffer[-SEQ_LEN:]

#             if len(sequence_buffer) == SEQ_LEN:
#                 input_sequence = np.array(sequence_buffer, dtype=np.float32)

#                 # Fit SARIMA only once
#                 if sarima_model.results is None:
#                     sarima_model.fit(input_sequence)
#                     print("✅ SARIMA model fitted.")

#                 # Anomaly detection
#                 anomalies, mse_per_feature = detect_anomaly_per_feature(input_sequence, model)

#                 # Forecasting
#                 forecast_values, upper_bounds, lower_bounds = forecast(
#                     model, input_sequence, sarima_model, forecast_steps=FORECAST_STEPS
#                 )

#                 # Update Prometheus metrics
#                 for i, feature in enumerate(FEATURES):
#                     mse_gauges[feature].set(mse_per_feature[i])
#                     anomaly_gauges[feature].set(anomalies[feature])
#                     forecast_gauges[feature].set(forecast_values[0, i])
#                     upper_bound_gauges[feature].set(upper_bounds[0, i])
#                     lower_bound_gauges[feature].set(lower_bounds[0, i])

#                     # Fuzzy evaluation
#                     prediction = forecast_values[0, i]
#                     upper = upper_bounds[0, i]
#                     lower = lower_bounds[0, i]
#                     actual = input_sequence[-1, i]

#                     fuzzy_risk = evaluate_fuzzy_anomaly(prediction, actual, upper, lower)
#                     print(f"Fuzzy Risk for {feature}: {fuzzy_risk}")
                    
#                     fuzzy_risk_gauges[feature].set(fuzzy_risk)
                    

#                     if fuzzy_risk > 0.8:
#                         print(f"🚨 {feature}: خطر بالا (fuzzy risk = {fuzzy_risk:.2f})")
#                     elif fuzzy_risk > 0.5:
#                         print(f"⚠️ {feature}: خطر متوسط (fuzzy risk = {fuzzy_risk:.2f})")
#                     else:
#                         print(f"✅ {feature}: وضعیت نرمال (fuzzy risk = {fuzzy_risk:.2f})")

#                 print("✅ Metrics updated:", dict(zip(FEATURES, mse_per_feature)))
#                 print("✅ Forecast values:", dict(zip(FEATURES, forecast_values[0])))

#         except Exception as e:
#             print(f"❌ Error in monitoring loop: {e}")

#         time.sleep(FETCH_INTERVAL)

# # Disable default collectors to keep metrics clean
# try:
#     REGISTRY.unregister(prom_client.GC_COLLECTOR)
#     REGISTRY.unregister(prom_client.PLATFORM_COLLECTOR)
#     REGISTRY.unregister(prom_client.PROCESS_COLLECTOR)
# except Exception as e:
#     print(f"ℹ️ Could not unregister some collectors: {e}")

# # Flask routes
# @app.route('/metrics')
# def metrics():
#     return Response(generate_latest(REGISTRY), mimetype='text/plain')

# if __name__ == '__main__':
#     monitor_thread = threading.Thread(target=monitor)
#     monitor_thread.daemon = True
#     monitor_thread.start()
#     app.run(host='0.0.0.0', port=8000)



# main.py
"""
Flask server for real-time anomaly detection using SARIMA-EE-LSTM and fuzzy logic.
Exports metrics to Prometheus and optionally saves plots.
"""

import os
import threading
import time
import re
import numpy as np
from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY
import prometheus_client as prom_client
from data_fetcher import fetch_historical_data
from detect_anomalies import detect_anomaly
from model import load_model
from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQ_LEN, INPUT_DIM , SCALER_PATH
try:
    from matplotlib_utils import save_multiple_metrics_to_files, save_anomaly_detection_plot_to_file
except ImportError:
    print("[⚠️] matplotlib_utils not found. Plot saving disabled.")
    save_multiple_metrics_to_files = None
    save_anomaly_detection_plot_to_file = None

def safe_gauge(name, documentation):
    """
    Create or reuse a Prometheus Gauge, ensuring no duplicate registration.
    
    Args:
        name (str): Metric name
        documentation (str): Metric description
    
    Returns:
        Gauge: Prometheus Gauge object
    """
    try:
        metric = Gauge(name, documentation, registry=None)
        REGISTRY.unregister(metric)
    except KeyError:
        pass
    return Gauge(name, documentation)

app = Flask(__name__)

def sanitize_feature_name(feature):
    """
    Sanitize feature name for Prometheus metric naming.
    
    Args:
        feature (str): Feature name
    
    Returns:
        str: Sanitized name
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

# Define path to save plots
save_dir = "./saved_plots"
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
        print(f"[✔️] Created directory {save_dir}")
    except Exception as e:
        print(f"[❌] Failed to create directory {save_dir}: {str(e)}")

# Prometheus Metrics
anomaly_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
    for feature in FEATURES
}
mse_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_mse", f"MSE error for {feature}")
    for feature in FEATURES
}
risk_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_fuzzy_risk", f"Fuzzy anomaly risk for {feature}")
    for feature in FEATURES
}
predicted_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_predicted_value", f"Predicted value for {feature}")
    for feature in FEATURES
}
actual_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_actual_value", f"Actual value for {feature}")
    for feature in FEATURES
}

# Load trained LSTM model
model_tuple = load_model(MODEL_PATH, SCALER_PATH)
if model_tuple:
    model, scaler, sarima_forecasters = model_tuple
    print(f"[✔️] Loaded SARIMA-EE-LSTM model from {MODEL_PATH}")
else:
    print(f"[❌] Failed to load model from {MODEL_PATH}")
    model = None
    scaler = None
    sarima_forecasters = None

# Data buffer
sequence_buffer = []

def monitor():
    """
    Monitor loop to fetch data, detect anomalies, and update Prometheus metrics.
    """
    global sequence_buffer
    while True:
        try:
            # Fetch latest data
            latest_data = fetch_historical_data()
            if latest_data is None:
                print("[⚠️] Failed to fetch data, retrying in next interval...")
                time.sleep(FETCH_INTERVAL)
                continue

            if latest_data.shape[1] != len(FEATURES):
                print(f"[⚠️] Invalid data shape: got {latest_data.shape[1]} features, expected {len(FEATURES)}")
                time.sleep(FETCH_INTERVAL)
                continue

            print(f"[DEBUG] Fetched data shape: {latest_data.shape}")
            
            # Add to sequence buffer
            sequence_buffer.append(latest_data)
            if len(sequence_buffer) > SEQ_LEN:
                sequence_buffer = sequence_buffer[-SEQ_LEN:]

            # Check if enough data for anomaly detection
            if len(sequence_buffer) == SEQ_LEN and model:
                # Stack sequences and ensure correct shape (SEQ_LEN, n_features)
                input_sequence = np.vstack(sequence_buffer)
                if input_sequence.shape[0] > SEQ_LEN:
                    input_sequence = input_sequence[-SEQ_LEN:]
                print(f"[DEBUG] Input sequence shape: {input_sequence.shape}")

                # Detect anomalies
                results = detect_anomaly(input_sequence, (model, scaler, sarima_forecasters))

                # Update Prometheus metrics
                for feature in FEATURES:
                    result = results.get(feature, {
                        "is_anomaly": False,
                        "mse": 0.0,
                        "risk_score": 0.0,
                        "predicted": 0.0,
                        "actual": 0.0
                    })
                    anomaly_gauges[feature].set(1 if result["is_anomaly"] else 0)
                    mse_gauges[feature].set(result["mse"])
                    risk_gauges[feature].set(result["risk_score"])
                    predicted_gauges[feature].set(result["predicted"])
                    actual_gauges[feature].set(result["actual"])

                    # Log risk levels
                    if result["risk_score"] > 0.8:
                        print(f"[🚨] {feature}: High risk (fuzzy risk = {result['risk_score']:.2f})")
                    elif result["risk_score"] > 0.5:
                        print(f"[⚠️] {feature}: Medium risk (fuzzy risk = {result['risk_score']:.2f})")
                    else:
                        print(f"[✅] {feature}: Normal (fuzzy risk = {result['risk_score']:.2f})")

                print(f"[✔️] Metrics updated: { {f: r['risk_score'] for f, r in results.items()} }")

                # Save plots if matplotlib_utils is available
                if save_multiple_metrics_to_files and save_anomaly_detection_plot_to_file:
                    try:
                        metrics_dict = {}
                        for i, feature in enumerate(FEATURES):
                            metrics_dict[feature] = {
                                'actual': input_sequence[:, i],
                                'predicted': results[feature]["predicted"],
                                'mse': results[feature]["mse"],
                                'risk_score': results[feature]["risk_score"],
                                'is_anomaly': results[feature]["is_anomaly"]
                            }
                        save_multiple_metrics_to_files(metrics_dict, save_dir)
                        for i, feature in enumerate(FEATURES):
                            save_anomaly_detection_plot_to_file(
                                input_sequence[:, i],
                                results[feature]["predicted"],
                                results[feature]["is_anomaly"],
                                feature,
                                save_dir,
                                mse=results[feature]["mse"],
                                risk_score=results[feature]["risk_score"]
                            )
                        print(f"[✔️] Plots saved to {save_dir}")
                    except Exception as e:
                        print(f"[❌] Error saving plots: {str(e)}")

        except Exception as e:
            print(f"[❌] Error in monitoring loop: {str(e)}")

        time.sleep(FETCH_INTERVAL)

# Disable default Prometheus collectors
try:
    REGISTRY.unregister(prom_client.GC_COLLECTOR)
    REGISTRY.unregister(prom_client.PLATFORM_COLLECTOR)
    REGISTRY.unregister(prom_client.PROCESS_COLLECTOR)
except Exception as e:
    print(f"[ℹ️] Could not unregister some collectors: {str(e)}")

# Flask routes
@app.route('/metrics')
def metrics():
    """
    Expose Prometheus metrics endpoint.
    
    Returns:
        Response: Prometheus metrics in text format
    """
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    app.run(host='0.0.0.0', port=8000)