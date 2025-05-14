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
from data_fetcher import fetch_historical_data, fetch_latest_data
from preprocessing import transform_data, inverse_transform_data, fit_scaler, save_scaler
from model import load_model, forecast
from detect_anomalies import detect_anomalies
from config import (
    FEATURES, MODEL_PATH, SAVE_DIR, DATASET_PATH, 
    SEQ_LEN, FORECAST_STEPS, SCALER_PATH, THRESHOLDS
)
import logging
try:
    from matplotlib_utils import save_multiple_metrics_to_files, save_anomaly_detection_plot_to_file
except ImportError:
    logging.warning("[⚠️] matplotlib_utils not found. Plot saving disabled.")
    save_multiple_metrics_to_files = None
    save_anomaly_detection_plot_to_file = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Sanitize feature names for Prometheus
def sanitize_feature_name(feature: str) -> str:
    """Sanitize feature name for Prometheus metric naming."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

# Prometheus Metrics
anomaly_gauges = {
    feature: Gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
    for feature in FEATURES
}
mse_gauges = {
    feature: Gauge(f"{sanitize_feature_name(feature)}_system_mse", f"MSE error for {feature}")
    for feature in FEATURES
}
risk_gauges = {
    feature: Gauge(f"{sanitize_feature_name(feature)}_fuzzy_risk", f"Fuzzy anomaly risk for {feature}")
    for feature in FEATURES
}
predicted_gauges = {
    feature: Gauge(f"{sanitize_feature_name(feature)}_predicted_value", f"Predicted value for {feature}")
    for feature in FEATURES
}
actual_gauges = {
    feature: Gauge(f"{sanitize_feature_name(feature)}_actual_value", f"Actual value for {feature}")
    for feature in FEATURES
}

# Create save directory
if not os.path.exists(SAVE_DIR):
    try:
        os.makedirs(SAVE_DIR)
        logger.info(f"[✔️] Created directory {SAVE_DIR}")
    except Exception as e:
        logger.error(f"[❌] Failed to create directory {SAVE_DIR}: {str(e)}")

# Load trained model
try:
    logger.info(f"[DEBUG] Attempting to load model from {MODEL_PATH}")
    model_tuple = load_model(MODEL_PATH, SCALER_PATH)
    if model_tuple and all(x is not None for x in model_tuple):
        model, scaler, sarima_forecasters = model_tuple
        logger.info(f"[✔️] Loaded model from {MODEL_PATH}")
        logger.info(f"[✔️] Scaler loaded from {SCALER_PATH}")
    else:
        logger.error(f"[❌] Failed to load model components from {MODEL_PATH}")
        logger.error(f"[DEBUG] Model tuple: {model_tuple}")
        model = None
        scaler = None
        sarima_forecasters = None
except Exception as e:
    logger.error(f"[❌] Error loading model: {str(e)}")
    model = None
    scaler = None
    sarima_forecasters = None

# Data buffer
sequence_buffer = []
buffer_labels = []

def monitor():
    """Monitor loop to process CSV data, detect anomalies, and update Prometheus metrics."""
    global sequence_buffer, buffer_labels, scaler, model, sarima_forecasters
    current_index = 0
    while True:
        try:
            if model is None or sarima_forecasters is None:
                logger.error("[❌] Model not loaded, attempting to reload...")
                try:
                    model_tuple = load_model(MODEL_PATH, SCALER_PATH)
                    if model_tuple and all(x is not None for x in model_tuple):
                        model, scaler, sarima_forecasters = model_tuple
                        logger.info("[✔️] Successfully reloaded model")
                    else:
                        logger.error("[❌] Failed to reload model")
                        time.sleep(10)
                        continue
                except Exception as e:
                    logger.error(f"[❌] Error reloading model: {str(e)}")
                    time.sleep(10)
                    continue

            # Load historical data
            historical_data, labels = fetch_historical_data(DATASET_PATH)
            if historical_data is None or historical_data.shape[1] != len(FEATURES):
                logger.error(f"[❌] Invalid data shape or failed to fetch: got {historical_data.shape if historical_data is not None else None}, expected {len(FEATURES)} features")
                time.sleep(10)
                continue

            # Simulate real-time monitoring by processing chunks
            if current_index + SEQ_LEN <= len(historical_data):
                input_sequence = historical_data[current_index:current_index + SEQ_LEN]
                input_labels = labels[current_index:current_index + SEQ_LEN] if labels is not None else None
                current_index += 1
            else:
                # Reset to start for continuous looping
                current_index = 0
                input_sequence = historical_data[current_index:current_index + SEQ_LEN]
                input_labels = labels[current_index:current_index + SEQ_LEN] if labels is not None else None
                current_index += 1

            logger.info(f"[✅] Processing data chunk: shape={input_sequence.shape}")

            # Ensure scaler exists
            if scaler is None:
                logger.warning("[⚠️] Scaler not available, creating new scaler...")
                scaler = fit_scaler(historical_data)
                save_scaler(scaler, SCALER_PATH)
                logger.info(f"[✔️] Created and saved new scaler to {SCALER_PATH}")

            # Transform data
            scaled_sequence = transform_data(input_sequence, scaler)

            # Forecast
            if model is None or sarima_forecasters is None:
                logger.error("[❌] Model not loaded")
                time.sleep(10)
                continue

            predictions, upper_bounds, lower_bounds = forecast(model, scaled_sequence, scaler, sarima_forecasters, FORECAST_STEPS)
            
            if predictions is None or upper_bounds is None or lower_bounds is None:
                logger.error("[❌] Forecast failed")
                time.sleep(10)
                continue

            predictions = inverse_transform_data(predictions, scaler)
            upper_bounds = inverse_transform_data(upper_bounds, scaler)
            lower_bounds = inverse_transform_data(lower_bounds, scaler)

            # Detect anomalies
            anomalies, metrics = detect_anomalies(
                input_sequence[-1:],  # Last actual value
                predictions,        # Predicted values
                model,
                scaler,
                sarima_forecasters,
                thresholds=THRESHOLDS
            )

            if anomalies is not None and metrics is not None:
                # Update Prometheus metrics
                for i, feature in enumerate(FEATURES):
                    mse_gauges[feature].set(metrics[feature]['mse'])
                    anomaly_gauges[feature].set(1.0 if anomalies[feature] else 0.0)
                    predicted_gauges[feature].set(predictions[0, i])
                    actual_gauges[feature].set(input_sequence[-1, i])
                    risk_gauges[feature].set(metrics[feature]['fuzzy_risk'])

                    # Log status
                    if metrics[feature]['fuzzy_risk'] > 0.8:
                        logger.warning(f"🚨 {feature}: خطر بالا (fuzzy risk = {metrics[feature]['fuzzy_risk']:.2f})")
                    elif metrics[feature]['fuzzy_risk'] > 0.5:
                        logger.warning(f"⚠️ {feature}: خطر متوسط (fuzzy risk = {metrics[feature]['fuzzy_risk']:.2f})")
                    else:
                        logger.info(f"✅ {feature}: وضعیت نرمال (fuzzy risk = {metrics[feature]['fuzzy_risk']:.2f})")

                logger.info("✅ Metrics updated:", dict(zip(FEATURES, [metrics[f]['mse'] for f in FEATURES])))
                logger.info("✅ Forecast values:", dict(zip(FEATURES, predictions[0])))

            # Save plots
            if save_multiple_metrics_to_files and save_anomaly_detection_plot_to_file:
                try:
                    metrics_dict = {}
                    for i, feature in enumerate(FEATURES):
                        metrics_dict[feature] = {
                            'actual': input_sequence[-FORECAST_STEPS:, i],
                            'predicted': predictions[:, i],
                            'mse': metrics[feature]['mse'],
                            'risk_score': metrics[feature]['fuzzy_risk'],
                            'is_anomaly': anomalies[feature]
                        }
                    save_multiple_metrics_to_files(metrics_dict, SAVE_DIR)
                    for i, feature in enumerate(FEATURES):
                        save_anomaly_detection_plot_to_file(
                            input_sequence[-FORECAST_STEPS:, i],
                            predictions[:, i],
                            anomalies[feature],
                            feature,
                            SAVE_DIR,
                            mse=metrics[feature]['mse'],
                            risk_score=metrics[feature]['fuzzy_risk']
                        )
                    logger.info(f"[✔️] Plots saved to {SAVE_DIR}")
                except Exception as e:
                    logger.error(f"[❌] Error saving plots: {str(e)}")

        except Exception as e:
            logger.error(f"[❌] Error in monitoring loop: {str(e)}")

        time.sleep(10)  # Adjust interval as needed

# Disable default Prometheus collectors
try:
    REGISTRY.unregister(prom_client.GC_COLLECTOR)
    REGISTRY.unregister(prom_client.PLATFORM_COLLECTOR)
    REGISTRY.unregister(prom_client.PROCESS_COLLECTOR)
except Exception as e:
    logger.info(f"[ℹ️] Could not unregister some collectors: {str(e)}")

# Flask routes
@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    app.run(host='0.0.0.0', port=8000)