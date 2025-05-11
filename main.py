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
from model import load_model, forecast, SARIMAForecaster
from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQ_LEN, FORECAST_STEPS, DEVICE
from fuzzy import evaluate_fuzzy_anomaly

from matplotlib_utils import save_plot_to_file, save_multiple_metrics_to_files, save_anomaly_detection_plot_to_file


def safe_gauge(name, documentation):
    try:
        metric = Gauge(name, documentation, registry=None)
        REGISTRY.unregister(metric)
    except KeyError:
        pass
    return Gauge(name, documentation)


app = Flask(__name__)

def sanitize_feature_name(feature):
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

# Define path to save plots
save_dir = "./saved_plots"

# Prometheus Metrics
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
fuzzy_risk_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_fuzzy_risk", f"Fuzzy anomaly risk for {feature}")
    for feature in FEATURES
}

# Load trained LSTM model
model = load_model(MODEL_PATH)
print(f"✅ Loaded SARIMA-EE-LSTM model and scaler from {MODEL_PATH}")

# Initialize SARIMA model (fit only once)
sarima_model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), index_col=0)

# Buffer & SARIMA
sequence_buffer = []

def monitor():
    global sequence_buffer, sarima_model

    while True:
        try:
            latest_data = fetch_latest_data()
            if len(latest_data) != len(FEATURES):
                raise ValueError(f"Mismatch between fetched data ({len(latest_data)}) and features ({len(FEATURES)})")

            sequence_buffer.append(latest_data)
            if len(sequence_buffer) > SEQ_LEN:
                sequence_buffer = sequence_buffer[-SEQ_LEN:]

            if len(sequence_buffer) == SEQ_LEN:
                input_sequence = np.array(sequence_buffer, dtype=np.float32)

                # Fit SARIMA only once
                if sarima_model.results is None:
                    sarima_model.fit(input_sequence)
                    print("✅ SARIMA model fitted.")

                # Anomaly detection
                anomalies, mse_per_feature = detect_anomaly_per_feature(input_sequence, model)

                # Forecasting
                forecast_values, upper_bounds, lower_bounds = forecast(
                    model, input_sequence, sarima_model, forecast_steps=FORECAST_STEPS
                )

                # چاپ داده‌های پیش‌بینی
                print(f"Forecast values: {forecast_values}")
                print(f"Upper bounds: {upper_bounds}")
                print(f"Lower bounds: {lower_bounds}")

                # Update Prometheus metrics
                for i, feature in enumerate(FEATURES):
                    mse_gauges[feature].set(mse_per_feature[i])
                    anomaly_gauges[feature].set(anomalies[feature])
                    forecast_gauges[feature].set(forecast_values[0, i])
                    upper_bound_gauges[feature].set(upper_bounds[0, i])
                    lower_bound_gauges[feature].set(lower_bounds[0, i])

                    # Fuzzy evaluation
                    prediction = forecast_values[0, i]
                    upper = upper_bounds[0, i]
                    lower = lower_bounds[0, i]
                    actual = input_sequence[-1, i]

                    fuzzy_risk = evaluate_fuzzy_anomaly(prediction, actual, upper, lower)
                    print(f"Fuzzy Risk for {feature}: {fuzzy_risk}")
                    
                    fuzzy_risk_gauges[feature].set(fuzzy_risk)

                    if fuzzy_risk > 0.8:
                        print(f"🚨 {feature}: خطر بالا (fuzzy risk = {fuzzy_risk:.2f})")
                    elif fuzzy_risk > 0.5:
                        print(f"⚠️ {feature}: خطر متوسط (fuzzy risk = {fuzzy_risk:.2f})")
                    else:
                        print(f"✅ {feature}: وضعیت نرمال (fuzzy risk = {fuzzy_risk:.2f})")

                print("✅ Metrics updated:", dict(zip(FEATURES, mse_per_feature)))
                print("✅ Forecast values:", dict(zip(FEATURES, forecast_values[0])))

                # Save forecast vs actual as files
                metrics_dict = {
                    feature: (input_sequence[:, i], forecast_values[0, i]) 
                    for i, feature in enumerate(FEATURES)
                }
                save_multiple_metrics_to_files(metrics_dict, save_dir)

                # Save anomalies for each feature as files
                for i, feature in enumerate(FEATURES):
                    save_anomaly_detection_plot_to_file(input_sequence[:, i], forecast_values[0, i], anomalies[feature], feature, save_dir)

        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")

        time.sleep(FETCH_INTERVAL)


# # Disable default collectors to keep metrics clean
try:
    REGISTRY.unregister(prom_client.GC_COLLECTOR)
    REGISTRY.unregister(prom_client.PLATFORM_COLLECTOR)
    REGISTRY.unregister(prom_client.PROCESS_COLLECTOR)
except Exception as e:
    print(f"ℹ️ Could not unregister some collectors: {e}")

# Flask routes
@app.route('/metrics')
def metrics():
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    app.run(host='0.0.0.0', port=8000)
