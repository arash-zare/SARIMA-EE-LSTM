import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Function to sanitize feature names
def sanitize_feature_name(feature):
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()


def save_plot_to_file(actual, predicted, feature_name, file_path):
    """
    این تابع برای ذخیره نمودار مقادیر واقعی و پیش‌بینی شده برای یک ویژگی خاص به‌صورت فایل استفاده می‌شود.
    
    Args:
    actual (array-like): داده‌های واقعی
    predicted (array-like): داده‌های پیش‌بینی شده
    feature_name (str): نام ویژگی که نمودار مربوط به آن است
    file_path (str): مسیر ذخیره فایل نمودار
    """
    time_steps = np.arange(len(actual))  # زمان گام‌ها را ایجاد می‌کنیم

    # رسم نمودار
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, actual, label='Actual', color='blue')  # داده‌های واقعی
    plt.plot(time_steps, predicted, label='Predicted', color='red', linestyle='--')  # داده‌های پیش‌بینی شده
    plt.title(f'Actual vs Predicted for {feature_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.grid(True)

    # ذخیره فایل به‌جای نمایش
    os.makedirs(file_path, exist_ok=True)  # اگر مسیر وجود ندارد، پوشه‌ها را می‌سازد
    file_name = f"{sanitize_feature_name(feature_name)}_plot.png"  # نام فایل بر اساس ویژگی
    full_path = os.path.join(file_path, file_name)
    
    plt.savefig(full_path)
    plt.close()  # بعد از ذخیره، نمودار را می‌بندیم تا مصرف حافظه کاهش یابد


def save_multiple_metrics_to_files(metrics_dict, save_dir):
    """
    Saves actual vs predicted plots for multiple features.

    Args:
    metrics_dict (dict): Dictionary where keys are feature names, and values are tuples of actual and predicted data
    save_dir (str): Directory to save the plots
    """
    for feature_name, (actual, predicted) in metrics_dict.items():
        save_plot_to_file(actual, predicted, feature_name, save_dir)

def save_anomaly_detection_plot_to_file(actual, predicted, anomaly, feature_name, file_path):
    """
    Saves a plot with actual, predicted values, and highlighted anomalies.

    Args:
    actual (array-like): Actual data
    predicted (array-like): Predicted data
    anomaly (array-like): Anomaly detection results (usually a binary array)
    feature_name (str): The name of the feature
    file_path (str): Path to save the plot
    """
    time_steps = np.arange(len(actual))
    
    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, actual, label='Actual', color='blue')
    plt.plot(time_steps, predicted, label='Predicted', color='red', linestyle='--')
    
    # Highlight anomalies
    anomaly_points = time_steps[anomaly == 1]
    plt.scatter(anomaly_points, actual[anomaly == 1], color='orange', label='Anomalies', zorder=5)
    
    plt.title(f'Actual vs Predicted for {feature_name} with Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the plot to a file
    os.makedirs(file_path, exist_ok=True)
    file_name = f"{sanitize_feature_name(feature_name)}_anomaly_plot.png"
    full_path = os.path.join(file_path, file_name)
    
    plt.savefig(full_path)
    plt.close()  # Close plot after saving
