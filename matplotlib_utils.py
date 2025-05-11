"""
matplotlib_utils.py
~~~~~~~~~~~~~~~~~~
This module provides functions to visualize and save plots for actual vs predicted values and anomaly detection.
It is designed to work with main.py and supports both scalar and array inputs for predictions and anomalies.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from datetime import datetime
import matplotlib as mpl

# Set consistent style
plt.style.use('default')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

# Define consistent colors
COLORS = {
    'actual': '#1f77b4',  # Blue
    'predicted': '#ff7f0e',  # Orange
    'anomaly': '#d62728',  # Red
    'error': '#2ca02c',  # Green
    'background': '#f8f9fa'  # Light gray
}

def sanitize_feature_name(feature):
    """Sanitize feature names for file naming by replacing invalid characters.

    Args:
        feature (str): The feature name.

    Returns:
        str: Sanitized feature name.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

def format_value(value):
    """Format value for display based on its magnitude."""
    if abs(value) >= 1e9:
        return f'{value/1e9:.2f}B'
    elif abs(value) >= 1e6:
        return f'{value/1e6:.2f}M'
    elif abs(value) >= 1e3:
        return f'{value/1e3:.2f}K'
    else:
        return f'{value:.4f}'

def save_plot_to_file(actual, predicted, feature_name, file_path, forecast_steps=1):
    """Save a plot of actual vs predicted values to a file.

    Args:
        actual (array-like): Actual data.
        predicted (array-like or float): Predicted data (scalar or array).
        feature_name (str): Name of the feature for the plot.
        file_path (str): Directory to save the plot.
        forecast_steps (int): Number of forecast steps (default=1).
    """
    actual = np.array(actual, dtype=np.float32)
    time_steps = np.arange(len(actual))  # Time steps for actual data

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    fig.suptitle(f'Time Series Analysis: {feature_name}', fontsize=16, y=0.95)

    # Main plot (top)
    ax1.plot(time_steps, actual, label='Actual', color=COLORS['actual'], linewidth=2, marker='o', markersize=4)
    
    # Handle scalar or array predicted values
    if np.isscalar(predicted) or len(np.shape(predicted)) == 0:
        last_time_step = len(actual) - 1
        ax1.scatter([last_time_step + 1], [predicted], color=COLORS['predicted'], label='Predicted', zorder=5, s=100, marker='*')
        ax1.axhline(y=predicted, xmin=(last_time_step + 1) / (last_time_step + 2), 
                   color=COLORS['predicted'], linestyle='--', alpha=0.5, label='Prediction Line')
        # Add prediction value annotation
        ax1.annotate(f'Pred: {format_value(predicted)}', 
                    xy=(last_time_step + 1, predicted),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        predicted = np.array(predicted, dtype=np.float32)
        predicted_steps = np.arange(len(actual) - len(predicted), len(actual))
        ax1.plot(predicted_steps, predicted, label='Predicted', color=COLORS['predicted'], 
                linestyle='--', linewidth=2, marker='*', markersize=6)

    # Add error bands if we have predictions
    if not np.isscalar(predicted):
        error = np.abs(actual[-len(predicted):] - predicted)
        ax1.fill_between(predicted_steps, 
                        predicted - error, 
                        predicted + error, 
                        color=COLORS['predicted'], alpha=0.1, 
                        label='Error Band')

    ax1.set_title('Actual vs Predicted Values', fontsize=12)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Error plot (bottom)
    if not np.isscalar(predicted):
        error = np.abs(actual[-len(predicted):] - predicted)
        ax2.bar(predicted_steps, error, color=COLORS['error'], alpha=0.6, label='Absolute Error')
        ax2.set_title('Prediction Error', fontsize=12)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Absolute Error')
        ax2.grid(True, alpha=0.3)
    else:
        error = np.abs(actual[-1] - predicted)
        ax2.bar([len(actual)], [error], color=COLORS['error'], alpha=0.6, label='Absolute Error')
        ax2.set_title('Prediction Error', fontsize=12)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Absolute Error')
        ax2.grid(True, alpha=0.3)

    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, color='gray')

    plt.tight_layout()

    # Save file
    os.makedirs(file_path, exist_ok=True)
    file_name = f"{sanitize_feature_name(feature_name)}_plot.png"
    full_path = os.path.join(file_path, file_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_multiple_metrics_to_files(metrics_dict, save_dir):
    """Save actual vs predicted plots for multiple features.

    Args:
        metrics_dict (dict): Dictionary with feature names as keys and (actual, predicted) tuples as values.
        save_dir (str): Directory to save the plots.
    """
    for feature_name, (actual, predicted) in metrics_dict.items():
        save_plot_to_file(actual, predicted, feature_name, save_dir)

def save_anomaly_detection_plot_to_file(actual, predicted, anomaly, feature_name, file_path):
    """Save a plot with actual, predicted values, and highlighted anomalies.

    Args:
        actual (array-like): Actual data.
        predicted (array-like or float): Predicted data (scalar or array).
        anomaly (array-like or int): Anomaly detection results (binary array or single value).
        feature_name (str): Name of the feature.
        file_path (str): Directory to save the plot.
    """
    actual = np.array(actual, dtype=np.float32)
    time_steps = np.arange(len(actual))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    fig.suptitle(f'Anomaly Detection Analysis: {feature_name}', fontsize=16, y=0.95)

    # Main plot (top)
    ax1.plot(time_steps, actual, label='Actual', color=COLORS['actual'], linewidth=2, marker='o', markersize=4)

    # Handle scalar or array predicted values
    if np.isscalar(predicted) or len(np.shape(predicted)) == 0:
        last_time_step = len(actual) - 1
        ax1.scatter([last_time_step + 1], [predicted], color=COLORS['predicted'], label='Predicted', zorder=5, s=100, marker='*')
        ax1.axhline(y=predicted, xmin=(last_time_step + 1) / (last_time_step + 2), 
                   color=COLORS['predicted'], linestyle='--', alpha=0.5, label='Prediction Line')
        # Add prediction value annotation
        ax1.annotate(f'Pred: {format_value(predicted)}', 
                    xy=(last_time_step + 1, predicted),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        predicted = np.array(predicted, dtype=np.float32)
        predicted_steps = np.arange(len(actual) - len(predicted), len(actual))
        ax1.plot(predicted_steps, predicted, label='Predicted', color=COLORS['predicted'], 
                linestyle='--', linewidth=2, marker='*', markersize=6)

    # Handle scalar or array anomaly values
    if np.isscalar(anomaly) or len(np.shape(anomaly)) == 0:
        if anomaly == 1:
            ax1.scatter([len(actual)], [predicted], color=COLORS['anomaly'], label='Anomaly', 
                       zorder=5, s=150, marker='*')
            # Add anomaly annotation
            ax1.annotate('ANOMALY DETECTED', 
                        xy=(len(actual), predicted),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        anomaly = np.array(anomaly, dtype=np.int32)
        anomaly_points = time_steps[anomaly == 1]
        if len(anomaly_points) > 0:
            ax1.scatter(anomaly_points, actual[anomaly == 1], color=COLORS['anomaly'], 
                       label='Anomalies', zorder=5, s=150, marker='*')
            # Add anomaly annotations
            for point in anomaly_points:
                ax1.annotate('ANOMALY', 
                            xy=(point, actual[point]),
                            xytext=(10, 20), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_title('Actual vs Predicted Values with Anomalies', fontsize=12)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Anomaly plot (bottom)
    if not np.isscalar(anomaly):
        anomaly = np.array(anomaly, dtype=np.int32)
        ax2.bar(time_steps, anomaly, color=COLORS['anomaly'], alpha=0.6, label='Anomaly Flag')
        ax2.set_title('Anomaly Detection Results', fontsize=12)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Anomaly (0/1)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.bar([len(actual)], [anomaly], color=COLORS['anomaly'], alpha=0.6, label='Anomaly Flag')
        ax2.set_title('Anomaly Detection Results', fontsize=12)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Anomaly (0/1)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)

    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, color='gray')

    plt.tight_layout()

    # Save file
    os.makedirs(file_path, exist_ok=True)
    file_name = f"{sanitize_feature_name(feature_name)}_anomaly_plot.png"
    full_path = os.path.join(file_path, file_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()