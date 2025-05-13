# matplotlib_utils.py
"""
This module provides functions to visualize and save plots for actual vs predicted values and anomaly detection.
It displays all parameters (actual, predicted, MSE, fuzzy risk, upper/lower bounds) and supports scalar/array inputs.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from datetime import datetime
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

# Try to use a commonly available font family
try:
    # Try DejaVu Sans first (commonly available on Linux)
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    try:
        # Fallback to sans-serif
        plt.rcParams['font.family'] = 'sans-serif'
    except:
        # If all else fails, use the default
        pass

# Define consistent colors
COLORS = {
    'actual': '#1f77b4',  # Blue
    'predicted': '#ff7f0e',  # Orange
    'anomaly': '#d62728',  # Red
    'error': '#2ca02c',  # Green
    'upper_bound': '#9467bd',  # Purple
    'lower_bound': '#8c564b',  # Brown
    'background': '#f8f9fa'  # Light gray
}

def sanitize_feature_name(feature):
    """
    Sanitize feature names for file naming by replacing invalid characters.
    
    Args:
        feature (str): The feature name.
    
    Returns:
        str: Sanitized feature name.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', feature).lower()

def format_value(value):
    """
    Format value for display based on its magnitude.
    
    Args:
        value (float): Value to format.
    
    Returns:
        str: Formatted value.
    """
    try:
        if np.isnan(value) or np.isinf(value):
            return 'N/A'
        if abs(value) >= 1e9:
            return f'{value/1e9:.2f}B'
        elif abs(value) >= 1e6:
            return f'{value/1e6:.2f}M'
        elif abs(value) >= 1e3:
            return f'{value/1e3:.2f}K'
        else:
            return f'{value:.4f}'
    except Exception:
        return 'N/A'

def save_plot_to_file(actual, predicted, feature_name, file_path, mse=None, risk_score=None, upper_bound=None, lower_bound=None):
    """
    Save a plot of actual vs predicted values with all parameters.
    
    Args:
        actual (array-like): Actual data.
        predicted (array-like or float): Predicted data (scalar or array).
        feature_name (str): Name of the feature for the plot.
        file_path (str): Directory to save the plot.
        mse (float, optional): Mean squared error.
        risk_score (float, optional): Fuzzy risk score (0 to 1).
        upper_bound (float, optional): Upper bound for the prediction.
        lower_bound (float, optional): Lower bound for the prediction.
    """
    try:
        # Validate inputs
        actual = np.array(actual, dtype=np.float32)
        if np.any(np.isnan(actual)) or np.any(np.isinf(actual)):
            actual = np.nan_to_num(actual, nan=0.0, posinf=0.0, neginf=0.0)
        time_steps = np.arange(len(actual))
        last_time_step = len(actual) - 1

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[3, 1, 1])
        fig.suptitle(f'Time Series Analysis: {feature_name}', fontsize=16, y=0.98)

        # Main plot (top)
        ax1.plot(time_steps, actual, label='Actual', color=COLORS['actual'], linewidth=2)
        
        # Handle scalar or array predicted values
        if np.isscalar(predicted) or len(np.shape(predicted)) == 0:
            ax1.scatter([last_time_step + 1], [predicted], color=COLORS['predicted'], label='Predicted', zorder=5, s=100, marker='*')
            ax1.axhline(y=predicted, xmin=(last_time_step + 1) / (last_time_step + 2), 
                       color=COLORS['predicted'], linestyle='--', alpha=0.5, label='Prediction Line')
            ax1.annotate(f'Pred: {format_value(predicted)}', 
                        xy=(last_time_step + 1, predicted),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        else:
            predicted = np.array(predicted, dtype=np.float32)
            predicted_steps = np.arange(len(actual) - len(predicted), len(actual))
            ax1.plot(predicted_steps, predicted, label='Predicted', color=COLORS['predicted'], 
                    linestyle='--', linewidth=2)

        # Add bounds if provided
        if upper_bound is not None and lower_bound is not None:
            if np.isscalar(upper_bound):
                ax1.axhline(y=upper_bound, color=COLORS['upper_bound'], linestyle=':', alpha=0.5, label='Upper Bound')
                ax1.axhline(y=lower_bound, color=COLORS['lower_bound'], linestyle=':', alpha=0.5, label='Lower Bound')
                ax1.fill_between([last_time_step, last_time_step + 1], 
                                [lower_bound, lower_bound], 
                                [upper_bound, upper_bound], 
                                color=COLORS['upper_bound'], alpha=0.1, label='Bound Range')
            else:
                ax1.fill_between(predicted_steps, lower_bound, upper_bound, 
                                color=COLORS['upper_bound'], alpha=0.1, label='Bound Range')

        # Mark high-risk points
        if risk_score is not None and np.isscalar(risk_score) and risk_score > 0.5:
            ax1.scatter([last_time_step + 1], [predicted if np.isscalar(predicted) else predicted[-1]], 
                       color=COLORS['anomaly'], label='High Risk', zorder=5, s=150, marker='*')
            ax1.annotate(f'Risk: {risk_score:.2f}', 
                        xy=(last_time_step + 1, predicted if np.isscalar(predicted) else predicted[-1]),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax1.set_title('Actual vs Predicted Values', fontsize=12)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Error plot (middle)
        if np.isscalar(predicted):
            error = np.abs(actual[-1] - predicted)
            ax2.bar([last_time_step + 1], [error], color=COLORS['error'], alpha=0.6, label='Absolute Error')
        else:
            error = np.abs(actual[-len(predicted):] - predicted)
            ax2.bar(predicted_steps, error, color=COLORS['error'], alpha=0.6, label='Absolute Error')
        ax2.set_title('Prediction Error', fontsize=12)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Absolute Error')
        ax2.grid(True, alpha=0.3)

        # Fuzzy risk plot (bottom)
        if risk_score is not None:
            norm = Normalize(vmin=0, vmax=1)
            cmap = plt.cm.RdYlGn_r
            if np.isscalar(risk_score):
                ax3.bar([last_time_step + 1], [1], color=cmap(norm(risk_score)), alpha=0.6, label='Fuzzy Risk')
                ax3.annotate(f'Risk: {risk_score:.2f}', 
                            xy=(last_time_step + 1, 0.5),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))
            else:
                risk_score = np.array(risk_score, dtype=np.float32)
                for i, t in enumerate(predicted_steps):
                    ax3.bar(t, 1, color=cmap(norm(risk_score[i])), alpha=0.6)
                sm = ScalarMappable(cmap=cmap, norm=norm)
                plt.colorbar(sm, ax=ax3, label='Fuzzy Risk')
            ax3.set_title('Fuzzy Anomaly Risk', fontsize=12)
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Risk Level')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)

        # Add parameter table
        table_data = [
            ['Last Actual', format_value(actual[-1])],
            ['Predicted', format_value(predicted if np.isscalar(predicted) else predicted[-1])],
            ['MSE', format_value(mse) if mse is not None else 'N/A'],
            ['Fuzzy Risk', f'{risk_score:.2f}' if risk_score is not None else 'N/A'],
            ['Upper Bound', format_value(upper_bound) if upper_bound is not None else 'N/A'],
            ['Lower Bound', format_value(lower_bound) if lower_bound is not None else 'N/A']
        ]
        table = plt.table(cellText=table_data,
                         colLabels=['Parameter', 'Value'],
                         loc='bottom',
                         bbox=[0.1, -0.3, 0.8, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, color='gray')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        # Save file
        os.makedirs(file_path, exist_ok=True)
        file_name = f"{sanitize_feature_name(feature_name)}_plot.png"
        full_path = os.path.join(file_path, file_name)
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"[✔️] Plot saved to {full_path}")
        except Exception as e:
            print(f"[❌] Error saving plot to {full_path}: {str(e)}")
        plt.close()

    except Exception as e:
        print(f"[❌] Error in save_plot_to_file for {feature_name}: {str(e)}")
        plt.close()

def save_multiple_metrics_to_files(metrics_dict, save_dir):
    """
    Save actual vs predicted plots for multiple features with all parameters.
    
    Args:
        metrics_dict (dict): Dictionary with feature names as keys and 
                           dicts containing 'actual', 'predicted', 'mse', 'risk_score', 'upper_bound', 'lower_bound'.
        save_dir (str): Directory to save the plots.
    """
    try:
        for feature_name, data in metrics_dict.items():
            save_plot_to_file(
                actual=data.get('actual', []),
                predicted=data.get('predicted', 0.0),
                feature_name=feature_name,
                file_path=save_dir,
                mse=data.get('mse', None),
                risk_score=data.get('risk_score', None),
                upper_bound=data.get('upper_bound', None),
                lower_bound=data.get('lower_bound', None)
            )
    except Exception as e:
        print(f"[❌] Error in save_multiple_metrics_to_files: {str(e)}")

def save_anomaly_detection_plot_to_file(actual, predicted, anomaly, feature_name, file_path, mse=None, risk_score=None, upper_bound=None, lower_bound=None):
    """
    Save a plot with actual, predicted values, and anomaly detection results.
    
    Args:
        actual (array-like): Actual data.
        predicted (array-like or float): Predicted data (scalar or array).
        anomaly (bool or float): Anomaly flag or fuzzy risk score.
        feature_name (str): Name of the feature.
        file_path (str): Directory to save the plot.
        mse (float, optional): Mean squared error.
        risk_score (float, optional): Fuzzy risk score (0 to 1).
        upper_bound (float, optional): Upper bound for the prediction.
        lower_bound (float, optional): Lower bound for the prediction.
    """
    try:
        # Use risk_score if provided, otherwise infer from anomaly
        effective_risk = risk_score if risk_score is not None else (1.0 if anomaly else 0.0)
        save_plot_to_file(
            actual=actual,
            predicted=predicted,
            feature_name=feature_name + "_anomaly",
            file_path=file_path,
            mse=mse,
            risk_score=effective_risk,
            upper_bound=upper_bound,
            lower_bound=lower_bound
        )
    except Exception as e:
        print(f"[❌] Error in save_anomaly_detection_plot_to_file for {feature_name}: {str(e)}")