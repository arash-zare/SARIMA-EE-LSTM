# fuzzy.py
"""
This module implements a fuzzy logic system for anomaly detection in time-series data.
It evaluates the risk of anomalies based on prediction residuals and deviations from bounds.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy input variables with adjusted ranges based on typical Node Exporter metrics
residual = ctrl.Antecedent(np.arange(-1000, 1001, 10), 'residual')  # prediction - actual
upper_diff = ctrl.Antecedent(np.arange(0, 1001, 10), 'upper_diff')  # distance from upper bound
lower_diff = ctrl.Antecedent(np.arange(0, 1001, 10), 'lower_diff')  # distance from lower bound

# Define fuzzy output variable
anomaly_risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'anomaly_risk')

# Define membership functions
residual['low'] = fuzz.trimf(residual.universe, [-1000, -500, 0])
residual['medium'] = fuzz.trimf(residual.universe, [-200, 0, 200])
residual['high'] = fuzz.trimf(residual.universe, [0, 500, 1000])

upper_diff['close'] = fuzz.trimf(upper_diff.universe, [0, 0, 200])
upper_diff['far'] = fuzz.trimf(upper_diff.universe, [100, 500, 1000])

lower_diff['close'] = fuzz.trimf(lower_diff.universe, [0, 0, 200])
lower_diff['far'] = fuzz.trimf(lower_diff.universe, [100, 500, 1000])

anomaly_risk['low'] = fuzz.trimf(anomaly_risk.universe, [0.0, 0.0, 0.4])
anomaly_risk['medium'] = fuzz.trimf(anomaly_risk.universe, [0.3, 0.5, 0.7])
anomaly_risk['high'] = fuzz.trimf(anomaly_risk.universe, [0.6, 1.0, 1.0])

# Define expanded fuzzy rules to cover more scenarios
rules = [
    # High risk rules
    ctrl.Rule(residual['high'] & upper_diff['close'], anomaly_risk['high']),
    ctrl.Rule(residual['low'] & lower_diff['close'], anomaly_risk['high']),
    ctrl.Rule(residual['high'] & lower_diff['close'], anomaly_risk['high']),  # Added
    ctrl.Rule(residual['low'] & upper_diff['close'], anomaly_risk['high']),  # Added
    
    # Medium risk rules
    ctrl.Rule(residual['medium'] & (upper_diff['close'] | lower_diff['close']), anomaly_risk['medium']),
    ctrl.Rule(residual['high'] & upper_diff['far'], anomaly_risk['medium']),
    ctrl.Rule(residual['low'] & lower_diff['far'], anomaly_risk['medium']),
    ctrl.Rule(residual['medium'] & upper_diff['far'], anomaly_risk['medium']),  # Added
    ctrl.Rule(residual['medium'] & lower_diff['far'], anomaly_risk['medium']),  # Added
    
    # Low risk rules
    ctrl.Rule(residual['medium'] & upper_diff['far'] & lower_diff['far'], anomaly_risk['low']),
    ctrl.Rule(residual['low'] & upper_diff['far'] & lower_diff['far'], anomaly_risk['low']),
    ctrl.Rule(residual['high'] & upper_diff['far'] & lower_diff['far'], anomaly_risk['low'])  # Added
]

# Create control system
anomaly_ctrl = ctrl.ControlSystem(rules)
anomaly_simulator = ctrl.ControlSystemSimulation(anomaly_ctrl)

def normalize_value(value, min_val, max_val):
    """
    Normalize a value to a specified range, preventing division by zero.
    
    Args:
        value (float): Value to normalize
        min_val (float): Minimum value of the range
        max_val (float): Maximum value of the range
        
    Returns:
        float: Normalized value
    """
    try:
        range_span = max_val - min_val
        if range_span < 1e-10:
            return 0.0  # Avoid division by zero
        return (value - min_val) / range_span
    except Exception as e:
        print(f"[❌] Error in normalize_value: {str(e)}")
        return 0.0

def evaluate_fuzzy_anomaly(y_pred, y_true, upper, lower):
    """
    Evaluate anomaly risk using fuzzy logic based on prediction residuals and bounds.
    
    Args:
        y_pred (float): Predicted value
        y_true (float): Actual value
        upper (float): Upper bound for anomaly detection
        lower (float): Lower bound for anomaly detection
        
    Returns:
        float: Anomaly risk score between 0 and 1
    """
    try:
        # Calculate raw differences
        residual_val = y_pred - y_true
        upper_diff_val = max(0.0, y_pred - upper)
        lower_diff_val = max(0.0, lower - y_pred)
        
        # Scale differences to match fuzzy membership ranges
        scale = max(abs(y_true), abs(y_pred), abs(upper), abs(lower), 1.0)
        residual_val = residual_val / scale * 1000  # Scale to [-1000, 1000]
        upper_diff_val = upper_diff_val / scale * 1000  # Scale to [0, 1000]
        lower_diff_val = lower_diff_val / scale * 1000  # Scale to [0, 1000]
        
        # Clip values to membership ranges
        residual_val = np.clip(residual_val, -1000, 1000)
        upper_diff_val = np.clip(upper_diff_val, 0, 1000)
        lower_diff_val = np.clip(lower_diff_val, 0, 1000)
        
        # Debug inputs
        print(f"[DEBUG] Fuzzy inputs: residual={residual_val:.2f}, upper_diff={upper_diff_val:.2f}, lower_diff={lower_diff_val:.2f}")
        
        # Set inputs to fuzzy system
        anomaly_simulator.input['residual'] = float(residual_val)
        anomaly_simulator.input['upper_diff'] = float(upper_diff_val)
        anomaly_simulator.input['lower_diff'] = float(lower_diff_val)
        
        # Compute fuzzy output
        anomaly_simulator.compute()
        
        # Get and validate risk score
        risk_score = float(anomaly_simulator.output['anomaly_risk'])
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        print(f"[DEBUG] Fuzzy output: risk_score={risk_score:.2f}")
        return risk_score
        
    except Exception as e:
        print(f"[❌] Error in fuzzy evaluation: {str(e)}")
        # Fallback: Calculate relative differences
        residual_val = y_pred - y_true if 'residual_val' not in locals() else residual_val
        upper_diff_val = max(0.0, y_pred - upper) if 'upper_diff_val' not in locals() else upper_diff_val
        lower_diff_val = max(0.0, lower - y_pred) if 'lower_diff_val' not in locals() else lower_diff_val
        
        scale = max(abs(y_true), abs(y_pred), abs(upper), abs(lower), 1.0)
        rel_residual = abs(residual_val) / (scale + 1e-10)
        rel_upper = upper_diff_val / (scale + 1e-10)
        rel_lower = lower_diff_val / (scale + 1e-10)
        
        # Fallback risk based on relative differences
        if rel_residual > 0.5 or rel_upper > 0.5 or rel_lower > 0.5:
            print("[DEBUG] Fallback: High risk")
            return 0.8
        elif rel_residual > 0.2 or rel_upper > 0.2 or rel_lower > 0.2:
            print("[DEBUG] Fallback: Medium risk")
            return 0.5
        else:
            print("[DEBUG] Fallback: Low risk")
            return 0.2