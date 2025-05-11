"""
fuzzy.py
~~~~~~~~
This module implements a fuzzy logic system for anomaly detection.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy input variables with ranges based on your data
residual = ctrl.Antecedent(np.arange(-1e6, 1e6, 1e4), 'residual')  # prediction - actual
upper_diff = ctrl.Antecedent(np.arange(0, 1e6, 1e4), 'upper_diff')  # distance from upper bound
lower_diff = ctrl.Antecedent(np.arange(0, 1e6, 1e4), 'lower_diff')  # distance from lower bound

# Define fuzzy output variable
anomaly_risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'anomaly_risk')

# Define membership functions based on your data ranges
residual['low'] = fuzz.trimf(residual.universe, [-1e6, -5e5, 0])
residual['medium'] = fuzz.trimf(residual.universe, [-2e5, 0, 2e5])
residual['high'] = fuzz.trimf(residual.universe, [0, 5e5, 1e6])

upper_diff['close'] = fuzz.trimf(upper_diff.universe, [0, 0, 2e5])
upper_diff['far'] = fuzz.trimf(upper_diff.universe, [1e5, 5e5, 1e6])

lower_diff['close'] = fuzz.trimf(lower_diff.universe, [0, 0, 2e5])
lower_diff['far'] = fuzz.trimf(lower_diff.universe, [1e5, 5e5, 1e6])

anomaly_risk['low'] = fuzz.trimf(anomaly_risk.universe, [0.0, 0.0, 0.4])
anomaly_risk['medium'] = fuzz.trimf(anomaly_risk.universe, [0.3, 0.5, 0.7])
anomaly_risk['high'] = fuzz.trimf(anomaly_risk.universe, [0.6, 1.0, 1.0])

# Define fuzzy rules
rules = [
    # High risk rules
    ctrl.Rule(residual['high'] & upper_diff['close'], anomaly_risk['high']),
    ctrl.Rule(residual['low'] & lower_diff['close'], anomaly_risk['high']),
    
    # Medium risk rules
    ctrl.Rule(residual['medium'] & (upper_diff['close'] | lower_diff['close']), anomaly_risk['medium']),
    ctrl.Rule(residual['high'] & upper_diff['far'], anomaly_risk['medium']),
    ctrl.Rule(residual['low'] & lower_diff['far'], anomaly_risk['medium']),
    
    # Low risk rules
    ctrl.Rule(residual['medium'] & upper_diff['far'] & lower_diff['far'], anomaly_risk['low']),
    ctrl.Rule(residual['low'] & upper_diff['far'] & lower_diff['far'], anomaly_risk['low']),
]

# Create control system
anomaly_ctrl = ctrl.ControlSystem(rules)
anomaly_simulator = ctrl.ControlSystemSimulation(anomaly_ctrl)

def normalize_value(value, min_val, max_val):
    """Normalize a value to the range [0, 1]."""
    return (value - min_val) / (max_val - min_val)

def evaluate_fuzzy_anomaly(y_pred, y_true, upper, lower):
    """
    Evaluate anomaly risk using fuzzy logic.
    
    Args:
        y_pred (float): Predicted value
        y_true (float): Actual value
        upper (float): Upper bound
        lower (float): Lower bound
        
    Returns:
        float: Anomaly risk score between 0 and 1
    """
    try:
        # Calculate raw differences
        residual_val = y_pred - y_true
        upper_diff_val = max(0.0, y_pred - upper)
        lower_diff_val = max(0.0, lower - y_pred)
        
        # Normalize differences based on the scale of the data
        scale = max(abs(y_true), abs(y_pred), abs(upper), abs(lower))
        if scale > 0:
            residual_val = residual_val / scale
            upper_diff_val = upper_diff_val / scale
            lower_diff_val = lower_diff_val / scale
        
        # Clip values to reasonable ranges
        residual_val = np.clip(residual_val, -1.0, 1.0)
        upper_diff_val = np.clip(upper_diff_val, 0.0, 1.0)
        lower_diff_val = np.clip(lower_diff_val, 0.0, 1.0)
        
        # Set inputs to fuzzy system
        anomaly_simulator.input['residual'] = float(residual_val)
        anomaly_simulator.input['upper_diff'] = float(upper_diff_val)
        anomaly_simulator.input['lower_diff'] = float(lower_diff_val)
        
        # Compute fuzzy output
        anomaly_simulator.compute()
        
        # Get risk score and ensure it's a float
        risk_score = float(anomaly_simulator.output['anomaly_risk'])
        
        # Ensure risk score is between 0 and 1
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        return risk_score
        
    except Exception as e:
        print(f"❌ Error in fuzzy evaluation: {str(e)}")
        # Calculate relative differences for fallback
        rel_residual = abs(residual_val) / (abs(y_true) + 1e-10)
        rel_upper = upper_diff_val / (abs(upper) + 1e-10)
        rel_lower = lower_diff_val / (abs(lower) + 1e-10)
        
        # Return risk based on relative differences
        if rel_residual > 0.5 or rel_upper > 0.5 or rel_lower > 0.5:
            return 0.8  # High risk
        elif rel_residual > 0.2 or rel_upper > 0.2 or rel_lower > 0.2:
            return 0.5  # Medium risk
        else:
            return 0.2  # Low risk
