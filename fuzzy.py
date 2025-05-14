# fuzzy.py
"""
This module implements a fuzzy logic system for anomaly detection in time-series data.
It evaluates the risk of anomalies based on prediction residuals and deviations from bounds.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging
import torch

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

class FuzzyLogic:
    """
    Fuzzy logic implementation for handling uncertainty in predictions.
    """
    
    def __init__(self):
        """Initialize the FuzzyLogic class."""
        self.setup_logger()
        self.params = {
            'lstm_weight': 0.6,
            'sarima_weight': 0.4
        }
        
        # Initialize fuzzy sets for risk evaluation with adjusted parameters
        self.risk_sets = {
            'low': {'center': 0.2, 'width': 0.4},
            'medium': {'center': 0.5, 'width': 0.4},
            'high': {'center': 0.8, 'width': 0.4}
        }
        
        # Initialize fuzzy control system
        self.setup_fuzzy_system()
    
    def setup_fuzzy_system(self):
        """Set up the fuzzy control system."""
        try:
            # Define input variables with adjusted ranges
            self.residual = ctrl.Antecedent(np.arange(-1000, 1001, 10), 'residual')
            self.upper_diff = ctrl.Antecedent(np.arange(0, 1001, 10), 'upper_diff')
            self.lower_diff = ctrl.Antecedent(np.arange(0, 1001, 10), 'lower_diff')
            
            # Define output variable
            self.anomaly_risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'anomaly_risk')
            
            # Define membership functions with adjusted ranges
            self.residual['low'] = fuzz.trimf(self.residual.universe, [-1000, -200, 0])
            self.residual['medium'] = fuzz.trimf(self.residual.universe, [-100, 0, 100])
            self.residual['high'] = fuzz.trimf(self.residual.universe, [0, 200, 1000])
            
            self.upper_diff['close'] = fuzz.trimf(self.upper_diff.universe, [0, 0, 100])
            self.upper_diff['far'] = fuzz.trimf(self.upper_diff.universe, [50, 500, 1000])
            
            self.lower_diff['close'] = fuzz.trimf(self.lower_diff.universe, [0, 0, 100])
            self.lower_diff['far'] = fuzz.trimf(self.lower_diff.universe, [50, 500, 1000])
            
            # Adjust risk output membership functions
            self.anomaly_risk['low'] = fuzz.trimf(self.anomaly_risk.universe, [0.0, 0.0, 0.3])
            self.anomaly_risk['medium'] = fuzz.trimf(self.anomaly_risk.universe, [0.2, 0.5, 0.8])
            self.anomaly_risk['high'] = fuzz.trimf(self.anomaly_risk.universe, [0.7, 1.0, 1.0])
            
            # Define more specific rules
            rules = [
                # High risk rules
                ctrl.Rule(self.residual['high'], self.anomaly_risk['high']),
                ctrl.Rule(self.residual['medium'] & (self.upper_diff['close'] | self.lower_diff['close']), 
                         self.anomaly_risk['high']),
                
                # Medium risk rules
                ctrl.Rule(self.residual['medium'] & self.upper_diff['far'] & self.lower_diff['far'], 
                         self.anomaly_risk['medium']),
                ctrl.Rule(self.residual['low'] & (self.upper_diff['close'] | self.lower_diff['close']), 
                         self.anomaly_risk['medium']),
                
                # Low risk rules
                ctrl.Rule(self.residual['low'] & self.upper_diff['far'] & self.lower_diff['far'], 
                         self.anomaly_risk['low'])
            ]
            
            # Create control system
            self.control_system = ctrl.ControlSystem(rules)
            self.simulator = ctrl.ControlSystemSimulation(self.control_system)
            
        except Exception as e:
            logging.error(f"[❌] Error setting up fuzzy system: {str(e)}")
    
    def evaluate_risk(self, residual, upper_diff, lower_diff):
        """
        Evaluate risk using fuzzy logic.
        
        Args:
            residual (float): Normalized residual
            upper_diff (float): Normalized upper bound difference
            lower_diff (float): Normalized lower bound difference
            
        Returns:
            float: Risk score between 0 and 1
        """
        try:
            # Handle extreme values first
            abs_residual = abs(residual)
            if abs_residual > 800:
                return 0.8  # High risk for extreme residuals
            elif abs_residual > 500:
                return 0.6  # Medium risk for high residuals
            
            # Scale inputs to match fuzzy system ranges
            residual = np.clip(residual * 1000, -1000, 1000)
            upper_diff = np.clip(upper_diff * 1000, 0, 1000)
            lower_diff = np.clip(lower_diff * 1000, 0, 1000)
            
            # Set inputs
            self.simulator.input['residual'] = float(residual)
            self.simulator.input['upper_diff'] = float(upper_diff)
            self.simulator.input['lower_diff'] = float(lower_diff)
            
            # Compute output
            self.simulator.compute()
            
            # Get risk score
            risk_score = float(self.simulator.output['anomaly_risk'])
            risk_score = np.clip(risk_score, 0.0, 1.0)
            
            # Adjust risk score based on MSE
            if abs_residual > 800:  # Very high residual
                risk_score = max(risk_score, 0.8)
            elif abs_residual > 500:  # High residual
                risk_score = max(risk_score, 0.6)
            
            return risk_score
            
        except Exception as e:
            logging.error(f"[❌] Error in fuzzy risk evaluation: {str(e)}")
            # Fallback to simple risk calculation based on residual
            abs_residual = abs(residual)
            if abs_residual > 800:
                return 0.8  # High risk
            elif abs_residual > 500:
                return 0.6  # Medium risk
            else:
                return 0.2  # Low risk
    
    def get_params(self):
        """Get fuzzy logic parameters."""
        return self.params
    
    def set_params(self, **params):
        """Set fuzzy logic parameters."""
        self.params.update(params)
    
    def setup_logger(self):
        """Set up logging configuration."""
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
            )
    
    def calculate_membership(self, error, threshold):
        """
        Calculate fuzzy membership for a given error value.
        
        Args:
            error (float): The error value to calculate membership for
            threshold (float): The threshold value for the feature
            
        Returns:
            float: Membership value between 0 and 1
        """
        try:
            # Normalize error by threshold
            normalized_error = error / threshold
            
            # Calculate membership using a triangular membership function
            if normalized_error <= 0:
                return 1.0  # Perfect match
            elif normalized_error >= 1:
                return 0.0  # Complete mismatch
            else:
                return 1.0 - normalized_error  # Linear decrease
            
        except Exception as e:
            logging.error(f"[❌] Error calculating fuzzy membership: {str(e)}")
            return 0.0
    
    def aggregate_scores(self, scores, method='mean'):
        """
        Aggregate multiple fuzzy scores into a single score.
        
        Args:
            scores (list): List of fuzzy scores to aggregate
            method (str): Aggregation method ('mean', 'min', 'max')
            
        Returns:
            float: Aggregated score
        """
        try:
            if not scores:
                return 0.0
                
            if method == 'mean':
                return np.mean(scores)
            elif method == 'min':
                return np.min(scores)
            elif method == 'max':
                return np.max(scores)
            else:
                logging.warning(f"[⚠️] Unknown aggregation method: {method}, using mean")
                return np.mean(scores)
                
        except Exception as e:
            logging.error(f"[❌] Error aggregating fuzzy scores: {str(e)}")
            return 0.0
    
    def calculate_fuzzy_metrics(self, predictions, actuals, thresholds):
        """
        Calculate fuzzy logic based metrics for predictions.
        
        Args:
            predictions (np.ndarray): Predicted values
            actuals (np.ndarray): Actual values
            thresholds (dict): Threshold values for each feature
            
        Returns:
            dict: Dictionary containing fuzzy metrics
        """
        try:
            fuzzy_scores = []
            
            for pred, actual in zip(predictions, actuals):
                feature_scores = []
                for i, (feature, threshold) in enumerate(thresholds.items()):
                    error = abs(pred[i] - actual[i])
                    membership = self.calculate_membership(error, threshold)
                    feature_scores.append(membership)
                
                # Aggregate feature scores
                fuzzy_scores.append(self.aggregate_scores(feature_scores))
            
            return {
                'Fuzzy_Accuracy': np.mean(fuzzy_scores),
                'Fuzzy_Precision': np.mean([score for score in fuzzy_scores if score > 0.5]),
                'Fuzzy_Recall': np.mean([score for score in fuzzy_scores if score > 0.7])
            }
            
        except Exception as e:
            logging.error(f"[❌] Error calculating fuzzy metrics: {str(e)}")
            return {
                'Fuzzy_Accuracy': 0.0,
                'Fuzzy_Precision': 0.0,
                'Fuzzy_Recall': 0.0
            }

    def combine_predictions(self, lstm_pred, sarima_preds):
        """
        Combine LSTM and SARIMA predictions using fuzzy logic.
        
        Args:
            lstm_pred (torch.Tensor): LSTM predictions
            sarima_preds (torch.Tensor): SARIMA predictions
            
        Returns:
            torch.Tensor: Combined predictions
        """
        try:
            # Convert to numpy for easier manipulation
            lstm_np = lstm_pred.detach().cpu().numpy()
            sarima_np = sarima_preds.detach().cpu().numpy()
            
            # Calculate confidence scores for each model
            lstm_confidence = np.ones_like(lstm_np) * 0.6  # Base confidence for LSTM
            sarima_confidence = np.ones_like(sarima_np) * 0.4  # Base confidence for SARIMA
            
            # Weighted combination
            combined = (lstm_np * lstm_confidence + sarima_np * sarima_confidence) / (lstm_confidence + sarima_confidence)
            
            # Convert back to tensor
            return torch.FloatTensor(combined).to(lstm_pred.device)
            
        except Exception as e:
            logging.error(f"[❌] Error combining predictions: {str(e)}")
            # Fallback to LSTM predictions
            return lstm_pred