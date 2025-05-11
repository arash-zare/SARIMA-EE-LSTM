import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# تعریف متغیرهای ورودی فازی
residual = ctrl.Antecedent(np.arange(-1.0, 1.01, 0.01), 'residual')  # پیش‌بینی - واقعی نرمال‌شده
upper_diff = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'upper_diff')  # فاصله از حد بالا
lower_diff = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'lower_diff')  # فاصله از حد پایین

# خروجی فازی
anomaly_risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'anomaly_risk')

# تعریف توابع عضویت
residual['low'] = fuzz.trimf(residual.universe, [-1.0, -0.5, 0.0])
residual['medium'] = fuzz.trimf(residual.universe, [-0.2, 0.0, 0.2])
residual['high'] = fuzz.trimf(residual.universe, [0.0, 0.5, 1.0])

upper_diff['close'] = fuzz.trimf(upper_diff.universe, [0, 0, 0.5])
upper_diff['far'] = fuzz.trimf(upper_diff.universe, [0.3, 1.0, 2.0])

lower_diff['close'] = fuzz.trimf(lower_diff.universe, [0, 0, 0.5])
lower_diff['far'] = fuzz.trimf(lower_diff.universe, [0.3, 1.0, 2.0])

anomaly_risk['low'] = fuzz.trimf(anomaly_risk.universe, [0.0, 0.0, 0.4])
anomaly_risk['medium'] = fuzz.trimf(anomaly_risk.universe, [0.3, 0.5, 0.7])
anomaly_risk['high'] = fuzz.trimf(anomaly_risk.universe, [0.6, 1.0, 1.0])

# قوانین فازی
rules = [
    ctrl.Rule(residual['medium'] & upper_diff['far'] & lower_diff['far'], anomaly_risk['low']),
    ctrl.Rule(residual['high'] & upper_diff['close'], anomaly_risk['high']),
    ctrl.Rule(residual['low'] & lower_diff['close'], anomaly_risk['high']),
    ctrl.Rule(residual['medium'] & (upper_diff['close'] | lower_diff['close']), anomaly_risk['medium']),
    ctrl.Rule(residual['low'] & upper_diff['far'] & lower_diff['far'], anomaly_risk['low']),
]

# کنترلر فازی
anomaly_ctrl = ctrl.ControlSystem(rules)
anomaly_simulator = ctrl.ControlSystemSimulation(anomaly_ctrl)



def evaluate_fuzzy_anomaly(y_pred, y_true, upper, lower):
    residual_val = y_pred - y_true
    upper_diff_val = max(0.0, y_pred - upper)
    lower_diff_val = max(0.0, lower - y_pred)

    # ورودی‌ها به سیستم فازی
    anomaly_simulator.input['residual'] = residual_val
    anomaly_simulator.input['upper_diff'] = upper_diff_val
    anomaly_simulator.input['lower_diff'] = lower_diff_val

    anomaly_simulator.compute()
    
    # بررسی مقدار خروجی
    try:
        risk_score = anomaly_simulator.output['anomaly_risk']
        print(f"Fuzzy anomaly risk score: {risk_score}")  # چاپ مقدار برای دیباگ
    except KeyError as e:
        print(f"❌ Error: Could not access 'anomaly_risk' in fuzzy system. Error: {e}")
        risk_score = None

    return risk_score if risk_score is not None else 0.0  # برگشت مقدار پیش‌فرض در صورت خطا
