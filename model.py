import torch
import torch.nn as nn
import numpy as np
import statsmodels.api as sm
from config import INPUT_DIM, SEQ_LEN, DEVICE, MODEL_PATH

# SARIMA-EE-LSTM model
class SARIMA_EELSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, num_layers=2):
        super(SARIMA_EELSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # فقط آخرین تایم‌استپ
        out = self.fc(out)
        return out  # shape: (batch_size, input_dim)

# Forecast function using the model
def forecast(model, input_sequence, forecast_steps=1):
    """
    Predict future steps based on input_sequence.

    Args:
        model: trained SARIMA_EELSTM model
        input_sequence: numpy array (seq_len, input_dim)
        forecast_steps: how many steps to forecast

    Returns:
        forecasts: np.array of shape (forecast_steps, input_dim)
        upper_bounds: np.array (forecast_steps, input_dim)
        lower_bounds: np.array (forecast_steps, input_dim)
    """
    model.eval()

    input_seq = torch.tensor(input_sequence[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    forecasts = []

    with torch.no_grad():
        current_input = input_seq.clone()

        for _ in range(forecast_steps):
            output = model(current_input)
            forecasts.append(output.squeeze(0).cpu().numpy())

            # Append the prediction to sequence for next prediction
            next_input = torch.cat([current_input[:, 1:, :], output.unsqueeze(1)], dim=1)
            current_input = next_input

    forecasts = np.stack(forecasts, axis=0)

    # Simple bounds (±10% as example)
    upper_bounds = forecasts * 1.1
    lower_bounds = forecasts * 0.9

    return forecasts, upper_bounds, lower_bounds

# Fit SARIMA model per feature
def fit_sarima_forecast(data, forecast_steps=1):
    """
    Apply SARIMA per feature to forecast.

    Args:
        data: numpy array (n_samples, n_features)
        forecast_steps: steps to forecast

    Returns:
        forecasted_data: (forecast_steps, n_features)
    """
    data = np.asarray(data)
    n_samples, n_features = data.shape
    forecasted = np.zeros((forecast_steps, n_features))

    for i in range(n_features):
        series = data[:, i]
        try:
            model = sm.tsa.ARIMA(series, order=(1, 0, 0))
            fitted_model = model.fit()
            pred = fitted_model.forecast(steps=forecast_steps)
            forecasted[:, i] = pred
        except Exception as e:
            print(f"⚠️ SARIMA fitting failed for feature {i}: {e}")
            # fallback: repeat last value
            forecasted[:, i] = series[-1]

    return forecasted

# Optional: load_model (فقط در صورتی که بخوای جداگانه لود کنی)
def load_model(model_path=MODEL_PATH):
    model = SARIMA_EELSTM()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
