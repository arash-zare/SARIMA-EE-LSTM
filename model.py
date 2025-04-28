import torch
import torch.nn as nn
import numpy as np
import statsmodels.api as sm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from config import INPUT_DIM, SEQ_LEN, DEVICE, MODEL_PATH

# --- Model Definition ---
class SARIMA_EELSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, num_layers=2):
        super(SARIMA_EELSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time-step
        out = self.fc(out)
        return out

# --- Forecast Function ---
def forecast(model, input_sequence, forecast_steps=1):
    model.eval()
    input_seq = torch.tensor(input_sequence[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    forecasts = []

    with torch.no_grad():
        current_input = input_seq.clone()

        for _ in range(forecast_steps):
            output = model(current_input)
            forecasts.append(output.squeeze(0).cpu().numpy())
            next_input = torch.cat([current_input[:, 1:, :], output.unsqueeze(1)], dim=1)
            current_input = next_input

    forecasts = np.stack(forecasts, axis=0)
    upper_bounds = forecasts * 1.1
    lower_bounds = forecasts * 0.9

    return forecasts, upper_bounds, lower_bounds

# --- SARIMA per Feature Forecast ---
def fit_sarima_forecast(data, forecast_steps=1):
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
            forecasted[:, i] = series[-1]  # fallback

    return forecasted

# --- Load Saved Model ---
def load_model(model_path=MODEL_PATH):
    model = SARIMA_EELSTM()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- Create Sequences for LSTM ---
def create_sequences(data, seq_len):
    sequences = []
    targets = []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        target = data[i+seq_len]
        sequences.append(seq)
        targets.append(target)

    sequences = np.stack(sequences)
    targets = np.stack(targets)

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# --- Training ---
def train_model(data, seq_len, epochs, batch_size, model_path):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    # Normalize data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.cpu().numpy())
    data = torch.tensor(data, dtype=torch.float32)

    sequences, targets = create_sequences(data, seq_len)
    dataset = TensorDataset(sequences, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SARIMA_EELSTM(input_dim=data.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at {model_path}")

    return model
