# model.py
import torch
import torch.nn as nn
import numpy as np
import statsmodels.api as sm
from torch.utils.data import DataLoader, TensorDataset
from config import INPUT_DIM, SEQ_LEN, DEVICE, MODEL_PATH, SCALER_PATH
from preprocessing import fit_scaler, transform_data, load_scaler, save_scaler

# --- Model Definition ---
class SARIMA_EELSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, num_layers=2):
        super(SARIMA_EELSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last timestep's output
        out = self.fc(out)
        return out

# --- Forecast Function ---
from preprocessing import inverse_transform_data  # 🔥 اضافه کن بالای فایل اگر نیست

def forecast(model, input_sequence, forecast_steps=1):
    model.eval()
    input_sequence = transform_data(input_sequence)  # 🔥 normalize input
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

    # 🔥 اینجا خروجی رو به مقیاس اصلی برگردون
    forecasts_real = inverse_transform_data(forecasts)

    # 🔥 سپس Bound ها رو حساب کن
    upper_bounds = forecasts_real * 1.1
    lower_bounds = forecasts_real * 0.9

    # 🔥 خروجی دقیقا مثل قبل: سه تا
    return forecasts_real, upper_bounds, lower_bounds


# --- Load Model and Scaler ---
def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model = SARIMA_EELSTM()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    load_scaler(scaler_path)  # 🔥 اسکیلر رو هم لود کن
    print(f"✅ Model and Scaler loaded from {model_path} and {scaler_path}")
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
# --- Training ---
def train_model(
    data,
    seq_len,
    epochs,
    batch_size,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    resume_training=False
):
    """
    Train the SARIMA_EELSTM model on given data.

    Args:
        data (np.ndarray or torch.Tensor): Input data.
        seq_len (int): Sequence length for LSTM input.
        epochs (int): Number of epochs.
        batch_size (int): Batch size for DataLoader.
        model_path (str): Path to save the trained model.
        scaler_path (str): Path to save the fitted scaler.
        resume_training (bool): If True, resume training from saved model.

    Returns:
        model (SARIMA_EELSTM): Trained model.
    """
    # --- Handle input ---
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    # --- Fit or Load Scaler ---
    if resume_training:
        try:
            load_scaler(scaler_path)
            print(f"[ℹ️] Loaded existing scaler from {scaler_path}.")
        except Exception as e:
            print(f"[⚠️] Failed to load scaler: {e}. Fitting new scaler...")
            fit_scaler(data.numpy())
            save_scaler(scaler_path)
    else:
        fit_scaler(data.numpy())
        save_scaler(scaler_path)

    # --- Scale data ---
    data = transform_data(data.numpy())
    data = torch.tensor(data, dtype=torch.float32)

    # --- Create sequences ---
    sequences, targets = create_sequences(data, seq_len)
    dataset = TensorDataset(sequences, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Load or Initialize Model ---
    model = SARIMA_EELSTM(input_dim=data.shape[1]).to(DEVICE)
    if resume_training:
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"[ℹ️] Resuming training from {model_path}.")
        except Exception as e:
            print(f"[⚠️] Failed to load model: {e}. Training from scratch...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
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
        print(f"[🧠] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # --- Save Model ---
    torch.save(model.state_dict(), model_path)
    print(f"[✅] Model saved at {model_path}")
    print(f"[✅] Scaler saved at {scaler_path}")

    return model
