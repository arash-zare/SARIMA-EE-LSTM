# train.py
"""
Training script for Fuzzy-SARIMA-EE-LSTM model with comprehensive metrics.
"""

import os
import numpy as np
import time
import logging
import argparse
from data_fetcher import fetch_historical_data
from model import EELSTM, SARIMAForecaster
from preprocessing import fit_scaler, transform_data, preprocess_for_training
from config import (
    FEATURES, MODEL_PATH, SAVE_DIR, DATASET_PATH, 
    SEQ_LEN, FORECAST_STEPS, INPUT_DIM, HIDDEN_DIM, 
    NUM_LAYERS, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE,
    SARIMA_CONFIGS, SCALER_PATH
)
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fuzzy import FuzzyLogic

DATA_CACHE_FILE = "cached_data.npy"

def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

class FuzzySARIMAEE:
    def __init__(self, input_dim, hidden_dim, num_layers, sarima_configs):
        self.lstm = EELSTM(input_dim, hidden_dim, num_layers).to(DEVICE)
        self.sarima_forecasters = {}
        self.sarima_configs = sarima_configs
        self.fuzzy = FuzzyLogic()
        
    def fit_sarima(self, x):
        """Fit SARIMA models for each feature using the input data."""
        for i, feature in enumerate(FEATURES):
            try:
                # Convert to numpy and reshape to 1D array for SARIMA
                feature_data = x[:, :, i].cpu().numpy().flatten()
                forecaster = SARIMAForecaster(**self.sarima_configs[feature])
                forecaster.fit(feature_data)
                self.sarima_forecasters[feature] = forecaster
                logging.info(f"[✔️] Fitted SARIMA model for {feature}")
            except Exception as e:
                logging.warning(f"[⚠️] Failed to fit SARIMA model for {feature}: {str(e)}")
                self.sarima_forecasters[feature] = None
        
    def forward(self, x):
        # LSTM prediction
        lstm_pred = self.lstm(x)
        
        # SARIMA predictions for each feature
        sarima_preds = []
        for i, feature in enumerate(FEATURES):
            try:
                if feature not in self.sarima_forecasters or self.sarima_forecasters[feature] is None:
                    # If SARIMA model is not fitted or failed, use LSTM prediction
                    sarima_pred = lstm_pred[:, i].detach().cpu().numpy()
                else:
                    # Convert to numpy and reshape to 1D array for SARIMA
                    feature_data = x[:, :, i].cpu().numpy().flatten()
                    # Get the last prediction from SARIMA
                    try:
                        sarima_pred = self.sarima_forecasters[feature].forecast(feature_data)
                        if sarima_pred is None or len(sarima_pred) == 0:
                            raise ValueError("Empty SARIMA prediction")
                        # Use the last prediction for all samples in the batch
                        last_pred = float(sarima_pred[-1])  # Convert to float explicitly
                        sarima_pred = np.full(x.size(0), last_pred, dtype=np.float32)
                    except (ValueError, TypeError, IndexError) as e:
                        logging.warning(f"[⚠️] Error in SARIMA forecast for {feature}: {str(e)}")
                        # Use LSTM prediction as fallback
                        sarima_pred = lstm_pred[:, i].detach().cpu().numpy()
                
                # Ensure correct type and check for NaN values
                sarima_pred = np.array(sarima_pred, dtype=np.float32)
                if np.isnan(sarima_pred).any():
                    logging.warning(f"[⚠️] NaN values in SARIMA predictions for {feature}, using LSTM predictions")
                    sarima_pred = lstm_pred[:, i].detach().cpu().numpy()
                
                sarima_preds.append(sarima_pred)
            except Exception as e:
                logging.warning(f"[⚠️] Error in SARIMA forecast for {feature}: {str(e)}")
                # Use LSTM prediction as fallback
                sarima_preds.append(lstm_pred[:, i].detach().cpu().numpy())
        
        # Convert predictions to numpy array and ensure correct shape and type
        sarima_preds = np.array(sarima_preds, dtype=np.float32).T
        sarima_preds = torch.FloatTensor(sarima_preds).to(DEVICE)
        
        # Combine predictions using fuzzy logic
        combined_pred = self.fuzzy.combine_predictions(lstm_pred, sarima_preds)
        
        return combined_pred, lstm_pred, sarima_preds

def calculate_feature_metrics(y_true, y_pred, feature_name):
    """
    Calculate accuracy metrics for a single feature.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        feature_name (str): Name of the feature
    
    Returns:
        dict: Dictionary containing metrics
    """
    metrics = {}
    
    try:
        # Check for NaN values
        if np.isnan(y_true).any() or np.isnan(y_pred).any():
            logging.warning(f"[⚠️] NaN values found in {feature_name} metrics calculation")
            # Replace NaN values with 0
            y_true = np.nan_to_num(y_true, nan=0.0)
            y_pred = np.nan_to_num(y_pred, nan=0.0)
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0  # Avoid division by zero
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = 0.0
        
        # Calculate R² score
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Calculate accuracy (1 - normalized RMSE)
        range_true = np.max(y_true) - np.min(y_true)
        if range_true != 0:
            metrics['accuracy'] = 1 - (metrics['rmse'] / range_true)
        else:
            metrics['accuracy'] = 1.0 if metrics['rmse'] == 0 else 0.0
        
    except Exception as e:
        logging.error(f"[❌] Error calculating metrics for {feature_name}: {str(e)}")
        # Return default metrics
        metrics = {
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'mape': 0.0,
            'r2': 0.0,
            'accuracy': 0.0
        }
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """
    Train the Fuzzy-SARIMA-EE-LSTM model.
    """
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    feature_metrics_history = {feature: [] for feature in FEATURES}
    
    # Fit SARIMA models using the first batch of training data
    for batch_x, _ in train_loader:
        model.fit_sarima(batch_x)
        break
    
    for epoch in range(epochs):
        # Training phase
        model.lstm.train()
        train_loss = 0
        batch_count = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Get predictions
            combined_pred, lstm_pred, sarima_preds = model.forward(batch_x)
            
            # Check for NaN values in predictions
            if torch.isnan(combined_pred).any() or torch.isnan(lstm_pred).any():
                logging.warning("[⚠️] NaN values in predictions, skipping batch")
                continue
            
            # Calculate loss only on LSTM predictions for backpropagation
            lstm_loss = criterion(lstm_pred, batch_y)
            
            # Calculate combined loss for monitoring
            combined_loss = criterion(combined_pred, batch_y)
            
            # Backpropagate through LSTM only
            lstm_loss.backward()
            optimizer.step()
            
            train_loss += combined_loss.item()
            batch_count += 1
        
        if batch_count > 0:
            train_loss /= batch_count
        train_losses.append(train_loss)
        
        # Validation phase
        model.lstm.eval()
        val_loss = 0
        val_batch_count = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                combined_pred, lstm_pred, sarima_preds = model.forward(batch_x)
                
                # Check for NaN values
                if torch.isnan(combined_pred).any():
                    logging.warning("[⚠️] NaN values in validation predictions, skipping batch")
                    continue
                
                loss = criterion(combined_pred, batch_y)
                val_loss += loss.item()
                val_batch_count += 1
                
                all_val_preds.append(combined_pred.cpu().numpy())
                all_val_targets.append(batch_y.cpu().numpy())
        
        if val_batch_count > 0:
            val_loss /= val_batch_count
        val_losses.append(val_loss)
        
        # Calculate per-feature metrics
        if all_val_preds and all_val_targets:
            all_val_preds = np.concatenate(all_val_preds, axis=0)
            all_val_targets = np.concatenate(all_val_targets, axis=0)
            
            epoch_metrics = {}
            for i, feature in enumerate(FEATURES):
                feature_metrics = calculate_feature_metrics(
                    all_val_targets[:, i],
                    all_val_preds[:, i],
                    feature
                )
                epoch_metrics[feature] = feature_metrics
                feature_metrics_history[feature].append(feature_metrics)
        
        # Log only training progress
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model state dictionary with all components
            torch.save({
                'lstm_state_dict': model.lstm.state_dict(),
                'sarima_configs': SARIMA_CONFIGS,
                'fuzzy_params': model.fuzzy.get_params(),
                'model_config': {
                    'input_dim': INPUT_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS
                }
            }, MODEL_PATH)
            logging.info(f"[✔️] Model saved to {MODEL_PATH}")
    
    return train_losses, val_losses, feature_metrics_history

def run_training(args):
    """
    Main training function for Fuzzy-SARIMA-EE-LSTM model.
    """
    try:
        # Ensure save directories exist
        model_dir = os.path.dirname(MODEL_PATH)
        scaler_dir = os.path.dirname(SCALER_PATH)
        for directory in [model_dir, scaler_dir]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"[✔️] Created directory {directory}")

        # Load or fetch data
        if os.path.exists(DATA_CACHE_FILE) and not args.force_refresh:
            data = np.load(DATA_CACHE_FILE)
            logging.info(f"[✔️] Loaded cached data with shape: {data.shape}")
        else:
            data, _ = fetch_historical_data(
                start_offset=args.start_offset,
                step=args.step,
                duration=args.duration
            )
            logging.info(f"[✔️] Fetched data with shape: {data.shape}")
            np.save(DATA_CACHE_FILE, data)

        logging.info(f"[DEBUG] Training model, saving to {MODEL_PATH}...")
        start_time = time.time()

        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        train_loader, val_loader, test_loader = preprocess_for_training(
            data=data,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE
        )
        
        # Initialize model
        model = FuzzySARIMAEE(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            sarima_configs=SARIMA_CONFIGS
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.lstm.parameters(), lr=LEARNING_RATE)
        
        # Train model
        logging.info("Starting training...")
        train_losses, val_losses, feature_metrics_history = train_model(
            model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS
        )
        
        # Evaluate on test set
        logging.info("\n=== Test Set Metrics ===")
        model.lstm.eval()
        all_test_preds = []
        all_test_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                combined_pred, lstm_pred, sarima_preds = model.forward(batch_x)
                all_test_preds.append(combined_pred.cpu().numpy())
                all_test_targets.append(batch_y.cpu().numpy())
        
        all_test_preds = np.concatenate(all_test_preds, axis=0)
        all_test_targets = np.concatenate(all_test_targets, axis=0)
        
        # Calculate overall metrics
        overall_metrics = calculate_feature_metrics(
            all_test_targets.flatten(),
            all_test_preds.flatten(),
            "overall"
        )
        
        logging.info("\nOverall Test Metrics:")
        logging.info(f"MSE: {overall_metrics['mse']:.4f}")
        logging.info(f"RMSE: {overall_metrics['rmse']:.4f}")
        logging.info(f"MAE: {overall_metrics['mae']:.4f}")
        logging.info(f"MAPE: {overall_metrics['mape']:.2f}%")
        logging.info(f"R²: {overall_metrics['r2']:.4f}")
        logging.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
        
        # Calculate and log per-feature test metrics
        logging.info("\nPer-Feature Test Metrics:")
        for i, feature in enumerate(FEATURES):
            feature_metrics = calculate_feature_metrics(
                all_test_targets[:, i],
                all_test_preds[:, i],
                feature
            )
            logging.info(f"\n{feature}:")
            logging.info(f"  Accuracy: {feature_metrics['accuracy']:.4f}")
            logging.info(f"  MSE: {feature_metrics['mse']:.4f}")
            logging.info(f"  RMSE: {feature_metrics['rmse']:.4f}")
            logging.info(f"  MAE: {feature_metrics['mae']:.4f}")
            logging.info(f"  MAPE: {feature_metrics['mape']:.2f}%")
            logging.info(f"  R²: {feature_metrics['r2']:.4f}")
        
        training_time = time.time() - start_time
        logging.info(f"\n[✔️] Training completed in {training_time:.2f} seconds")

    except Exception as e:
        logging.error(f"[❌] Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(description="Train Fuzzy-SARIMA-EE-LSTM model.")
    parser.add_argument("--start_offset", type=str, default="10m", help="Start offset for data fetch")
    parser.add_argument("--step", type=str, default="1s", help="Step size for data fetch")
    parser.add_argument("--duration", type=str, default="8m", help="Duration for data fetch")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--force_refresh", action="store_true", help="Force refetch data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use for training (cuda/cpu)")
    args = parser.parse_args()

    run_training(args)