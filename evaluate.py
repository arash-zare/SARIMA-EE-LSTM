import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SARIMA_EE_LSTM
from config import *
import joblib
import logging
import json
from fuzzy import FuzzyLogic
from matplotlib_utils import save_multiple_metrics_to_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def validate_config():
    """Validate configuration parameters"""
    try:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not FEATURES:
            raise ValueError("FEATURES list is empty")
        if len(FEATURES) != INPUT_DIM:
            raise ValueError(f"INPUT_DIM ({INPUT_DIM}) does not match number of FEATURES ({len(FEATURES)})")
        if not all(feature in THRESHOLDS for feature in FEATURES):
            raise ValueError("Not all FEATURES have corresponding THRESHOLDS")
        logging.info("Configuration validated successfully")
    except Exception as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        raise

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(DATASET_PATH)
        if not all(feature in df.columns for feature in FEATURES):
            raise ValueError("Not all FEATURES found in dataset columns")
        logging.info(f"Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def prepare_sequences(data, seq_len):
    """Prepare sequences for LSTM input"""
    try:
        if seq_len <= 0:
            raise ValueError("Sequence length must be positive")
        sequences = []
        targets = []
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
            targets.append(data[i+seq_len])
        sequences = np.array(sequences)
        targets = np.array(targets)
        if sequences.shape[0] == 0:
            raise ValueError("No sequences generated; dataset too small for given sequence length")
        logging.info(f"Prepared {sequences.shape[0]} sequences of length {seq_len}")
        return sequences, targets
    except Exception as e:
        logging.error(f"Error preparing sequences: {str(e)}")
        raise

def calculate_fuzzy_metrics(predictions, actuals, thresholds):
    """Calculate fuzzy logic based metrics"""
    try:
        fuzzy = FuzzyLogic()
        fuzzy_scores = []
        
        for pred, actual in zip(predictions, actuals):
            feature_scores = []
            for i, feature in enumerate(FEATURES):
                threshold = thresholds[feature]
                error = abs(pred[i] - actual[i])
                membership = fuzzy.calculate_membership(error, threshold)
                feature_scores.append(membership)
            fuzzy_scores.append(np.mean(feature_scores))
        
        fuzzy_accuracy = np.mean(fuzzy_scores)
        fuzzy_precision = np.mean([score for score in fuzzy_scores if score > 0.5] or [0])
        fuzzy_recall = np.mean([score for score in fuzzy_scores if score > 0.7] or [0])
        
        return {
            'Fuzzy_Accuracy': fuzzy_accuracy,
            'Fuzzy_Precision': fuzzy_precision,
            'Fuzzy_Recall': fuzzy_recall
        }
    except Exception as e:
        logging.error(f"Error calculating fuzzy metrics: {str(e)}")
        return {
            'Fuzzy_Accuracy': 0.0,
            'Fuzzy_Precision': 0.0,
            'Fuzzy_Recall': 0.0
        }

def evaluate_model(model, X_test, y_test, scaler, batch_size=32):
    """Evaluate the model and calculate metrics"""
    try:
        model.eval()
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                pred = model(x_batch)
                predictions.append(pred.cpu().numpy())
                actuals.append(y_batch.numpy())
        
        predictions = np.concatenate(predictions).squeeze()
        actuals = np.concatenate(actuals)
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions)
        actuals = scaler.inverse_transform(actuals)
        
        # Calculate traditional metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / actuals, where=actuals != 0)) * 100
        
        # Calculate fuzzy metrics
        fuzzy_metrics = calculate_fuzzy_metrics(predictions, actuals, THRESHOLDS)
        
        # Combine metrics
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            **fuzzy_metrics
        }
        
        # Prepare data for plotting
        metrics_dict = {}
        for i, feature in enumerate(FEATURES):
            metrics_dict[feature] = {
                'actual': actuals[:, i],
                'predicted': predictions[:, i],
                'mse': mean_squared_error(actuals[:, i], predictions[:, i]),
                'risk_score': fuzzy_metrics['Fuzzy_Accuracy']  # Using average fuzzy score
            }
        
        # Save plots
        save_multiple_metrics_to_files(metrics_dict, save_dir='plots')
        
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        raise

def main():
    """Main function to evaluate the SARIMA-EE-LSTM model"""
    try:
        # Validate configuration
        validate_config()
        
        # Load data
        df = load_and_preprocess_data()
        
        # Split data into features
        features = df[FEATURES].values
        
        # Load scaler
        scaler = joblib.load(SCALER_PATH)
        scaled_features = scaler.transform(features)
        
        # Prepare sequences
        X, y = prepare_sequences(scaled_features, SEQ_LEN)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        # Load model
        model = SARIMA_EE_LSTM(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_dim=INPUT_DIM
        ).to(DEVICE)
        
        model.load_state_dict(torch.load(MODEL_PATH))
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, scaler)
        
        # Save metrics
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved to metrics.json")
        
        # Log results
        logging.info("\nEvaluation Metrics:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()