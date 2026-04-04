import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score, classification_report
import logging

# ---------------------------------------------------------
# 0. Setup & Configuration (Production Standards)
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    'window_size': 50,
    'batch_size': 64,
    'hidden_dim': 16, # Kept small for Edge/RAM constraints
    'learning_rate': 0.001,
    'epochs': 100,
    'features': ['voltage', 'latency', 'sensor_1', 'sensor_2']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Data Simulation & Preprocessing
# ---------------------------------------------------------
def generate_synthetic_can_data(num_samples=10000):
    """Simulates CAN bus telemetry with specific fault injections."""
    logger.info("Generating synthetic CAN bus data...")
    time = np.linspace(0, 100, num_samples)
    
    # Normal behavior
    df = pd.DataFrame({
        'voltage': 12.0 + np.sin(time) * 0.5 + np.random.normal(0, 0.1, num_samples),
        'latency': 5.0 + np.random.normal(0, 0.5, num_samples),
        'sensor_1': np.cos(time) + np.random.normal(0, 0.1, num_samples),
        'sensor_2': np.sin(time + np.pi/4) + np.random.normal(0, 0.1, num_samples),
        'label': 0
    })

    # Inject anomalies (Voltage spikes, latency jitters, dropouts)
    anomaly_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
    for idx in anomaly_indices:
        fault_type = np.random.choice(['spike', 'jitter', 'dropout'])
        if fault_type == 'spike':
            df.loc[idx:idx+5, 'voltage'] += np.random.uniform(3, 6) # Voltage spike
        elif fault_type == 'jitter':
            df.loc[idx:idx+10, 'latency'] += np.random.uniform(10, 20) # Latency jitter
        elif fault_type == 'dropout':
            df.loc[idx:idx+3, ['sensor_1', 'sensor_2']] = 0 # Signal dropout
        
        df.loc[idx:idx+10, 'label'] = 1 # Mark region as anomalous

    return df

class VehicleSensorDataset(Dataset):
    """Sliding window dataset for time-series models."""
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        return torch.tensor(window, dtype=torch.float32)

# ---------------------------------------------------------
# 2. Edge-Optimized LSTM Autoencoder
# ---------------------------------------------------------
class EdgeLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EdgeLSTMAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden state to reconstruct the sequence
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder(hidden_repeated)
        return reconstructed

    def quantize_for_edge(self):
        """Applies dynamic quantization to meet strict MB RAM/Flash budgets."""
        logger.info("Applying dynamic quantization for Edge NPU/CPU deployment...")
        quantized_model = torch.quantization.quantize_dynamic(
            self, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        return quantized_model

# ---------------------------------------------------------
# 3. Training & Inference Pipeline
# ---------------------------------------------------------
class PredictiveHealthMonitor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.lstm_ae = EdgeLSTMAutoencoder(len(config['features']), config['hidden_dim']).to(device)
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.lstm_threshold = 0.0

    def prepare_data(self, df):
        scaled_data = self.scaler.fit_transform(df[self.config['features']])
        dataset = VehicleSensorDataset(scaled_data, self.config['window_size'])
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True), scaled_data

    def train_lstm(self, dataloader):
        logger.info("Training LSTM Autoencoder...")
        optimizer = torch.optim.Adam(self.lstm_ae.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        
        self.lstm_ae.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed = self.lstm_ae(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - Loss: {total_loss/len(dataloader):.4f}")

    def train_isolation_forest(self, scaled_data):
        logger.info("Training Isolation Forest on raw sensor points...")
        self.iso_forest.fit(scaled_data)

    def determine_threshold(self, dataloader):
        self.lstm_ae.eval()
        errors = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                reconstructed = self.lstm_ae(batch)
                mse = torch.mean((batch - reconstructed) ** 2, dim=[1, 2])
                errors.extend(mse.cpu().numpy())
        # Set threshold at the 95th percentile of training reconstruction errors
        self.lstm_threshold = np.percentile(errors, 95)
        logger.info(f"LSTM Anomaly Threshold set to: {self.lstm_threshold:.4f}")

    def correlate_signals(self, df_test, lstm_errors, if_preds):
        """
        Signal Correlation Logic (As stated in your CV).
        Associates anomalies across modalities to find the root cause.
        """
        logger.info("Running signal correlation logic to identify root causes...")
        diagnostics = []
        
        # Pad LSTM errors to match dataframe length due to sliding window
        pad_length = self.config['window_size']
        lstm_errors_padded = np.pad(lstm_errors, (pad_length, 0), mode='constant', constant_values=0)
        
        for i in range(len(df_test)):
            is_lstm_anomaly = lstm_errors_padded[i] > self.lstm_threshold
            is_if_anomaly = if_preds[i] == -1
            
            if is_lstm_anomaly or is_if_anomaly:
                voltage = df_test.iloc[i]['voltage']
                latency = df_test.iloc[i]['latency']
                
                # Rule-based correlation mimicking embedded diagnostic logic
                if voltage > 14.5 and latency > 10.0:
                    diagnostics.append("CRITICAL: ECU Power Surge causing Network Jitter")
                elif latency > 15.0:
                    diagnostics.append("WARNING: CAN Bus Network Congestion")
                elif voltage < 10.0:
                    diagnostics.append("WARNING: Battery Voltage Drop")
                else:
                    diagnostics.append("UNKNOWN: Generic Sensor Drift")
            else:
                diagnostics.append("HEALTHY")
                
        return diagnostics

    def evaluate(self, df_test):
        logger.info("Evaluating model performance...")
        _, scaled_test = self.prepare_data(df_test)
        
        # 1. Get IF Predictions (-1 is anomaly, 1 is normal -> map to 1 and 0)
        if_preds = self.iso_forest.predict(scaled_test)
        if_preds_mapped = np.where(if_preds == -1, 1, 0)

        # 2. Get LSTM Predictions
        self.lstm_ae.eval()
        lstm_errors = []
        test_dataset = VehicleSensorDataset(scaled_test, self.config['window_size'])
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                reconstructed = self.lstm_ae(batch)
                mse = torch.mean((batch - reconstructed) ** 2, dim=[1, 2])
                lstm_errors.extend(mse.cpu().numpy())
        
        lstm_preds = (np.array(lstm_errors) > self.lstm_threshold).astype(int)
        
        # Pad LSTM predictions to match df length
        pad_length = self.config['window_size']
        lstm_preds_padded = np.pad(lstm_preds, (pad_length, 0), mode='constant', constant_values=0)
        
        # Ensemble Logic: If either model detects it, flag it (High Recall strategy for safety)
        ensemble_preds = np.logical_or(lstm_preds_padded, if_preds_mapped).astype(int)
        
        # Calculate Metrics
        labels = df_test['label'].values
        f1 = f1_score(labels, ensemble_preds)
        logger.info(f"F1 Score Achieved: {f1:.2f}")
        logger.info("\n" + classification_report(labels, ensemble_preds))
        
        # Run correlation logic
        diagnostics = self.correlate_signals(df_test, lstm_errors, if_preds)
        df_test['System_Diagnostic'] = diagnostics
        return df_test

# ---------------------------------------------------------
# 4. Execution 
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate Data
    df_train = generate_synthetic_can_data(10000)
    df_test = generate_synthetic_can_data(3000) # Contains novel anomalies

    # 2. Initialize Monitor
    monitor = PredictiveHealthMonitor(CONFIG)
    
    # 3. Train Pipeline
    dataloader, scaled_train = monitor.prepare_data(df_train)
    monitor.train_lstm(dataloader)
    monitor.train_isolation_forest(scaled_train)
    monitor.determine_threshold(dataloader)
    
    # 4. Evaluate & Correlate
    results_df = monitor.evaluate(df_test)
    
    # 5. Optimize for Edge (Fulfills JD requirement)
    edge_ready_model = monitor.lstm_ae.quantize_for_edge()
    logger.info(f"Edge Model Ready: {type(edge_ready_model)}")
    
    # Show a snippet of detected anomalies
    anomalies = results_df[results_df['System_Diagnostic'] != "HEALTHY"]
    logger.info(f"Sample of correlated anomalies detected:\n{anomalies[['voltage', 'latency', 'System_Diagnostic']].head()}")