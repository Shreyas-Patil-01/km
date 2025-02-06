import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
import os
from sklearn.metrics import mean_squared_error
import joblib
import tensorflow as tf
# Disable GPU if not available
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        # Adjust contamination for fewer anomalies
        self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.01)  # Decreased contamination
        self.lstm_model = None
        self.sequence_length = 10  # Number of timesteps to look back
        self.n_features = None     # Will be determined from data
    def build_lstm_model(self):
        """Build LSTM model with dynamic input shape"""
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(64, activation='relu', return_sequences=False),
            Dense(self.n_features)  # Predict all features
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())  # Use actual function
        return model
    def preprocess_data(self, filepath):
        """Load and preprocess data with proper handling"""
        # Load JSON data
        with open(filepath, 'r') as file:
            raw_data = []
            for line in file:
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
                    continue
        # Convert to DataFrame
        records = []
        for entry in raw_data:
            timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
            ist_timestamp = timestamp.astimezone(timezone(timedelta(hours=5, minutes=30)))  # Convert to IST
            record = {'timestamp': ist_timestamp, 'metric_name': entry['metric_name'], 'value': entry['value']}
            record.update(entry['attributes'])  # Include attributes (e.g., CPU state, device)
            records.append(record)
        df = pd.DataFrame(records)
        df.fillna(0, inplace=True)
        # Pivot data to have metric columns
        pivot_df = df.pivot_table(index='timestamp', columns='metric_name', values='value', aggfunc='mean')
        pivot_df = pivot_df.fillna(0)  # Replace NaNs with 0 if some metrics are missing at a timestamp
        return pivot_df
    def create_sequences(self, data):
        """Create time-series sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    def train_model(self, data):
        """Train models with proper shape handling"""
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        self.n_features = scaled_data.shape[1]
        # Build and train LSTM
        self.lstm_model = self.build_lstm_model()
        X, y = self.create_sequences(scaled_data)
        # Add validation split
        split = int(0.8 * len(X))
        self.lstm_model.fit(
            X[:split], y[:split],
            epochs=10,
            batch_size=32,
            validation_data=(X[split:], y[split:]),
            verbose=1
        )
    def detect_anomalies(self, data):
        """Detect anomalies using Isolation Forest"""
        scaled_data = self.scaler.transform(data)
        anomalies = self.isolation_forest.fit_predict(scaled_data)
        return anomalies
    def generate_root_cause_analysis(self, anomalies, data, correlation_matrix):
        """Analyze anomalies and suggest potential solutions"""
        anomaly_indices = np.where(anomalies == -1)[0]
        root_cause_results = pd.DataFrame()
        for idx in anomaly_indices:
            row = data.iloc[idx]
            solutions = self.suggest_solution(row, correlation_matrix)
            new_row = pd.DataFrame([{
                'timestamp': row.name,
                'anomaly': 'Yes',
                'suggested_solution': solutions
            }])
            root_cause_results = pd.concat([root_cause_results, new_row], ignore_index=True)
        return root_cause_results
    def suggest_solution(self, row, correlation_matrix):
        """Analyze the pattern and suggest solutions"""
        solutions = []
        if correlation_matrix.get('system.cpu.time', {}).get('system.memory.usage', 0) > 0.5 and row.get('system.memory.usage', 0) > 75:
            solutions.append('High memory usage detected, consider optimizing processes or checking for memory leaks.')
        if row.get('system.cpu.time', 0) > 80:
            solutions.append('High CPU usage detected, consider optimizing processes.')
        if row.get('system.disk.io', 0) > 1000000:
            solutions.append('High disk IO detected, check for bottlenecks.')
        if row.get('system.network.bytes_sent', 0) > 1000000:
            solutions.append('High network traffic detected, check for unnecessary traffic or processes consuming bandwidth.')
        if not solutions:
            solutions.append("Investigate further based on correlations.")
        return solutions
    def visualize_anomalies(self, data, anomalies):
        """Visualize anomalies"""
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['system.cpu.time'], label='CPU Time')
        plt.scatter(data.index[anomalies == -1], data['system.cpu.time'][anomalies == -1], color='red', label='Anomalies')
        plt.title('Detected Anomalies in CPU Time')
        plt.legend()
        plt.savefig('anomaly_visualization.png')
        plt.close()
    def calculate_rmse(self, true_values, predicted_values):
        """Calculate RMSE to assess model performance"""
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        return rmse
    def save_models(self):
        """Save the Isolation Forest and LSTM models"""
        # Save Isolation Forest model
        joblib.dump(self.isolation_forest, 'isolation_forest_model.pkl')
        print("Isolation Forest model saved to 'isolation_forest_model.pkl'")
        # Save LSTM model
        self.lstm_model.save('lstm_model.keras')  # Use .keras format for better compatibility
        print("LSTM model saved to 'lstm_model.keras'")
        # Save scaler and expected columns
        joblib.dump(self.scaler, 'scaler.pkl')
        with open('expected_columns.json', 'w') as f:
            json.dump(list(self.scaler.feature_names_in_), f)
        print("Scaler and expected columns saved.")
    def run(self, filepath):
        print("Loading and preprocessing data...")
        data = self.preprocess_data(filepath)
        print("Training model...")
        self.train_model(data)
        print("Detecting anomalies...")
        anomalies = self.detect_anomalies(data)
        print(f"Found {np.sum(anomalies == -1)} anomalies. Visualizing...")
        self.visualize_anomalies(data, anomalies)
        print("Calculating RMSE...")
        predictions = self.lstm_model.predict(data.values.reshape((data.shape[0], 1, data.shape[1])))  # Adjust shape for LSTM input
        rmse = self.calculate_rmse(data.values, predictions)
        print(f"RMSE of the model: {rmse}")
        print("Analyzing anomalies for root cause...")
        correlation_matrix = data.corr()  # Correlation matrix for all features
        root_cause_results = self.generate_root_cause_analysis(anomalies, data, correlation_matrix)
        print("Root Cause Analysis Results:\n", root_cause_results)
        # Save results to CSV
        root_cause_results.to_csv('root_cause_analysis.csv', index=False)
        print("Results saved to 'root_cause_analysis.csv'.")
        # Save models
        self.save_models()
if __name__ == '__main__':
    detector = AnomalyDetector()
    detector.run('metrics_data.json')