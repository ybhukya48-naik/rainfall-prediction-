import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path):
    """Load dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean and preprocess weather data."""
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Feature selection (example columns)
    features = ['Temperature', 'Humidity', 'Pressure', 'WindSpeed']
    target = 'Rainfall'
    
    X = df[features]
    y = df[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for future predictions
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_scaled, y

def generate_dummy_data(file_path):
    """Generate a dummy weather dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Temperature': np.random.uniform(10, 40, n_samples),
        'Humidity': np.random.uniform(30, 95, n_samples),
        'Pressure': np.random.uniform(990, 1030, n_samples),
        'WindSpeed': np.random.uniform(0, 30, n_samples)
    }
    
    # Simple logic for rainfall: high humidity and low pressure increase probability
    data['Rainfall'] = (
        (data['Humidity'] > 70) & (data['Pressure'] < 1010) | 
        (data['Humidity'] > 85)
    ).astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Dummy dataset created at {file_path}")

if __name__ == "__main__":
    generate_dummy_data('data/weather_data.csv')
