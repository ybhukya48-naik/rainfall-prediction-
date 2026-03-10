import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path):
    """Load dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df, is_training=True):
    """Clean and preprocess weather data with advanced feature engineering."""
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Advanced Feature Engineering
    # 1. Temperature-Humidity Index (THI)
    df['THI'] = 0.8 * df['Temperature'] + (df['Humidity'] / 100) * (df['Temperature'] - 14.4) + 46.4
    
    # 2. Dew Point Approximation (Simple formula)
    df['DewPoint'] = df['Temperature'] - ((100 - df['Humidity']) / 5)
    
    # 3. Wind Chill (Simplified)
    df['WindChill'] = 13.12 + 0.6215 * df['Temperature'] - 11.37 * (df['WindSpeed']**0.16) + 0.3965 * df['Temperature'] * (df['WindSpeed']**0.16)

    features = ['Temperature', 'Humidity', 'Pressure', 'WindSpeed', 'THI', 'DewPoint', 'WindChill']
    
    if is_training:
        target = 'Rainfall'
        X = df[features]
        y = df[target]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler and feature names
        joblib.dump(scaler, 'models/advanced_scaler.pkl')
        joblib.dump(features, 'models/feature_names.pkl')
        
        return X_scaled, y, features
    else:
        X = df[features]
        scaler = joblib.load('models/advanced_scaler.pkl')
        X_scaled = scaler.transform(X)
        return X_scaled

def generate_advanced_dummy_data(file_path):
    """Generate a more complex weather dataset."""
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'Temperature': np.random.uniform(5, 45, n_samples),
        'Humidity': np.random.uniform(20, 100, n_samples),
        'Pressure': np.random.uniform(980, 1040, n_samples),
        'WindSpeed': np.random.uniform(0, 50, n_samples)
    }
    
    # Complex non-linear rainfall logic
    # Interaction between low pressure, high humidity, and moderate temperature
    prob = (
        0.4 * (data['Humidity'] / 100) + 
        0.3 * (1 - (data['Pressure'] - 980) / 60) + 
        0.2 * (1 - np.abs(data['Temperature'] - 20) / 25) +
        0.1 * (data['WindSpeed'] / 50)
    )
    
    data['Rainfall'] = (prob > 0.6).astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Advanced dummy dataset created at {file_path}")

if __name__ == "__main__":
    generate_advanced_dummy_data('data/advanced_weather_data.csv')
