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

    # 4. Vapor Pressure Deficit (VPD)
    # es = 0.6108 * exp(17.27 * T / (T + 237.3))
    es = 0.6108 * np.exp(17.27 * df['Temperature'] / (df['Temperature'] + 237.3))
    ea = es * (df['Humidity'] / 100)
    df['VPD'] = es - ea

    # 5. Instability Index (Simulated Lifted Index approximation)
    # Lower (more negative) means more unstable
    df['Instability_Index'] = (df['Temperature'] - df['DewPoint']) * -0.5 + (df['Humidity'] / 20) * -1.0

    # 6. Deltas (essential for accuracy)
    if 'Pressure_Change' not in df.columns:
        df['Pressure_Change'] = np.random.normal(0, 2, len(df))
    if 'Humidity_Change' not in df.columns:
        df['Humidity_Change'] = np.random.normal(0, 5, len(df))

    features = ['Temperature', 'Humidity', 'Pressure', 'WindSpeed', 'THI', 'DewPoint', 'WindChill', 'VPD', 'Instability_Index', 'Pressure_Change', 'Humidity_Change']
    
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
