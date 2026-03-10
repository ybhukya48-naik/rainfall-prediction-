import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_iot_satellite_data(file_path, n_samples=3000):
    """Generate high-precision IoT and Satellite/Radar feature data."""
    np.random.seed(42)
    
    # 1. IoT Sensor Data (Simulated high frequency)
    data = {
        'IoT_Temp': np.random.uniform(5, 45, n_samples),
        'IoT_Hum': np.random.uniform(20, 100, n_samples),
        'IoT_Press': np.random.uniform(980, 1040, n_samples),
        'IoT_Wind': np.random.uniform(0, 60, n_samples),
    }
    
    # 2. Satellite/Radar Features (Simulated image analysis metrics)
    # Brightness Temp (Satellite IR): Lower usually means higher clouds/rain
    data['Satellite_BrightnessTemp'] = np.random.uniform(200, 300, n_samples)
    # Cloud Reflectivity (Satellite VIS/IR)
    data['Satellite_Reflectivity'] = np.random.uniform(0, 1, n_samples)
    # Radar Echo Intensity (dBZ)
    data['Radar_EchoIntensity'] = np.random.uniform(0, 75, n_samples)
    
    # 3. Derived Rainfall logic (Ground Truth)
    # Strong correlation with Radar Echo and high humidity
    prob = (
        0.5 * (data['Radar_EchoIntensity'] / 75) + 
        0.2 * (data['IoT_Hum'] / 100) + 
        0.2 * (1 - (data['Satellite_BrightnessTemp'] - 200) / 100) +
        0.1 * (data['Satellite_Reflectivity'])
    )
    
    data['Rainfall'] = (prob > 0.65).astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"IoT & Satellite-enhanced dataset created at {file_path}")

def preprocess_iot_data(df, is_training=True):
    """Advanced preprocessing for combined IoT/Satellite data."""
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Feature Engineering
    # Satellite-IoT Consistency (Correlation check)
    df['Atmo_Instability'] = (df['IoT_Hum'] * (300 - df['Satellite_BrightnessTemp'])) / 1000
    
    features = [
        'IoT_Temp', 'IoT_Hum', 'IoT_Press', 'IoT_Wind',
        'Satellite_BrightnessTemp', 'Satellite_Reflectivity', 'Radar_EchoIntensity',
        'Atmo_Instability'
    ]
    
    if is_training:
        target = 'Rainfall'
        X = df[features]
        y = df[target]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        joblib.dump(scaler, 'models/iot_satellite_scaler.pkl')
        joblib.dump(features, 'models/iot_feature_names.pkl')
        
        return X_scaled, y, features
    else:
        X = df[features]
        scaler = joblib.load('models/iot_satellite_scaler.pkl')
        X_scaled = scaler.transform(X)
        return X_scaled

if __name__ == "__main__":
    generate_iot_satellite_data('data/iot_satellite_weather.csv')
