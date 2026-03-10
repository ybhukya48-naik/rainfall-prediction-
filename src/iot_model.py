import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import pandas as pd
import numpy as np
from iot_data_utils import preprocess_iot_data, generate_iot_satellite_data
import os

def train_iot_lgbm_model():
    """Train a LightGBM model for IoT and Satellite data."""
    data_path = 'data/iot_satellite_weather.csv'
    if not os.path.exists(data_path):
        generate_iot_satellite_data(data_path)
    
    # Load and preprocess
    df = pd.read_csv(data_path)
    X, y, features = preprocess_iot_data(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define LightGBM Classifier
    lgbm_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        random_state=42,
        metric='auc',
        verbosity=-1
    )
    
    print("Training LightGBM model for IoT/Satellite rainfall analysis...")
    lgbm_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = lgbm_model.predict(X_test)
    print("\nLightGBM (IoT/Satellite) Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(lgbm_model, 'models/iot_satellite_lgbm_model.pkl')
    print("LightGBM model saved to models/iot_satellite_lgbm_model.pkl")

if __name__ == "__main__":
    train_iot_lgbm_model()
