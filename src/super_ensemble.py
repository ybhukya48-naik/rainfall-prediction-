from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import torch
import torch.nn as nn
from dl_models import AutoencoderLSTM, BiLSTMModel
from trend_analysis import train_prophet_model
from iot_data_utils import preprocess_iot_data, generate_iot_satellite_data
import os
import pandas as pd
import numpy as np

def train_super_stacking_ensemble():
    """Train a Super Stacking Ensemble: (XGB + LGBM + Cat + BiLSTM + Prophet)."""
    data_path = 'data/iot_satellite_weather.csv'
    if not os.path.exists(data_path):
        generate_iot_satellite_data(data_path)
    
    # Load and preprocess
    df = pd.read_csv(data_path)
    X, y, features = preprocess_iot_data(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base ML models
    estimators = [
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=-1)),
        ('cat', CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, verbose=0, random_state=42))
    ]
    
    # 1. Train Stacking Classifier (ML part)
    stacking_ml = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    print("Training ML Stacking (XGB, LightGBM, CatBoost)...")
    stacking_ml.fit(X_train, y_train)
    
    # 2. Train DL part (BiLSTM & Autoencoder)
    # Reshape for LSTM: (batch, seq_len, features)
    X_train_dl = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_dl = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Autoencoder for Noise Removal
    print("Training Autoencoder-LSTM for noise removal...")
    ae_model = AutoencoderLSTM(X_train.shape[1], hidden_dim=16)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.01)
    ae_criterion = nn.MSELoss()
    ae_model.train()
    for epoch in range(50):
        ae_optimizer.zero_grad()
        decoded, _ = ae_model(torch.FloatTensor(X_train_dl))
        loss = ae_criterion(decoded, torch.FloatTensor(X_train_dl))
        loss.backward()
        ae_optimizer.step()
    
    # BiLSTM for Prediction
    bilstm_model = BiLSTMModel(X_train.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print("Training BiLSTM DL model...")
    bilstm_model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = bilstm_model(torch.FloatTensor(X_train_dl))
        loss = criterion(outputs, torch.FloatTensor(y_train.values).view(-1, 1))
        loss.backward()
        optimizer.step()
    
    # 3. Prophet Model (Trend part)
    prophet_model = train_prophet_model(df)
    
    # 4. Save all components
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(stacking_ml, 'models/stacking_ml_ensemble.pkl')
    torch.save(ae_model.state_dict(), 'models/autoencoder_lstm.pth')
    torch.save(bilstm_model.state_dict(), 'models/super_bilstm_model.pth')
    
    # Evaluate Final Hybrid Prediction
    # Combine outputs (simplified averaging for the demo)
    stack_preds = stacking_ml.predict_proba(X_test)[:, 1]
    bilstm_model.eval()
    with torch.no_grad():
        bilstm_preds = bilstm_model(torch.FloatTensor(X_test_dl)).numpy().flatten()
    
    final_preds = (stack_preds + bilstm_preds) / 2
    final_labels = (final_preds > 0.5).astype(int)
    
    print("\nSuper Stacking Ensemble Evaluation:")
    print(classification_report(y_test, final_labels))
    print(f"Accuracy: {accuracy_score(y_test, final_labels):.4f}")
    
    print("Super Ensemble saved to models/ directory.")

if __name__ == "__main__":
    train_super_stacking_ensemble()
