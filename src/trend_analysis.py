import pandas as pd
from prophet import Prophet
import os
import joblib

def train_prophet_model(df, target_col='Rainfall'):
    """Train a Prophet model to capture seasonal trends in rainfall."""
    # Prophet requires 'ds' (date) and 'y' (target) columns
    # Create mock timestamps if not present
    if 'Timestamp' not in df.columns:
        df['ds'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    else:
        df['ds'] = pd.to_datetime(df['Timestamp'])
        
    df['y'] = df[target_col]
    
    # Train Prophet model
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(df[['ds', 'y']])
    
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(m, 'models/prophet_model.pkl')
    print("Prophet model saved to models/prophet_model.pkl")
    return m

def get_prophet_forecast(m, periods=24):
    """Generate future forecast using Prophet."""
    future = m.make_future_dataframe(periods=periods, freq='H')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
