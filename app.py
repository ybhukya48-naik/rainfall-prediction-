from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from owm_utils import get_live_weather, get_live_forecast
from src.dl_models import BiLSTMModel
import torch

app = Flask(__name__)

# Paths for models
MODELS_DIR = 'models'
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, 'advanced_hybrid_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'advanced_scaler.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')
TS_LSTM_PATH = os.path.join(MODELS_DIR, 'timeseries_lstm.pth')
TS_META_PATH = os.path.join(MODELS_DIR, 'timeseries_lstm_meta.pkl')

def load_advanced_assets():
    if all(os.path.exists(p) for p in [HYBRID_MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        return (
            joblib.load(HYBRID_MODEL_PATH),
            joblib.load(SCALER_PATH),
            joblib.load(FEATURES_PATH)
        )
    return None, None, None

def engineer_features(temp, hum, press, wind):
    thi = 0.8 * temp + (hum / 100) * (temp - 14.4) + 46.4
    dew_point = temp - ((100 - hum) / 5)
    wind_chill = 13.12 + 0.6215 * temp - 11.37 * (wind**0.16) + 0.3965 * temp * (wind**0.16)
    return [temp, hum, press, wind, thi, dew_point, wind_chill]

@app.route('/')
def index():
    return render_template('index.html', mode="Advanced Hybrid")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, scaler, feature_names = load_advanced_assets()
        if model is None:
            return render_template('index.html', 
                                 prediction_text="Error: Advanced model not trained. Run src/advanced_model.py first.")
        
        # Get basic input values
        temp = float(request.form['temperature'])
        hum = float(request.form['humidity'])
        press = float(request.form['pressure'])
        wind = float(request.form['wind_speed'])
        
        # Feature Engineering (must match training)
        thi = 0.8 * temp + (hum / 100) * (temp - 14.4) + 46.4
        dew_point = temp - ((100 - hum) / 5)
        wind_chill = 13.12 + 0.6215 * temp - 11.37 * (wind**0.16) + 0.3965 * temp * (wind**0.16)
        
        # Prepare input array
        input_data = np.array([[temp, hum, press, wind, thi, dew_point, wind_chill]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        result = "Rain Expected" if prediction == 1 else "No Rain"
        
        return render_template('index.html', 
                               prediction_text=f"Prediction: {result}", 
                               probability=f"Probability: {prob:.2%}",
                               mode="Advanced Hybrid (Stacking: XGB+LGBM+Cat)")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

@app.route('/api/live', methods=['GET'])
def api_live():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'success': False, 'error': 'lat and lon required'}), 400
    model, scaler, feature_names = load_advanced_assets()
    if model is None:
        return jsonify({'success': False, 'error': 'Advanced model not trained'}), 500
    w = get_live_weather(lat, lon)
    if not w.get('success'):
        return jsonify(w), 502
    feats = engineer_features(w['temp'], w['hum'], w['press'], w['wind'])
    input_scaled = scaler.transform(np.array([feats]))
    pred = int(model.predict(input_scaled)[0])
    prob = float(model.predict_proba(input_scaled)[0][1])
    return jsonify({'success': True, 'city': w.get('city'), 'prediction': pred, 'probability': prob})

@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'success': False, 'error': 'lat and lon required'}), 400
    model, scaler, feature_names = load_advanced_assets()
    if model is None:
        return jsonify({'success': False, 'error': 'Advanced model not trained'}), 500
    f = get_live_forecast(lat, lon)
    if not f.get('success'):
        return jsonify(f), 502
    entries = f['data']['list']
    results = []
    for item in entries:
        temp = item['main']['temp']
        hum = item['main']['humidity']
        press = item['main']['pressure']
        wind = item['wind']['speed']
        feats = engineer_features(temp, hum, press, wind)
        input_scaled = scaler.transform(np.array([feats]))
        prob = float(model.predict_proba(input_scaled)[0][1])
        pred = int(prob >= 0.5)
        results.append({'time': item['dt_txt'], 'prediction': pred, 'probability': prob})
    return jsonify({'success': True, 'results': results})

def load_ts_lstm():
    if os.path.exists(TS_LSTM_PATH) and os.path.exists(TS_META_PATH):
        meta = joblib.load(TS_META_PATH)
        model = BiLSTMModel(meta['input_dim'], hidden_dim=32)
        state = torch.load(TS_LSTM_PATH, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        return model, meta
    return None, None

@app.route('/api/forecast_ensemble', methods=['GET'])
def api_forecast_ensemble():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'success': False, 'error': 'lat and lon required'}), 400
    model, scaler, feature_names = load_advanced_assets()
    if model is None:
        return jsonify({'success': False, 'error': 'Advanced model not trained'}), 500
    lstm_model, meta = load_ts_lstm()
    f = get_live_forecast(lat, lon)
    if not f.get('success'):
        return jsonify(f), 502
    entries = f['data']['list']
    feats_list = []
    times = []
    for item in entries:
        temp = item['main']['temp']
        hum = item['main']['humidity']
        press = item['main']['pressure']
        wind = item['wind']['speed']
        feats = engineer_features(temp, hum, press, wind)
        feats_list.append(feats)
        times.append(item['dt_txt'])
    X_scaled = scaler.transform(np.array(feats_list))
    results = []
    for i in range(len(X_scaled)):
        ml_prob = float(model.predict_proba(X_scaled[i:i+1])[0][1])
        if lstm_model and meta:
            start = max(0, i - meta['seq_len'] + 1)
            end = i + 1
            seq = X_scaled[start:end]
            if len(seq) < meta['seq_len']:
                seq = np.pad(seq, ((meta['seq_len'] - len(seq), 0), (0, 0)), mode='edge')
            seq_tensor = torch.FloatTensor(seq.reshape(1, meta['seq_len'], meta['input_dim']))
            with torch.no_grad():
                dl_prob = float(lstm_model(seq_tensor).numpy().flatten()[0])
            prob = (ml_prob + dl_prob) / 2
        else:
            prob = ml_prob
        pred = int(prob >= 0.5)
        results.append({'time': times[i], 'prediction': pred, 'probability': prob})
    return jsonify({'success': True, 'results': results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
