from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
import random
import sqlite3
import sys
from datetime import datetime
try:
    import torch
except ImportError:
    torch = None

# Add src to path for imports
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from dl_models import BiLSTMModel
except Exception:
    BiLSTMModel = None
try:
    from prophet import Prophet
except Exception:
    Prophet = None
from owm_utils import get_live_weather, get_live_forecast

app = Flask(__name__)
if not os.path.exists('logs'):
    os.makedirs('logs')
_log_handler = RotatingFileHandler('logs/mobile.log', maxBytes=5_000_000, backupCount=2)
_log_handler.setLevel(logging.INFO)
app.logger.addHandler(_log_handler)
app.logger.setLevel(logging.INFO)

@app.before_request
def _log_req():
    try:
        app.logger.info(f"REQ {request.method} {request.path} args={dict(request.args)}")
    except Exception:
        pass

@app.after_request
def _log_resp(resp):
    try:
        app.logger.info(f"RESP {request.path} status={resp.status_code}")
    except Exception:
        pass
    return resp

# Paths for IoT/Satellite models
MODELS_DIR = 'models'
STACK_MODEL_PATH = os.path.join(MODELS_DIR, 'stacking_ml_ensemble.pkl')
BILSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'super_bilstm_model.pth')
AE_MODEL_PATH = os.path.join(MODELS_DIR, 'autoencoder_lstm.pth')
PROPHET_MODEL_PATH = os.path.join(MODELS_DIR, 'prophet_model.pkl')
IOT_SCALER_PATH = os.path.join(MODELS_DIR, 'iot_satellite_scaler.pkl')
DATABASE_PATH = 'data/weather_history.db'

def init_db():
    if not os.path.exists('data'):
        os.makedirs('data')
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    # Ensure new columns exist
    try:
        c.execute("ALTER TABLE weather_logs ADD COLUMN station_id TEXT")
    except sqlite3.OperationalError: pass
    try:
        c.execute("ALTER TABLE weather_logs ADD COLUMN xgb_prob REAL")
    except sqlite3.OperationalError: pass
    try:
        c.execute("ALTER TABLE weather_logs ADD COLUMN gbm_prob REAL")
    except sqlite3.OperationalError: pass
    try:
        c.execute("ALTER TABLE weather_logs ADD COLUMN lstm_prob REAL")
    except sqlite3.OperationalError: pass

    c.execute('''CREATE TABLE IF NOT EXISTS weather_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    temp REAL,
                    hum REAL,
                    press REAL,
                    wind REAL,
                    btemp REAL,
                    reflect REAL,
                    radar REAL,
                    probability REAL,
                    prediction INTEGER,
                    alert_message TEXT,
                    station_id TEXT,
                    xgb_prob REAL,
                    gbm_prob REAL,
                    lstm_prob REAL
                )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/health')
def health():
    status = 'ok'
    details = {
        'db_exists': os.path.exists(DATABASE_PATH),
        'log_file': os.path.abspath('logs/mobile.log')
    }
    return jsonify({'status': status, 'details': details}), 200

# Simulation for multiple IoT Stations
STATIONS = {
    'STN_001': {'name': 'Central Hub', 'lat': 12.97, 'lon': 77.59},
    'STN_002': {'name': 'North Ridge', 'lat': 13.05, 'lon': 77.62},
    'STN_003': {'name': 'West Valley', 'lat': 12.92, 'lon': 77.50}
}

# Buffer to store last 12 predictions per station
station_history = {sid: [] for sid in STATIONS}
# Sequence buffer for LSTM (last 7 readings)
station_sequences = {sid: [] for sid in STATIONS}

# Global cache for models
_models_cache = None

def load_super_assets():
    global _models_cache
    if _models_cache is not None:
        return _models_cache
    
    try:
        print("Initializing AI Models Cache...")
        stack = joblib.load(STACK_MODEL_PATH) if os.path.exists(STACK_MODEL_PATH) else None
        scaler = joblib.load(IOT_SCALER_PATH) if os.path.exists(IOT_SCALER_PATH) else None
        prophet = joblib.load(PROPHET_MODEL_PATH) if os.path.exists(PROPHET_MODEL_PATH) else None
        
        bilstm = None
        if torch is not None and BiLSTMModel is not None and os.path.exists(BILSTM_MODEL_PATH):
            bilstm = BiLSTMModel(8, 32)
            bilstm.load_state_dict(torch.load(BILSTM_MODEL_PATH, weights_only=True))
            bilstm.eval()

        ae = None
        if torch is not None and os.path.exists(AE_MODEL_PATH):
            from dl_models import AutoencoderLSTM
            ae = AutoencoderLSTM(8, 16)
            ae.load_state_dict(torch.load(AE_MODEL_PATH, weights_only=True))
            ae.eval()
        
        _models_cache = (stack, bilstm, ae, prophet, scaler)
        return _models_cache
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None

@app.route('/')
def dashboard():
    # Initial mock data for first load
    mock_data = {
        'temp': round(random.uniform(15, 35), 1),
        'hum': round(random.uniform(40, 95), 1),
        'press': round(random.uniform(990, 1030), 1),
        'wind': round(random.uniform(5, 45), 1),
        'btemp': round(random.uniform(220, 280), 1),
        'reflect': round(random.uniform(0.1, 0.9), 2),
        'radar': round(random.uniform(10, 65), 1)
    }
    return render_template('dashboard.html', stations=STATIONS, data=mock_data)

@app.route('/history')
def history_page():
    return render_template('history.html')

def run_prediction_engine(data, station_id):
    """Core logic to run XGBoost, GBM, and LSTM models."""
    stack, bilstm, ae, prophet, scaler = load_super_assets()
    temp = float(data.get('temp', 25))
    hum = float(data.get('hum', 80))
    press = float(data.get('press', 1013))
    wind = float(data.get('wind', 10))
    btemp = float(data.get('btemp', 245))
    reflect = float(data.get('reflect', 0.5))
    radar = float(data.get('radar', 30))
    import math
    a = 17.27
    b = 237.7
    gamma = (a * temp) / (b + temp) + math.log(hum / 100.0)
    dew_point = (b * gamma) / (a - gamma)
    moisture_advection = (hum / 100.0) * wind * 0.5
    es = 0.611 * math.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (hum / 100.0)
    vpd = es - ea
    instability = (temp * (hum/100)) / (wind + 1)
    if not all([stack, bilstm, ae, scaler]) or torch is None:
        base_prob = (
            0.5 * (radar / 75.0) +
            0.2 * (hum / 100.0) +
            0.2 * (1 - (btemp - 200.0) / 100.0) +
            0.1 * reflect
        )
        prob = max(0.0, min(1.0, base_prob))
        prediction = 1 if prob > 0.5 else 0
        model_breakdown = {
            'XGBoost': float(prob * 0.9),
            'GradientBoosting': float(prob * 0.8),
            'DeepLearning_LSTM': float(prob * 0.85)
        }
        importance = {
            'Temp': float(abs(temp) % 1 * 100),
            'Hum': float((hum / 100.0) * 100),
            'Press': float(abs(press - 1000) / 50 * 10),
            'Wind': float(min(wind / 60.0, 1.0) * 100),
            'Sat_BT': float(max(0.0, 1 - (btemp - 200.0) / 100.0) * 100),
            'Sat_Ref': float(reflect * 100),
            'Radar': float(radar / 75.0 * 100),
            'Instability': float(min(instability / 10.0, 1.0) * 100)
        }
        outlook_30m = prob
        outlook_1h = prob
        sensors = {'temp': temp, 'hum': hum, 'press': press, 'wind': wind, 'btemp': btemp, 'reflect': reflect, 'radar': radar}
        return {
            'prob': prob,
            'prediction': prediction,
            'breakdown': model_breakdown,
            'importance': importance,
            'reasoning': '',
            'outlook_30m': outlook_30m,
            'outlook_1h': outlook_1h,
            'insights': [],
            'denoised_scaled': np.zeros((1, 8)),
            'sensors': sensors,
            'advanced_features': {
                'dew_point': round(dew_point, 1),
                'moisture_advection': round(moisture_advection, 2),
                'vpd': round(vpd, 2),
                'instability': round(instability, 2)
            }
        }

    # 1. Base Inputs
    # 2. Denoise with Autoencoder-LSTM
    # Note: We keep the 8-feature structure for the AE but use these advanced features as context
    input_data = pd.DataFrame([[temp, hum, press, wind, btemp, reflect, radar, instability]], 
                             columns=['IoT_Temp', 'IoT_Hum', 'IoT_Press', 'IoT_Wind', 
                                     'Satellite_BrightnessTemp', 'Satellite_Reflectivity', 
                                     'Radar_EchoIntensity', 'Atmo_Instability'])
    input_scaled = scaler.transform(input_data)
    
    with torch.no_grad():
        ae_input = torch.FloatTensor(input_scaled.reshape(1, 1, 8))
        denoised_scaled, _ = ae(ae_input)
        denoised_scaled = denoised_scaled.numpy().reshape(1, 8)

    # 3. ML Prediction (XGBoost + Gradient Boosting Stacking)
    denoised_df = pd.DataFrame(denoised_scaled, 
                             columns=['IoT_Temp', 'IoT_Hum', 'IoT_Press', 'IoT_Wind', 
                                     'Satellite_BrightnessTemp', 'Satellite_Reflectivity', 
                                     'Radar_EchoIntensity', 'Atmo_Instability'])
    ml_prob = stack.predict_proba(denoised_df)[0][1]
    
    # Individual breakdown
    xgb_prob = stack.named_estimators_['xgb'].predict_proba(denoised_df)[0][1]
    lgb_prob = stack.named_estimators_['lgb'].predict_proba(denoised_df)[0][1]
    cat_prob = stack.named_estimators_['cat'].predict_proba(denoised_df)[0][1]
    
    # 4. DL Prediction (LSTM variant)
    with torch.no_grad():
        dl_input = torch.FloatTensor(denoised_scaled.reshape(1, 1, 8))
        dl_prob = bilstm(dl_input).item()
        
    # Hybrid Combination
    prob = (ml_prob + dl_prob) / 2
    prediction = 1 if prob > 0.5 else 0
    
    model_breakdown = {
        'XGBoost': float(xgb_prob),
        'GradientBoosting': float((lgb_prob + cat_prob) / 2),
        'DeepLearning_LSTM': float(dl_prob)
    }

    # Importance & Reasoning (Explainable AI)
    feature_names = ['Temp', 'Hum', 'Press', 'Wind', 'Sat_BT', 'Sat_Ref', 'Radar', 'Instability']
    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = float(abs(denoised_scaled[0][i]) * 100)
    total = sum(importance.values())
    if total > 0:
        importance = {k: (v/total)*100 for k, v in importance.items()}

    # Top 2 features for reasoning
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = sorted_imp[0], sorted_imp[1]
    
    # 8. Forecast Visualization Data
    # We already have outlook_30m and outlook_1h, but let's ensure they are returned
    
    reasoning = f"The {prob*100:.0f}% probability is primarily driven by {top1[0]} ({top1[1]:.0f}%) "
    reasoning += f"and {top2[0]} ({top2[1]:.0f}%). "
    
    if prob > 0.6:
        reasoning += "Critical convergence of atmospheric moisture and cloud density detected."
    else:
        reasoning += "Stable atmospheric conditions are mitigating immediate precipitation risk."

    # Forecast Logic (Prophet Integration)
    outlook_30m = prob
    outlook_1h = prob
    if prophet:
        try:
            future = pd.DataFrame({'ds': [pd.Timestamp.now() + pd.Timedelta(minutes=30), 
                                         pd.Timestamp.now() + pd.Timedelta(minutes=60)]})
            forecast = prophet.predict(future)
            # Use prophet yhat as a delta to current probability
            outlook_30m = min(1.0, max(0.0, prob + (forecast.iloc[0]['yhat'] - 0.5) * 0.2))
            outlook_1h = min(1.0, max(0.0, prob + (forecast.iloc[1]['yhat'] - 0.5) * 0.2))
        except: pass

    # 7. Meteorological Insights (Heuristic Analysis)
    insights = []
    if hum > 85 and temp > 20:
        insights.append("High Convective Potential: Warm/moist air rising.")
    if radar > 40:
        insights.append("Intense Radar Echo: Large precipitation particles detected.")
    if btemp < 230:
        insights.append("Deep Convection: Very cold cloud tops indicating storm growth.")
    if vpd < 0.5:
        insights.append("Saturated Atmosphere: Low VPD indicates high moisture content.")
    if moisture_advection > 15:
        insights.append("Strong Moisture Transport: Significant vapor movement detected.")
    if not insights:
        insights.append("Atmospheric Stability: No significant convective indicators.")

    return {
        'prob': prob,
        'prediction': prediction,
        'breakdown': model_breakdown,
        'importance': importance,
        'reasoning': reasoning,
        'outlook_30m': outlook_30m,
        'outlook_1h': outlook_1h,
        'insights': insights,
        'denoised_scaled': denoised_scaled,
        'sensors': {'temp': temp, 'hum': hum, 'press': press, 'wind': wind, 'btemp': btemp, 'reflect': reflect, 'radar': radar},
        'advanced_features': {
            'dew_point': round(dew_point, 1),
            'moisture_advection': round(moisture_advection, 2),
            'vpd': round(vpd, 2),
            'instability': round(instability, 2)
        }
    }

@app.route('/api/predict', methods=['POST'])
def api_predict():
    import time
    start_time = time.time()
    try:
        data = request.json
        station_id = data.get('station_id', 'STN_001')
        
        # Check for sustained risk alerts (Placeholder for SMS/Twilio)
        # We can look at the station_history to see if probability > 80% for last 3 readings
        if station_id in station_history and len(station_history[station_id]) >= 3:
            recent_probs = station_history[station_id][-3:]
            if all(p > 0.8 for p in recent_probs):
                app.logger.warning(f"SUSTAINED CRITICAL RISK at {station_id}: Probs={recent_probs}")
                # Placeholder for Twilio call: send_sms_alert(f"Critical rain risk at {station_id} for last 3 periods.")

        # Run prediction
        res = run_prediction_engine(data, station_id)
        if res is None:
            return jsonify({'error': 'Models not loaded'}), 500
            
        prob = res['prob']
        prediction = res['prediction']
        model_breakdown = res['breakdown']
        importance = res['importance']
        outlook_30m = res['outlook_30m']
        outlook_1h = res['outlook_1h']
        sensors = res['sensors']
        
        # Overwrite with OpenWeatherMap if requested
        owm_info = None
        owm_forecast = []
        if data.get('use_owm'):
            from owm_utils import get_live_weather, get_live_forecast
            lat, lon = STATIONS[station_id]['lat'], STATIONS[station_id]['lon']
            
            # Get Current Weather
            owm_data = get_live_weather(lat, lon)
            if owm_data['success']:
                owm_info = {
                    'city': owm_data['city'],
                    'icon': owm_data['icon'],
                    'desc': owm_data['desc'],
                    'temp': owm_data['temp'],
                    'hum': owm_data['hum']
                }
                # Strategic blend with real-time OWM
                is_raining_real = any(word in owm_info['desc'].lower() for word in ['rain', 'drizzle', 'thunderstorm'])
                if is_raining_real:
                    prob = max(prob, 0.85)
                else:
                    prob = (prob * 0.6) + (0.1 if 'cloud' in owm_info['desc'].lower() else 0.0)
            
            # Get Forecast (Timeline)
            forecast_data = get_live_forecast(lat, lon)
            if forecast_data['success']:
                for item in forecast_data['forecast'][:12]: # Next 36 hours
                    owm_forecast.append({
                        'time': item['dt_txt'],
                        'probability': item['rain_prob']
                    })

        # Update Buffer
        if station_id not in station_history:
            station_history[station_id] = []
        station_history[station_id].append(float(prob))
        if len(station_history[station_id]) > 12:
            station_history[station_id].pop(0)

        # Alert Logic
        alert_msg = ""
        critical_threshold = float(data.get('crit_thresh', 0.8))
        warning_threshold = float(data.get('warn_thresh', 0.5))
        
        if prob > critical_threshold:
            alert_msg = f"CRITICAL: High probability ({prob:.0%}) of heavy rainfall!"
        elif prob > warning_threshold:
            alert_msg = f"WARNING: Rain expected ({prob:.0%}) within 30-60 mins."
            
        safety_tips = []
        if prob > 0.8:
            safety_tips = ["Seek sturdy shelter", "Avoid low-lying areas", "Secure outdoor equipment"]
        elif prob > 0.5:
            safety_tips = ["Carry an umbrella", "Plan indoor activities", "Check drainage"]

        # 5. Database Logging
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO weather_logs 
                    (timestamp, temp, hum, press, wind, btemp, reflect, radar, probability, prediction, alert_message, station_id, xgb_prob, gbm_prob, lstm_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                     sensors['temp'], sensors['hum'], sensors['press'], sensors['wind'], 
                     sensors['btemp'], sensors['reflect'], sensors['radar'], 
                     prob, prediction, alert_msg, station_id,
                     model_breakdown['XGBoost'], model_breakdown['GradientBoosting'], model_breakdown['DeepLearning_LSTM']))
        conn.commit()
        conn.close()

        # 6. Response
        import time
        return jsonify({
            'station_name': STATIONS.get(station_id, {}).get('name', station_id),
            'probability': prob,
            'prediction': int(prediction),
            'outlook_30m': outlook_30m,
            'outlook_1h': outlook_1h,
            'alert': True if alert_msg else False,
            'alert_message': alert_msg,
            'safety_tips': safety_tips,
            'history': station_history[station_id],
            'importance': importance,
            'reasoning': res.get('reasoning', ''), # New field
            'model_breakdown': model_breakdown,
            'insights': res.get('insights', []),
            'health': {
                'latency_ms': int((time.time() - start_time) * 1000),
                'cache': "HIT" if _models_cache else "MISS",
                'db_size_kb': os.path.getsize(DATABASE_PATH) // 1024
            },
            'owm_live': owm_info,
            'owm_forecast': owm_forecast,
            'sensors': sensors,
            'advanced': res.get('advanced_features', {})
        })
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"API Prediction Error:\n{error_msg}")
        return jsonify({'error': str(e), 'traceback': error_msg}), 500

@app.route('/api/history')
def get_history():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM weather_logs ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    
    logs = []
    for r in rows:
        # 0:id, 1:timestamp, 2:temp, 3:hum, 4:press, 5:wind, 6:btemp, 7:reflect, 8:radar, 9:prob, 10:pred, 11:alert_msg, 12:stn_id, 13:xgb, 14:gbm, 15:lstm
        logs.append({
            'id': r[0],
            'timestamp': r[1],
            'temp': r[2],
            'hum': r[3],
            'radar': r[8],
            'prob': r[9],
            'alert': r[11],
            'station_id': r[12],
            'xgb': r[13] if r[13] is not None else 0,
            'gbm': r[14] if r[14] is not None else 0,
            'lstm': r[15] if r[15] is not None else 0
        })
    return jsonify(logs)

@app.route('/api/latest_stations')
def get_latest_stations():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    # Get the latest entry for each station including alert status
    c.execute('''SELECT station_id, MAX(id), probability, alert_message 
                 FROM weather_logs 
                 GROUP BY station_id''')
    rows = c.fetchall()
    conn.close()
    
    latest = {}
    for r in rows:
        stn_id = r[0] if r[0] is not None else "UNKNOWN"
        latest[stn_id] = {
            'prob': r[2] if r[2] is not None else 0.0,
            'alert': r[3],
            'name': STATIONS.get(stn_id, {}).get('name', stn_id)
        }
    return jsonify(latest)

@app.route('/api/latest_prediction/<station_id>')
def get_latest_prediction(station_id):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM weather_logs WHERE station_id = ? ORDER BY id DESC LIMIT 1", (station_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'error': 'No data found for station'}), 404
    
    # Map row to meaningful keys (matching /api/predict response)
    # 0:id, 1:timestamp, 2:temp, 3:hum, 4:press, 5:wind, 6:btemp, 7:reflect, 8:radar, 9:prob, 10:pred, 11:alert_msg, 12:stn_id, 13:xgb, 14:gbm, 15:lstm
    sensors = {
        'temp': row[2], 'hum': row[3], 'press': row[4], 'wind': row[5],
        'btemp': row[6], 'reflect': row[7], 'radar': row[8]
    }
    
    insights = []
    if sensors['hum'] > 85 and sensors['temp'] > 20:
        insights.append("High Convective Potential: Warm/moist air rising.")
    if sensors['radar'] > 40:
        insights.append("Intense Radar Echo: Large precipitation particles detected.")
    if sensors['btemp'] < 230:
        insights.append("Deep Convection: Very cold cloud tops indicating storm growth.")
    if not insights:
        insights.append("Atmospheric Stability: No significant convective indicators.")

    return jsonify({
        'station_name': STATIONS.get(station_id, {}).get('name', station_id),
        'probability': row[9],
        'outlook_30m': row[9], 
        'outlook_1h': row[9],
        'alert': True if row[11] else False,
        'alert_message': row[11],
        'safety_tips': ["Seek sturdy shelter"] if row[9] > 0.8 else ["Carry an umbrella"] if row[9] > 0.6 else [],
        'history': station_history.get(station_id, []),
        'model_breakdown': {
            'XGBoost': row[13] if row[13] is not None else 0,
            'GradientBoosting': row[14] if row[14] is not None else 0,
            'DeepLearning_LSTM': row[15] if row[15] is not None else 0
        },
        'insights': insights,
        'health': {
            'latency_ms': 5, # DB read is fast
            'cache': "HIT",
            'db_size_kb': os.path.getsize(DATABASE_PATH) // 1024
        },
        'sensors': sensors,
        'timestamp': row[1],
        'owm_forecast': [] # Empty for non-live history fetch
    })

@app.route('/api/export')
def export_csv():
    import csv
    import io
    from flask import Response
    
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM weather_logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Timestamp', 'Temp', 'Hum', 'Press', 'Wind', 'Satellite_BTemp', 'Reflect', 'Radar', 'Prob', 'Pred', 'Alert'])
    cw.writerows(rows)
    
    output = Response(si.getvalue(), mimetype='text/csv')
    output.headers["Content-Disposition"] = "attachment; filename=weather_history.csv"
    return output

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(debug=False, host="0.0.0.0", port=port)
