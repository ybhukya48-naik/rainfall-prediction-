# RainAlert IoT: Advanced Rainfall Detection & Early Warning System

RainAlert IoT is a comprehensive, multi-layered rainfall detection system that integrates IoT sensor data, satellite imagery analysis, and radar echo intensity using a hybrid of advanced machine learning models.

## 🚀 Key Features

### 1. **Advanced Hybrid AI Engine**
- **LightGBM Analytics**: Optimized for high-speed IoT data streams and satellite features.
- **Stacking Ensemble**: Combines XGBoost, LightGBM, and CatBoost for superior prediction accuracy (97.25%).
- **Explainable AI (XAI)**: Provides real-time "Model Reasoning" to show exactly which sensors are driving each prediction.

### 2. **IoT & Satellite Integration**
- **Multi-Station Support**: Monitor multiple regional stations (Central, North, West) simultaneously.
- **Satellite Analysis**: Simulates Infrared (IR) Brightness Temperature and Visible (VIS) Reflectivity data.
- **Radar Echo**: Integrates simulated dBZ radar intensity for precise precipitation detection.

### 3. **Predictive Analytics**
- **Temporal Trends**: Real-time trend analysis using rolling history buffers.
- **Predictive Outlook**: Provides 30-minute and 1-hour "Early Warning" probability forecasts.

### 4. **Mobile & Web Interfaces**
- **Web Dashboard**: Responsive, dark-themed Flask interface with Chart.js and HTML5 Canvas.
- **Flutter Mobile App**: A dedicated mobile application built with Flutter for cross-platform (iOS/Android) support, providing real-time alerts and station monitoring.

### 5. **Data Persistence & Analysis**
- **SQLite Logging**: Automated persistent storage of all sensor readings and predictions.
- **History Module**: Dedicated historical analysis view with session-based statistics.
- **Data Export**: One-click CSV export for external meteorological research.

## 🛠️ Tech Stack
- **Backend**: Python, Flask, SQLite3
- **ML/AI**: LightGBM, XGBoost, CatBoost, Scikit-Learn
- **Frontend**: HTML5 (Canvas/SVG), CSS3 (Mobile-First), JavaScript (Chart.js)
- **Data**: Pandas, Numpy, Joblib

## 📦 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the AI Engine**:
   ```bash
   # Train the baseline model
   python src/model.py
   # Train the advanced hybrid stacking model
   python src/advanced_model.py
   # Train the IoT-optimized LightGBM engine
   python src/iot_model.py
   ```

3. **Launch the Mobile Dashboard**:
   ```bash
   python mobile_app.py
   ```
   Navigate to `http://127.0.0.1:5001/` in your browser.

## 📡 IoT Integration (Developer Guide)
The system exposes a RESTful API at `/api/predict` for IoT data ingestion. 
Example Payload:
```json
{
  "station_id": "STN_001",
  "temp": 25.4,
  "hum": 82.1,
  "press": 1012.5,
  "wind": 15.2,
  "btemp": 245.0,
  "reflect": 0.65,
  "radar": 45.2
}
```

---
Developed as a high-fidelity rainfall detection and early warning solution.
