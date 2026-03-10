import requests
import time
import random
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
BACKEND_URL = "http://127.0.0.1:5003/api/predict"
OWM_API_KEY = os.getenv("OWM_API_KEY")
USE_REAL_WEATHER = True # Set to True to fetch real-world data as base

STATIONS = [
    {"id": "STN_001", "name": "Central Hub", "lat": 12.97, "lon": 77.59, "profile": "urban"},
    {"id": "STN_002", "name": "North Ridge", "lat": 13.05, "lon": 77.62, "profile": "mountain"},
    {"id": "STN_003", "name": "West Valley", "lat": 12.92, "lon": 77.50, "profile": "valley"}
]

def get_real_weather(lat, lon):
    """Fetch real-world weather to use as a base for simulator."""
    if not OWM_API_KEY:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def generate_sensor_data(station):
    """Generate realistic weather data based on the station's location/profile."""
    profile = station.get("profile", "urban")
    
    # Attempt to get real-world base values
    real_data = None
    if USE_REAL_WEATHER:
        real_data = get_real_weather(station["lat"], station["lon"])
    
    if real_data:
        temp = real_data['main']['temp']
        hum = real_data['main']['humidity']
        press = real_data['main']['pressure']
        wind = real_data['wind']['speed']
        # Correlate radar/satellite with real description
        desc = real_data['weather'][0]['description'].lower()
        is_raining = 'rain' in desc or 'drizzle' in desc
        is_cloudy = 'cloud' in desc or 'overcast' in desc
        
        reflect = 0.7 if is_cloudy else (0.9 if is_raining else random.uniform(0.1, 0.4))
        radar = 45 if is_raining else (20 if is_cloudy else random.uniform(5, 15))
        btemp = 235 if is_cloudy else (225 if is_raining else random.uniform(250, 270))
    else:
        # Fallback to random base values
        temp = random.uniform(22, 32)
        hum = random.uniform(60, 95)
        press = random.uniform(1005, 1015)
        wind = random.uniform(5, 25)
        reflect = random.uniform(0.1, 0.9)
        radar = random.uniform(10, 65)
        btemp = random.uniform(230, 260)
    
    # Profile adjustments
    if profile == "mountain":
        temp -= 5 # Celerity cooling
        press -= 10 # Altitude drop
        wind += 10 # Ridge winds
    elif profile == "valley":
        hum += 10 # Moisture trap
        wind -= 5 # Shielded
        
    return {
        "station_id": station["id"],
        "temp": round(temp + random.uniform(-1, 1), 1),
        "hum": round(min(100, hum + random.uniform(-5, 5)), 1),
        "press": round(press + random.uniform(-1, 1), 1),
        "wind": round(wind + random.uniform(-2, 2), 1),
        "btemp": round(btemp + random.uniform(-5, 5), 1),
        "reflect": round(min(1.0, reflect + random.uniform(-0.05, 0.05)), 2),
        "radar": round(max(0, radar + random.uniform(-5, 5)), 1),
        "use_owm": True # Request backend to cross-reference OWM
    }

def run_simulator():
    print(f"--- RainAlert IoT Simulator Started ---")
    print(f"Target Backend: {BACKEND_URL}")
    print(f"Simulating {len(STATIONS)} stations...")
    
    try:
        while True:
            for stn in STATIONS:
                data = generate_sensor_data(stn)
                
                try:
                    response = requests.post(BACKEND_URL, json=data, timeout=30)
                    if response.status_code == 200:
                        res_json = response.json()
                        prob = res_json.get('probability', 0)
                        alert = " ALERT!" if res_json.get('alert') else ""
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {stn['name']} Sent -> Prob: {prob:.1%}{alert}")
                    else:
                        print(f"Error sending data for {stn['name']}: HTTP {response.status_code}")
                except requests.exceptions.ConnectionError:
                    print(f"Error: Could not connect to backend at {BACKEND_URL}. Is mobile_app.py running?")
                
                # Small delay between stations
                time.sleep(1)
                
            print("-" * 40)
            # Wait for the next reporting cycle
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nSimulator stopped by user.")

if __name__ == "__main__":
    run_simulator()
