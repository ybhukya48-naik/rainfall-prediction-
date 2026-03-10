import requests
import os
from dotenv import load_dotenv

load_dotenv()

# OpenWeatherMap Configuration
OWM_API_KEY = os.getenv("OWM_API_KEY", "YOUR_OPENWEATHER_API_KEY") # User will need to replace this

def get_live_weather(lat, lon):
    """Fetch real-time weather data from OpenWeatherMap."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'temp': data['main']['temp'],
                'hum': data['main']['humidity'],
                'press': data['main']['pressure'],
                'wind': data['wind']['speed'],
                'city': data['name'],
                'icon': data['weather'][0]['icon'],
                'desc': data['weather'][0]['description'],
                'success': True
            }
        else:
            print(f"OWM Error: {response.status_code}")
            return {'success': False, 'error': f"API Error: {response.status_code}"}
    except Exception as e:
        print(f"OWM Connection Error: {e}")
        return {'success': False, 'error': str(e)}

def get_live_forecast(lat, lon):
    """Fetch 5-day / 3-hour forecast for trend analysis."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None
